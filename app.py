# -*- coding: utf-8 -*-
"""
dept-faiss-backend — 共用 FAISS 向量資料庫服務

支援多 project 管理，提供純向量搜尋 API（BM25 路由由呼叫方負責），
以及管理端點（上傳文件、切分、建立索引）。

環境變數
--------
PROJECTS          project 清單，格式：default:index_map.json,icu:icu_index_map.json
EMBEDDING_MODEL   OpenAI embedding 模型（預設 text-embedding-3-small）
TOP_K             每個 SOP 回傳的 chunk 數（預設 5）
VECTOR_CACHE_MAX  每個 project 最多快取幾個 FAISS index（預設 50）
CHUNK_SIZE        文字切分大小（預設 500 字元）
CHUNK_OVERLAP     切分重疊（預設 100 字元）
MAX_UPLOAD_SIZE   上傳大小上限（bytes，預設 50MB）
API_TOKEN         若設定，受保護路由需 X-API-Key 或 Authorization: Bearer
ALLOWED_ORIGINS   CORS 白名單（逗號分隔）
DEBUG             1 = 啟用 debug logging
PORT              監聽埠（預設 80）
"""
import os
import re
import io
import json
import shutil
import hashlib
import functools
import hmac
import logging
import uuid
from typing import Dict, Any, List, Tuple, Optional
from time import perf_counter
from datetime import datetime

import pdfplumber
from dotenv import load_dotenv
from flask import Flask, request, jsonify, g, send_from_directory
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document as LCDocument

# =========================
# 基本設定
# =========================
load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FAISS_STORE_DIR = os.getenv("FAISS_STORE_DIR", "faiss_index_store")


def abspath_from_base(p: str) -> str:
    if not p:
        return p
    return p if os.path.isabs(p) else os.path.normpath(os.path.join(BASE_DIR, p))


# PROJECTS env: "default:index_map.json,icu:icu_index_map.json"
PROJECTS_RAW = os.getenv("PROJECTS", "default:index_map.json").strip()
_PROJECT_ID_RE = re.compile(r'^[a-zA-Z0-9_\-]{1,64}$')

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
TOP_K = int(os.getenv("TOP_K", "5"))
_VECTOR_CACHE_MAX = int(os.getenv("VECTOR_CACHE_MAX", "50"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
MAX_UPLOAD_SIZE = int(os.getenv("MAX_UPLOAD_SIZE", str(50 * 1024 * 1024)))
DEBUG = os.getenv("DEBUG", "0").strip() == "1"
API_TOKEN = os.getenv("API_TOKEN", "").strip()

_raw_origins = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()]
ALLOWED_ORIGINS = _raw_origins or ["http://localhost:3000", "http://localhost:5173", "http://localhost:8080"]

LOG_LEVEL = logging.DEBUG if DEBUG else logging.INFO
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("nurse-faiss-service")

embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)


# =========================
# Per-project state
# =========================
class ProjectState:
    def __init__(self):
        self.index_map: Dict[str, Any] = {}
        self.sop_titles: List[str] = []
        self.vector_cache: Dict[str, FAISS] = {}
        self.index_map_file: str = ""


_projects: Dict[str, ProjectState] = {}


# =========================
# FAISS Index Helpers
# =========================
def _load_index_map_file(fp: str) -> Dict[str, Any]:
    fp = abspath_from_base(fp)
    if not os.path.exists(fp):
        raise FileNotFoundError(f"找不到 index_map 檔案：{fp}")
    with open(fp, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"index_map 格式錯誤（需為 dict）：{fp}")
    return data


def _compute_faiss_index_sha256(index_dir: str) -> str:
    h = hashlib.sha256()
    for fn in sorted(os.listdir(index_dir)):
        fp = os.path.join(index_dir, fn)
        if not os.path.isfile(fp):
            continue
        with open(fp, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
    return h.hexdigest()


def _verify_faiss_index_sha256(index_dir: str, expected: str) -> None:
    actual = _compute_faiss_index_sha256(index_dir)
    if actual != expected:
        raise ValueError(
            f"FAISS index 完整性驗證失敗：{index_dir}\n"
            f"  預期 SHA256: {expected}\n"
            f"  實際 SHA256: {actual}"
        )
    logger.info("FAISS index SHA256 驗證通過：%s", index_dir)


def get_vectorstore_for_title(title: str, state: ProjectState) -> FAISS:
    if title in state.vector_cache:
        return state.vector_cache[title]

    info = state.index_map.get(title) or {}
    index_path = info.get("index_path")
    if not index_path:
        raise ValueError(f"index_map 中找不到 {title} 的 index_path")

    index_path = os.path.normpath(abspath_from_base(index_path.replace("\\", "/")))

    base_real = os.path.realpath(BASE_DIR)
    if not os.path.realpath(index_path).startswith(base_real + os.sep):
        raise ValueError(f"index_path is outside BASE_DIR: {index_path}")

    if not os.path.exists(index_path):
        raise FileNotFoundError(f"找不到 FAISS index：{index_path}")

    expected_sha256 = (info.get("sha256") or "").strip().lower()
    if expected_sha256:
        _verify_faiss_index_sha256(index_path, expected_sha256)
    else:
        logger.warning(
            "SECURITY: FAISS index '%s' 未設定 sha256 驗證，存在反序列化攻擊風險。",
            index_path,
        )

    vs = FAISS.load_local(
        index_path,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    if len(state.vector_cache) >= _VECTOR_CACHE_MAX:
        state.vector_cache.pop(next(iter(state.vector_cache)))
    state.vector_cache[title] = vs
    return vs


# =========================
# BM25 Routing
# =========================
def route_sop_bm25(
    original: str,
    optimized: str,
    state: ProjectState,
    top_n: int = ROUTE_TOP_N,
    min_score: float = 0.0,
) -> List[Tuple[str, float]]:
    if not state.bm25_router or not state.sop_titles:
        return []
    scores_orig = state.bm25_router.score(original)
    scores_opt = state.bm25_router.score(optimized)
    scores = [max(a, b) for a, b in zip(scores_orig, scores_opt)]
    pairs = list(zip(state.sop_titles, scores))
    pairs.sort(key=lambda x: x[1], reverse=True)
    top = pairs[:max(3, top_n)]
    filtered = [(t, s) for t, s in top if s > min_score]
    return filtered if filtered else top


# =========================
# Init / Project管理
# =========================
def _parse_projects() -> Dict[str, str]:
    """解析 PROJECTS env var → {project_id: index_map_file}。"""
    result: Dict[str, str] = {}
    for entry in PROJECTS_RAW.split(","):
        entry = entry.strip()
        if not entry:
            continue
        parts = entry.split(":", 1)
        if len(parts) != 2:
            logger.warning("PROJECTS 格式錯誤（應為 id:file）：%s", entry)
            continue
        pid, fp = parts[0].strip(), parts[1].strip()
        if not _PROJECT_ID_RE.match(pid):
            logger.warning("PROJECTS project_id 格式不合法：%s", pid)
            continue
        result[pid] = fp
    return result


def _init_project(project_id: str, index_map_file: str) -> ProjectState:
    state = ProjectState()
    state.index_map_file = abspath_from_base(index_map_file)

    try:
        state.index_map = _load_index_map_file(index_map_file)
    except FileNotFoundError:
        # 允許空 project（index_map 尚未建立）
        logger.warning("[%s] index_map 不存在，初始化為空 project", project_id)
        state.index_map = {}

    # 補算缺少 sha256 的索引
    sha_updated = 0
    for title, info in state.index_map.items():
        if (info.get("sha256") or "").strip():
            continue
        idx_path = info.get("index_path", "")
        idx_path = os.path.normpath(abspath_from_base(idx_path.replace("\\", "/")))
        if not os.path.isdir(idx_path):
            continue
        info["sha256"] = _compute_faiss_index_sha256(idx_path)
        sha_updated += 1
    if sha_updated:
        _save_index_map_state(state)
        logger.info("[%s] 補算 %d 筆 SHA256", project_id, sha_updated)

    state.sop_titles = list(state.index_map.keys())

    preloaded = 0
    for title in state.sop_titles:
        try:
            get_vectorstore_for_title(title, state)
            preloaded += 1
        except Exception:
            logger.warning("[%s] 預載 FAISS 索引失敗: %s", project_id, title, exc_info=True)

    logger.info("✅ [project=%s] %d SOP indexes loaded, %d preloaded", project_id, len(state.sop_titles), preloaded)
    return state


_state_initialized = False


def init_app_state():
    global _projects, _state_initialized
    for pid, fp in _parse_projects().items():
        try:
            _projects[pid] = _init_project(pid, fp)
        except Exception as e:
            logger.error("初始化 project '%s' 失敗: %s", pid, e, exc_info=True)
    logger.info("✅ 所有 projects 初始化完成：%s", list(_projects.keys()))
    _state_initialized = True


def ensure_state():
    global _state_initialized
    if not _state_initialized:
        init_app_state()


# =========================
# index_map 儲存
# =========================
def _save_index_map_state(state: ProjectState):
    """原子寫入 index_map.json。"""
    fp = state.index_map_file
    if not fp:
        return
    tmp = fp + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state.index_map, f, ensure_ascii=False, indent=2)
    os.replace(tmp, fp)


def _save_index_map(project_id: str):
    state = _projects.get(project_id)
    if state:
        _save_index_map_state(state)


# =========================
# 文字提取與切分
# =========================
def _make_safe_name(title: str) -> str:
    """由 SOP 標題產生唯一的 safe_name（用於目錄名）。"""
    suffix = hashlib.md5(title.encode("utf-8")).hexdigest()[:12]
    return f"sop_{suffix}"


def _find_title_by_sop_key(state: "ProjectState", sop_key: str) -> Optional[str]:
    """從 index_map 中查找 sop_key 對應的 title。"""
    for t, info in state.index_map.items():
        if (info.get("safe_name") or t) == sop_key:
            return t
    return None


# =========================
# Chunk / Source 管理
# =========================

def _load_chunks(index_dir: str) -> List[Dict]:
    """載入 chunks.json；不存在時回傳空清單。"""
    fp = os.path.join(index_dir, "chunks.json")
    if not os.path.exists(fp):
        return []
    with open(fp, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_chunks(index_dir: str, chunks: List[Dict]) -> None:
    """原子寫入 chunks.json。"""
    fp = os.path.join(index_dir, "chunks.json")
    tmp = fp + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    os.replace(tmp, fp)


def _load_source(index_dir: str) -> Dict:
    """載入 source.json；不存在時回傳空 dict。"""
    fp = os.path.join(index_dir, "source.json")
    if not os.path.exists(fp):
        return {}
    with open(fp, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_source(index_dir: str, source: Dict) -> None:
    """原子寫入 source.json。"""
    fp = os.path.join(index_dir, "source.json")
    tmp = fp + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(source, f, ensure_ascii=False, indent=2)
    os.replace(tmp, fp)


def _source_to_chunks(
    source_data: Dict,
    title: str,
    chunk_size: int,
    chunk_overlap: int,
) -> Tuple[List[LCDocument], List[Dict]]:
    """從 source_data 切分，回傳 (LCDocument list, chunk_dicts list)。"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", " ", ""],
    )
    lc_docs: List[LCDocument] = []
    chunk_dicts: List[Dict] = []

    src_type = source_data.get("type")
    if src_type == "pdf":
        for page_info in source_data.get("pages", []):
            page_num = page_info["page"]
            text = page_info.get("text", "")
            for c in splitter.split_text(text):
                meta = {"page": page_num, "sop_title": title}
                lc_docs.append(LCDocument(page_content=c, metadata=meta))
                chunk_dicts.append({
                    "id": str(len(chunk_dicts)),
                    "content": c,
                    "page": page_num,
                    "metadata": meta,
                })
    else:
        text = source_data.get("text", "")
        for c in splitter.split_text(text):
            meta = {"sop_title": title}
            lc_docs.append(LCDocument(page_content=c, metadata=meta))
            chunk_dicts.append({
                "id": str(len(chunk_dicts)),
                "content": c,
                "page": None,
                "metadata": meta,
            })
    return lc_docs, chunk_dicts


def _rebuild_faiss_from_chunk_dicts(
    state: "ProjectState",
    title: str,
    index_dir: str,
    chunk_dicts: List[Dict],
) -> "FAISS":
    """從 chunk_dicts 重建 FAISS index，清除快取，回傳新 vectorstore。"""
    docs = [
        LCDocument(page_content=c["content"], metadata=c.get("metadata") or {})
        for c in chunk_dicts
    ]
    if os.path.exists(index_dir):
        shutil.rmtree(index_dir)
    os.makedirs(index_dir)
    vs = FAISS.from_documents(docs, embeddings)
    vs.save_local(index_dir)
    state.vector_cache.pop(title, None)
    return vs


# =========================
# Flask App
# =========================
app = Flask(__name__, static_folder="static", static_url_path="/static")
CORS(app, resources={r"/api/*": {"origins": ALLOWED_ORIGINS}})

limiter = Limiter(
    key_func=get_remote_address,
    app=app,
    default_limits=[],
    storage_uri="memory://",
)


@app.before_request
def _before():
    g.request_id = str(uuid.uuid4())[:8]
    g.start_time = perf_counter()


@app.after_request
def _after(response):
    duration_ms = round((perf_counter() - g.start_time) * 1000, 1) if hasattr(g, "start_time") else 0
    logger.info(
        "method=%s path=%s status=%s duration_ms=%s rid=%s",
        request.method, request.path, response.status_code,
        duration_ms, g.get("request_id", "-"),
    )
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "SAMEORIGIN")
    return response


# =========================
# API Key 認證
# =========================
def require_api_key(f):
    @functools.wraps(f)
    def decorated(*args, **kwargs):
        if not API_TOKEN:
            return f(*args, **kwargs)
        auth_header = request.headers.get("Authorization", "")
        token = ""
        if auth_header.startswith("Bearer "):
            token = auth_header[len("Bearer "):].strip()
        if not token:
            token = request.headers.get("X-API-Key", "").strip()
        if not token or not hmac.compare_digest(token, API_TOKEN):
            return jsonify({"error": "API 金鑰無效或缺少"}), 401
        return f(*args, **kwargs)
    return decorated


# =========================
# 前端靜態檔案
# =========================
@app.get("/")
@app.get("/admin")
@app.get("/admin/")
def serve_frontend():
    """回傳管理前端（需 static/index.html 存在）。"""
    static_dir = os.path.join(BASE_DIR, "static")
    if os.path.exists(os.path.join(static_dir, "index.html")):
        return send_from_directory(static_dir, "index.html")
    return jsonify({"message": "nurse-faiss-service is running. Frontend not deployed yet."}), 200


@app.get("/admin/<path:filename>")
def serve_frontend_assets(filename):
    return send_from_directory(os.path.join(BASE_DIR, "static"), filename)


# =========================
# Search Routes
# =========================
@app.get("/api/health")
def health():
    ensure_state()
    return jsonify({
        "status": "ok",
        "projects": list(_projects.keys()),
        "time": datetime.utcnow().isoformat(),
    })


@app.get("/api/projects")
def list_projects():
    """列出所有 projects 及其 SOP 清單（含 sop_key）。"""
    ensure_state()
    result: Dict[str, Any] = {}
    for pid, state in _projects.items():
        sops = {}
        for title in state.sop_titles:
            info = state.index_map.get(title) or {}
            sops[title] = {
                "sop_key": (info.get("safe_name") or title),
                "num_chunks": info.get("num_chunks", 0),
            }
        result[pid] = {
            "sop_count": len(state.sop_titles),
            "sops": sops,
        }
    return jsonify({"projects": result})


@app.post("/api/projects/<project_id>/search")
@require_api_key
@limiter.limit("60 per minute")
def search(project_id: str):
    """
    向量搜尋（BM25 路由由呼叫方負責）。

    Request body:
        query     (str, required)
        sop_keys  (list[str], optional) 指定搜尋範圍；省略時搜尋 project 所有 SOP
        top_k     (int, optional)
    """
    ensure_state()

    state = _projects.get(project_id)
    if state is None:
        return jsonify({"error": f"project '{project_id}' 不存在，可用：{list(_projects.keys())}"}), 404

    body = request.get_json(silent=True) or {}
    query: str = (body.get("query") or "").strip()
    top_k: int = max(1, min(int(body.get("top_k", TOP_K)), 20))
    requested_keys: List[str] = body.get("sop_keys") or []

    if not query:
        return jsonify({"error": "query 不得為空"}), 400

    # 決定要搜尋的 SOP title 清單
    if requested_keys:
        key_to_title = {
            (info.get("safe_name") or t): t
            for t, info in state.index_map.items()
        }
        search_titles = [key_to_title[k] for k in requested_keys if k in key_to_title]
        if not search_titles:
            return jsonify({"error": "指定的 sop_keys 均不存在於此 project"}), 404
    else:
        search_titles = list(state.sop_titles)

    if not search_titles:
        return jsonify({"error": "此 project 尚無任何 SOP"}), 404

    t1 = perf_counter()
    try:
        query_vector = embeddings.embed_query(query)
    except Exception as e:
        logger.error("embed_query failed: %s", e, exc_info=True)
        return jsonify({"error": "向量嵌入失敗，請稍後再試"}), 500
    embed_ms = round((perf_counter() - t1) * 1000, 2)

    t2 = perf_counter()
    merged: List[Tuple[Any, float, str]] = []
    for sop_title in search_titles:
        try:
            vs = get_vectorstore_for_title(sop_title, state)
        except Exception:
            logger.warning("載入 FAISS index 失敗：%s", sop_title, exc_info=True)
            continue
        for doc, score in vs.similarity_search_with_score_by_vector(query_vector, k=top_k):
            merged.append((doc, score, sop_title))
    search_ms = round((perf_counter() - t2) * 1000, 2)

    merged.sort(key=lambda x: x[1])
    top = merged[:top_k]

    chunks = []
    for doc, score, sop_title in top:
        info = state.index_map.get(sop_title) or {}
        chunks.append({
            "content": doc.page_content,
            "sop_title": sop_title,
            "sop_key": (info.get("safe_name") or sop_title),
            "score": float(score),
            "page": doc.metadata.get("page"),
            "metadata": dict(doc.metadata),
        })

    logger.info(
        "search project=%s embed_ms=%s search_ms=%s sops=%d chunks=%d",
        project_id, embed_ms, search_ms, len(search_titles), len(chunks),
    )

    return jsonify({
        "chunks": chunks,
        "timings": {"embed_ms": embed_ms, "search_ms": search_ms},
    })


# =========================
# Admin Routes
# =========================
@app.post("/api/admin/projects")
@require_api_key
def create_project():
    """建立新 project（自動建立空的 index_map.json）。"""
    ensure_state()
    body = request.get_json(silent=True) or {}
    project_id = (body.get("project_id") or "").strip()

    if not _PROJECT_ID_RE.match(project_id):
        return jsonify({"error": "project_id 只能包含英數字、底線、連字號，長度 1-64"}), 400
    if project_id in _projects:
        return jsonify({"error": f"project '{project_id}' 已存在"}), 409

    index_map_file = abspath_from_base(f"{project_id}_index_map.json")
    if os.path.exists(index_map_file):
        return jsonify({"error": f"index_map 檔案已存在：{index_map_file}"}), 409

    with open(index_map_file, "w", encoding="utf-8") as f:
        json.dump({}, f)

    state = ProjectState()
    state.index_map = {}
    state.index_map_file = index_map_file
    state.sop_titles = []
    _projects[project_id] = state

    logger.info("建立新 project: %s", project_id)
    return jsonify({"ok": True, "project_id": project_id}), 201


@app.delete("/api/admin/projects/<project_id>")
@require_api_key
def delete_project(project_id: str):
    """刪除 project（同時刪除 index_map.json 及所有 FAISS index）。"""
    ensure_state()

    if project_id == "default":
        return jsonify({"error": "不得刪除 default project"}), 400

    state = _projects.get(project_id)
    if not state:
        return jsonify({"error": f"project '{project_id}' 不存在"}), 404

    # 刪除所有 FAISS index 目錄
    for title, info in state.index_map.items():
        idx_path = info.get("index_path", "")
        if idx_path:
            idx_path = abspath_from_base(idx_path)
            if os.path.isdir(idx_path):
                shutil.rmtree(idx_path, ignore_errors=True)
                logger.info("已刪除 FAISS index 目錄：%s", idx_path)

    # 刪除 index_map 檔案
    if state.index_map_file and os.path.exists(state.index_map_file):
        os.remove(state.index_map_file)

    del _projects[project_id]
    logger.info("已刪除 project: %s", project_id)
    return jsonify({"ok": True})


@app.post("/api/admin/projects/<project_id>/upload")
@require_api_key
@limiter.limit("20 per minute")
def upload_sop(project_id: str):
    """
    上傳 PDF 或文字檔，自動切分並建立 FAISS 向量索引。

    multipart/form-data 欄位：
        file          (required) PDF / TXT / MD 檔案
        title         (optional) SOP 標題，預設為檔名
        chunk_size    (optional) 切分大小，預設 CHUNK_SIZE
        chunk_overlap (optional) 重疊字元數，預設 CHUNK_OVERLAP
        overwrite     (optional) "true" = 覆蓋現有 SOP
    """
    ensure_state()

    state = _projects.get(project_id)
    if state is None:
        return jsonify({"error": f"project '{project_id}' 不存在"}), 404

    if "file" not in request.files:
        return jsonify({"error": "缺少 file 欄位"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "檔名為空"}), 400

    title = (request.form.get("title") or "").strip()
    if not title:
        title = os.path.splitext(file.filename)[0]

    chunk_size = max(100, int(request.form.get("chunk_size", CHUNK_SIZE)))
    chunk_overlap = max(0, int(request.form.get("chunk_overlap", CHUNK_OVERLAP)))
    overwrite = request.form.get("overwrite", "").lower() in ("1", "true", "yes")

    file_bytes = file.read()
    if len(file_bytes) > MAX_UPLOAD_SIZE:
        return jsonify({"error": f"檔案超過大小限制（最大 {MAX_UPLOAD_SIZE // 1024 // 1024}MB）"}), 413
    if not file_bytes:
        return jsonify({"error": "檔案內容為空"}), 400

    # 檢查是否已存在
    if title in state.index_map and not overwrite:
        return jsonify({"error": f"SOP「{title}」已存在，請勾選覆蓋或先刪除"}), 409

    # 提取原始文字並切分
    ext = os.path.splitext(file.filename)[1].lower()
    try:
        if ext == ".pdf":
            source_data: Dict = {"type": "pdf", "pages": []}
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    text = page.extract_text() or ""
                    if text.strip():
                        source_data["pages"].append({"page": page_num, "text": text})
        elif ext in (".txt", ".md"):
            source_data = {"type": "text", "text": file_bytes.decode("utf-8", errors="ignore")}
        else:
            return jsonify({"error": "僅支援 PDF、TXT、MD 格式"}), 400

        docs, chunk_dicts = _source_to_chunks(source_data, title, chunk_size, chunk_overlap)
    except Exception as e:
        logger.exception("文字提取失敗 project=%s title=%s", project_id, title)
        return jsonify({"error": f"文字提取失敗：{e}"}), 500

    if not docs:
        return jsonify({"error": "無法從檔案中提取文字（可能為掃描版 PDF 或空白檔案）"}), 400

    # 決定 safe_name（覆蓋時沿用原有）
    if title in state.index_map and overwrite:
        safe_name = state.index_map[title].get("safe_name") or _make_safe_name(title)
    else:
        safe_name = _make_safe_name(title)

    # 建立 FAISS index
    store_dir = abspath_from_base(FAISS_STORE_DIR)
    index_dir = os.path.join(store_dir, safe_name)
    try:
        os.makedirs(store_dir, exist_ok=True)
        if os.path.exists(index_dir):
            shutil.rmtree(index_dir)
        os.makedirs(index_dir)
        vs = FAISS.from_documents(docs, embeddings)
        vs.save_local(index_dir)
        _save_source(index_dir, source_data)
        _save_chunks(index_dir, chunk_dicts)
    except Exception as e:
        logger.exception("建立 FAISS index 失敗 project=%s title=%s", project_id, title)
        return jsonify({"error": f"建立向量索引失敗：{e}"}), 500

    sha256 = _compute_faiss_index_sha256(index_dir)
    rel_path = os.path.relpath(index_dir, BASE_DIR)

    # 更新 state
    state.index_map[title] = {
        "safe_name": safe_name,
        "index_path": rel_path,
        "num_chunks": len(chunk_dicts),
        "sha256": sha256,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
    }
    state.sop_titles = list(state.index_map.keys())
    state.vector_cache[title] = vs

    # 寫回 index_map.json
    _save_index_map(project_id)

    logger.info("上傳完成 project=%s title=%s sop_key=%s chunks=%d", project_id, title, safe_name, len(docs))
    return jsonify({
        "ok": True,
        "title": title,
        "sop_key": safe_name,
        "num_chunks": len(docs),
    })


@app.get("/api/projects/<project_id>/sops/<sop_key>/chunks")
def get_sop_chunks(project_id: str, sop_key: str):
    """列出指定 SOP 的所有 chunks（支援 offset/limit 分頁）。"""
    ensure_state()
    state = _projects.get(project_id)
    if not state:
        return jsonify({"error": f"project '{project_id}' 不存在"}), 404

    title = _find_title_by_sop_key(state, sop_key)
    if not title:
        return jsonify({"error": f"SOP '{sop_key}' 不存在"}), 404

    info = state.index_map[title]
    index_dir = abspath_from_base(info.get("index_path", ""))
    all_chunks = _load_chunks(index_dir)

    offset = max(0, int(request.args.get("offset", 0)))
    limit = max(1, min(int(request.args.get("limit", 200)), 500))
    page_chunks = all_chunks[offset: offset + limit]

    return jsonify({
        "project_id": project_id,
        "sop_key": sop_key,
        "title": title,
        "total": len(all_chunks),
        "offset": offset,
        "limit": limit,
        "chunk_size": info.get("chunk_size", CHUNK_SIZE),
        "chunk_overlap": info.get("chunk_overlap", CHUNK_OVERLAP),
        "chunks": page_chunks,
    })


@app.delete("/api/admin/projects/<project_id>/sops/<sop_key>/chunks/<chunk_id>")
@require_api_key
def delete_chunk(project_id: str, sop_key: str, chunk_id: str):
    """刪除單一 chunk 並重建 FAISS index。"""
    ensure_state()
    state = _projects.get(project_id)
    if not state:
        return jsonify({"error": f"project '{project_id}' 不存在"}), 404

    title = _find_title_by_sop_key(state, sop_key)
    if not title:
        return jsonify({"error": f"SOP '{sop_key}' 不存在"}), 404

    info = state.index_map[title]
    index_dir = abspath_from_base(info.get("index_path", ""))
    chunks = _load_chunks(index_dir)

    new_chunks = [c for c in chunks if c["id"] != chunk_id]
    if len(new_chunks) == len(chunks):
        return jsonify({"error": f"chunk '{chunk_id}' 不存在"}), 404
    if not new_chunks:
        return jsonify({"error": "無法刪除最後一個 chunk，請直接刪除整個 SOP"}), 400

    try:
        _rebuild_faiss_from_chunk_dicts(state, title, index_dir, new_chunks)
    except Exception as e:
        logger.exception("重建 FAISS index 失敗 project=%s sop_key=%s", project_id, sop_key)
        return jsonify({"error": f"重建索引失敗：{e}"}), 500

    # 重新指派連續 id
    for i, c in enumerate(new_chunks):
        c["id"] = str(i)
    _save_chunks(index_dir, new_chunks)

    sha256 = _compute_faiss_index_sha256(index_dir)
    info["num_chunks"] = len(new_chunks)
    info["sha256"] = sha256
    _save_index_map(project_id)

    logger.info("刪除 chunk project=%s sop_key=%s chunk_id=%s 剩餘=%d", project_id, sop_key, chunk_id, len(new_chunks))
    return jsonify({"ok": True, "num_chunks": len(new_chunks)})


@app.post("/api/admin/projects/<project_id>/sops/<sop_key>/rechunk")
@require_api_key
def rechunk_sop(project_id: str, sop_key: str):
    """以新切分參數重建 FAISS index（需有 source.json）。"""
    ensure_state()
    state = _projects.get(project_id)
    if not state:
        return jsonify({"error": f"project '{project_id}' 不存在"}), 404

    title = _find_title_by_sop_key(state, sop_key)
    if not title:
        return jsonify({"error": f"SOP '{sop_key}' 不存在"}), 404

    info = state.index_map[title]
    index_dir = abspath_from_base(info.get("index_path", ""))
    source_data = _load_source(index_dir)
    if not source_data:
        return jsonify({"error": "找不到原始文字（source.json），請重新上傳文件"}), 404

    body = request.get_json(silent=True) or {}
    chunk_size = max(100, int(body.get("chunk_size", info.get("chunk_size", CHUNK_SIZE))))
    chunk_overlap = max(0, int(body.get("chunk_overlap", info.get("chunk_overlap", CHUNK_OVERLAP))))
    if chunk_overlap >= chunk_size:
        return jsonify({"error": "chunk_overlap 必須小於 chunk_size"}), 400

    try:
        docs, chunk_dicts = _source_to_chunks(source_data, title, chunk_size, chunk_overlap)
    except Exception as e:
        return jsonify({"error": f"切分失敗：{e}"}), 500

    if not chunk_dicts:
        return jsonify({"error": "切分結果為空"}), 400

    try:
        _rebuild_faiss_from_chunk_dicts(state, title, index_dir, chunk_dicts)
    except Exception as e:
        logger.exception("重建 FAISS index 失敗 project=%s sop_key=%s", project_id, sop_key)
        return jsonify({"error": f"重建索引失敗：{e}"}), 500

    _save_chunks(index_dir, chunk_dicts)
    _save_source(index_dir, source_data)

    sha256 = _compute_faiss_index_sha256(index_dir)
    info["num_chunks"] = len(chunk_dicts)
    info["sha256"] = sha256
    info["chunk_size"] = chunk_size
    info["chunk_overlap"] = chunk_overlap
    _save_index_map(project_id)

    logger.info(
        "重新切分 project=%s sop_key=%s chunk_size=%d overlap=%d chunks=%d",
        project_id, sop_key, chunk_size, chunk_overlap, len(chunk_dicts),
    )
    return jsonify({
        "ok": True,
        "num_chunks": len(chunk_dicts),
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
    })


@app.delete("/api/admin/projects/<project_id>/sops/<sop_key>")
@require_api_key
def delete_sop(project_id: str, sop_key: str):
    """刪除指定 SOP 及其 FAISS index。"""
    ensure_state()

    state = _projects.get(project_id)
    if state is None:
        return jsonify({"error": f"project '{project_id}' 不存在"}), 404

    title = _find_title_by_sop_key(state, sop_key)
    if title is None:
        return jsonify({"error": f"SOP '{sop_key}' 不存在"}), 404

    info = state.index_map[title]

    # 刪除 FAISS index 目錄
    idx_path = info.get("index_path", "")
    if idx_path:
        idx_path = abspath_from_base(idx_path)
        if os.path.isdir(idx_path):
            shutil.rmtree(idx_path, ignore_errors=True)

    # 移除 state
    del state.index_map[title]
    state.vector_cache.pop(title, None)
    state.sop_titles = list(state.index_map.keys())

    _save_index_map(project_id)

    logger.info("刪除 SOP project=%s title=%s sop_key=%s", project_id, title, sop_key)
    return jsonify({"ok": True, "deleted_title": title})


if __name__ == "__main__":
    ensure_state()
    app.run(
        host="0.0.0.0",
        port=int(os.getenv("PORT", "80")),
        debug=DEBUG,
    )
