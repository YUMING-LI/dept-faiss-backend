# -*- coding: utf-8 -*-
"""
ihd-faiss-backend — 共用 FAISS 向量資料庫服務

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
import jwt as _jwt
from typing import Dict, Any, List, Tuple, Optional
from time import perf_counter
from datetime import datetime, timezone

import pdfplumber
from pg_store import pg as _pg
from dotenv import load_dotenv
from flask import Flask, request, jsonify, g, send_from_directory
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from openai import OpenAI as _OpenAI
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
CHUNK_MODEL = os.getenv("CHUNK_MODEL", "gpt-5.4-mini")
TOP_K = int(os.getenv("TOP_K", "5"))
_VECTOR_CACHE_MAX = int(os.getenv("VECTOR_CACHE_MAX", "50"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
MAX_UPLOAD_SIZE = int(os.getenv("MAX_UPLOAD_SIZE", str(50 * 1024 * 1024)))
DEBUG = os.getenv("DEBUG", "0").strip() == "1"
API_TOKEN = os.getenv("API_TOKEN", "").strip()
REQUIRE_API_TOKEN = os.getenv("REQUIRE_API_TOKEN", "0").strip() == "1"
if REQUIRE_API_TOKEN and not API_TOKEN:
    raise RuntimeError("REQUIRE_API_TOKEN=1 但 API_TOKEN 未設定，拒絕啟動")

_raw_origins = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()]
ALLOWED_ORIGINS = _raw_origins or ["http://localhost:3000", "http://localhost:5173", "http://localhost:8080"]

LOG_LEVEL = logging.DEBUG if DEBUG else logging.INFO
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("ihd-faiss-service")

# PostgreSQL 持久化
_PG_ENABLED = os.getenv("PG_STORE_ENABLED", "1").strip() == "1"
if _PG_ENABLED:
    _pg.init(
        host=os.getenv("DB_HOST", "10.1.207.19"),
        port=int(os.getenv("DB_PORT", "5432")),
        dbname=os.getenv("DB_NAME", "edah_sh"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD", ""),
        sslmode=os.getenv("DB_SSLMODE", "disable"),
    )
else:
    logger.warning("PG_STORE_ENABLED=0，FAISS index 僅存在容器檔案系統（重建後遺失）")

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


def get_vectorstore_for_title(title: str, state: ProjectState, skip_verify: bool = False) -> FAISS:
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

    if not skip_verify:
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
    failed: List[str] = []
    for title in state.sop_titles:
        try:
            # 不 skip_verify：剛在上方補算/載入 sha256，預載時驗證可在啟動階段
            # 即時偵測索引被竄改（pickle 反序列化攻擊面），避免 runtime 才爆。
            get_vectorstore_for_title(title, state, skip_verify=False)
            preloaded += 1
        except Exception:
            failed.append(title)
            logger.error("[%s] 預載 FAISS 索引失敗: %s", project_id, title, exc_info=True)

    if failed:
        logger.error(
            "[%s] %d / %d FAISS 索引預載失敗（這些 SOP 在 runtime 將無法回應）：%s",
            project_id, len(failed), len(state.sop_titles), failed,
        )

    logger.info("✅ [project=%s] %d SOP indexes loaded, %d preloaded", project_id, len(state.sop_titles), preloaded)
    return state


_state_initialized = False

_DYNAMIC_PROJECTS_FILE = os.path.join(BASE_DIR, "dynamic_projects.json")


def _load_dynamic_projects() -> Dict[str, str]:
    """讀取 runtime 建立的 project 清單（{project_id: index_map_file}）。"""
    if not os.path.exists(_DYNAMIC_PROJECTS_FILE):
        return {}
    try:
        with open(_DYNAMIC_PROJECTS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        logger.warning("無法讀取 dynamic_projects.json，略過")
        return {}


def _save_dynamic_projects(data: Dict[str, str]) -> None:
    tmp = _DYNAMIC_PROJECTS_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, _DYNAMIC_PROJECTS_FILE)


def _restore_from_pg():
    """從 PG 還原所有 FAISS index 到檔案系統。"""
    if not _PG_ENABLED:
        return
    try:
        rows = _pg.load_all_sops()
    except Exception:
        logger.error("PG load_all_sops 失敗，略過還原", exc_info=True)
        return

    store_dir = abspath_from_base(FAISS_STORE_DIR)
    os.makedirs(store_dir, exist_ok=True)
    restored = 0
    for row in rows:
        pid = row["project_id"]
        title = row["title"]
        safe_name = row["safe_name"]
        index_dir = os.path.join(store_dir, safe_name)
        if os.path.exists(os.path.join(index_dir, "index.faiss")):
            continue
        try:
            _pg.restore_sop_to_dir(pid, title, index_dir)
            restored += 1
        except Exception:
            logger.error("PG 還原失敗 project=%s title=%s", pid, title, exc_info=True)
    if restored:
        logger.info("從 PG 還原 %d 筆 FAISS index 到檔案系統", restored)


def init_app_state():
    global _projects, _state_initialized

    _restore_from_pg()

    all_projects = {**_parse_projects(), **_load_dynamic_projects()}
    success = 0
    for pid, fp in all_projects.items():
        try:
            _projects[pid] = _init_project(pid, fp)
            success += 1
        except Exception as e:
            logger.error("初始化 project '%s' 失敗: %s", pid, e, exc_info=True)
    logger.info("✅ 所有 projects 初始化完成：%s", list(_projects.keys()))
    if success > 0 or not all_projects:
        _state_initialized = True
    else:
        logger.error("所有 projects 初始化失敗，下次請求將重新嘗試")


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


_GPT_CHUNK_SYSTEM = """\
你是一位醫療 SOP 文件切分專家。
請將使用者提供的 SOP 全文切分成多個語意完整的 chunk，每個 chunk 須符合：
1. 涵蓋一個完整概念、程序步驟或段落（例如：目的、適應症、準備事項、執行步驟、注意事項）
2. 可獨立作為知識庫的一筆查詢條目，不依賴前後文即可理解
3. 保留原始條列符號、數字序號與表格結構
4. 長度以 100–600 字為宜；若段落本身較短，可合併至相鄰段落

僅回傳 JSON 陣列，每個元素為一個 chunk 的純文字內容，不要加任何說明文字：
["chunk 1 內容", "chunk 2 內容", ...]
"""


def _source_to_chunks_gpt(
    source_data: Dict,
    title: str,
) -> Tuple[List[LCDocument], List[Dict]]:
    """用 GPT 語意切分，失敗時 raise RuntimeError。"""
    # 組合全文（PDF 依頁次拼接，保留頁碼資訊）
    if source_data.get("type") == "pdf":
        pages = source_data.get("pages", [])
        full_text = "\n\n".join(
            f"[第 {p['page']} 頁]\n{p['text']}" for p in pages if p.get("text", "").strip()
        )
        page_map = {p["page"]: p["text"] for p in pages}
    else:
        full_text = source_data.get("text", "")
        page_map = {}

    if not full_text.strip():
        raise RuntimeError("文件內容為空，無法切分")

    client = _OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
    response = client.chat.completions.create(
        model=CHUNK_MODEL,
        messages=[
            {"role": "system", "content": _GPT_CHUNK_SYSTEM},
            {"role": "user", "content": f"SOP 標題：{title}\n\n{full_text}"},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content or ""
    # GPT 回傳 {"chunks": [...]} 或直接 [...]
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"GPT 回傳非合法 JSON：{e}\n原始內容：{raw[:200]}")

    if isinstance(parsed, list):
        texts = parsed
    elif isinstance(parsed, dict):
        # 取第一個 list 值
        texts = next((v for v in parsed.values() if isinstance(v, list)), None)
        if texts is None:
            raise RuntimeError(f"GPT 回傳 JSON 結構不符預期：{raw[:200]}")
    else:
        raise RuntimeError(f"GPT 回傳型別不符：{type(parsed)}")

    texts = [str(t).strip() for t in texts if str(t).strip()]
    if not texts:
        raise RuntimeError("GPT 切分結果為空")

    lc_docs: List[LCDocument] = []
    chunk_dicts: List[Dict] = []
    for text in texts:
        # 嘗試找最接近的頁碼（在 PDF 模式下）
        page = None
        if page_map:
            best_overlap = 0
            for pg, pg_text in page_map.items():
                overlap = sum(1 for c in text if c in pg_text)
                if overlap > best_overlap:
                    best_overlap, page = overlap, pg

        meta = {"sop_title": title, "page": page, "chunker": "gpt"}
        lc_docs.append(LCDocument(page_content=text, metadata=meta))
        chunk_dicts.append({
            "id": str(len(chunk_dicts)),
            "content": text,
            "page": page,
            "metadata": meta,
        })

    return lc_docs, chunk_dicts


def _rebuild_faiss_from_chunk_dicts(
    state: "ProjectState",
    title: str,
    index_dir: str,
    chunk_dicts: List[Dict],
) -> "FAISS":
    """從 chunk_dicts 重建 FAISS index，清除快取，回傳新 vectorstore。
    使用 tmp 目錄確保失敗時不破壞現有 index。
    """
    docs = [
        LCDocument(page_content=c["content"], metadata=c.get("metadata") or {})
        for c in chunk_dicts
    ]
    tmp_dir = index_dir + ".tmp"
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir)
    try:
        vs = FAISS.from_documents(docs, embeddings)
        vs.save_local(tmp_dir)
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    if os.path.exists(index_dir):
        shutil.rmtree(index_dir)
    os.rename(tmp_dir, index_dir)
    state.vector_cache.pop(title, None)
    return vs


# =========================
# Flask App
# =========================
app = Flask(__name__, static_folder="static", static_url_path="/static")
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_SIZE
CORS(app, supports_credentials=True, resources={r"/api/*": {"origins": ALLOWED_ORIGINS}})

from routes.auth import bp as _auth_bp
app.register_blueprint(_auth_bp)

_REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379")
_REQUIRE_REDIS = os.getenv("RATE_LIMIT_REQUIRE_REDIS", "0").strip() == "1"
_rate_limit_storage = "memory://"
try:
    import redis as _redis_mod
    _redis_mod.from_url(_REDIS_URL).ping()
    _rate_limit_storage = _REDIS_URL
    logger.info("Rate limiter 使用 Redis: %s", _REDIS_URL)
except Exception as _redis_err:
    if _REQUIRE_REDIS:
        raise RuntimeError(
            f"RATE_LIMIT_REQUIRE_REDIS=1 但無法連線 Redis ({_REDIS_URL}): {_redis_err}"
        )
    logger.warning("Redis 連線失敗，rate limiter 改用 memory://（多 worker 下計數不準確）")

limiter = Limiter(
    key_func=get_remote_address,
    app=app,
    default_limits=[],
    storage_uri=_rate_limit_storage,
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
def _verify_jwt(token: str) -> bool:
    """回傳 True 若 token 是有效的 manager JWT。"""
    secret = os.getenv("JWT_SECRET", "").strip()
    if not secret:
        return False
    try:
        payload = _jwt.decode(token, secret, algorithms=["HS256"])
        return payload.get("project") == "ihd-faiss" and payload.get("permission") == "manager"
    except Exception:
        return False


def require_api_key(f):
    @functools.wraps(f)
    def decorated(*args, **kwargs):
        if not API_TOKEN:
            return f(*args, **kwargs)
        auth_header = request.headers.get("Authorization", "")
        bearer = ""
        if auth_header.startswith("Bearer "):
            bearer = auth_header[len("Bearer "):].strip()
        api_key = request.headers.get("X-API-Key", "").strip()

        # Server-to-server: X-API-Key 或 Bearer == API_TOKEN
        check_token = bearer or api_key
        if check_token and hmac.compare_digest(check_token, API_TOKEN):
            return f(*args, **kwargs)

        # 前端登入：Bearer JWT
        if bearer and _verify_jwt(bearer):
            return f(*args, **kwargs)

        return jsonify({"error": "API 金鑰無效或缺少"}), 401
    return decorated


# =========================
# 前端靜態檔案
# =========================
@app.get("/")
@app.get("/ihd-faiss")
@app.get("/ihd-faiss/")
def serve_frontend():
    """回傳管理前端（本地開發用；生產環境由 ihd-faiss-fronted 容器提供）。"""
    static_dir = os.path.join(BASE_DIR, "static")
    if os.path.exists(os.path.join(static_dir, "index.html")):
        return send_from_directory(static_dir, "index.html")
    return jsonify({"message": "ihd-faiss-backend is running. Frontend not deployed yet."}), 200


@app.get("/ihd-faiss/<path:filename>")
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
        "time": datetime.now(timezone.utc).isoformat(),
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

    # 寫入 dynamic_projects.json，確保重啟後仍可載入
    dynamic = _load_dynamic_projects()
    dynamic[project_id] = f"{project_id}_index_map.json"
    _save_dynamic_projects(dynamic)

    if _PG_ENABLED:
        try:
            _pg.save_project(project_id, f"{project_id}_index_map.json", is_dynamic=True)
        except Exception:
            logger.error("PG save_project 失敗", exc_info=True)

    logger.info("建立新 project: %s", project_id)
    return jsonify({"ok": True, "project_id": project_id}), 201


@app.delete("/api/admin/projects/<project_id>")
@require_api_key
def delete_project(project_id: str):
    """刪除 project（同時刪除 index_map.json 及所有 FAISS index）。"""
    ensure_state()

    if project_id == "nurse-healthhub":
        return jsonify({"error": "不得刪除 nurse-healthhub project"}), 400

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

    # 從 dynamic_projects.json 移除
    dynamic = _load_dynamic_projects()
    if project_id in dynamic:
        del dynamic[project_id]
        _save_dynamic_projects(dynamic)

    if _PG_ENABLED:
        try:
            _pg.delete_project(project_id)
        except Exception:
            logger.error("PG delete_project 失敗", exc_info=True)

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
    use_gpt_chunker = request.form.get("chunker", "gpt").lower() != "char"

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

        if use_gpt_chunker:
            try:
                docs, chunk_dicts = _source_to_chunks_gpt(source_data, title)
                logger.info("GPT 語意切分完成 project=%s title=%s chunks=%d", project_id, title, len(chunk_dicts))
            except Exception as gpt_err:
                _err_str = str(gpt_err).lower()
                if any(k in _err_str for k in ("does not exist", "invalid model", "model_not_found", "no such model")):
                    logger.warning(
                        "GPT 切分失敗：CHUNK_MODEL='%s' 可能不存在或名稱有誤，請檢查 env var。"
                        "fallback 到字數切分 project=%s title=%s err=%s",
                        CHUNK_MODEL, project_id, title, gpt_err,
                    )
                else:
                    logger.warning("GPT 切分失敗，fallback 到字數切分 project=%s title=%s err=%s", project_id, title, gpt_err)
                docs, chunk_dicts = _source_to_chunks(source_data, title, chunk_size, chunk_overlap)
        else:
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

    # 建立 FAISS index（使用 tmp 目錄，確保失敗時不破壞現有 index）
    store_dir = abspath_from_base(FAISS_STORE_DIR)
    index_dir = os.path.join(store_dir, safe_name)
    tmp_dir = index_dir + ".tmp"
    try:
        os.makedirs(store_dir, exist_ok=True)
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
        os.makedirs(tmp_dir)
        vs = FAISS.from_documents(docs, embeddings)
        vs.save_local(tmp_dir)
        _save_source(tmp_dir, source_data)
        _save_chunks(tmp_dir, chunk_dicts)
        if os.path.exists(index_dir):
            shutil.rmtree(index_dir)
        os.rename(tmp_dir, index_dir)
    except Exception as e:
        shutil.rmtree(tmp_dir, ignore_errors=True)
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

    # 持久化到 PostgreSQL
    if _PG_ENABLED:
        try:
            _pg.save_sop(
                project_id, title, safe_name, index_dir,
                len(chunk_dicts), chunk_size, chunk_overlap, sha256,
            )
        except Exception:
            logger.error("PG save_sop 失敗（檔案系統已儲存）", exc_info=True)

    logger.info("上傳完成 project=%s title=%s sop_key=%s chunks=%d", project_id, title, safe_name, len(docs))
    return jsonify({
        "ok": True,
        "title": title,
        "sop_key": safe_name,
        "num_chunks": len(docs),
    })


@app.get("/api/projects/<project_id>/sops/<sop_key>/chunks")
@require_api_key
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
    if not all_chunks:
        all_chunks = _pg.get_chunks(project_id, title)

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

    if _PG_ENABLED:
        try:
            safe_name = info.get("safe_name", sop_key)
            _pg.save_sop(
                project_id, title, safe_name, index_dir,
                len(new_chunks), info.get("chunk_size", CHUNK_SIZE),
                info.get("chunk_overlap", CHUNK_OVERLAP), sha256,
            )
        except Exception:
            logger.error("PG save_sop (delete_chunk) 失敗", exc_info=True)

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
    use_gpt_chunker = body.get("chunker", "gpt").lower() != "char"
    chunk_size = max(100, int(body.get("chunk_size", info.get("chunk_size", CHUNK_SIZE))))
    chunk_overlap = max(0, int(body.get("chunk_overlap", info.get("chunk_overlap", CHUNK_OVERLAP))))
    if not use_gpt_chunker and chunk_overlap >= chunk_size:
        return jsonify({"error": "chunk_overlap 必須小於 chunk_size"}), 400

    try:
        if use_gpt_chunker:
            try:
                docs, chunk_dicts = _source_to_chunks_gpt(source_data, title)
                logger.info("GPT 語意切分完成 project=%s sop_key=%s chunks=%d", project_id, sop_key, len(chunk_dicts))
            except Exception as gpt_err:
                _err_str = str(gpt_err).lower()
                if any(k in _err_str for k in ("does not exist", "invalid model", "model_not_found", "no such model")):
                    logger.warning(
                        "GPT 切分失敗：CHUNK_MODEL='%s' 可能不存在或名稱有誤，請檢查 env var。"
                        "fallback 到字數切分 sop_key=%s err=%s",
                        CHUNK_MODEL, sop_key, gpt_err,
                    )
                else:
                    logger.warning("GPT 切分失敗，fallback 到字數切分 sop_key=%s err=%s", sop_key, gpt_err)
                docs, chunk_dicts = _source_to_chunks(source_data, title, chunk_size, chunk_overlap)
        else:
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

    if _PG_ENABLED:
        try:
            safe_name = info.get("safe_name", sop_key)
            _pg.save_sop(
                project_id, title, safe_name, index_dir,
                len(chunk_dicts), chunk_size, chunk_overlap, sha256,
            )
        except Exception:
            logger.error("PG save_sop (rechunk) 失敗", exc_info=True)

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

    if _PG_ENABLED:
        try:
            _pg.delete_sop(project_id, title)
        except Exception:
            logger.error("PG delete_sop 失敗", exc_info=True)

    logger.info("刪除 SOP project=%s title=%s sop_key=%s", project_id, title, sop_key)
    return jsonify({"ok": True, "deleted_title": title})


if __name__ == "__main__":
    ensure_state()
    app.run(
        host="0.0.0.0",
        port=int(os.getenv("PORT", "80")),
        debug=DEBUG,
    )
