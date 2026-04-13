# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Commands

```bash
pip install -r requirements.txt    # install dependencies
python app.py                      # run dev server (DEBUG=1 recommended)

# Docker
docker compose up -d --build       # build and start
docker compose logs -f             # follow logs
```

No linter or test runner is configured.

## Docker / Deployment

- **Container name**: `dept-faiss-backend`
- **Docker network**: `shdnetwork` (external)
- **env_file**: `/home/shduser/Docker_static/dept-faiss-backend/.env`
- **Port**: exposes 80 (no host binding)
- **WSGI**: `gunicorn --workers 2 --timeout 180`
- **Healthcheck**: `curl /api/health`

## Architecture

Single-file Flask service (`app.py`). Manages multiple FAISS vector databases by project.
Called by `nurse-healthhub-backend` via HTTP.

### Core flow
1. At startup, `init_app_state()` loads all projects defined in `PROJECTS` env var
2. Each project has: `index_map.json`, FAISS index directories, BM25 router, vector cache
3. `POST /api/projects/<id>/search` performs BM25 routing + vector search, returns chunks

### Key env vars
| Var | Description |
|-----|-------------|
| `PROJECTS` | project 清單：`default:index_map.json,icu:icu_index_map.json` |
| `OPENAI_API_KEY` | OpenAI API key（用於 embedding） |
| `EMBEDDING_MODEL` | 預設 `text-embedding-3-small` |
| `TOP_K` | 每次搜尋回傳的 chunk 數（預設 5） |
| `ROUTE_TOP_N` | BM25 路由選取前 N 個 SOP（預設 3） |
| `VECTOR_CACHE_MAX` | 每個 project 最多快取幾個 FAISS index（預設 50） |
| `API_TOKEN` | 若設定，POST 路由需 X-API-Key 或 Bearer token |
| `ALLOWED_ORIGINS` | CORS 白名單 |

### Adding a new project
1. 建立 `<project_id>_index_map.json`（格式與 `index_map.json` 相同）
2. 確認 FAISS index 目錄存在（`index_path` 欄位指定的路徑）
3. 在 `.env` 加入：`PROJECTS=default:index_map.json,<project_id>:<project_id>_index_map.json`
4. 重新部署

### API routes
| Method | Path | Auth | Notes |
|--------|------|------|-------|
| GET | `/api/health` | — | 健康檢查，回傳 projects 清單 |
| GET | `/api/projects` | — | 列出所有 projects + SOP 清單（含 sop_key） |
| POST | `/api/projects/<id>/search` | API_TOKEN | BM25 路由 + 向量搜尋 |

### Search API
`POST /api/projects/<project_id>/search`

Request:
```json
{
  "query": "優化後的查詢字串",
  "original": "原始查詢（選填，用於 BM25）",
  "top_k": 5,
  "route_top_n": 3,
  "route_min_score": 0.0
}
```

Response:
```json
{
  "routed": [{"title": "SOP標題", "sop_key": "safe_name", "score": 0.8}],
  "chunks": [
    {
      "content": "chunk 內容",
      "sop_title": "SOP標題",
      "sop_key": "safe_name",
      "score": 0.12,
      "page": 3,
      "metadata": {}
    }
  ],
  "timings": {"bm25_ms": 1.2, "embed_ms": 120.5, "search_ms": 8.3}
}
```

Note: FAISS L2 distance — score 越小代表越相關。
