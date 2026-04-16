# -*- coding: utf-8 -*-
"""
PostgreSQL 持久化層 — 將 FAISS index 二進位檔 + metadata 存入 edah_sh。

使用方式：
    from pg_store import pg
    pg.ensure_tables()
    pg.save_sop(project_id, title, safe_name, index_dir, num_chunks, ...)
    rows = pg.load_all_sops()
"""
import io
import os
import json
import logging
import tarfile
from typing import Dict, List, Any, Optional

import psycopg2
import psycopg2.extras

logger = logging.getLogger("ihd-faiss-service")


class PgStore:
    def __init__(self):
        self._dsn: Optional[str] = None

    def init(
        self,
        host: str,
        port: int,
        dbname: str,
        user: str,
        password: str,
        sslmode: str = "disable",
    ):
        self._dsn = (
            f"host={host} port={port} dbname={dbname} "
            f"user={user} password={password} sslmode={sslmode}"
        )
        self.ensure_tables()
        logger.info("PgStore 已連線 %s@%s:%s/%s", user, host, port, dbname)

    def _conn(self):
        if not self._dsn:
            raise RuntimeError("PgStore 尚未初始化，請先呼叫 init()")
        return psycopg2.connect(self._dsn)

    # ------------------------------------------------------------------
    # DDL
    # ------------------------------------------------------------------
    def ensure_tables(self):
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS faiss_projects (
                        project_id  VARCHAR(64) PRIMARY KEY,
                        index_map_file VARCHAR(256),
                        is_dynamic  BOOLEAN DEFAULT FALSE,
                        created_at  TIMESTAMPTZ DEFAULT NOW()
                    );
                """)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS faiss_indexes (
                        id           SERIAL PRIMARY KEY,
                        project_id   VARCHAR(64) NOT NULL,
                        title        TEXT NOT NULL,
                        safe_name    VARCHAR(128) NOT NULL,
                        index_bin    BYTEA NOT NULL,
                        chunks       JSONB NOT NULL DEFAULT '[]'::jsonb,
                        source       JSONB NOT NULL DEFAULT '{}'::jsonb,
                        num_chunks   INTEGER NOT NULL DEFAULT 0,
                        chunk_size   INTEGER NOT NULL DEFAULT 500,
                        chunk_overlap INTEGER NOT NULL DEFAULT 100,
                        sha256       VARCHAR(64),
                        created_at   TIMESTAMPTZ DEFAULT NOW(),
                        updated_at   TIMESTAMPTZ DEFAULT NOW(),
                        UNIQUE(project_id, title)
                    );
                """)
            conn.commit()

    # ------------------------------------------------------------------
    # SOP CRUD
    # ------------------------------------------------------------------
    def save_sop(
        self,
        project_id: str,
        title: str,
        safe_name: str,
        index_dir: str,
        num_chunks: int,
        chunk_size: int,
        chunk_overlap: int,
        sha256: str,
    ):
        """將 index_dir 內的所有檔案打包成 tar.gz 存入 PG。"""
        index_bin = self._pack_index_dir(index_dir)
        chunks = self._read_json(os.path.join(index_dir, "chunks.json"), [])
        source = self._read_json(os.path.join(index_dir, "source.json"), {})

        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO faiss_indexes
                        (project_id, title, safe_name, index_bin,
                         chunks, source, num_chunks, chunk_size, chunk_overlap, sha256)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (project_id, title) DO UPDATE SET
                        safe_name     = EXCLUDED.safe_name,
                        index_bin     = EXCLUDED.index_bin,
                        chunks        = EXCLUDED.chunks,
                        source        = EXCLUDED.source,
                        num_chunks    = EXCLUDED.num_chunks,
                        chunk_size    = EXCLUDED.chunk_size,
                        chunk_overlap = EXCLUDED.chunk_overlap,
                        sha256        = EXCLUDED.sha256,
                        updated_at    = NOW()
                """, (
                    project_id, title, safe_name,
                    psycopg2.Binary(index_bin),
                    json.dumps(chunks, ensure_ascii=False),
                    json.dumps(source, ensure_ascii=False),
                    num_chunks, chunk_size, chunk_overlap, sha256,
                ))
            conn.commit()
        logger.info("PG save_sop project=%s title=%s size=%dKB", project_id, title, len(index_bin) // 1024)

    def load_all_sops(self) -> List[Dict[str, Any]]:
        """載入所有 SOP 記錄（不含 index_bin，用於 index_map 重建）。"""
        with self._conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("""
                    SELECT project_id, title, safe_name,
                           num_chunks, chunk_size, chunk_overlap, sha256
                    FROM faiss_indexes
                    ORDER BY project_id, title
                """)
                return [dict(r) for r in cur.fetchall()]

    def restore_sop_to_dir(self, project_id: str, title: str, target_dir: str) -> bool:
        """從 PG 取出 index_bin 並解壓到 target_dir。回傳是否成功。"""
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT index_bin, chunks, source FROM faiss_indexes WHERE project_id=%s AND title=%s",
                    (project_id, title),
                )
                row = cur.fetchone()
        if not row:
            return False

        index_bin, chunks_json, source_json = row
        os.makedirs(target_dir, exist_ok=True)
        self._unpack_index_dir(bytes(index_bin), target_dir)

        with open(os.path.join(target_dir, "chunks.json"), "w", encoding="utf-8") as f:
            json.dump(chunks_json if isinstance(chunks_json, list) else json.loads(chunks_json), f, ensure_ascii=False, indent=2)
        with open(os.path.join(target_dir, "source.json"), "w", encoding="utf-8") as f:
            json.dump(source_json if isinstance(source_json, dict) else json.loads(source_json), f, ensure_ascii=False, indent=2)

        logger.info("PG restore_sop project=%s title=%s → %s", project_id, title, target_dir)
        return True

    def delete_sop(self, project_id: str, title: str):
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM faiss_indexes WHERE project_id=%s AND title=%s",
                    (project_id, title),
                )
            conn.commit()

    def delete_project(self, project_id: str):
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM faiss_indexes WHERE project_id=%s", (project_id,))
                cur.execute("DELETE FROM faiss_projects WHERE project_id=%s", (project_id,))
            conn.commit()

    # ------------------------------------------------------------------
    # Project CRUD
    # ------------------------------------------------------------------
    def save_project(self, project_id: str, index_map_file: str, is_dynamic: bool = False):
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO faiss_projects (project_id, index_map_file, is_dynamic)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (project_id) DO UPDATE SET
                        index_map_file = EXCLUDED.index_map_file,
                        is_dynamic     = EXCLUDED.is_dynamic
                """, (project_id, index_map_file, is_dynamic))
            conn.commit()

    def load_projects(self) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("SELECT project_id, index_map_file, is_dynamic FROM faiss_projects")
                return [dict(r) for r in cur.fetchall()]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _pack_index_dir(index_dir: str) -> bytes:
        """將 index_dir 中的 index.faiss + index.pkl 打包成 tar.gz bytes。"""
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            for fn in sorted(os.listdir(index_dir)):
                fp = os.path.join(index_dir, fn)
                if not os.path.isfile(fp):
                    continue
                if fn in ("chunks.json", "source.json"):
                    continue
                tar.add(fp, arcname=fn)
        return buf.getvalue()

    @staticmethod
    def _unpack_index_dir(data: bytes, target_dir: str):
        """從 tar.gz bytes 解壓 index.faiss + index.pkl 到 target_dir。"""
        buf = io.BytesIO(data)
        with tarfile.open(fileobj=buf, mode="r:gz") as tar:
            tar.extractall(target_dir)

    @staticmethod
    def _read_json(fp: str, default):
        if not os.path.exists(fp):
            return default
        with open(fp, "r", encoding="utf-8") as f:
            return json.load(f)


pg = PgStore()
