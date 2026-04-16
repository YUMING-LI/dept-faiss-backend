#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
一次性遷移腳本：將現有檔案系統的 FAISS index 匯入 PostgreSQL。

用法：
    # 在容器內執行
    docker exec -it ihd-faiss-backend python migrate_fs_to_pg.py

    # 或本機執行（需設定環境變數）
    DB_HOST=10.1.207.19 DB_PASSWORD=shdadmin python migrate_fs_to_pg.py
"""
import os
import json
import sys

from dotenv import load_dotenv

load_dotenv()

from pg_store import pg

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FAISS_STORE_DIR = os.getenv("FAISS_STORE_DIR", "faiss_index_store")
PROJECTS_RAW = os.getenv("PROJECTS", "nurse-healthhub:index_map.json").strip()


def abspath_from_base(p: str) -> str:
    if not p:
        return p
    return p if os.path.isabs(p) else os.path.normpath(os.path.join(BASE_DIR, p))


def main():
    pg.init(
        host=os.getenv("DB_HOST", "10.1.207.19"),
        port=int(os.getenv("DB_PORT", "5432")),
        dbname=os.getenv("DB_NAME", "edah_sh"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD", ""),
        sslmode=os.getenv("DB_SSLMODE", "disable"),
    )

    projects = {}
    for entry in PROJECTS_RAW.split(","):
        entry = entry.strip()
        if not entry:
            continue
        parts = entry.split(":", 1)
        if len(parts) != 2:
            continue
        projects[parts[0].strip()] = parts[1].strip()

    dyn_file = os.path.join(BASE_DIR, "dynamic_projects.json")
    if os.path.exists(dyn_file):
        with open(dyn_file, "r") as f:
            projects.update(json.load(f))

    total = 0
    for pid, index_map_file in projects.items():
        fp = abspath_from_base(index_map_file)
        if not os.path.exists(fp):
            print(f"[SKIP] {pid}: index_map 不存在 {fp}")
            continue

        with open(fp, "r", encoding="utf-8") as f:
            index_map = json.load(f)

        pg.save_project(pid, index_map_file, is_dynamic=(pid not in PROJECTS_RAW))

        for title, info in index_map.items():
            safe_name = info.get("safe_name", "")
            index_path = info.get("index_path", "")
            index_dir = abspath_from_base(index_path.replace("\\", "/"))

            if not os.path.isdir(index_dir):
                print(f"  [SKIP] {pid}/{title}: index_dir 不存在 {index_dir}")
                continue

            if not os.path.exists(os.path.join(index_dir, "index.faiss")):
                print(f"  [SKIP] {pid}/{title}: 缺少 index.faiss")
                continue

            pg.save_sop(
                project_id=pid,
                title=title,
                safe_name=safe_name,
                index_dir=index_dir,
                num_chunks=info.get("num_chunks", 0),
                chunk_size=info.get("chunk_size", 500),
                chunk_overlap=info.get("chunk_overlap", 100),
                sha256=info.get("sha256", ""),
            )
            total += 1
            print(f"  [OK] {pid}/{title} ({safe_name})")

    print(f"\n遷移完成：{total} 筆 SOP 已寫入 PostgreSQL")


if __name__ == "__main__":
    main()
