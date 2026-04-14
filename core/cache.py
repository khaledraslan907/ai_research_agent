from __future__ import annotations

import json
import sqlite3
import time
from typing import Any, Optional

from core.config import CACHE_DB


CACHE_VERSION = "v3_0"
DEFAULT_TTL_SECONDS = 60 * 60 * 24 * 7   # 7 days


class CacheManager:
    def __init__(self, db_path: str = str(CACHE_DB), ttl_seconds: int = DEFAULT_TTL_SECONDS):
        self.db_path = db_path
        self.ttl = ttl_seconds
        self._init_db()

    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    namespace  TEXT NOT NULL,
                    cache_key  TEXT NOT NULL,
                    value      TEXT NOT NULL,
                    created_at REAL NOT NULL DEFAULT 0,
                    PRIMARY KEY (namespace, cache_key)
                )
            """)
            # add created_at column to existing DBs that pre-date this version
            try:
                conn.execute("ALTER TABLE cache ADD COLUMN created_at REAL NOT NULL DEFAULT 0")
            except Exception:
                pass
            conn.commit()

    def _get(self, namespace: str, cache_key: str) -> Optional[Any]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT value, created_at FROM cache WHERE namespace=? AND cache_key=?",
                (namespace, cache_key),
            ).fetchone()

            if not row:
                return None

            age = time.time() - (row["created_at"] or 0)
            if age > self.ttl:
                conn.execute(
                    "DELETE FROM cache WHERE namespace=? AND cache_key=?",
                    (namespace, cache_key),
                )
                conn.commit()
                return None

            try:
                return json.loads(row["value"])
            except Exception:
                return None

    def _set(self, namespace: str, cache_key: str, value: Any) -> None:
        with self._connect() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO cache (namespace, cache_key, value, created_at)
                   VALUES (?, ?, ?, ?)""",
                (namespace, cache_key, json.dumps(value, default=str), time.time()),
            )
            conn.commit()

    def _key(self, *parts) -> str:
        return f"{CACHE_VERSION}::" + "::".join(str(p).strip().lower() for p in parts)

    # --- public API ---

    def get_query(self, provider: str, query: str, max_results: int) -> Optional[Any]:
        return self._get("query_cache", self._key(provider, query, max_results))

    def set_query(self, provider: str, query: str, max_results: int, value: Any) -> None:
        self._set("query_cache", self._key(provider, query, max_results), value)

    def get_scrape(self, url_or_domain: str) -> Optional[Any]:
        return self._get("scrape_cache", self._key(url_or_domain))

    def set_scrape(self, url_or_domain: str, value: Any) -> None:
        self._set("scrape_cache", self._key(url_or_domain), value)

    def get_llm(self, key: str) -> Optional[Any]:
        return self._get("llm_cache", self._key(key))

    def set_llm(self, key: str, value: Any) -> None:
        self._set("llm_cache", self._key(key), value)

    def get_generic(self, namespace: str, key: str) -> Optional[Any]:
        return self._get(namespace, self._key(key))

    def set_generic(self, namespace: str, key: str, value: Any) -> None:
        self._set(namespace, self._key(key), value)

    def clear_expired(self) -> int:
        cutoff = time.time() - self.ttl
        with self._connect() as conn:
            cur = conn.execute(
                "DELETE FROM cache WHERE created_at < ? AND created_at > 0", (cutoff,)
            )
            conn.commit()
            return cur.rowcount
