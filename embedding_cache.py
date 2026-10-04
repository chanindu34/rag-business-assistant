"""Disk cache for embeddings, keyed by (model, task type, text).

Re-running ingestion after a quota error, or after changing unrelated
settings, re-embeds nothing that was already embedded. Stored as float32
blobs in SQLite, so ~1,000 vectors of 3072 dims is about 12 MB.
"""

import hashlib
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


class EmbeddingCache:
    def __init__(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.execute("CREATE TABLE IF NOT EXISTS emb (key TEXT PRIMARY KEY, vec BLOB)")

    @staticmethod
    def key(model: str, task_type: Optional[str], text: str) -> str:
        return hashlib.sha256(f"{model}|{task_type}|{text}".encode("utf-8")).hexdigest()

    def get_many(self, keys: List[str]) -> Dict[str, List[float]]:
        out = {}
        for i in range(0, len(keys), 500):
            part = keys[i:i + 500]
            rows = self.conn.execute(
                f"SELECT key, vec FROM emb WHERE key IN ({','.join('?' * len(part))})", part
            ).fetchall()
            out.update({k: np.frombuffer(v, dtype=np.float32).tolist() for k, v in rows})
        return out

    def put_many(self, items: Dict[str, List[float]]) -> None:
        self.conn.executemany(
            "INSERT OR REPLACE INTO emb (key, vec) VALUES (?, ?)",
            [(k, np.asarray(v, dtype=np.float32).tobytes()) for k, v in items.items()],
        )
        self.conn.commit()
