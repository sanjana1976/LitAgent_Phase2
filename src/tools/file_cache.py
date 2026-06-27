"""Small JSON/text file cache on disk to avoid repeat downloads and API calls."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class FileCache:
    """Namespaced key-value cache under a root directory."""

    def __init__(self, root: Path, *, namespace: str = "default") -> None:
        self._root = (root / namespace).resolve()
        self._root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _key_path(root: Path, key: str) -> Path:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return root / f"{digest}.json"

    def path_for_key(self, key: str) -> Path:
        return self._key_path(self._root, key)

    def get_json(self, key: str) -> Any | None:
        path = self.path_for_key(key)
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Cache read failed for %s: %s", path, exc)
            return None

    def set_json(self, key: str, value: Any) -> None:
        path = self.path_for_key(key)
        tmp = path.with_suffix(".tmp")
        serialized = json.dumps(value, ensure_ascii=False, indent=2)
        tmp.write_text(serialized, encoding="utf-8")
        tmp.replace(path)

    def get_bytes_key(self, subdir: str, binary_key: str) -> Path | None:
        """Return path to cached binary file if it exists."""
        d = (self._root.parent / subdir).resolve()
        fname = hashlib.sha256(binary_key.encode("utf-8")).hexdigest() + ".bin"
        p = d / fname
        return p if p.is_file() else None

    def write_bytes_key(self, subdir: str, binary_key: str, data: bytes) -> Path:
        d = (self._root.parent / subdir).resolve()
        d.mkdir(parents=True, exist_ok=True)
        fname = hashlib.sha256(binary_key.encode("utf-8")).hexdigest() + ".pdf"
        path = d / fname
        path.write_bytes(data)
        return path
