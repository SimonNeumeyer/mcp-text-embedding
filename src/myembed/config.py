"""Runtime configuration, read from the environment.

Nothing here is committed as data: the repo only declares *which* model by name
(and optionally a pinned HF revision) and *where* the store lives. The weights
themselves are fetched into the HuggingFace cache (`HF_HOME`), and the store is
per-deployment state pointed to by `MYEMBED_STORE_PATH`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_MODEL = "all-mpnet-base-v2"  # semantics over syntax


def _default_store_path() -> Path:
    base = os.environ.get("XDG_DATA_HOME") or (Path.home() / ".local" / "share")
    return Path(base) / "myembed" / "embeddings.npz"


@dataclass(frozen=True)
class Config:
    model: str
    revision: str | None  # HF model commit; None = whatever HF resolves to
    store_path: Path

    @classmethod
    def from_env(cls) -> "Config":
        store = os.environ.get("MYEMBED_STORE_PATH")
        return cls(
            model=os.environ.get("MYEMBED_MODEL", DEFAULT_MODEL),
            revision=os.environ.get("MYEMBED_MODEL_REVISION") or None,
            store_path=Path(store) if store else _default_store_path(),
        )
