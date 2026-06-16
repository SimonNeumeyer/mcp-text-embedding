"""Runtime configuration, read from the environment.

Nothing here is committed as data: the repo only declares *which* model by name
(and optionally a pinned HF revision) and *where* the stores live. The weights
themselves are fetched into the HuggingFace cache (`HF_HOME`), and the stores are
per-deployment state under the directory pointed to by `TEXT_EMBEDDING_STORE_DIR`.

Each *context* is an independent `.npz` store living at `<store_dir>/<context>.npz`,
so distinct purposes keep separate embedding spaces. Context names are validated
(not sanitized) into filenames to keep them human-readable and reject traversal.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

DEFAULT_MODEL = "all-mpnet-base-v2"  # semantics over syntax

# A context maps 1:1 to a `<context>.npz` filename, so restrict it to a safe,
# collision-free, traversal-free charset and reject anything else.
CONTEXT_RE = re.compile(r"^[A-Za-z0-9._-]+$")


def _default_store_dir() -> Path:
    base = os.environ.get("XDG_DATA_HOME") or (Path.home() / ".local" / "share")
    return Path(base) / "text-embedding"


@dataclass(frozen=True)
class Config:
    model: str
    revision: str | None  # HF model commit; None = whatever HF resolves to
    store_dir: Path

    @classmethod
    def from_env(cls) -> "Config":
        store = os.environ.get("TEXT_EMBEDDING_STORE_DIR")
        return cls(
            model=os.environ.get("TEXT_EMBEDDING_MODEL", DEFAULT_MODEL),
            revision=os.environ.get("TEXT_EMBEDDING_MODEL_REVISION") or None,
            store_dir=Path(store) if store else _default_store_dir(),
        )

    def path_for(self, context: str) -> Path:
        """Map a context name to its `.npz` path, rejecting unsafe names."""
        if context in (".", "..") or not CONTEXT_RE.match(context):
            raise ValueError(
                f"invalid context {context!r}: must match {CONTEXT_RE.pattern}"
            )
        return self.store_dir / f"{context}.npz"

    def list_contexts(self) -> list[str]:
        """Names of all contexts that currently have a store on disk."""
        if not self.store_dir.is_dir():
            return []
        return sorted(p.stem for p in self.store_dir.glob("*.npz"))
