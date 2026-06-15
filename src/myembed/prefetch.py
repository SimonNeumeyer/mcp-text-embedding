"""Warm the HuggingFace cache so the server never downloads at request time.

Run this once on a node with internet (e.g. a cluster login node), then run the
server with HF_HUB_OFFLINE=1 / TRANSFORMERS_OFFLINE=1 on the compute node.
"""

from __future__ import annotations

from .config import Config


def main() -> None:
    cfg = Config.from_env()
    from sentence_transformers import SentenceTransformer

    print(f"prefetching {cfg.model!r} (revision={cfg.revision or 'latest'}) ...")
    SentenceTransformer(cfg.model, revision=cfg.revision)
    print("done — weights are in the HF cache")


if __name__ == "__main__":
    main()
