"""Command-line interface to the keyed semantic embedding store.

Same `EmbeddingStore` the MCP server uses, so the CLI and the server are one
source of truth. Defaults come from the environment (see `config.py`) and can be
overridden per invocation with `--store` / `--model` / `--revision`.

  myembed add --id doc1 --text "The cat sat on the mat."
  echo "long paragraph" | myembed add --id doc2
  myembed query --text "a feline on a carpet" -k 5
  myembed query --id doc1 -k 5
"""

from __future__ import annotations

import argparse
import sys

from .config import Config
from .store import EmbeddingStore


def main() -> None:
    cfg = Config.from_env()
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--store", default=str(cfg.store_path), help="path to the .npz store")
    p.add_argument("--model", default=cfg.model,
                   help="embedding model (only used when creating a new store)")
    p.add_argument("--revision", default=cfg.revision, help="pinned HF model revision")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("add", help="embed (id, text) and persist alongside history")
    a.add_argument("--id", required=True)
    a.add_argument("--text", help="text to embed; omit to read from stdin")
    a.add_argument("--overwrite", action="store_true")

    q = sub.add_parser("query", help="keys of the k closest embeddings")
    src = q.add_mutually_exclusive_group(required=True)
    src.add_argument("--id", help="use an embedding already in the store")
    src.add_argument("--text", help="embed this text, then search")
    q.add_argument("-k", type=int, default=5)

    args = p.parse_args()
    store = EmbeddingStore.load(args.store, model=args.model, revision=args.revision)

    if args.cmd == "add":
        text = args.text if args.text is not None else sys.stdin.read()
        store.add(args.id, text, overwrite=args.overwrite)
        store.save(args.store)
        print(f"stored id={args.id!r} ({len(store.ids)} total) -> {args.store}")

    elif args.cmd == "query":
        if args.id is not None:
            vec, exclude = store.vector_for(args.id), args.id
        else:
            vec, exclude = store.encode(args.text), None
        for key, sim in store.closest(vec, args.k, exclude=exclude):
            print(f"{sim:.4f}\t{key}")


if __name__ == "__main__":
    main()
