"""Command-line interface to the keyed semantic embedding store.

The CLI is the *ingestion* side of the store: it gets text into a context (one
entry, or a batch file) and reports what's there. Querying and classifying are the
agent's job over the MCP server -- both front-ends share one `EmbeddingStore`, so
they stay one source of truth. Every command operates on one *context* (its own
`.npz` under the store directory); `contexts` lists them. Defaults come from the
environment (see `config.py`) and `--store-dir` overrides the store location.

  myembed add animals --id doc1 --text "The cat sat on the mat." --meta class=animal
  echo "long paragraph" | myembed add animals --id doc2
  myembed seed animals corpus.jsonl --overwrite
  myembed info animals
  myembed contexts

`seed` reads either a JSON array of objects or JSONL (one object per line); each
object is `{"id": ..., "text": ..., "metadata": {...}}` with `metadata` optional.

Note: the CLI writes the `.npz` directly, while a running MCP server loads each
context once and holds it in memory. So entries added here are not visible to an
already-running server until it restarts -- the intended flow is to seed offline,
then start the server.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import Config
from .store import EmbeddingStore


def _read_records(path: str) -> list[dict]:
    """Parse a seed file as a JSON array (or single object) or, failing that, JSONL."""
    raw = Path(path).read_text()
    if not raw.strip():
        return []
    try:
        parsed = json.loads(raw)  # whole-file JSON: array or single object
        return parsed if isinstance(parsed, list) else [parsed]
    except json.JSONDecodeError:
        pass
    records: list[dict] = []
    for n, line in enumerate(raw.splitlines(), 1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise SystemExit(f"seed: line {n}: invalid JSON ({e})")
    return records


def _to_items(records: list[dict]) -> list[tuple[str, str, dict]]:
    items: list[tuple[str, str, dict]] = []
    for i, rec in enumerate(records):
        if not isinstance(rec, dict) or "id" not in rec or "text" not in rec:
            raise SystemExit(f"seed: record {i} must be an object with 'id' and 'text'")
        meta = rec.get("metadata", {})
        if not isinstance(meta, dict):
            raise SystemExit(f"seed: record {i} 'metadata' must be an object")
        items.append((str(rec["id"]), str(rec["text"]), meta))
    return items


def _parse_meta(pairs: list[str] | None) -> dict:
    """Turn repeated `KEY=VALUE` flags into a metadata map (string values)."""
    meta: dict = {}
    for pair in pairs or []:
        key, sep, value = pair.partition("=")
        if not sep or not key:
            raise SystemExit(f"add: --meta expects KEY=VALUE, got {pair!r}")
        meta[key] = value
    return meta


def main() -> None:
    cfg = Config.from_env()
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--store-dir", default=str(cfg.store_dir),
                   help="directory holding the per-context .npz stores")
    p.add_argument("--model", default=cfg.model,
                   help="embedding model (only used when creating a new store)")
    p.add_argument("--revision", default=cfg.revision, help="pinned HF model revision")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("add", help="embed one (id, text) into a context and persist")
    a.add_argument("context")
    a.add_argument("--id", required=True)
    a.add_argument("--text", help="text to embed; omit to read from stdin")
    a.add_argument("--meta", action="append", metavar="KEY=VALUE",
                   help="metadata entry (repeatable); e.g. --meta class=animal")
    a.add_argument("--overwrite", action="store_true")

    s = sub.add_parser("seed", help="bulk-import a JSON/JSONL file into a context")
    s.add_argument("context")
    s.add_argument("file", help="path to a JSON array or JSONL seed file")
    s.add_argument("--overwrite", action="store_true",
                   help="replace ids already in the store instead of erroring")

    i = sub.add_parser("info", help="report a context's model, size, and classes")
    i.add_argument("context")

    sub.add_parser("contexts", help="list all contexts with a store on disk")

    args = p.parse_args()
    cfg = Config(model=args.model, revision=args.revision, store_dir=Path(args.store_dir))

    if args.cmd == "contexts":
        for name in cfg.list_contexts():
            print(name)
        return

    try:
        path = cfg.path_for(args.context)
    except ValueError as e:
        raise SystemExit(f"{args.cmd}: {e}")
    store = EmbeddingStore.load(path, model=cfg.model, revision=cfg.revision)

    if args.cmd == "add":
        text = args.text if args.text is not None else sys.stdin.read()
        try:
            store.add(args.id, text, overwrite=args.overwrite,
                      metadata=_parse_meta(args.meta))
        except KeyError as e:
            raise SystemExit(f"add: {e.args[0]}")
        store.save(path)
        print(f"stored id={args.id!r} in {args.context!r} ({len(store.ids)} total) -> {path}")

    elif args.cmd == "seed":
        items = _to_items(_read_records(args.file))
        try:
            n = store.add_many(items, overwrite=args.overwrite)
        except KeyError as e:
            raise SystemExit(f"seed: {e.args[0]}")
        store.save(path)
        print(f"seeded {n} item(s) into {args.context!r} ({len(store.ids)} total) -> {path}")

    elif args.cmd == "info":
        rev = f" @ {store.revision}" if store.revision else ""
        print(f"context:    {args.context}")
        print(f"store:      {path}")
        print(f"model:      {store.model}{rev}")
        print(f"count:      {len(store.ids)}")
        print(f"classified: {store.num_classified}")
        print(f"classes:    {', '.join(store.classes) or '(none)'}")
        print(f"dim:        {store.dim}")


if __name__ == "__main__":
    main()
