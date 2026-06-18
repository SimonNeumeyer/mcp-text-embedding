"""Command-line interface to the keyed semantic embedding store.

The CLI gets text into a context (one entry, or a batch file), reports what's there,
and runs the batch queries (`classify`, `density`) that scripts need without a running
server -- the interactive querying still lives on the MCP server; both front-ends share
one `EmbeddingStore`, so they stay one source of truth. Every command operates on one
*context* (its own `.npz` under the store directory); `contexts` lists them. Defaults come
from the environment (see `config.py`) and `--store-dir` overrides the store location.

  text-embedding add animals --id doc1 --text "The cat sat on the mat." --meta class=animal
  echo "long paragraph" | text-embedding add animals --id doc2
  text-embedding seed animals corpus.jsonl --overwrite
  text-embedding delete animals --id doc1 --id doc2
  text-embedding info animals
  text-embedding ids animals
  text-embedding plot animals --method tsne
  text-embedding contexts
  text-embedding classify animals queries.jsonl
  text-embedding density animals queries.jsonl

`classify` and `density` are query-only batch commands: they read a JSON array or JSONL of
`{"id": ..., "text": ...}` and print one JSON result per line (input order) to stdout, never
mutating the store -- `classify` -> `{"id", "classes": [{"class", "probability"}, ...]}`,
`density` -> `{"id", "density", "neighbors", "count", ...}`.

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


def _batch_rows(store, items, *, kind, kappa=10.0, prior="uniform", radius=0.5):
    """Yield one output dict per (id, text) query for the `classify`/`density` batch commands.

    An empty store yields an empty/zero result per id *without* loading the model (matching the
    store methods' own empty returns); otherwise one batched encode feeds the per-query estimate.
    These are query-only -- the store is never mutated or saved.
    """
    if not store.ids:
        for id_, _text, _meta in items:
            if kind == "classify":
                yield {"id": id_, "classes": []}
            else:
                yield {"id": id_, "density": 0.0, "neighbors": 0, "count": 0,
                       "kappa": kappa, "radius": radius}
        return
    vecs = store.encode_many([text for _id, text, _meta in items])
    for (id_, _text, _meta), vec in zip(items, vecs):
        if kind == "classify":
            classes = [
                {"class": cls_, "probability": prob}
                for cls_, prob in store.class_probabilities(
                    vec, kappa=kappa, prior=prior, exclude=None)
            ]
            yield {"id": id_, "classes": classes}
        else:
            yield {"id": id_, **store.density(vec, kappa=kappa, radius=radius, exclude=None)}


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

    d = sub.add_parser("delete", help="remove one or more ids from a context")
    d.add_argument("context")
    d.add_argument("--id", required=True, action="append", dest="ids", metavar="ID",
                   help="id to delete (repeatable)")

    i = sub.add_parser("info", help="report a context's model, size, and classes")
    i.add_argument("context")

    ids_p = sub.add_parser("ids", help="list all ids stored in a context")
    ids_p.add_argument("context")

    pl = sub.add_parser(
        "plot", help="render a 2-D scatter of a context's embeddings, colored by class"
    )
    pl.add_argument("context")
    pl.add_argument("--method", choices=("pca", "tsne"), default="pca")
    pl.add_argument("--out",
                    help="output PNG path (default: <store-dir>/<context>.<method>.png)")
    pl.add_argument("--seed", type=int, default=0, help="random seed for t-SNE")

    sub.add_parser("contexts", help="list all contexts with a store on disk")

    cl = sub.add_parser(
        "classify", help="batch-classify a JSONL file of {id, text} queries against a context"
    )
    cl.add_argument("context")
    cl.add_argument("file", help="path to a JSON array or JSONL file of {id, text} queries")
    cl.add_argument("--kappa", type=float, default=10.0,
                    help="vMF concentration/bandwidth (higher -> peakier)")
    cl.add_argument("--prior", choices=("uniform", "empirical"), default="uniform")

    de = sub.add_parser(
        "density", help="batch density estimate for a JSONL file of {id, text} queries"
    )
    de.add_argument("context")
    de.add_argument("file", help="path to a JSON array or JSONL file of {id, text} queries")
    de.add_argument("--kappa", type=float, default=10.0,
                    help="vMF concentration/bandwidth (higher -> more local)")
    de.add_argument("--radius", type=float, default=0.5,
                    help="cosine threshold for the neighbor count")

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

    elif args.cmd == "delete":
        try:
            n = store.delete_many(args.ids)
        except KeyError as e:
            raise SystemExit(f"delete: {e.args[0]}")
        store.save(path)
        print(f"deleted {n} id(s) from {args.context!r} ({len(store.ids)} total) -> {path}")

    elif args.cmd == "info":
        rev = f" @ {store.revision}" if store.revision else ""
        print(f"context:    {args.context}")
        print(f"store:      {path}")
        print(f"model:      {store.model}{rev}")
        print(f"count:      {len(store.ids)}")
        print(f"classified: {store.num_classified}")
        print(f"classes:    {', '.join(store.classes) or '(none)'}")
        print(f"dim:        {store.dim}")

    elif args.cmd == "ids":
        for id_ in store.ids:
            print(id_)

    elif args.cmd == "classify":
        rows = _batch_rows(store, _to_items(_read_records(args.file)),
                           kind="classify", kappa=args.kappa, prior=args.prior)
        for row in rows:
            print(json.dumps(row, ensure_ascii=False))

    elif args.cmd == "density":
        rows = _batch_rows(store, _to_items(_read_records(args.file)),
                           kind="density", kappa=args.kappa, radius=args.radius)
        for row in rows:
            print(json.dumps(row, ensure_ascii=False))

    elif args.cmd == "plot":
        try:
            ids, coords, classes = store.project(args.method, seed=args.seed)
        except ValueError as e:
            raise SystemExit(f"plot: {e}")
        out = Path(args.out) if args.out else path.with_suffix(f".{args.method}.png")

        import matplotlib  # local: only the plot path pays for matplotlib
        matplotlib.use("Agg")  # no display; must be set before importing pyplot
        import matplotlib.pyplot as plt

        # one scatter per distinct class so each gets its own color and legend entry
        labels = [c if c is not None else "(unclassified)" for c in classes]
        fig, ax = plt.subplots(figsize=(8, 6))
        for label in sorted(set(labels)):
            pts = coords[[i for i, lab in enumerate(labels) if lab == label]]
            ax.scatter(pts[:, 0], pts[:, 1], label=label, s=30, alpha=0.8)
        ax.set_title(f"{args.context} ({args.method}, {len(ids)} points)")
        ax.legend(loc="best", fontsize="small")
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"plotted {len(ids)} point(s) from {args.context!r} -> {out}")


if __name__ == "__main__":
    main()
