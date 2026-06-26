# mcp-text-embedding

MCP server over a **keyed semantic text-embedding store**. Embed `(id, text)` with a
sentence-transformer (semantics over syntax), persist vectors in a `.npz` keyed map, and
query the closest-k by cosine similarity — or estimate a query's **class probabilities**
by kernel density estimation. `classify` / `density` default to a **validated geometry
calibration** that de-corrupts the ranking at high dimension / low sample count (see
[Calibration](#calibration-trustworthy-ranking)).

Two levels of hierarchy:

- **Context** — every endpoint takes an obligatory `context`, and each context is its own
  independent store (`<store_dir>/<context>.npz`). Distinct purposes keep separate
  embedding spaces.
- **Metadata** — each sample carries an optional key-value map. The class used by
  `classify` lives under the `"class"` key of that map; everything else is free-form.

The repo is **pure code**: model weights live in the HuggingFace cache and the embedding
stores are per-deployment state — neither is committed.

## Install

```bash
conda create -n text-embedding python=3.12 && conda activate text-embedding
pip install -e .
```

## Configuration (environment variables)

| var | default | purpose |
|---|---|---|
| `TEXT_EMBEDDING_MODEL` | `all-mpnet-base-v2` | embedding model (the "mechanism") |
| `TEXT_EMBEDDING_MODEL_REVISION` | _(latest)_ | pin the HF model commit for reproducibility |
| `TEXT_EMBEDDING_STORE_DIR` | `~/.local/share/text-embedding/` | directory holding the per-context `.npz` stores |
| `HF_HOME` | `~/.cache/huggingface` | where weights cache (point at scratch on a cluster) |
| `HF_HUB_OFFLINE` | `0` | set `1` to forbid network at request time |

All contexts share the one configured model; each store records the model + revision it
was built with and **refuses to mix mechanisms** — adding with a different model raises
`ModelMismatch`. Context names must match `^[A-Za-z0-9._-]+$` (they become filenames).

## Run

```bash
mcp-text-embedding          # starts the MCP server over stdio
```

Register with Claude Code (`~/.claude.json` or project `.mcp.json`):

```json
{
  "mcpServers": {
    "mcp-text-embedding": {
      "command": "mcp-text-embedding",
      "env": { "TEXT_EMBEDDING_STORE_DIR": "/path/to/stores" }
    }
  }
}
```

### Tools

Every tool except `list_contexts` takes an obligatory `context`.

- `add_texts(context, items, overwrite=False)` — embed and persist a batch; `items` is a
  list of `{"id", "text", "metadata"}` objects (`metadata` optional, free-form — its
  `class` entry is what `classify` scores). Pass a one-element list for a single add. One
  batched encode and one write; the whole batch is rejected if any id is a duplicate or
  already present unless `overwrite=True` (which also replaces each matched key's metadata).
- `closest(context, k=5, text=None, id=None)` — keys of the k closest embeddings to a
  fresh `text` or an existing `id`.
- `classify(context, text=None, id=None, kappa=10.0, prior="uniform", calibrate=True)` —
  class probabilities for a fresh `text` or an existing `id`, via a von Mises–Fisher
  kernel density estimate over samples that carry a `class`. `kappa` is the
  concentration/bandwidth (higher → peakier); `prior` is `"uniform"` or `"empirical"`.
  With `calibrate=True` (default) the ranking runs in the validated geometry transform
  (see [Calibration](#calibration-trustworthy-ranking)) and returns
  `{"classes": [...], "n_eff", "low_confidence"}`; `calibrate=False` returns the raw-cosine
  `[{"class", "probability"}, ...]` list, byte-for-byte with the pre-calibration behaviour.
- `density(context, text=None, id=None, kappa=10.0, radius=0.5, calibrate=True)` — how
  crowded the space is around a fresh `text` or an existing `id`, over every **non-background**
  point. Always returns `density` (smooth weight in `(0, 1]`, comparable across queries),
  `neighbors` (count within `radius`), and `count`. With `calibrate=True` (default) it adds an
  honesty layer — `percentile` (rank of the query's local density against the reference),
  `lof_score`, `n_eff` (effective sample size), `rank_ci` (bootstrap 95% CI on the percentile),
  and `low_confidence`; `calibrate=False` is the raw-cosine estimate, byte-for-byte with before.
- `list_keys(context)` — all stored keys.
- `list_contexts()` — all contexts with a store on disk.
- `store_info(context)` — pinned model/revision, vector count, dimension, the classes
  present, the background count, and the adopted calibration config (if any).

## CLI

The CLI handles ingestion and inspection, plus the batch queries (`classify`, `density`) that
scripts run without a server; interactive querying still lives on the MCP server. Every command
operates on one context (a required positional), except `contexts`.

```bash
text-embedding add animals --id doc1 --text "The cat sat on the mat." --meta class=animal
echo "long paragraph" | text-embedding add animals --id doc2    # text from stdin
text-embedding add animals --id doc1 --text "..." --overwrite    # replace a key
text-embedding add animals --id bg1 --text "..." --background    # unlabelled background point
text-embedding seed animals corpus.jsonl --overwrite            # bulk import
text-embedding delete animals --id doc1 --id doc2               # remove ids (repeatable)
text-embedding info animals                                     # count / classes / dim / calibration
text-embedding ids animals                                      # list all stored ids
text-embedding plot animals --method tsne                       # 2-D scatter -> PNG
text-embedding contexts                                         # list all contexts
text-embedding evaluate animals --apply                         # validate + adopt a calibration
text-embedding classify animals queries.jsonl                   # batch class probabilities
text-embedding density animals queries.jsonl --no-calibrate     # batch density (raw cosine)
```

`classify` and `density` are query-only: they read the same `{"id", "text"}` JSON array / JSONL
as `seed` (`metadata` ignored) and print one JSON result per input line to stdout, never touching
the store — `classify` → `{"id", "classes": [{"class", "probability"}, ...], "n_eff", ...}` (sorted,
`--kappa` / `--prior`), `density` → `{"id", "density", "neighbors", "count", "percentile", ...}`
(`--kappa` / `--radius`). Both calibrate by default; pass `--no-calibrate` for the raw-cosine output
(no `n_eff` / `percentile` fields). An empty/unseeded context yields an empty/zero result per id
without loading the model.

`plot` projects a context's embeddings to 2-D and writes a scatter PNG colored by
`class` (points without one fall under `(unclassified)`). `--method` is `pca` (linear,
deterministic; the default) or `tsne` (nonlinear; `--seed` makes it reproducible). The
image goes to `--out` or, by default, `<store-dir>/<context>.<method>.png`. Needs more
than two points; t-SNE additionally requires `matplotlib` + `scikit-learn`.

`--meta KEY=VALUE` is repeatable (`--meta class=animal --meta src=wiki`). `seed` reads a
JSON array or JSONL (one object per line), each `{"id": ..., "text": ..., "metadata": {...}}`
with `metadata` optional:

```jsonl
{"id": "doc1", "text": "The cat sat on the mat.", "metadata": {"class": "animal"}}
{"id": "doc2", "text": "The engine roared."}
```

Defaults come from the same env vars as the server; override per call with
`--store-dir` / `--model` / `--revision`.

> The CLI writes the `.npz` directly, while a running server holds each context in memory.
> Entries added here aren't visible to an already-running server until it restarts — seed
> offline, then start the server.

## Calibration (trustworthy ranking)

At high embedding dimension with few labelled samples, the **raw cosine ranking is
corrupted** — not just rescaled — by a dominant common-mode direction (anisotropy), by
hubness (a few points that are everyone's neighbour), and by the Marchenko–Pastur noise
bulk drowning the fine semantic signal. No monotone remap of the output fixes a ranking
that is already in the wrong order. `classify` and `density` therefore default to
`calibrate=True`, which scores in a learned, **validated geometry transform** instead of
raw cosine.

The transform is fit **unsupervised** (it never sees labels): denoise to the signal
subspace (Gavish–Donoho optimal hard threshold), drop the top common-mode directions
(all-but-the-top), Ledoit–Wolf-shrunk whitening, optional mutual-proximity hubness
reduction. The data-starved step is *geometry estimation*, so you can lift it with
**background** points — unlabelled, in-domain samples added with `--background` (or
metadata `{"background": "1"}`). They feed the transform fit but never vote in `classify`
and are not part of the `density` reference.

Because the points are labelled, the calibration is **validated, not assumed**:

```bash
text-embedding evaluate animals          # nested-CV report; print only
text-embedding evaluate animals --apply  # ...and adopt the winning pipeline (persisted)
```

`evaluate` runs leak-free nested cross-validation over the labelled points — each outer
fold's test points are held out of *both* the transform fit and the pipeline selection —
and reports the raw-cosine baseline vs. the selected pipeline: kNN accuracy / hit-rate
with a **paired bootstrap CI on the delta**, hubness before/after, a selection-frequency
table, and the recommended config. The adopted config is stored in the `.npz` and picked
up by the server on its next mtime reload. (Known limitation: the transform is fit on a
synthetic-heavy pool and validated only against the labelled points — a transform that
whitens away real signal co-aligned with a *synthetic-generator* artifact would not be
caught without a reserved real holdout.)

> **Multilingual:** the pipeline is model-agnostic — it operates on stored vectors at
> whatever dimension the configured model emits — so point `TEXT_EMBEDDING_MODEL` at a
> multilingual sentence-transformer and calibration works unchanged.

## Offline / cluster (Palma)

Prefetch the weights once on a node with internet, then run offline:

```bash
text-embedding-prefetch               # on a login node (warms HF_HOME)
HF_HUB_OFFLINE=1 mcp-text-embedding   # on a compute node
```

## Scaling

`.npz` + brute-force cosine is fine to ~10⁴–10⁵ vectors per context. The `EmbeddingStore`
API (`add` / `closest` / `class_probabilities`) is stable, so the backend can later be
swapped for sqlite / faiss / lancedb without touching the MCP tool definitions. One
`SentenceTransformer` is shared across all contexts in a process, so many contexts don't
multiply model memory.
