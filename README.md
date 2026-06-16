# myembed-mcp

MCP server over a **keyed semantic text-embedding store**. Embed `(id, text)` with a
sentence-transformer (semantics over syntax), persist vectors in a `.npz` keyed map, and
query the closest-k by cosine similarity — or estimate a query's **class probabilities**
by kernel density estimation.

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
conda create -n myembed python=3.12 && conda activate myembed
pip install -e .
```

## Configuration (environment variables)

| var | default | purpose |
|---|---|---|
| `MYEMBED_MODEL` | `all-mpnet-base-v2` | embedding model (the "mechanism") |
| `MYEMBED_MODEL_REVISION` | _(latest)_ | pin the HF model commit for reproducibility |
| `MYEMBED_STORE_DIR` | `~/.local/share/myembed/` | directory holding the per-context `.npz` stores |
| `HF_HOME` | `~/.cache/huggingface` | where weights cache (point at scratch on a cluster) |
| `HF_HUB_OFFLINE` | `0` | set `1` to forbid network at request time |

All contexts share the one configured model; each store records the model + revision it
was built with and **refuses to mix mechanisms** — adding with a different model raises
`ModelMismatch`. Context names must match `^[A-Za-z0-9._-]+$` (they become filenames).

## Run

```bash
myembed-mcp          # starts the MCP server over stdio
```

Register with Claude Code (`~/.claude.json` or project `.mcp.json`):

```json
{
  "mcpServers": {
    "myembed": {
      "command": "myembed-mcp",
      "env": { "MYEMBED_STORE_DIR": "/path/to/stores" }
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
- `classify(context, text=None, id=None, kappa=10.0, prior="uniform")` — class
  probabilities for a fresh `text` or an existing `id`, via a von Mises–Fisher kernel
  density estimate over samples that carry a `class`. `kappa` is the
  concentration/bandwidth (higher → peakier); `prior` is `"uniform"` or `"empirical"`.
- `density(context, text=None, id=None, kappa=10.0, radius=0.5)` — how crowded the space
  is around a fresh `text` or an existing `id`, via a von Mises–Fisher kernel density
  estimate over all stored points. Returns `density` (smooth weight in `(0, 1]`, higher →
  denser, comparable across queries), `neighbors` (count within cosine `radius`), and
  `count`. `kappa` is the concentration/bandwidth (higher → more local).
- `list_keys(context)` — all stored keys.
- `list_contexts()` — all contexts with a store on disk.
- `store_info(context)` — pinned model/revision, vector count, dimension, and the
  classes present.

## CLI

The CLI is the **ingestion + inspection** side of the store — getting text into a context
and seeing what's there. Querying and classifying are the agent's job over the MCP server.
Every command operates on one context (a required positional), except `contexts`.

```bash
myembed add animals --id doc1 --text "The cat sat on the mat." --meta class=animal
echo "long paragraph" | myembed add animals --id doc2          # text from stdin
myembed add animals --id doc1 --text "..." --overwrite          # replace a key
myembed seed animals corpus.jsonl --overwrite                   # bulk import
myembed info animals                                            # count / classes / dim
myembed contexts                                                # list all contexts
```

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

## Offline / cluster (Palma)

Prefetch the weights once on a node with internet, then run offline:

```bash
myembed-prefetch                      # on a login node (warms HF_HOME)
HF_HUB_OFFLINE=1 myembed-mcp          # on a compute node
```

## Scaling

`.npz` + brute-force cosine is fine to ~10⁴–10⁵ vectors per context. The `EmbeddingStore`
API (`add` / `closest` / `class_probabilities`) is stable, so the backend can later be
swapped for sqlite / faiss / lancedb without touching the MCP tool definitions. One
`SentenceTransformer` is shared across all contexts in a process, so many contexts don't
multiply model memory.
