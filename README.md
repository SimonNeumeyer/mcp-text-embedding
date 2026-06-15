# myembed-mcp

MCP server over a **keyed semantic text-embedding store**. Embed `(id, text)` with a
sentence-transformer (semantics over syntax), persist vectors in a single `.npz` keyed
map, and query the closest-k by cosine similarity.

The repo is **pure code**: model weights live in the HuggingFace cache and the embedding
store is per-deployment state — neither is committed.

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
| `MYEMBED_STORE_PATH` | `~/.local/share/myembed/embeddings.npz` | where the keyed map lives |
| `HF_HOME` | `~/.cache/huggingface` | where weights cache (point at scratch on a cluster) |
| `HF_HUB_OFFLINE` | `0` | set `1` to forbid network at request time |

The store records the model + revision it was built with and **refuses to mix
mechanisms** — adding with a different model raises `ModelMismatch`.

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
      "env": { "MYEMBED_STORE_PATH": "/path/to/embeddings.npz" }
    }
  }
}
```

### Tools

- `add_text(id, text, overwrite=False)` — embed and persist under `id`.
- `closest(k=5, text=None, id=None)` — keys of the k closest embeddings to a fresh
  `text` or an existing `id`.
- `list_keys()` — all stored keys.
- `store_info()` — pinned model/revision, vector count, dimension.

## CLI

The same store is usable from the shell (handy for bulk ingest / debugging):

```bash
myembed add --id doc1 --text "The cat sat on the mat."
echo "long paragraph" | myembed add --id doc2     # text from stdin
myembed add --id doc1 --text "..." --overwrite     # replace a key
myembed query --text "a feline on a carpet" -k 5   # closest to fresh text
myembed query --id doc1 -k 5                        # closest to a stored key
```

Defaults come from the same env vars as the server; override per call with
`--store` / `--model` / `--revision`.

## Offline / cluster (Palma)

Prefetch the weights once on a node with internet, then run offline:

```bash
myembed-prefetch                      # on a login node (warms HF_HOME)
HF_HUB_OFFLINE=1 myembed-mcp          # on a compute node
```

## Scaling

`.npz` + brute-force cosine is fine to ~10⁴–10⁵ vectors. The `EmbeddingStore` API
(`add` / `closest`) is stable, so the backend can later be swapped for sqlite / faiss /
lancedb without touching the MCP tool definitions.
