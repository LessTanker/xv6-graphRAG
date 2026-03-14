# xv6-graphRAG

This project is a GraphRAG-style assistant for exploring the [xv6-riscv](https://github.com/mit-pdos/xv6-riscv) kernel. The project has been refactored into a four-class pipeline with a unified runtime entrypoint.

## Architecture

![Pipeline](pipeline.png)

The `src` directory now keeps only the core modules:

```text
src/
├── KnowledgeIndexer.py   # Offline pipeline: parse code, build graph, build embeddings
├── QueryProcessor.py     # Online query understanding: HyDE + query embedding + planning
├── GraphRetriever.py     # Retrieval core: FAISS seed search + graph traversal expansion
├── ResponseGenerator.py  # Prompt assembly + final LLM response
├── main.py               # Unified CLI entrypoint
├── config.py             # Paths, model settings, planner strategy config
├── utils.py              # Shared IO, text formatting, and LLM call helpers
└── __init__.py
```

## Data Flow

1. `KnowledgeIndexer` reads xv6 source and `compile_commands.json`, then writes:
   - `data/chunks_metadata.json`
   - `data/graph_edges.json`
   - `data/faiss.index`
2. `QueryProcessor` receives user query, runs HyDE, builds query embedding, and generates query plan.
3. `GraphRetriever` runs Top-K retrieval on FAISS and expands related chunks using plan-guided graph traversal.
4. `ResponseGenerator` assembles markdown context, calls LLM, and writes final output to:
   - `data/search_results_with_graph.json`
   - `data/prompt.md`

Index building is no longer a set of standalone scripts. It is part of the integrated pipeline executed by `main.py`.

## Tech Stack

- Retrieval architecture: GraphRAG (vector retrieval + graph traversal)
- Embeddings: `sentence-transformers` (`BAAI/bge-small-en` by default)
- Vector index: `faiss-cpu` (`IndexFlatL2`)
- Code parsing: `libclang` Python bindings (`clang.cindex`) over xv6 compile DB
- Graph construction: AST-based relations (`CALLS`, `USES_STRUCT`, `USES_GLOBAL`, `MACRO_USE`)
- Query understanding: HyDE query expansion + LLM-based query planning
- Prompting/answering: chat-completions style LLM API (configurable via `.env`)
- Runtime: Python 3, `python-dotenv`, standard JSON/CLI pipeline

## Setup

1. Prerequisites:
   - Python 3.10+
   - `libclang` (`sudo apt install libclang-dev`)
   - `bear` (`sudo apt install bear`)
   - xv6 build toolchain
2. Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

3. Configure `.env` (optional if you only want retrieval without LLM answers):

```env
LLM_API_URL="https://api.deepseek.com/chat/completions"
LLM_TOKEN="your-token"
LLM_MODEL="deepseek-chat"
EMBEDDING_MODEL_NAME="BAAI/bge-small-en"
EMBEDDING_BATCH_SIZE="64"
# Optional: use static file-path partition instead of Leiden communities.
# true/1/yes/on => static partition by file path prefix (for example kernel/fs/*)
COMMUNITY_USE_STATIC_PATH_PARTITION="false"
```

## Usage

### Full rebuild

```bash
make rebuild
```

### Build compile database only

```bash
make compile-db
```

### Build index artifacts only

```bash
make index
```

### Ask a query

```bash
make query Q="how does the trap path reach usertrap?"
```

or run directly:

```bash
.venv/bin/python -m src.main "how does the scheduler switch process context?"
```

To force re-indexing in direct mode:

```bash
.venv/bin/python -m src.main "index warmup" --rebuild-index
```

## Community Discovery and Module Routing

The pipeline now includes two extra stages:

1. Community discovery over code graph nodes (`CommunityManager`)
2. Module-aware routing in query planning (`QueryProcessor` + `GraphRetriever`)

### What is built automatically

When you run `make index`, `make rebuild`, or `src.main --rebuild-index`, the system will:

1. Build base artifacts (`chunks_metadata.json`, `graph_edges.json`, `faiss.index`)
2. Build communities (`data/communities.json`)
3. Build community summary vector index (`data/community_faiss.index`)

No extra command is required for community generation.

### Community partition modes

- `COMMUNITY_USE_STATIC_PATH_PARTITION=false` (default):
   - Use Leiden algorithm on graph edges (fallback to NetworkX greedy modularity if Leiden deps are unavailable).
- `COMMUNITY_USE_STATIC_PATH_PARTITION=true`:
   - Skip Leiden and prefer static grouping by file/module path prefix.
   - Useful for explicit xv6 module boundaries like `kernel/fs/*`, `kernel/proc*`, `user/*`.

### Planner and retrieval behavior

During query planning:

1. Query embedding first searches `community_faiss.index`
2. Top related module summaries are injected into planner prompt
3. Planner is required to output `restricted_community_id` in plan JSON

During graph traversal:

1. If `restricted_community_id` exists, BFS/shortest-path traversal is filtered to that community only
2. For `FILE_OR_MODULE` queries with locked community, traversal depth is slightly increased for better local recall

### Quick verification

1. Rebuild once:

```bash
make rebuild
```

2. Confirm generated files:

```bash
ls data/communities.json data/community_faiss.index
```

3. Run a module-oriented query:

```bash
make query Q="explain the file system logging path in kernel/log.c"
```

4. Inspect planned routing field in output:

```bash
cat data/search_results_with_graph.json | jq '.query_plan'
```

Expected plan includes:

```json
{
   "query_type": "FILE_OR_MODULE",
   "traversal_strategy": "module_nodes_expansion",
   "restricted_community_id": 3
}
```

## Outputs

- `data/chunks_metadata.json`: extracted code chunks
- `data/graph_edges.json`: graph relationships
- `data/faiss.index`: vector index
- `data/communities.json`: community_id -> node_ids + community summaries
- `data/community_faiss.index`: vector index for community summaries
- `data/prompt.md`: merged retrieval context for LLM
- `data/search_results_with_graph.json`: final response payload

## License

[MIT License](LICENSE)
