# Copilot Instructions — NVIDIA Generative AI Examples

## Repository Overview

This is NVIDIA's reference repository for generative AI examples, covering RAG pipelines, agentic workflows, fine-tuning, and NeMo microservices integrations. All deployments are Docker/Docker Compose–based; there is no single top-level build or test runner.

Top-level sections:
- `RAG/` — Core RAG framework and examples (primary development area)
- `nemo/` — NeMo microservices tutorials (Evaluator, Guardrails, Data Designer, etc.)
- `nemotron/` — LLM and VLM fine-tuning with Nemotron
- `finetuning/` — Model fine-tuning examples (Codegemma, Gemma, StarCoder2)
- `industries/` — Vertical-specific examples (healthcare, energy, asset lifecycle)
- `community/` — Community-contributed examples
- `vision_workflows/` — Git submodule for vision NIM workflows (clone with `--recurse-submodules`)

## Running RAG Examples

Each RAG example lives under `RAG/examples/<type>/<framework>/` and is self-contained with its own `docker-compose.yaml`.

```bash
# Required environment variable
export NVIDIA_API_KEY=nvapi-...

# Build and start a specific example (e.g., basic LangChain RAG)
cd RAG/examples/basic_rag/langchain/
docker compose up -d --build

# View chain server API docs (once running)
open http://localhost:8081/docs

# RAG Playground UI
open http://localhost:8090/

# Stop and clean up
docker compose down
```

Local NIM deployment uses shared compose files from `RAG/examples/local_deploy/`:
- `docker-compose-vectordb.yaml` — Milvus or pgvector
- `docker-compose-nim-ms.yaml` — NIM inference/embedding microservices

## Architecture: How a RAG Example Works

Each example wires together three Docker services via `docker-compose.yaml`:

1. **Chain Server** (`RAG/src/chain_server/`) — FastAPI server (port 8081) that implements `/generate`, `/documents`, and `/health` endpoints. It dynamically imports the example's `chains.py` at runtime based on `EXAMPLE_PATH` env var.

2. **RAG Playground** (`RAG/src/rag_playground/`) — Frontend UI (port 8090) that calls the chain server API.

3. **Vector DB** — Milvus (default, GPU-accelerated) or pgvector, included via the shared `local_deploy/` compose files.

The Chain Server's Dockerfile uses `EXAMPLE_PATH` as a build arg to copy the specific example's `chains.py` (and optional `requirements.txt`) into the image at build time.

## Adding or Modifying a RAG Example

Every RAG example must implement the `BaseExample` ABC from `RAG/src/chain_server/base.py`:

```python
from RAG.src.chain_server.base import BaseExample

class MyExample(BaseExample):
    def llm_chain(self, query, chat_history, **kwargs) -> Generator[str, None, None]: ...
    def rag_chain(self, query, chat_history, **kwargs) -> Generator[str, None, None]: ...
    def ingest_docs(self, data_dir, filename) -> None: ...
    def get_documents(self) -> List[str]: ...
    def delete_documents(self, filenames) -> None: ...
```

Helper utilities are in `RAG/src/chain_server/utils.py`: `get_llm()`, `get_embedding_model()`, `get_vectorstore()`, `get_text_splitter()`, `get_config()`, `get_prompts()`, `create_vectorstore_langchain()`, etc.

## Configuration System

Configuration is read from environment variables with the `APP_` prefix. Defaults live in `RAG/src/chain_server/configuration.py` using the `@configclass` / `configfield` decorators from `configuration_wizard.py`.

Key env vars for the chain server:

| Variable | Purpose | Default |
|---|---|---|
| `APP_LLM_MODELNAME` | LLM model name | `meta/llama3-70b-instruct` |
| `APP_LLM_SERVERURL` | Local NIM URL; empty = NVIDIA API Catalog | `""` |
| `APP_EMBEDDINGS_MODELNAME` | Embedding model | `nvidia/nv-embedqa-e5-v5` |
| `APP_VECTORSTORE_NAME` | `milvus` or `pgvector` | `milvus` |
| `APP_VECTORSTORE_URL` | Vector DB URL | `http://milvus:19530` |
| `APP_TEXTSPLITTER_MODELNAME` | Tokenizer for chunking | `Snowflake/snowflake-arctic-embed-l` |
| `APP_RETRIEVER_TOPK` | Docs to retrieve | `4` |
| `COLLECTION_NAME` | Milvus/pgvector collection | `nvidia_api_catalog` |
| `ENABLE_TRACING` | OpenTelemetry tracing | `false` |
| `LOGLEVEL` | Server log level | `INFO` |

## Prompt Customization

Each example has a `prompt.yaml` alongside its `docker-compose.yaml`. It's volume-mounted into the chain server container. Standard keys are `chat_template` and `rag_template`.

## Code Style

The `RAG/` directory is linted with **black** and **isort** via pre-commit:
- Black: `--skip-string-normalization --line-length=119`
- isort: `--multi-line=3 --trailing-comma --line-width=119`

Run pre-commit locally:
```bash
pre-commit run --files RAG/...
```

## License Headers

All Python files under `RAG/` must include the Apache-2.0 SPDX header:
```python
# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
```

## Contributing

- All commits must be signed off: `git commit -s -m "message"`
- Fork the repo, work on a branch, open a PR to `main`
- There is no automated CI/CD; PRs are reviewed and tested manually
