# Smoke Tests — Multimodal ORAN RAG Chatbot

## Overview

This document describes the smoke test suite for the **oran-chatbot-multimodal** deployment. Smoke tests are lightweight, fast checks that verify the core subsystems are wired together correctly — they do **not** replace full integration or evaluation tests (see `evals/`). Run them after any fresh deployment, dependency change, or code edit to catch regressions early.

---

## Running the Tests

### Prerequisites

```bash
cd /home/rayhuang/workshop/oran-chatbot-multimodal
source .venv/bin/activate          # or use .venv/bin/python directly
```

Set your NVIDIA API key if running tests that reach the LLM API:

```bash
export NVIDIA_API_KEY="nvapi-..."
# OR edit config.yaml:  nvidia_api_key: "nvapi-..."
```

### Run all smoke tests

```bash
.venv/bin/python docs/smoke_tests.py
```

### Run a single category

```bash
.venv/bin/python docs/smoke_tests.py --category imports
.venv/bin/python docs/smoke_tests.py --category config
.venv/bin/python docs/smoke_tests.py --category components
.venv/bin/python docs/smoke_tests.py --category server
```

### Expected output (all passing)

```
[PASS] imports      :: all core modules importable
[PASS] config       :: config.yaml parsed, required keys present
[PASS] components   :: LLMClient, CrossEncoder, ConversationSummaryMemory
[PASS] server       :: HTTP 200 on /, health endpoint returns ok

4/4 tests passed.
```

---

## Test Cases

### Category 1 — `imports`

Verifies every core module can be imported without error. A failure here usually means a missing package or a broken install.

| # | What is tested | Module / symbol |
|---|---|---|
| 1.1 | LangChain core | `langchain_core.prompts`, `langchain_core.output_parsers` |
| 1.2 | LangChain community | `langchain_community.vectorstores.FAISS` |
| 1.3 | NVIDIA AI endpoints | `langchain_nvidia_ai_endpoints.ChatNVIDIA`, `NVIDIAEmbeddings` |
| 1.4 | LLM factory | `llm.llm.create_llm` |
| 1.5 | LLM client | `llm.llm_client.LLMClient` |
| 1.6 | Retriever | `retriever.retriever.get_relevant_docs` |
| 1.7 | Embedder | `retriever.embedder.NVIDIAEmbedders` |
| 1.8 | Memory helpers | `utils.memory.init_memory`, `get_summary`, `add_history_to_memory` |
| 1.9 | PDF parser | `vectorstore.custom_pdf_parser.get_pdf_documents` |
| 1.10 | PowerPoint parser | `vectorstore.custom_powerpoint_parser` |
| 1.11 | Guardrails | `guardrails.fact_check.fact_check` |
| 1.12 | Streamlit | `streamlit` |
| 1.13 | CrossEncoder | `sentence_transformers.CrossEncoder` |
| 1.14 | PyMuPDF | `pymupdf` (or `fitz`) |

---

### Category 2 — `config`

Verifies `config.yaml` is valid and contains all required keys.

| # | What is tested | Expected |
|---|---|---|
| 2.1 | YAML parses without error | No exception |
| 2.2 | `nvidia_api_key` present | Key exists (value may be placeholder) |
| 2.3 | `llm_model` present | Non-empty string |
| 2.4 | `embedding_model` present | Non-empty string |
| 2.5 | `reranker_model` present | Non-empty string |
| 2.6 | Boolean flags exist | `NIM`, `NREM`, `Reranker_NIM` all present |
| 2.7 | Bot config loads | `bot_config/multimodal_oran.config` parseable JSON with required keys |
| 2.8 | `bot_config/oran.config` loads | Same check for secondary config |

---

### Category 3 — `components`

Verifies instantiation of the main runtime objects. Does **not** make live API calls.

| # | What is tested | Expected |
|---|---|---|
| 3.1 | `LLMClient` init (NVIDIA type) | Object created, `llm` attribute is `ChatNVIDIA` |
| 3.2 | `LLMClient` init (NIM type, offline) | Object created with `base_url` set |
| 3.3 | `CrossEncoder` loads | Model `cross-encoder/ms-marco-MiniLM-L-6-v2` loads from cache |
| 3.4 | `CrossEncoder.predict` runs | Returns a float score for a (query, passage) pair |
| 3.5 | `ConversationSummaryMemory` init | `init_memory(llm, prompt_str)` returns a memory object |
| 3.6 | `NVIDIAEmbeddings` init | Object created (no API call yet) |
| 3.7 | FAISS load (if vectorstore exists) | `FAISS.load_local(...)` succeeds when `vectorstore/oran/vectorstore_nv` exists |
| 3.8 | `augment_multiple_query` is callable | Function importable; signature matches `(query, model)` |
| 3.9 | `fact_check` is callable | Function importable; signature matches `(evidence, query, response)` |

---

### Category 4 — `server`

Verifies the running Streamlit process responds correctly.

| # | What is tested | Expected |
|---|---|---|
| 4.1 | Health endpoint | `GET /_stcore/health` → `ok` (200) |
| 4.2 | Main page loads | `GET /` → HTTP 200, body contains `<title>` |
| 4.3 | No crash in log | `/tmp/oran-chatbot.log` contains no `Traceback` or `ModuleNotFoundError` |
| 4.4 | Process alive | PID from startup is still running |

---

## Smoke Test Script

Save as `docs/smoke_tests.py` and run directly.

```python
#!/usr/bin/env python3
"""
Smoke tests for oran-chatbot-multimodal.
Run from the project root:  .venv/bin/python docs/smoke_tests.py
"""
import argparse
import importlib
import json
import os
import subprocess
import sys
import traceback

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
os.environ.setdefault("NVIDIA_API_KEY", yaml.safe_load(open("config.yaml"))["nvidia_api_key"])

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
results = []


def check(name, fn):
    try:
        fn()
        print(f"{PASS} {name}")
        results.append((name, True, ""))
    except Exception as e:
        msg = traceback.format_exc(limit=3)
        print(f"{FAIL} {name}\n       {e}")
        results.append((name, False, msg))


# ── Category 1: imports ──────────────────────────────────────────────────────

def test_imports():
    modules = [
        "langchain_core.prompts",
        "langchain_core.output_parsers",
        "langchain_community.vectorstores",
        "langchain_nvidia_ai_endpoints",
        "streamlit",
        "sentence_transformers",
        "pymupdf",
        "yaml",
    ]
    for m in modules:
        importlib.import_module(m)
    from llm.llm import create_llm
    from llm.llm_client import LLMClient
    from retriever.retriever import get_relevant_docs
    from retriever.embedder import NVIDIAEmbedders
    from utils.memory import init_memory, get_summary, add_history_to_memory
    from vectorstore.custom_pdf_parser import get_pdf_documents
    from guardrails.fact_check import fact_check


# ── Category 2: config ───────────────────────────────────────────────────────

def test_config():
    cfg = yaml.safe_load(open("config.yaml"))
    for key in ["nvidia_api_key", "llm_model", "embedding_model", "reranker_model",
                "NIM", "NREM", "Reranker_NIM"]:
        assert key in cfg, f"Missing key: {key}"
    for cfg_file in ["bot_config/multimodal_oran.config", "bot_config/oran.config"]:
        bc = json.load(open(cfg_file))
        for k in ["name", "header", "page_title", "core_docs_directory_name"]:
            assert k in bc, f"{cfg_file}: missing key {k}"


# ── Category 3: components ───────────────────────────────────────────────────

def test_components():
    from llm.llm_client import LLMClient
    from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings

    # 3.1 LLMClient (NVIDIA)
    client = LLMClient("mistralai/mixtral-8x7b-instruct-v0.1", "NVIDIA")
    assert isinstance(client.llm, ChatNVIDIA)

    # 3.3 + 3.4 CrossEncoder
    from sentence_transformers import CrossEncoder
    ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    score = ce.predict([("What is ORAN?", "O-RAN is an open radio access network.")])
    assert isinstance(float(score[0]), float)

    # 3.5 ConversationSummaryMemory
    from utils.memory import init_memory
    summary_prompt = (
        "Summarize: {summary}\nNew lines: {new_lines}\nNew summary:"
    )
    mem = init_memory(client.llm, summary_prompt)
    assert mem is not None

    # 3.6 NVIDIAEmbeddings init
    emb = NVIDIAEmbeddings(model="nvidia/nv-embedqa-e5-v5")
    assert emb is not None

    # 3.8 + 3.9 callable checks
    from retriever.retriever import get_relevant_docs
    from guardrails.fact_check import fact_check
    import inspect
    assert callable(get_relevant_docs)
    assert callable(fact_check)


# ── Category 4: server ───────────────────────────────────────────────────────

def test_server():
    import urllib.request
    # 4.1 health
    res = urllib.request.urlopen("http://localhost:8011/_stcore/health", timeout=5)
    assert res.read().decode().strip() == "ok"
    # 4.2 main page
    res = urllib.request.urlopen("http://localhost:8011/", timeout=5)
    assert res.status == 200
    # 4.3 no crash in log
    log_path = "/tmp/oran-chatbot.log"
    if os.path.exists(log_path):
        log = open(log_path).read()
        assert "Traceback" not in log, "Traceback found in server log"
        assert "ModuleNotFoundError" not in log, "ModuleNotFoundError in server log"
    # 4.4 process alive
    result = subprocess.run(["pgrep", "-f", "streamlit"], capture_output=True, text=True)
    assert result.returncode == 0, "No streamlit process found"


# ── Runner ───────────────────────────────────────────────────────────────────

CATEGORIES = {
    "imports":    test_imports,
    "config":     test_config,
    "components": test_components,
    "server":     test_server,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", choices=list(CATEGORIES), default=None,
                        help="Run a single category (default: all)")
    args = parser.parse_args()

    to_run = {args.category: CATEGORIES[args.category]} if args.category else CATEGORIES
    for name, fn in to_run.items():
        check(f"{name:<12}", fn)

    passed = sum(1 for _, ok, _ in results if ok)
    total = len(results)
    print(f"\n{passed}/{total} tests passed.")
    sys.exit(0 if passed == total else 1)
```

---

## Adding New Test Cases

### Step 1 — Decide on a category

| Category | Add here when… |
|---|---|
| `imports` | You add a new module or third-party dependency |
| `config` | You add a new required key to `config.yaml` or a bot config file |
| `components` | You add a new class/function that must be instantiable without live API calls |
| `server` | You add a new route, page, or health check |

If your change doesn't fit any existing category, add a new function named `test_<category>` and register it in the `CATEGORIES` dict.

### Step 2 — Write the test function

Keep tests fast (< 5 s each). Mock or skip anything that requires a live NVIDIA API key or internet:

```python
def test_openai_backend():
    """Example: test the new OpenAI LLM backend (see feature-plan-openai-endpoint.md)."""
    import yaml
    cfg = yaml.safe_load(open("config.yaml"))
    if not cfg.get("OPENAI"):
        print("  (skipped — OPENAI flag is false)")
        return

    from llm.llm_client import LLMClient
    from langchain_openai import ChatOpenAI
    client = LLMClient(cfg["openai_model_name"], "OPENAI")
    assert isinstance(client.llm, ChatOpenAI), f"Expected ChatOpenAI, got {type(client.llm)}"
```

### Step 3 — Register it

Add to the `CATEGORIES` dict at the bottom of `smoke_tests.py`:

```python
CATEGORIES = {
    "imports":    test_imports,
    "config":     test_config,
    "components": test_components,
    "server":     test_server,
    "openai":     test_openai_backend,   # ← new entry
}
```

### Step 4 — Run and verify

```bash
.venv/bin/python docs/smoke_tests.py --category openai
```

### Guidelines

- **No live API calls** — instantiate objects but do not call `.invoke()`, `.embed_query()`, or `.stream()` unless a real key is available. Guard with `if cfg["nvidia_api_key"].startswith("nvapi-placeholder"): return`.
- **Assert clearly** — use descriptive assertion messages: `assert x, f"Expected X, got {x}"`.
- **One concern per test** — split unrelated checks into separate `check()` calls rather than one monolithic function.
- **Idempotent** — tests must leave no side effects (no files written, no DB mutations).

---

## Relationship to Full Evaluation Tests

The smoke tests here are **not** a replacement for the full RAG evaluation pipeline in `evals/`:

| | Smoke tests (this file) | Full evals (`evals/`) |
|---|---|---|
| Speed | < 30 s total | Minutes to hours |
| API calls | None (offline) | Yes — requires live key |
| What's checked | Wiring + imports + server health | Answer quality, faithfulness, context recall |
| When to run | Every deploy / code change | Before releases or major prompt changes |

For full evaluation, follow the Jupyter notebooks in `evals/` in order:
1. `01_synthetic_data_generation.ipynb` — generate Q&A test set from your documents
2. `02_filling_RAG_outputs_for_Evaluation.ipynb` — run the RAG pipeline on the test set
3. `03_eval_ragas.ipynb` — compute RAGAS metrics (faithfulness, answer relevancy, context recall)
4. `04_Human_Like_RAG_Evaluation-AIP.ipynb` — LLM-as-judge scoring
5. `05_complexquery_advancedRAG.ipynb` — advanced multi-hop query evaluation
