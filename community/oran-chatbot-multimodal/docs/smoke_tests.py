#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Smoke tests for oran-chatbot-multimodal.

Run from the project root:
    .venv/bin/python docs/smoke_tests.py            # all categories
    .venv/bin/python docs/smoke_tests.py --category imports
    .venv/bin/python docs/smoke_tests.py --category config
    .venv/bin/python docs/smoke_tests.py --category components
    .venv/bin/python docs/smoke_tests.py --category server

See docs/smoke-tests.md for full documentation and a guide on adding tests.
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
SKIP = "\033[93m[SKIP]\033[0m"
results = []


def check(label, fn):
    try:
        fn()
        print(f"{PASS} {label}")
        results.append((label, True, ""))
    except _SkipTest as s:
        print(f"{SKIP} {label}  ({s})")
        results.append((label, True, f"skipped: {s}"))
    except Exception as e:
        print(f"{FAIL} {label}\n       {e}")
        results.append((label, False, traceback.format_exc(limit=3)))


class _SkipTest(Exception):
    pass


def skip(reason):
    raise _SkipTest(reason)


# ── Category 1: imports ──────────────────────────────────────────────────────

def test_imports():
    """All core modules must be importable without errors."""
    stdlib_modules = [
        "langchain_core.prompts",
        "langchain_core.output_parsers",
        "langchain_community.vectorstores",
        "langchain_nvidia_ai_endpoints",
        "streamlit",
        "sentence_transformers",
        "pymupdf",
        "yaml",
    ]
    for m in stdlib_modules:
        importlib.import_module(m)

    from llm.llm import create_llm                              # noqa: F401
    from llm.llm_client import LLMClient                        # noqa: F401
    from retriever.retriever import get_relevant_docs           # noqa: F401
    from retriever.embedder import NVIDIAEmbedders              # noqa: F401
    from utils.memory import (                                  # noqa: F401
        init_memory, get_summary, add_history_to_memory,
    )
    from vectorstore.custom_pdf_parser import get_pdf_documents # noqa: F401
    from guardrails.fact_check import fact_check                # noqa: F401


# ── Category 2: config ───────────────────────────────────────────────────────

def test_config():
    """config.yaml and bot config files must be valid and complete."""
    cfg = yaml.safe_load(open("config.yaml"))

    required_keys = [
        "nvidia_api_key", "llm_model", "embedding_model", "reranker_model",
        "NIM", "NREM", "Reranker_NIM",
    ]
    for key in required_keys:
        assert key in cfg, f"config.yaml: missing required key '{key}'"

    for cfg_file in ["bot_config/multimodal_oran.config", "bot_config/oran.config"]:
        bc = json.load(open(cfg_file))
        for k in ["name", "header", "page_title", "core_docs_directory_name",
                  "summary_prompt"]:
            assert k in bc, f"{cfg_file}: missing key '{k}'"


# ── Category 3: components ───────────────────────────────────────────────────

def test_components():
    """Core runtime objects must be instantiable offline."""
    from llm.llm_client import LLMClient
    from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings

    # LLMClient — NVIDIA API Catalog backend
    client = LLMClient("mistralai/mixtral-8x7b-instruct-v0.1", "NVIDIA")
    assert isinstance(client.llm, ChatNVIDIA), (
        f"Expected ChatNVIDIA, got {type(client.llm)}"
    )

    # CrossEncoder reranker — local model, no internet needed after first download
    from sentence_transformers import CrossEncoder
    ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    score = ce.predict([("What is ORAN?", "O-RAN is an open radio access network.")])
    assert isinstance(float(score[0]), float), "CrossEncoder score must be a float"

    # ConversationSummaryMemory
    from utils.memory import init_memory
    summary_prompt = (
        "Summarize: {summary}\nNew lines: {new_lines}\nNew summary:"
    )
    mem = init_memory(client.llm, summary_prompt)
    assert mem is not None, "init_memory returned None"

    # NVIDIAEmbeddings init (no API call)
    emb = NVIDIAEmbeddings(model="nvidia/nv-embedqa-e5-v5")
    assert emb is not None, "NVIDIAEmbeddings init returned None"

    # FAISS vectorstore — only test if the index already exists
    vs_path = os.path.join("vectorstore", "oran", "vectorstore_nv")
    if os.path.exists(vs_path):
        from langchain_community.vectorstores import FAISS
        store = FAISS.load_local(
            os.path.join("vectorstore", "oran", "vectorstore_nv"),
            emb,
            allow_dangerous_deserialization=True,
        )
        assert store is not None, "FAISS.load_local returned None"
    else:
        skip("vectorstore/oran/vectorstore_nv not found — run Knowledge Base ingestion first")

    # Callable signatures
    from retriever.retriever import get_relevant_docs
    from guardrails.fact_check import fact_check
    assert callable(get_relevant_docs), "get_relevant_docs must be callable"
    assert callable(fact_check), "fact_check must be callable"


# ── Category 4: server ───────────────────────────────────────────────────────

def test_server():
    """Running Streamlit instance must be healthy and serving the main page."""
    import urllib.request
    import urllib.error

    base = "http://localhost:8011"

    # Health endpoint
    try:
        res = urllib.request.urlopen(f"{base}/_stcore/health", timeout=5)
        body = res.read().decode().strip()
        assert body == "ok", f"Health endpoint returned: {body!r}"
    except urllib.error.URLError as e:
        raise AssertionError(
            f"Could not reach {base} — is the server running? ({e})"
        ) from e

    # Main page HTTP 200
    res = urllib.request.urlopen(f"{base}/", timeout=5)
    assert res.status == 200, f"Main page returned HTTP {res.status}"

    # No crash lines in server log
    log_path = "/tmp/oran-chatbot.log"
    if os.path.exists(log_path):
        log = open(log_path).read()
        for bad in ["Traceback (most recent call last)", "ModuleNotFoundError"]:
            assert bad not in log, f"Error found in {log_path}: '{bad}'"

    # Streamlit process is alive
    result = subprocess.run(["pgrep", "-f", "streamlit"], capture_output=True, text=True)
    assert result.returncode == 0, "No streamlit process found — server may have crashed"


# ── Registry ─────────────────────────────────────────────────────────────────
#
# To add a new test category:
#   1. Define a function  test_<name>(). Raise AssertionError on failure.
#      Use skip(reason) to skip without failing.
#   2. Add it to CATEGORIES below.
#   3. Run:  .venv/bin/python docs/smoke_tests.py --category <name>
#
# See docs/smoke-tests.md for full guidelines.

CATEGORIES: dict = {
    "imports":    test_imports,
    "config":     test_config,
    "components": test_components,
    "server":     test_server,
}

# ── Runner ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Smoke tests for oran-chatbot-multimodal"
    )
    parser.add_argument(
        "--category",
        choices=list(CATEGORIES),
        default=None,
        help="Run a single category (default: all)",
    )
    args = parser.parse_args()

    to_run = (
        {args.category: CATEGORIES[args.category]}
        if args.category
        else CATEGORIES
    )

    print(f"\nRunning {len(to_run)} smoke test category/categories...\n")
    for label, fn in to_run.items():
        check(f"{label:<12}", fn)

    passed = sum(1 for _, ok, _ in results if ok)
    total = len(results)
    print(f"\n{'─'*40}")
    print(f"{passed}/{total} tests passed.")
    sys.exit(0 if passed == total else 1)
