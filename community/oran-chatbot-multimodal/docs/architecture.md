# oran-chatbot-multimodal — Architecture

## System Overview

A Streamlit-based multimodal RAG chatbot for O-RAN technical specifications, running at `localhost:8011`.

```
User Query (Streamlit UI)
       │
       ▼
┌──────────────────────────────────────────────────────┐
│               Multimodal_Assistant.py                │
│  (Main Streamlit app — orchestration layer)          │
└───┬──────────────┬──────────────┬────────────────────┘
    │              │              │
    ▼              ▼              ▼
[RAG Pipeline] [Memory]      [Guardrails]
```

---

## Component Reference

### 1. Frontend — `Multimodal_Assistant.py`

- **Streamlit** app (port 8011)
- 2 subpages: `pages/1_Knowledge_Base.py`, `pages/2_Evaluation_Metrics.py`
- Selectable RAG pipeline via `--rag_type`:

| Value | Pipeline |
|---|---|
| `0` | Basic RAG |
| `1` | *(default)* Augmented Query RAG + CrossEncoder reranking |
| `2` | HyDE (Hypothetical Document Embeddings) + Reranker |
| `3` | LangChain MultiQueryRetriever + Reranker |

---

### 2. Document Ingestion — `vectorstore/`

```
PDF / PPT / HTML
      │
      ▼
custom_pdf_parser.py / custom_powerpoint_parser.py
  ├─ Text blocks  (PyMuPDF / fitz, grouped by ~500 chars)
  ├─ Tables  ──► DePlot NIM (chart linearization)
  │                └─► Mixtral (plain-English description)
  └─ Images  ──► NeVA-22B  (captioning + graph detection)
                   └─► DePlot NIM (if graph detected)
      │
      ▼
embedder.py → FAISS index  (vectorstore/vectorstore_nv)
```

Images and tables are described by multimodal LLMs and their descriptions embedded alongside text, enabling truly multimodal retrieval without storing raw pixel data in the vector DB.

---

### 3. LLM Layer — `llm/`

| File | Role |
|---|---|
| `llm.py` | `NvidiaLLM`, `NimLLM`, `LocalLLM` — creates a `ChatNVIDIA` or `HuggingFacePipeline` |
| `llm_client.py` | `LLMClient` — wraps `chat_with_prompt` (text) and `multimodal_invoke` (vision) |

**Supported backends** (toggled in `config.yaml`):

| Backend | `NIM` flag | Model |
|---|---|---|
| NVIDIA AI Foundation Endpoints | `false` | `mistralai/mixtral-8x7b-instruct-v0.1` (default) |
| Self-hosted NVIDIA NIM | `true` | `meta/llama3-8b-instruct` or `llama3-70b-instruct` |
| Local HuggingFace model | `LOCAL` type | Any HF causal LM |

---

### 4. Retrieval — `retriever/`

| File | Role |
|---|---|
| `embedder.py` | `NVIDIAEmbedders` (API Catalog) or `HuggingFaceEmbedders`; NREM NIM supported |
| `vector.py` | Abstract `VectorClient` over Milvus or Qdrant |
| `retriever.py` | `get_relevant_docs()` (FAISS similarity) and `get_relevant_docs_mq()` (MultiQueryRetriever) |

**Advanced techniques in `Multimodal_Assistant.py`**:

- **Query Augmentation** — LLM generates 5 ORAN-specific sub-queries; all 6 are searched
- **HyDE** — LLM generates a hypothetical answer; its embedding is used for retrieval
- **Query Rewriting** — Rewrites follow-up questions to be self-contained using conversation history
- **Cross-Encoder Reranking** — `cross-encoder/ms-marco-MiniLM-L-6-v2` (or `NVIDIARerank` NIM) reranks top-k retrieved docs to top-4

---

### 5. Memory — `utils/memory.py`

- `ConversationSummaryMemory` (LangChain) — rolling LLM-generated summary of chat history
- Used by `query_rewriting()` to resolve ambiguous follow-up questions

---

### 6. Guardrails — `guardrails/fact_check.py`

- A second LLM call post-response verifies the answer against retrieved context
- Returns `TRUE` (green) or `FALSE` (red) with an explanation and suggested follow-ups
- Uses the same `llm_model` configured in `config.yaml`

---

### 7. Configuration — `config.yaml`

| Key | Default | Description |
|---|---|---|
| `nvidia_api_key` | — | NVIDIA API key (`nvapi-…`) |
| `llm_model` | `mistralai/mixtral-8x7b-instruct-v0.1` | LLM for chat and guardrails |
| `embedding_model` | `nvidia/nv-embedqa-e5-v5` | Embedding model |
| `reranker_model` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Local CrossEncoder reranker |
| `NIM` | `false` | Use self-hosted LLM NIM |
| `nim_model_name` | `meta/llama3-8b-instruct` | NIM model name |
| `nim_base_url` | `http://localhost:8000/v1` | NIM endpoint |
| `NREM` | `false` | Use self-hosted embedding NIM |
| `Reranker_NIM` | `false` | Use NVIDIARerank NIM |

---

## End-to-End Query Flow (RAG type 1)

```
User Query
   │
   ├─► query_rewriting()         [if follow-up, uses ConversationSummaryMemory]
   │         └─► rewritten query
   │
   ├─► augment_multiple_query()  → 5 ORAN-specific sub-queries
   │
   ├─► get_relevant_docs()       × 6 (original + 5 augmented)  →  top-k docs
   │
   ├─► CrossEncoder reranking    →  top-4 docs
   │
   ├─► nemo_rag()                →  LLM streams answer to UI
   │
   ├─► add_history_to_memory()   →  update ConversationSummaryMemory
   │
   └─► fact_check()  [optional]  →  TRUE / FALSE verification
```

---

## Directory Structure

```
oran-chatbot-multimodal/
├── Multimodal_Assistant.py      # Main Streamlit app (orchestration)
├── config.yaml                  # Runtime configuration
├── requirements.txt
├── bot_config/                  # System prompt / footer templates
├── docs/                        # Design documentation
├── evals/                       # Evaluation scripts
├── guardrails/
│   └── fact_check.py            # Post-response LLM fact verification
├── llm/
│   ├── llm.py                   # LLM backend factory (NVIDIA / NIM / Local)
│   └── llm_client.py            # Unified text + multimodal client
├── pages/
│   ├── 1_Knowledge_Base.py      # Document upload / vector DB management
│   └── 2_Evaluation_Metrics.py  # Synthetic test set + RAG metrics
├── retriever/
│   ├── embedder.py              # Embedding models (NVIDIA / HuggingFace / NREM)
│   ├── retriever.py             # Retrieval functions (basic + multi-query)
│   └── vector.py                # Vector DB abstraction (Milvus / Qdrant)
├── utils/
│   ├── api_key_check.py
│   ├── feedback.py              # Google Sheets feedback integration
│   └── memory.py                # ConversationSummaryMemory helpers
└── vectorstore/
    ├── custom_pdf_parser.py     # Multimodal PDF parsing (text/table/image)
    ├── custom_powerpoint_parser.py
    ├── embedder.py              # Ingestion-time embedder
    └── vectorstore_updater.py   # Add / delete documents from FAISS index
```
