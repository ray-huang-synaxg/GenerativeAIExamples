# Feature Plan: OpenAI API Endpoint Support in the LLM Layer

## Summary

Add an `OpenAILLM` backend to `llm/llm.py` and expose it through `config.yaml` so users can route all LLM calls (chat, multimodal, guardrails, query augmentation, reranking) through any OpenAI-compatible endpoint — including the official OpenAI API, Azure OpenAI, and self-hosted OpenAI-compatible servers (e.g., vLLM, LM Studio, Ollama, **SynaXG AIHub**).

> **Note — local LLM not required.**
> As long as an OpenAI-compatible endpoint is reachable (such as `http://aihub.synaxg.com:3000`), the chatbot can answer questions without any locally deployed model. The endpoint handles inference; the app only sends HTTP requests to it.

---

## Motivation

The current LLM layer supports three backends:

| Backend | How selected |
|---|---|
| NVIDIA AI Foundation Endpoints | `NIM: false` (default) |
| Self-hosted NVIDIA NIM | `NIM: true` |
| Local HuggingFace model | `model_type = "LOCAL"` |

OpenAI and OpenAI-compatible APIs (GPT-4o, GPT-4-turbo, Azure OpenAI, vLLM with OpenAI shim) are not supported. Adding this backend makes the chatbot usable by teams already provisioned with OpenAI API keys, and gives a low-friction path to test against stronger frontier models without GPU infrastructure.

---

## Scope

| In scope | Out of scope |
|---|---|
| Text chat (`chat_with_prompt`) via OpenAI chat completions | Fine-tuning or batch APIs |
| Multimodal invocation (`multimodal_invoke`) for vision models (GPT-4o, GPT-4-vision) | OpenAI embeddings (tracked separately as NREM equivalent) |
| Guardrails (`fact_check.py`) | Streaming function-calling / tool use |
| Query augmentation, rewriting, HyDE in `Multimodal_Assistant.py` | |
| Azure OpenAI endpoint variant | |

---

## Proposed Changes

### 1. `config.yaml` — new keys

```yaml
# OpenAI-compatible backend
OPENAI: false
openai_api_key: ""           # sk-... or Azure key; overrides OPENAI_API_KEY env var
openai_model_name: "gpt-4o"
openai_base_url: ""          # leave empty for api.openai.com; set for Azure or local
openai_api_version: ""       # required for Azure OpenAI (e.g. "2024-02-01")
openai_temperature: 0.1
openai_top_p: 0.5
openai_max_tokens: 1600
```

`OPENAI: true` takes precedence over `NIM: true`. Priority order:

```
OPENAI > NIM > NVIDIA (default)
```

---

### 2. `llm/llm.py` — add `OpenAILLM` class and update `create_llm()`

```python
from langchain_openai import ChatOpenAI, AzureChatOpenAI

class OpenAILLM:
    """Wraps any OpenAI-compatible endpoint via langchain-openai."""

    def __init__(self, model_name: str):
        cfg = yaml.safe_load(open("config.yaml"))
        api_key   = cfg.get("openai_api_key") or os.environ.get("OPENAI_API_KEY", "")
        base_url  = cfg.get("openai_base_url") or None
        api_ver   = cfg.get("openai_api_version") or None

        kwargs = dict(
            model=model_name,
            api_key=api_key,
            temperature=cfg.get("openai_temperature", 0.1),
            top_p=cfg.get("openai_top_p", 0.5),
            max_tokens=cfg.get("openai_max_tokens", 1600),
        )

        if api_ver:
            # Azure OpenAI path
            self.llm = AzureChatOpenAI(
                azure_endpoint=base_url,
                openai_api_version=api_ver,
                **kwargs,
            )
        else:
            if base_url:
                kwargs["base_url"] = base_url
            self.llm = ChatOpenAI(**kwargs)


def create_llm(model_name: str, model_type: str = "NVIDIA"):
    if model_type == "OPENAI":
        return OpenAILLM(model_name).llm
    elif model_type == "NVIDIA":
        return NvidiaLLM(model_name).llm
    elif model_type == "NIM":
        return NimLLM(model_name).llm
    elif model_type == "LOCAL":
        return LocalLLM(model_name).llm
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
```

`ChatOpenAI` implements the same `BaseChatModel` interface as `ChatNVIDIA`, so all downstream LangChain chains (prompts, `StrOutputParser`, `chain.stream`) work without modification.

---

### 3. `llm/llm_client.py` — detect OPENAI flag at init

```python
class LLMClient:
    def __init__(self, model_name="mixtral_8x7b", model_type="NVIDIA"):
        cfg = yaml.safe_load(open("config.yaml"))
        if cfg.get("OPENAI"):
            model_type = "OPENAI"
            model_name = cfg.get("openai_model_name", model_name)
        self.llm = create_llm(model_name, model_type)
```

`multimodal_invoke` already constructs a `HumanMessage` with an `image_url` content part — this is the standard format for GPT-4o and GPT-4-vision. No changes are needed to the call site; it just works when `self.llm` is a `ChatOpenAI` pointed at a vision-capable model.

---

### 4. `Multimodal_Assistant.py` — extend backend selection

In the `llm_client` initialization block near the top of the file:

```python
if config_yaml.get('OPENAI'):
    OPENAI_FLAG = True
    llm_client = LLMClient(config_yaml['openai_model_name'], "OPENAI")
    print("Initialized OpenAI-compatible endpoint for LLMs")
elif config_yaml['NIM']:
    NIM_FLAG = True
    llm_client = LLMClient(config_yaml['nim_model_name'], "NIM")
    print("Initialized NVIDIA NIM for LLMs")
else:
    llm_client = LLMClient(config_yaml['llm_model'])
    print("Initialized NVIDIA API Catalog for LLMs")
```

No changes needed to `nemo_rag()`, `augment_multiple_query()`, `augment_query_generated()`, or `query_rewriting()` — they already consume `llm_client.llm` or create a `ChatNVIDIA` inline (those inline calls should be updated to prefer `llm_client.llm` in a follow-up cleanup pass).

---

### 5. `requirements.txt` — add dependency

```
langchain-openai>=0.1.0
```

`langchain-openai` is the official LangChain integration package for OpenAI and Azure OpenAI. It is separate from `openai` but depends on it, so no direct `openai` pin is needed.

---

## Configuration Examples

### SynaXG AIHub (recommended for this deployment)

`http://aihub.synaxg.com:3000` exposes an OpenAI-compatible API. Set `openai_base_url` to its `/v1` path and supply the AIHub API key:

```yaml
OPENAI: true
openai_api_key: "<your-aihub-api-key>"
openai_model_name: "<model-name-as-listed-on-aihub>"  # e.g. "llama-3.1-8b-instruct"
openai_base_url: "http://aihub.synaxg.com:3000/v1"
openai_api_version: ""          # leave empty — not Azure
openai_temperature: 0.1
openai_top_p: 0.5
openai_max_tokens: 1600
```

To discover what models the AIHub endpoint offers:

```bash
curl http://aihub.synaxg.com:3000/v1/models \
  -H "Authorization: Bearer <your-aihub-api-key>" | python3 -m json.tool
```

### Official OpenAI API (GPT-4o)

```yaml
OPENAI: true
openai_api_key: "sk-..."
openai_model_name: "gpt-4o"
openai_base_url: ""
openai_api_version: ""
openai_temperature: 0.1
openai_max_tokens: 1600
```

### Azure OpenAI

```yaml
OPENAI: true
openai_api_key: "<azure-key>"
openai_model_name: "gpt-4o"
openai_base_url: "https://<resource>.openai.azure.com/"
openai_api_version: "2024-02-01"
```

### Self-hosted vLLM / Ollama (OpenAI-compatible shim)

```yaml
OPENAI: true
openai_api_key: "none"           # vLLM accepts any non-empty string
openai_model_name: "meta-llama/Llama-3-8B-Instruct"
openai_base_url: "http://localhost:8000/v1"
openai_api_version: ""
```

### Self-hosted LM Studio

```yaml
OPENAI: true
openai_api_key: "lm-studio"
openai_model_name: "llama-3-8b-instruct"
openai_base_url: "http://localhost:1234/v1"
```

---

## Multimodal Considerations

`multimodal_invoke` in `llm_client.py` sends:

```python
HumanMessage(content=[
    {"type": "text",      "text": "Describe this image in detail:"},
    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_string}"}},
])
```

This is the OpenAI vision message format. Compatibility matrix:

| Backend | Vision support |
|---|---|
| `gpt-4o` | ✅ Full support |
| `gpt-4-turbo` (with vision) | ✅ Full support |
| `gpt-4-vision-preview` | ✅ Full support |
| Azure GPT-4o | ✅ Full support |
| SynaXG AIHub (vision-capable model) | ✅ If the deployed model supports vision |
| SynaXG AIHub (text-only model) | ⚠️ Graceful fallback (no crash) |
| vLLM with LLaVA / Phi-3-vision | ✅ With compatible model |
| Ollama with LLaVA | ✅ With compatible model |
| GPT-3.5-turbo / text-only models | ❌ Will raise API error |

If a text-only model is configured, `multimodal_invoke` should fall back gracefully. A `try/except` guard in `llm_client.py` is recommended:

```python
def multimodal_invoke(self, b64_string, **kwargs):
    try:
        message = HumanMessage(content=[...])
        return self.llm.invoke([message])
    except Exception as e:
        print(f"[multimodal_invoke] Warning: vision call failed ({e}). Returning empty.")
        from langchain_core.messages import AIMessage
        return AIMessage(content="")
```

---

## Testing Plan

| Test | Expected |
|---|---|
| `config.yaml` with `OPENAI: true`, valid key, `gpt-4o` | Chatbot answers ORAN queries; fact check returns TRUE/FALSE |
| SynaXG AIHub endpoint (`openai_base_url: http://aihub.synaxg.com:3000/v1`) | Queries route to AIHub; answers stream back |
| Azure endpoint with `openai_api_version` set | `AzureChatOpenAI` used; answers return |
| `openai_base_url` pointing to local vLLM | Queries route to local server |
| `OPENAI: false`, `NIM: false` | Falls back to NVIDIA API Catalog (no regression) |
| `OPENAI: true`, `NIM: true` | `OPENAI` wins; NIM not used |
| Text-only model with multimodal call | Graceful fallback, no crash |
| Missing `openai_api_key`, `OPENAI_API_KEY` env var set | Env var used as fallback |

---

## Rollout Steps

1. Add `langchain-openai` to `requirements.txt`
2. Add new keys to `config.yaml` (all default to empty / `false` → no behaviour change)
3. Implement `OpenAILLM` in `llm/llm.py`
4. Update `LLMClient.__init__` in `llm/llm_client.py`
5. Update backend selection block in `Multimodal_Assistant.py`
6. Add `multimodal_invoke` fallback guard
7. Document the new config keys in `README.md` Step 7 section
8. Run manual smoke test with each configuration example above
