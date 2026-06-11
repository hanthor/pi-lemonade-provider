# pi-lemonade-provider

A pi extension that auto-discovers a [Lemonade Server](https://github.com/lemonade-sdk/lemonade-server) and registers its chat models as a pi provider.

## What is Lemonade Server?

Lemonade Server is an OpenAI-compatible local AI server that runs GGUF, vLLM, Stable Diffusion, Whisper, and TTS models — all on AMD GPUs (ROCm). It auto-downloads models from Hugging Face on first use and serves them through `/v1/chat/completions`.

## How Discovery Works

The extension finds your Lemonade Server in this order:

1. **`LEMONADE_API_BASE`** env var — full base URL, e.g. `http://192.168.0.6:13305/v1`
2. **`LEMONADE_HOST`** env var — host:port, e.g. `192.168.0.6:13305`
3. **Localhost defaults** — probes `http://localhost:13305/v1` and `http://localhost:8000/v1`

Once discovered, it calls `GET /v1/models?show_all=true` and registers all chat-capable models (llamacpp + vLLM backends). Models are prefixed `lemonade/` (e.g. `lemonade/Qwen3-4B-GGUF`).

## Installation

```bash
# Symlink into pi's global extensions directory
ln -s $(pwd) ~/.pi/agent/extensions/pi-lemonade-provider

# Or copy directly
cp -r . ~/.pi/agent/extensions/pi-lemonade-provider
```

## Usage

Set the env var to point at your Lemonade Server:

```bash
export LEMONADE_HOST=192.168.0.6:13305
pi
```

Then use `/model` to select a model like `lemonade/Qwen3-4B-GGUF`.

### Via Tailscale

```bash
export LEMONADE_API_BASE=https://lemonade.manatee-basking.ts.net/v1
pi
```

### Local server

If Lemonade Server is running on `localhost:13305`, no env vars needed — the extension discovers it automatically.

## Model Capabilities

Model labels from the Lemonade catalog are mapped to pi capabilities:

| Label | pi Mapping |
|-------|-----------|
| `reasoning` | `reasoning: true` |
| `vision` | `input: ["text", "image"]` |
| `tool-calling` | (server handles tools natively) |
| `coding` | (informational only) |

Non-chat models (image generation, transcription, embeddings, reranking, TTS) are excluded from the provider.

## Model Selection Tips

- **Fast & small**: `lemonade/Qwen3.5-4B-GGUF` (3.6 GB, vision, tool-calling) — great for quick coding tasks
- **Reasoning**: `lemonade/Qwen3-8B-GGUF` (5.2 GB, reasoning) — good for complex logic
- **Coding**: `lemonade/Qwen3-Coder-30B-A3B-Instruct-GGUF` (18.6 GB, coding, tool-calling)
- **Vision**: `lemonade/Qwen3-VL-8B-Instruct-GGUF` (6.2 GB, vision)

## Troubleshooting

**"no server found"** — Make sure Lemonade Server is running and reachable. Set `LEMONADE_HOST` explicitly.

```bash
# Test connectivity
curl http://localhost:13305/api/version
```

**"Stream ended without finish_reason" / context overflow** — pi's system prompt is large. Increase the server's context size:

```bash
# On the Lemonade Server host, set ctx_size before starting:
export LEMONADE_CTX_SIZE=32768
# Or in the k8s Deployment:
kubectl set env deploy/lemonade LEMONADE_CTX_SIZE=32768
```

**"failed to fetch models"** — The server is reachable but the `/v1/models` endpoint returned an error. Check server logs.

**Models don't appear** — Models with `downloaded: false` auto-download on first use via Lemonade's `/v1/chat/completions` endpoint. Select a model and send a prompt — Lemonade handles the download transparently.
