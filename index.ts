/**
 * Lemonade Server Provider for pi
 *
 * Auto-discovers a Lemonade Server on the local network or at known hosts,
 * fetches the model catalog, and registers chat/text models as a pi provider.
 *
 * Discovery order:
 *   1. LEMONADE_API_BASE env var (e.g. "http://192.168.0.6:13305/v1")
 *   2. LEMONADE_HOST env var (e.g. "http://192.168.0.6:13305")
 *   3. http://localhost:13305/v1  (default port)
 *
 * Once discovered, it calls GET /v1/models?show_all=true, filters to
 * chat-capable models (llamacpp, vllm backends), and registers them
 * with pi using the openai-completions API.
 *
 * Model labels are mapped to pi capabilities:
 *   - "reasoning"          → reasoning: true
 *   - "vision"             → input: ["text", "image"]
 *   - "tool-calling"       → (keeps text input, server handles tools)
 *   - "coding"             → (no special mapping, informational)
 *   - "hot"                → (no special mapping, informational)
 *
 * Non-chat models (image generation, transcription, embeddings, reranking,
 * TTS, upscaling, omni collections) are excluded.
 */

import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface LemonadeModelEntry {
  id: string;
  object: "model";
  owned_by: string;
  recipe: string;
  labels: string[];
  size: number; // GB
  downloaded: boolean;
  suggested: boolean;
  checkpoint: string;
  checkpoints: Record<string, string>;
  components: string[];
  created: number;
  max_context_window?: number;
  recipe_options: Record<string, unknown>;
  image_defaults?: Record<string, unknown>;
}

interface LemonadeModelList {
  data: LemonadeModelEntry[];
  object: "list";
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/** Recipes that produce chat / text-generation models. */
const CHAT_RECIPES = new Set(["llamacpp", "vllm"]);

/** Labels that indicate a model is NOT usable for chat. */
const NON_CHAT_LABELS = new Set([
  "image",          // Stable Diffusion / Flux
  "transcription",  // Whisper
  "realtime-transcription",
  "embeddings",     // embedding models
  "reranking",      // reranker models
  "tts",            // text-to-speech
  "upscaling",      // RealESRGAN
  "edit",           // image editing
  "chat-transcription", // omni audio (skip — audio endpoint only)
]);

/** Default context windows per rough model size bracket.
 *  Note: Lemonade Server defaults to ctx_size=4096 (configurable via
 *  LEMONADE_CTX_SIZE env var on the server). We use 4096 as a safe
 *  default and use the model's max_context_window when available. */
const DEFAULT_CTX: Record<string, number> = {
  tiny: 4096,
  small: 4096,
  medium: 4096,
  large: 4096,
  xlarge: 4096,
};

/** Default max output tokens per model size bracket. */
const DEFAULT_MAX_TOKENS: Record<string, number> = {
  tiny: 4096,
  small: 8192,
  medium: 16384,
  large: 32768,
  xlarge: 65536,
};

/** Max tokens for reasoning models — doubled since they spend tokens thinking. */
const REASONING_MAX_TOKENS: Record<string, number> = {
  tiny: 8192,
  small: 16384,
  medium: 32768,
  large: 65536,
  xlarge: 65536,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function sizeBracket(sizeGb: number): string {
  if (sizeGb <= 0) return "tiny";
  if (sizeGb < 1) return "tiny";
  if (sizeGb < 3) return "small";
  if (sizeGb < 10) return "medium";
  if (sizeGb < 30) return "large";
  return "xlarge";
}

/**
 * Returns true if the model is a chat / instruction-following text model.
 *
 * We include entries whose recipe is in CHAT_RECIPES, then exclude any
 * that are *purely* non-chat (e.g. embedding-only, reranking-only).
 * Models with both chat and non-chat labels (e.g. "vision" + "tool-calling")
 * are kept.
 */
function isChatModel(entry: LemonadeModelEntry): boolean {
  // Must be a chat recipe.
  if (!CHAT_RECIPES.has(entry.recipe)) return false;

  // Exclude models whose *only* labels are non-chat.
  if (entry.labels.length > 0) {
    const remainingLabels = entry.labels.filter((l) => !NON_CHAT_LABELS.has(l));
    if (remainingLabels.length === 0) return false;
  }

  return true;
}

/** Parse a human-readable size string like "2.38" GB. */
function parseSizeGb(size: number | undefined): number {
  if (size === undefined || size === null) return 2; // default guess
  return Number(size);
}

// ---------------------------------------------------------------------------
// Discovery
// ---------------------------------------------------------------------------

interface DiscoveredServer {
  baseUrl: string; // full OpenAI-compatible base (e.g. "http://host:13305/v1")
  host: string;
  port: number;
  version?: string;
}

/**
 * Try to discover a Lemonade Server.
 *
 * Strategy (in order):
 *   1. LEMONADE_API_BASE  — full base URL (may already end with /v1)
 *   2. LEMONADE_HOST      — host:port (no /v1 suffix)
 *   3. Localhost defaults  — 13305 (Lemonade default), 8000 (LiteLLM default)
 */
async function discoverServer(): Promise<DiscoveredServer | null> {
  // 1. Explicit env var — full base URL
  const apiBase = process.env.LEMONADE_API_BASE?.trim();
  if (apiBase) {
    const url = ensureV1(apiBase);
    const { host, port } = parseHostPort(url);
    const version = await fetchVersion(url);
    return { baseUrl: url, host, port, version };
  }

  // 2. LEMONADE_HOST — host:port without /v1
  const lemonadeHost = process.env.LEMONADE_HOST?.trim();
  if (lemonadeHost) {
    const base = ensureV1(`http://${lemonadeHost.replace(/^https?:\/\//, "")}`);
    const { host, port } = parseHostPort(base);
    const version = await fetchVersion(base);
    return { baseUrl: base, host, port, version };
  }

  // 3. Known localhost ports
  const localPorts = [13305, 8000];
  for (const port of localPorts) {
    const base = `http://localhost:${port}/v1`;
    const version = await fetchVersion(base);
    if (version !== undefined) {
      return { baseUrl: base, host: "localhost", port, version };
    }
  }

  return null;
}

/** Make sure the URL ends with /v1 */
function ensureV1(url: string): string {
  let u = url.replace(/\/+$/, "");
  if (!u.endsWith("/v1")) {
    u = u + "/v1";
  }
  if (!u.startsWith("http")) {
    u = `http://${u}`;
  }
  return u;
}

/** Extract host & port from a URL. */
function parseHostPort(baseUrl: string): { host: string; port: number } {
  try {
    const u = new URL(baseUrl);
    return {
      host: u.hostname,
      port: parseInt(u.port) || 13305,
    };
  } catch {
    return { host: "localhost", port: 13305 };
  }
}

/** Try to fetch /api/version; returns version string or undefined. */
async function fetchVersion(baseUrl: string): Promise<string | undefined> {
  try {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 2000);

    const resp = await fetch(`${baseUrl.replace(/\/v1\/?$/, "")}/api/version`, {
      signal: controller.signal,
    });
    clearTimeout(timeout);

    if (resp.ok) {
      const data = (await resp.json()) as { version?: string };
      return data.version;
    }
    return undefined;
  } catch {
    return undefined;
  }
}

// ---------------------------------------------------------------------------
// Model fetching
// ---------------------------------------------------------------------------

/** Fetch all models from the server. */
async function fetchModels(baseUrl: string): Promise<LemonadeModelEntry[]> {
  const resp = await fetch(`${baseUrl}/models?show_all=true`);
  if (!resp.ok) {
    throw new Error(
      `Lemonade Server returned ${resp.status} for /v1/models`
    );
  }
  const json = (await resp.json()) as LemonadeModelList;
  return json.data ?? [];
}

// ---------------------------------------------------------------------------
// Model mapping
// ---------------------------------------------------------------------------

interface ProviderModel {
  id: string;
  name: string;
  reasoning: boolean;
  input: string[];
  contextWindow: number;
  maxTokens: number;
  cost: {
    input: number;
    output: number;
    cacheRead: number;
    cacheWrite: number;
  };
  compat?: {
    thinkingFormat?: string;
    supportsDeveloperRole?: boolean;
    supportsReasoningEffort?: boolean;
  };
}

/** Map a Lemonade model entry to a pi provider model definition. */
function mapModel(entry: LemonadeModelEntry): ProviderModel {
  const sizeGb = parseSizeGb(entry.size);
  const bracket = sizeBracket(sizeGb);

  const hasVision = entry.labels.includes("vision");
  const hasReasoning = entry.labels.includes("reasoning");

  const contextWindow =
    entry.max_context_window ?? DEFAULT_CTX[bracket] ?? 4096;
  const maxTokens = hasReasoning
    ? (REASONING_MAX_TOKENS[bracket] ?? 32768)
    : (DEFAULT_MAX_TOKENS[bracket] ?? 16384);

  // Build a human-readable name with size info and labels
  const sizeStr = sizeGb >= 1 ? `${sizeGb.toFixed(1)} GB` : `${(sizeGb * 1024).toFixed(0)} MB`;
  const labelHints = entry.labels
    .filter((l) => !NON_CHAT_LABELS.has(l) && l !== "reasoning" && l !== "llamacpp")
    .join(", ");

  let name = entry.id;
  if (labelHints) {
    name = `${entry.id} (${labelHints})`;
  }

  // Reasoning models: enable deepseek thinking format so pi parses
  // reasoning_content streams and sends thinking: { type: "enabled" }.
  // Provider-level compat already sets supportsDeveloperRole: false and
  // supportsReasoningEffort: false, so they won't interfere.
  let compat: ProviderModel["compat"] | undefined;
  if (hasReasoning) {
    compat = {
      thinkingFormat: "deepseek",
    };
  }

  return {
    id: entry.id,
    name,
    reasoning: hasReasoning,
    input: hasVision ? ["text", "image"] : ["text"],
    contextWindow,
    maxTokens,
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    ...(compat ? { compat } : {}),
  };
}

// ---------------------------------------------------------------------------
// Extension Entry Point
// ---------------------------------------------------------------------------

export default async function (pi: ExtensionAPI) {
  // ── Phase 1: Discover the Lemonade Server ──
  const server = await discoverServer();

  if (!server) {
    // No server found — register an empty provider so the user can
    // still configure it manually via models.json if desired.
    pi.registerProvider("lemonade", {
      baseUrl: "http://localhost:13305/v1",
      api: "openai-completions",
      apiKey: "lemonade",
      compat: {
        supportsDeveloperRole: false,
        supportsReasoningEffort: false,
      },
      models: [],
    });

    pi.on("session_start", async (_event, ctx) => {
      ctx.ui.notify(
        "Lemonade: no server found. Set LEMONADE_HOST env var or check http://localhost:13305",
        "info"
      );
    });
    return;
  }

  // ── Phase 2: Fetch the model catalog ──
  let rawModels: LemonadeModelEntry[];
  try {
    rawModels = await fetchModels(server.baseUrl);
  } catch (err) {
    const errMsg = err instanceof Error ? err.message : String(err);
    pi.registerProvider("lemonade", {
      name: `Lemonade (${server.host}:${server.port})`,
      baseUrl: server.baseUrl,
      api: "openai-completions",
      apiKey: "lemonade",
      compat: {
        supportsDeveloperRole: false,
        supportsReasoningEffort: false,
      },
      models: [],
    });

    pi.on("session_start", async (_event, ctx) => {
      ctx.ui.notify(
        `Lemonade: connected to ${server.host}:${server.port} but failed to fetch models: ${errMsg}`,
        "warning"
      );
    });
    return;
  }

  // ── Phase 3: Filter and map chat models ──
  const totalFound = rawModels.length;

  const chatModels = rawModels
    .filter((m) => m.id && !m.id.startsWith("LMX-") && m.id !== "Lite Collection" && m.id !== "Ultra Collection")
    .filter(isChatModel)
    .sort((a, b) => a.id.localeCompare(b.id))
    .map(mapModel);

  const reasoningCount = chatModels.filter((m) => m.reasoning).length;
  const visionCount = chatModels.filter((m) => m.input.includes("image")).length;

  // ── Phase 4: Register the provider ──
  pi.registerProvider("lemonade", {
    name: `Lemonade (${server.host}:${server.port})`,
    baseUrl: server.baseUrl,
    api: "openai-completions",
    apiKey: "lemonade",
    compat: {
      supportsDeveloperRole: false,
      supportsReasoningEffort: false,
    },
    models: chatModels,
  });

  // ── Notify on session start ──
  pi.on("session_start", async (_event, ctx) => {
    const ver = server.version ? ` v${server.version}` : "";
    ctx.ui.notify(
      `Lemonade${ver} @ ${server.host}:${server.port} — ${chatModels.length} chat models (${reasoningCount} reasoning, ${visionCount} vision) from ${totalFound} total`,
      "info"
    );
  });

  // ── Rewrite Lemonade context-overflow errors so pi can auto-compact ──
  // Lemonade returns: {"error":{"code":400,"message":"request (N tokens) exceeds ...","type":"exceed_context_size_error"}}
  // This is a non-streaming error in a 200 response. pi sees it as a stream
  // that never produces finish_reason. Normalize to context_length_exceeded.
  pi.on("message_end", (event, ctx) => {
    const message = event.message;
    if (message.role !== "assistant") return;
    if (message.stopReason !== "error") return;
    if (message.provider !== "lemonade" && ctx.model?.provider !== "lemonade") return;

    const em = message.errorMessage ?? "";
    if (em.includes("context_length_exceeded")) return;
    if (!em.includes("exceeds the available context size")) return;

    return {
      message: {
        ...message,
        errorMessage: `context_length_exceeded: ${em}`,
      },
    };
  });
}
