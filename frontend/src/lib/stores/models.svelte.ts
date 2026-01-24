/**
 * Model configuration state management using Svelte 5 runes.
 *
 * Lazy-loads model characteristics (config.json) on demand and caches results.
 * Falls back to parsing model name/tags when config.json is unavailable.
 */

/* eslint-disable svelte/prefer-svelte-reactivity -- Using Map with reassignment for reactivity */
import { models as modelsApi } from "$api";
import type { ModelCharacteristics } from "$api";

export interface ConfigState {
  characteristics: ModelCharacteristics | null;
  loading: boolean;
  error: string | null;
}

/**
 * Architecture family patterns to detect from model names.
 * Order matters - more specific patterns first.
 */
const ARCHITECTURE_PATTERNS: [RegExp, string][] = [
  [/\bllama[-_]?3/i, "Llama"],
  [/\bllama/i, "Llama"],
  [/\bqwen/i, "Qwen"],
  [/\bglm/i, "GLM"],
  [/\bmistral/i, "Mistral"],
  [/\bmixtral/i, "Mixtral"],
  [/\bphi[-_]?[234]/i, "Phi"],
  [/\bgemma/i, "Gemma"],
  [/\bdeepseek/i, "DeepSeek"],
  [/\byi[-_]/i, "Yi"],
  [/\bstarcoder/i, "StarCoder"],
  [/\bcodellama/i, "CodeLlama"],
  [/\bfalcon/i, "Falcon"],
  [/\bvicuna/i, "Vicuna"],
  [/\binternlm/i, "InternLM"],
  [/\bbaichuan/i, "Baichuan"],
  [/\bchatglm/i, "ChatGLM"],
  [/\borca/i, "Orca"],
  [/\bneuralchat/i, "NeuralChat"],
  [/\bopenchat/i, "OpenChat"],
  [/\bzephyr/i, "Zephyr"],
  [/\bhermes/i, "Hermes"],
  [/\bsolar/i, "Solar"],
  [/\bcommand[-_]?r/i, "Command-R"],
  [/\bdbrx/i, "DBRX"],
  [/\bolmo/i, "OLMo"],
  [/\bkokoro/i, "Kokoro"],
  [/\bwhisper/i, "Whisper"],
];

/**
 * Quantization patterns to detect from model names.
 */
const QUANTIZATION_PATTERNS: [RegExp, number][] = [
  [/\b2[-_]?bit\b/i, 2],
  [/\b3[-_]?bit\b/i, 3],
  [/\b4[-_]?bit\b/i, 4],
  [/\b5[-_]?bit\b/i, 5],
  [/\b6[-_]?bit\b/i, 6],
  [/\b8[-_]?bit\b/i, 8],
  [/\b16[-_]?bit\b/i, 16],
  [/\bbf16\b/i, 16],
  [/\bfp16\b/i, 16],
  [/\bmxfp4\b/i, 4],
  [/[-_]2bit\b/i, 2],
  [/[-_]3bit\b/i, 3],
  [/[-_]4bit\b/i, 4],
  [/[-_]5bit\b/i, 5],
  [/[-_]6bit\b/i, 6],
  [/[-_]8bit\b/i, 8],
];

/**
 * Multimodal detection patterns.
 */
const MULTIMODAL_PATTERNS = [
  /\bvision\b/i,
  /\bvl\b/i,
  /\bmultimodal\b/i,
  /\bimage[-_]?text/i,
  /\bllava/i,
  /\bpixtral/i,
  /\bqwen[-_]?vl/i,
  /\binternvl/i,
];

/**
 * Tool-use detection patterns (from HuggingFace tags).
 */
const TOOL_USE_PATTERNS = [
  /\btool[-_]?use\b/i,
  /\bfunction[-_]?calling\b/i,
  /\btool[-_]?calling\b/i,
  /\btools\b/i,
];

/**
 * Parse model characteristics from model name and tags.
 * Used as fallback when config.json is unavailable.
 */
export function parseCharacteristicsFromName(
  modelId: string,
  tags: string[] = []
): ModelCharacteristics {
  const modelName = modelId.split("/").pop() || modelId;
  const searchText = `${modelName} ${tags.join(" ")}`;

  const characteristics: ModelCharacteristics = {};

  // Detect architecture family
  for (const [pattern, family] of ARCHITECTURE_PATTERNS) {
    if (pattern.test(searchText)) {
      characteristics.architecture_family = family;
      break;
    }
  }

  // Detect quantization
  for (const [pattern, bits] of QUANTIZATION_PATTERNS) {
    if (pattern.test(modelName)) {
      characteristics.quantization_bits = bits;
      break;
    }
  }

  // Detect multimodal
  for (const pattern of MULTIMODAL_PATTERNS) {
    if (pattern.test(searchText)) {
      characteristics.is_multimodal = true;
      characteristics.multimodal_type = "vision";
      break;
    }
  }

  // Detect tool-use (check tags)
  for (const pattern of TOOL_USE_PATTERNS) {
    if (pattern.test(searchText)) {
      characteristics.is_tool_use = true;
      break;
    }
  }

  return characteristics;
}

class ModelConfigStore {
  private configs = $state<Map<string, ConfigState>>(new Map());

  /**
   * Get cached config state for a model (if any).
   */
  getConfig(modelId: string): ConfigState | undefined {
    return this.configs.get(modelId);
  }

  /**
   * Fetch config for a model (lazy load).
   * Does nothing if already loaded, loading, or previously attempted.
   * Falls back to parsing model name/tags when API fails.
   *
   * @param modelId - HuggingFace model ID (e.g., "mlx-community/Llama-3-8B-4bit")
   * @param tags - Optional HuggingFace tags for fallback parsing
   */
  async fetchConfig(modelId: string, tags: string[] = []): Promise<void> {
    // Don't refetch if already loaded, loading, or attempted (prevents infinite loops)
    const existing = this.configs.get(modelId);
    if (existing) return;

    // Set loading state (create new Map for reactivity)
    this.configs = new Map(this.configs).set(modelId, {
      characteristics: null,
      loading: true,
      error: null,
    });

    try {
      const characteristics = await modelsApi.getConfig(modelId, tags.length > 0 ? tags : undefined);
      this.configs = new Map(this.configs).set(modelId, {
        characteristics,
        loading: false,
        error: null,
      });
    } catch {
      // API failed - fall back to parsing model name and tags
      const fallbackCharacteristics = parseCharacteristicsFromName(
        modelId,
        tags
      );
      const hasAnyCharacteristic =
        fallbackCharacteristics.architecture_family ||
        fallbackCharacteristics.quantization_bits ||
        fallbackCharacteristics.is_multimodal ||
        fallbackCharacteristics.is_tool_use;

      this.configs = new Map(this.configs).set(modelId, {
        characteristics: hasAnyCharacteristic ? fallbackCharacteristics : null,
        loading: false,
        error: null, // Don't show error if we have fallback data
      });
    }
  }

  /**
   * Clear cached config for a specific model.
   */
  clearConfig(modelId: string): void {
    const newMap = new Map(this.configs);
    newMap.delete(modelId);
    this.configs = newMap;
  }

  /**
   * Clear all cached configs.
   */
  clearAll(): void {
    this.configs = new Map();
  }
}

export const modelConfigStore = new ModelConfigStore();
