/**
 * Filter state and constants for model filtering
 */

export interface FilterState {
  architectures: string[];
  multimodal: boolean | null;
  quantization: number[];
}

export const ARCHITECTURE_OPTIONS = [
  "Llama",
  "Qwen",
  "Mistral",
  "Gemma",
  "Phi",
  "DeepSeek",
  "StarCoder",
  "GLM",
  "MiniMax",
];

export const QUANTIZATION_OPTIONS = [2, 3, 4, 8, 16];

export function createEmptyFilters(): FilterState {
  return {
    architectures: [],
    multimodal: null,
    quantization: [],
  };
}
