/**
 * @vitest-environment jsdom
 */
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { modelConfigStore, parseCharacteristicsFromName } from "./models.svelte";
import { flushSync } from "svelte";

// Mock the API client
vi.mock("$api", () => ({
  models: {
    getConfig: vi.fn(),
  },
}));

import { models as modelsApi } from "$api";

describe("ModelConfigStore", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    modelConfigStore.clearAll();
  });

  afterEach(() => {
    modelConfigStore.clearAll();
  });

  describe("getConfig", () => {
    it("returns undefined for uncached model", () => {
      const result = modelConfigStore.getConfig("test-model");
      expect(result).toBeUndefined();
    });

    it("returns cached state after fetch", async () => {
      const mockCharacteristics = {
        architecture_family: "Llama",
        is_multimodal: false,
        quantization_bits: 4,
      };
      vi.mocked(modelsApi.getConfig).mockResolvedValue(mockCharacteristics);

      await modelConfigStore.fetchConfig("test-model");
      flushSync();

      const result = modelConfigStore.getConfig("test-model");
      expect(result).toBeDefined();
      expect(result?.characteristics).toEqual(mockCharacteristics);
      expect(result?.loading).toBe(false);
      expect(result?.error).toBeNull();
    });
  });

  describe("fetchConfig", () => {
    it("sets loading state before fetch completes", async () => {
      vi.mocked(modelsApi.getConfig).mockImplementation(
        () => new Promise(() => {}) // Never resolves
      );

      // Start fetch but don't await
      void modelConfigStore.fetchConfig("loading-model");
      flushSync();

      const state = modelConfigStore.getConfig("loading-model");
      expect(state?.loading).toBe(true);
      expect(state?.characteristics).toBeNull();
      expect(state?.error).toBeNull();

      // Cleanup - cancel by clearing
      modelConfigStore.clearAll();
    });

    it("does not refetch if already loaded", async () => {
      const mockCharacteristics = {
        architecture_family: "Qwen",
        is_multimodal: true,
        quantization_bits: 8,
      };
      vi.mocked(modelsApi.getConfig).mockResolvedValue(mockCharacteristics);

      // First fetch
      await modelConfigStore.fetchConfig("cached-model");
      flushSync();

      // Second fetch should be a no-op
      await modelConfigStore.fetchConfig("cached-model");

      // Should only be called once
      expect(modelsApi.getConfig).toHaveBeenCalledTimes(1);
    });

    it("does not refetch if currently loading", async () => {
      vi.mocked(modelsApi.getConfig).mockImplementation(
        () =>
          new Promise((resolve) =>
            setTimeout(
              () => resolve({ architecture_family: "Test" } as never),
              100
            )
          )
      );

      // Start first fetch
      const promise1 = modelConfigStore.fetchConfig("loading-model");
      flushSync();

      // Try second fetch while first is still loading
      const promise2 = modelConfigStore.fetchConfig("loading-model");

      await Promise.all([promise1, promise2]);

      // Should only be called once
      expect(modelsApi.getConfig).toHaveBeenCalledTimes(1);
    });

    it("handles fetch errors by using name-based fallback", async () => {
      vi.mocked(modelsApi.getConfig).mockRejectedValue(
        new Error("Network error")
      );

      // Model name contains parseable info: "Llama" and "8bit"
      await modelConfigStore.fetchConfig("mlx-community/Llama-3-8B-8bit", [
        "mlx",
      ]);
      flushSync();

      const state = modelConfigStore.getConfig("mlx-community/Llama-3-8B-8bit");
      expect(state?.loading).toBe(false);
      expect(state?.error).toBeNull(); // No error when fallback works
      expect(state?.characteristics?.architecture_family).toBe("Llama");
      expect(state?.characteristics?.quantization_bits).toBe(8);
    });

    it("handles fetch errors with no parseable info", async () => {
      vi.mocked(modelsApi.getConfig).mockRejectedValue("string error");

      // Model name has no parseable architecture or quantization
      await modelConfigStore.fetchConfig("some-random-model");
      flushSync();

      const state = modelConfigStore.getConfig("some-random-model");
      expect(state?.loading).toBe(false);
      expect(state?.characteristics).toBeNull();
      expect(state?.error).toBeNull(); // No error shown to user
    });

    it("calls API with correct model ID", async () => {
      vi.mocked(modelsApi.getConfig).mockResolvedValue({
        architecture_family: "Mistral",
      } as never);

      await modelConfigStore.fetchConfig("mlx-community/Mistral-7B-4bit");

      expect(modelsApi.getConfig).toHaveBeenCalledWith(
        "mlx-community/Mistral-7B-4bit"
      );
    });
  });

  describe("clearConfig", () => {
    it("removes cached config for specific model", async () => {
      vi.mocked(modelsApi.getConfig).mockResolvedValue({
        architecture_family: "Llama",
      } as never);

      await modelConfigStore.fetchConfig("model-a");
      await modelConfigStore.fetchConfig("model-b");
      flushSync();

      expect(modelConfigStore.getConfig("model-a")).toBeDefined();
      expect(modelConfigStore.getConfig("model-b")).toBeDefined();

      modelConfigStore.clearConfig("model-a");
      flushSync();

      expect(modelConfigStore.getConfig("model-a")).toBeUndefined();
      expect(modelConfigStore.getConfig("model-b")).toBeDefined();
    });

    it("handles clearing non-existent model", () => {
      // Should not throw
      expect(() => modelConfigStore.clearConfig("non-existent")).not.toThrow();
    });
  });

  describe("clearAll", () => {
    it("removes all cached configs", async () => {
      vi.mocked(modelsApi.getConfig).mockResolvedValue({
        architecture_family: "Phi",
      } as never);

      await modelConfigStore.fetchConfig("model-1");
      await modelConfigStore.fetchConfig("model-2");
      await modelConfigStore.fetchConfig("model-3");
      flushSync();

      modelConfigStore.clearAll();
      flushSync();

      expect(modelConfigStore.getConfig("model-1")).toBeUndefined();
      expect(modelConfigStore.getConfig("model-2")).toBeUndefined();
      expect(modelConfigStore.getConfig("model-3")).toBeUndefined();
    });
  });

  describe("reactivity", () => {
    it("updates state reactively on successful fetch", async () => {
      const mockCharacteristics = {
        architecture_family: "DeepSeek",
        is_multimodal: false,
        quantization_bits: 4,
        max_position_embeddings: 8192,
      };
      vi.mocked(modelsApi.getConfig).mockResolvedValue(mockCharacteristics);

      // Initially undefined
      expect(modelConfigStore.getConfig("reactive-model")).toBeUndefined();

      await modelConfigStore.fetchConfig("reactive-model");
      flushSync();

      // Now should have data
      const state = modelConfigStore.getConfig("reactive-model");
      expect(state?.characteristics?.architecture_family).toBe("DeepSeek");
      expect(state?.characteristics?.max_position_embeddings).toBe(8192);
    });
  });
});

describe("parseCharacteristicsFromName", () => {
  it("detects Llama architecture", () => {
    const result = parseCharacteristicsFromName("mlx-community/Llama-3.1-8B-4bit");
    expect(result.architecture_family).toBe("Llama");
  });

  it("detects GLM architecture", () => {
    const result = parseCharacteristicsFromName("mlx-community/GLM-4.7-Flash-8bit");
    expect(result.architecture_family).toBe("GLM");
  });

  it("detects Qwen architecture", () => {
    const result = parseCharacteristicsFromName("Qwen2.5-0.5B-Instruct-4bit");
    expect(result.architecture_family).toBe("Qwen");
  });

  it("detects quantization from model name", () => {
    expect(
      parseCharacteristicsFromName("Model-4bit").quantization_bits
    ).toBe(4);
    expect(
      parseCharacteristicsFromName("Model-8bit").quantization_bits
    ).toBe(8);
    expect(
      parseCharacteristicsFromName("Model-bf16").quantization_bits
    ).toBe(16);
  });

  it("detects multimodal from name patterns", () => {
    expect(
      parseCharacteristicsFromName("Qwen-VL-4bit").is_multimodal
    ).toBe(true);
    expect(
      parseCharacteristicsFromName("LLaVA-1.5-7B").is_multimodal
    ).toBe(true);
  });

  it("detects multimodal from tags", () => {
    const result = parseCharacteristicsFromName("some-model", ["vision", "mlx"]);
    expect(result.is_multimodal).toBe(true);
  });

  it("combines architecture and quantization", () => {
    const result = parseCharacteristicsFromName(
      "mlx-community/Mistral-7B-Instruct-v0.3-4bit"
    );
    expect(result.architecture_family).toBe("Mistral");
    expect(result.quantization_bits).toBe(4);
  });

  it("returns empty object for unparseable names", () => {
    const result = parseCharacteristicsFromName("random-model-name");
    expect(result.architecture_family).toBeUndefined();
    expect(result.quantization_bits).toBeUndefined();
    expect(result.is_multimodal).toBeUndefined();
  });
});
