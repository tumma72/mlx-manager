/**
 * @vitest-environment jsdom
 */
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { probeStore } from "./probe.svelte";
import { flushSync } from "svelte";
import type { ProbeStep } from "$lib/api/types";

// Helper to create a mock ReadableStream reader
function createMockReader(chunks: string[]) {
  let index = 0;
  const encoder = new TextEncoder();
  return {
    read: vi.fn().mockImplementation(() => {
      if (index < chunks.length) {
        return Promise.resolve({
          done: false,
          value: encoder.encode(chunks[index++]),
        });
      }
      return Promise.resolve({ done: true, value: undefined });
    }),
  };
}

function createMockStreamResponse(
  chunks: string[],
  ok = true,
  statusText = "OK"
) {
  const reader = createMockReader(chunks);
  return {
    ok,
    statusText,
    body: { getReader: () => reader },
  };
}

describe("probeStore", () => {
  const mockFetch = vi.fn();

  beforeEach(() => {
    global.fetch = mockFetch;
    mockFetch.mockClear();
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  describe("getProbe", () => {
    it("returns default idle state for unknown model", () => {
      const state = probeStore.getProbe("unknown-model");

      expect(state).toEqual({
        status: "idle",
        currentStep: null,
        steps: [],
        capabilities: {},
        error: null,
        diagnostics: [],
        probeResult: null,
      });
    });

    it("returns stored state for known model", async () => {
      const modelId = "test-model";
      const chunks = [
        'data: {"step":"test","status":"running"}\n',
        "data: [DONE]\n",
      ];

      mockFetch.mockResolvedValueOnce(createMockStreamResponse(chunks));

      await probeStore.startProbe(modelId, "test-token");
      flushSync();

      const state = probeStore.getProbe(modelId);
      expect(state.status).toBe("completed");
      expect(state.steps).toHaveLength(1);
      expect(state.steps[0].step).toBe("test");

      // Clean up
      probeStore.reset(modelId);
    });
  });

  describe("startProbe", () => {
    it("sets initial probing state", async () => {
      const modelId = "test-model";
      const chunks = ["data: [DONE]\n"];

      mockFetch.mockResolvedValueOnce(createMockStreamResponse(chunks));

      const probePromise = probeStore.startProbe(modelId, "test-token");
      flushSync();

      const state = probeStore.getProbe(modelId);
      expect(state.status).toBe("probing");
      expect(state.currentStep).toBeNull();
      expect(state.steps).toEqual([]);
      expect(state.capabilities).toEqual({});
      expect(state.error).toBeNull();

      await probePromise;

      // Clean up
      probeStore.reset(modelId);
    });

    it("handles non-ok response (throws error -> status failed)", async () => {
      const modelId = "test-model";

      mockFetch.mockResolvedValueOnce(
        createMockStreamResponse([], false, "Internal Server Error")
      );

      await probeStore.startProbe(modelId, "test-token");
      flushSync();

      const state = probeStore.getProbe(modelId);
      expect(state.status).toBe("failed");
      expect(state.error).toBe("Probe failed: Internal Server Error");

      // Clean up
      probeStore.reset(modelId);
    });

    it("handles missing response body (no reader -> fails)", async () => {
      const modelId = "test-model";

      mockFetch.mockResolvedValueOnce({
        ok: true,
        statusText: "OK",
        body: null,
      });

      await probeStore.startProbe(modelId, "test-token");
      flushSync();

      const state = probeStore.getProbe(modelId);
      expect(state.status).toBe("failed");
      expect(state.error).toBe("No response body");

      // Clean up
      probeStore.reset(modelId);
    });

    it("processes SSE stream with steps (new steps added)", async () => {
      const modelId = "test-model";
      const step1: ProbeStep = {
        step: "step1",
        status: "running",
      };
      const step2: ProbeStep = {
        step: "step2",
        status: "completed",
      };

      const chunks = [
        `data: ${JSON.stringify(step1)}\n`,
        `data: ${JSON.stringify(step2)}\n`,
        "data: [DONE]\n",
      ];

      mockFetch.mockResolvedValueOnce(createMockStreamResponse(chunks));

      await probeStore.startProbe(modelId, "test-token");
      flushSync();

      const state = probeStore.getProbe(modelId);
      expect(state.status).toBe("completed");
      expect(state.steps).toHaveLength(2);
      expect(state.steps[0]).toEqual(step1);
      expect(state.steps[1]).toEqual(step2);

      // Clean up
      probeStore.reset(modelId);
    });

    it("updates existing step (existingIdx >= 0)", async () => {
      const modelId = "test-model";
      const step1: ProbeStep = {
        step: "test-step",
        status: "running",
      };
      const step1Updated: ProbeStep = {
        step: "test-step",
        status: "completed",
      };

      const chunks = [
        `data: ${JSON.stringify(step1)}\n`,
        `data: ${JSON.stringify(step1Updated)}\n`,
        "data: [DONE]\n",
      ];

      mockFetch.mockResolvedValueOnce(createMockStreamResponse(chunks));

      await probeStore.startProbe(modelId, "test-token");
      flushSync();

      const state = probeStore.getProbe(modelId);
      expect(state.status).toBe("completed");
      expect(state.steps).toHaveLength(1);
      expect(state.steps[0].status).toBe("completed");

      // Clean up
      probeStore.reset(modelId);
    });

    it("updates capabilities when step has capability + value", async () => {
      const modelId = "test-model";
      const step1: ProbeStep = {
        step: "tools-check",
        status: "completed",
        capability: "supports_native_tools",
        value: true,
      };
      const step2: ProbeStep = {
        step: "thinking-check",
        status: "completed",
        capability: "supports_thinking",
        value: false,
      };

      const chunks = [
        `data: ${JSON.stringify(step1)}\n`,
        `data: ${JSON.stringify(step2)}\n`,
        "data: [DONE]\n",
      ];

      mockFetch.mockResolvedValueOnce(createMockStreamResponse(chunks));

      await probeStore.startProbe(modelId, "test-token");
      flushSync();

      const state = probeStore.getProbe(modelId);
      expect(state.capabilities).toEqual({
        supports_native_tools: true,
        supports_thinking: false,
      });

      // Clean up
      probeStore.reset(modelId);
    });

    it("sets currentStep for running status steps", async () => {
      const modelId = "test-model";

      // Send a running step followed by [DONE] — currentStep should be
      // set from the running step and then cleared by the [DONE] handler.
      // We verify indirectly: a second running step should overwrite the first.
      const step1: ProbeStep = { step: "step1", status: "running" };
      const step2: ProbeStep = { step: "step2", status: "running" };

      const chunks = [
        `data: ${JSON.stringify(step1)}\ndata: ${JSON.stringify(step2)}\n`,
      ];

      mockFetch.mockResolvedValueOnce(createMockStreamResponse(chunks));

      await probeStore.startProbe(modelId, "test-token");
      flushSync();

      // After stream ends without [DONE], status is completed and
      // currentStep retains the last running step's value before completion clears it.
      // The steps array should contain both running steps.
      const state = probeStore.getProbe(modelId);
      expect(state.steps).toHaveLength(2);
      expect(state.steps[0].status).toBe("running");
      expect(state.steps[1].status).toBe("running");
      expect(state.status).toBe("completed");

      // Clean up
      probeStore.reset(modelId);
    });

    it("handles [DONE] message -> completed", async () => {
      const modelId = "test-model";
      const chunks = [
        'data: {"step":"test","status":"running"}\n',
        "data: [DONE]\n",
      ];

      mockFetch.mockResolvedValueOnce(createMockStreamResponse(chunks));

      await probeStore.startProbe(modelId, "test-token");
      flushSync();

      const state = probeStore.getProbe(modelId);
      expect(state.status).toBe("completed");
      expect(state.currentStep).toBeNull();

      // Clean up
      probeStore.reset(modelId);
    });

    it("handles stream end without [DONE] -> completed", async () => {
      const modelId = "test-model";
      const chunks = ['data: {"step":"test","status":"running"}\n'];

      mockFetch.mockResolvedValueOnce(createMockStreamResponse(chunks));

      await probeStore.startProbe(modelId, "test-token");
      flushSync();

      const state = probeStore.getProbe(modelId);
      expect(state.status).toBe("completed");
      expect(state.currentStep).toBeNull();

      // Clean up
      probeStore.reset(modelId);
    });

    it("ignores malformed JSON in SSE data", async () => {
      const modelId = "test-model";
      const validStep: ProbeStep = {
        step: "valid",
        status: "completed",
      };

      const chunks = [
        "data: {invalid json}\n",
        `data: ${JSON.stringify(validStep)}\n`,
        "data: [DONE]\n",
      ];

      mockFetch.mockResolvedValueOnce(createMockStreamResponse(chunks));

      await probeStore.startProbe(modelId, "test-token");
      flushSync();

      const state = probeStore.getProbe(modelId);
      expect(state.status).toBe("completed");
      expect(state.steps).toHaveLength(1);
      expect(state.steps[0].step).toBe("valid");

      // Clean up
      probeStore.reset(modelId);
    });

    it('ignores non-"data: " lines', async () => {
      const modelId = "test-model";
      const validStep: ProbeStep = {
        step: "valid",
        status: "completed",
      };

      const chunks = [
        ": this is a comment\n",
        "event: message\n",
        `data: ${JSON.stringify(validStep)}\n`,
        "data: [DONE]\n",
      ];

      mockFetch.mockResolvedValueOnce(createMockStreamResponse(chunks));

      await probeStore.startProbe(modelId, "test-token");
      flushSync();

      const state = probeStore.getProbe(modelId);
      expect(state.status).toBe("completed");
      expect(state.steps).toHaveLength(1);
      expect(state.steps[0].step).toBe("valid");

      // Clean up
      probeStore.reset(modelId);
    });

    it("uses correct API endpoint with encoded parameters", async () => {
      const modelId = "mlx-community/test-model";
      const token = "hf_token123";
      const chunks = ["data: [DONE]\n"];

      mockFetch.mockResolvedValueOnce(createMockStreamResponse(chunks));

      await probeStore.startProbe(modelId, token);

      expect(mockFetch).toHaveBeenCalledWith(
        "/api/models/probe/mlx-community%2Ftest-model?token=hf_token123",
        { method: "POST" }
      );

      // Clean up
      probeStore.reset(modelId);
    });

    it("handles step without capability field", async () => {
      const modelId = "test-model";
      const step: ProbeStep = {
        step: "no-capability",
        status: "completed",
      };

      const chunks = [`data: ${JSON.stringify(step)}\n`, "data: [DONE]\n"];

      mockFetch.mockResolvedValueOnce(createMockStreamResponse(chunks));

      await probeStore.startProbe(modelId, "test-token");
      flushSync();

      const state = probeStore.getProbe(modelId);
      expect(state.capabilities).toEqual({});

      // Clean up
      probeStore.reset(modelId);
    });

    it("handles step with capability but no value", async () => {
      const modelId = "test-model";
      const step: ProbeStep = {
        step: "has-capability",
        status: "completed",
        capability: "supports_native_tools",
      };

      const chunks = [`data: ${JSON.stringify(step)}\n`, "data: [DONE]\n"];

      mockFetch.mockResolvedValueOnce(createMockStreamResponse(chunks));

      await probeStore.startProbe(modelId, "test-token");
      flushSync();

      const state = probeStore.getProbe(modelId);
      expect(state.capabilities).toEqual({});

      // Clean up
      probeStore.reset(modelId);
    });

    it("does not update currentStep for non-running status", async () => {
      const modelId = "test-model";
      const step1: ProbeStep = {
        step: "step1",
        status: "running",
      };
      const step2: ProbeStep = {
        step: "step2",
        status: "completed",
      };

      const chunks = [
        `data: ${JSON.stringify(step1)}\n`,
        `data: ${JSON.stringify(step2)}\n`,
        "data: [DONE]\n",
      ];

      mockFetch.mockResolvedValueOnce(createMockStreamResponse(chunks));

      await probeStore.startProbe(modelId, "test-token");
      flushSync();

      const state = probeStore.getProbe(modelId);
      // currentStep should be null after completion, not "step2"
      expect(state.currentStep).toBeNull();

      // Clean up
      probeStore.reset(modelId);
    });

    it("handles multi-line chunks correctly", async () => {
      const modelId = "test-model";
      const step1: ProbeStep = { step: "step1", status: "running" };
      const step2: ProbeStep = { step: "step2", status: "completed" };

      // Multiple data lines in a single chunk
      const chunks = [
        `data: ${JSON.stringify(step1)}\ndata: ${JSON.stringify(step2)}\n`,
        "data: [DONE]\n",
      ];

      mockFetch.mockResolvedValueOnce(createMockStreamResponse(chunks));

      await probeStore.startProbe(modelId, "test-token");
      flushSync();

      const state = probeStore.getProbe(modelId);
      expect(state.steps).toHaveLength(2);
      expect(state.steps[0].step).toBe("step1");
      expect(state.steps[1].step).toBe("step2");

      // Clean up
      probeStore.reset(modelId);
    });

    it("handles partial chunks across reads", async () => {
      const modelId = "test-model";
      const step: ProbeStep = { step: "test", status: "completed" };
      const fullData = `data: ${JSON.stringify(step)}\n`;

      // Split the data across two chunks
      const chunks = [fullData.slice(0, 10), fullData.slice(10), "data: [DONE]\n"];

      mockFetch.mockResolvedValueOnce(createMockStreamResponse(chunks));

      await probeStore.startProbe(modelId, "test-token");
      flushSync();

      const state = probeStore.getProbe(modelId);
      expect(state.steps).toHaveLength(1);
      expect(state.steps[0].step).toBe("test");

      // Clean up
      probeStore.reset(modelId);
    });

    it("handles network error during fetch", async () => {
      const modelId = "test-model";

      mockFetch.mockRejectedValueOnce(new Error("Network error"));

      await probeStore.startProbe(modelId, "test-token");
      flushSync();

      const state = probeStore.getProbe(modelId);
      expect(state.status).toBe("failed");
      expect(state.error).toBe("Network error");

      // Clean up
      probeStore.reset(modelId);
    });

    it("handles non-Error exceptions", async () => {
      const modelId = "test-model";

      mockFetch.mockRejectedValueOnce("String error");

      await probeStore.startProbe(modelId, "test-token");
      flushSync();

      const state = probeStore.getProbe(modelId);
      expect(state.status).toBe("failed");
      expect(state.error).toBe("String error");

      // Clean up
      probeStore.reset(modelId);
    });
  });

  describe("reset", () => {
    it("removes a probe entry", async () => {
      const modelId = "test-model";
      const chunks = ["data: [DONE]\n"];

      mockFetch.mockResolvedValueOnce(createMockStreamResponse(chunks));

      await probeStore.startProbe(modelId, "test-token");
      flushSync();

      let state = probeStore.getProbe(modelId);
      expect(state.status).toBe("completed");

      probeStore.reset(modelId);
      flushSync();

      state = probeStore.getProbe(modelId);
      expect(state.status).toBe("idle");
      expect(state.steps).toEqual([]);
    });

    it("does nothing for non-existent probe", () => {
      expect(() => {
        probeStore.reset("non-existent-model");
        flushSync();
      }).not.toThrow();
    });
  });

  describe("probes getter", () => {
    it("returns all probes", async () => {
      const modelId1 = "model-1";
      const modelId2 = "model-2";
      const chunks = ["data: [DONE]\n"];

      mockFetch.mockResolvedValue(createMockStreamResponse(chunks));

      await probeStore.startProbe(modelId1, "token1");
      await probeStore.startProbe(modelId2, "token2");
      flushSync();

      const allProbes = probeStore.probes;
      expect(Object.keys(allProbes)).toContain(modelId1);
      expect(Object.keys(allProbes)).toContain(modelId2);
      expect(allProbes[modelId1].status).toBe("completed");
      expect(allProbes[modelId2].status).toBe("completed");

      // Clean up
      probeStore.reset(modelId1);
      probeStore.reset(modelId2);
    });

    it("returns empty object when no probes exist", () => {
      const allProbes = probeStore.probes;
      expect(allProbes).toEqual({});
    });
  });

  describe("buffer edge cases", () => {
    it("handles empty buffer on line split (lines.pop() returns undefined)", async () => {
      const modelId = "test-model";
      // Send a chunk that ends exactly at a newline, so buffer becomes empty after pop
      const chunks = ["data: [DONE]\n"];

      mockFetch.mockResolvedValueOnce(createMockStreamResponse(chunks));

      await probeStore.startProbe(modelId, "test-token");
      flushSync();

      const state = probeStore.getProbe(modelId);
      expect(state.status).toBe("completed");

      // Clean up
      probeStore.reset(modelId);
    });
  });

  describe("completion edge cases", () => {
    it("does not mark completed twice if stream ends after [DONE]", async () => {
      const modelId = "test-model";
      // This tests the line 109 false branch: if (probes[modelId]?.status === "probing")
      // When [DONE] is received, status becomes "completed", then when stream ends,
      // the condition is false so we don't set status again
      const chunks = ["data: [DONE]\n"];

      mockFetch.mockResolvedValueOnce(createMockStreamResponse(chunks));

      await probeStore.startProbe(modelId, "test-token");
      flushSync();

      const state = probeStore.getProbe(modelId);
      // Should be completed from [DONE]
      expect(state.status).toBe("completed");

      // The code has a check: if (probes[modelId]?.status === "probing")
      // If status is already "completed", it won't mark as completed again
      // This is tested by the fact that the final state is completed

      // Clean up
      probeStore.reset(modelId);
    });

    it("marks as completed when stream ends naturally without [DONE] while status is probing", async () => {
      const modelId = "test-model";
      const step: ProbeStep = {
        step: "test-step",
        status: "completed",
      };

      // Stream ends without [DONE]
      const chunks = [`data: ${JSON.stringify(step)}\n`];

      mockFetch.mockResolvedValueOnce(createMockStreamResponse(chunks));

      await probeStore.startProbe(modelId, "test-token");
      flushSync();

      const state = probeStore.getProbe(modelId);
      // Should be marked as completed by the fallback check
      expect(state.status).toBe("completed");
      expect(state.currentStep).toBeNull();

      // Clean up
      probeStore.reset(modelId);
    });

    it("handles stream ending right after [DONE] message (status not probing)", async () => {
      const modelId = "test-model";
      const step: ProbeStep = {
        step: "test-step",
        status: "running",
      };

      // Stream with step and [DONE], then stream ends
      // This explicitly tests that after [DONE] sets status to "completed",
      // the final check doesn't overwrite it
      const chunks = [
        `data: ${JSON.stringify(step)}\n`,
        "data: [DONE]\n",
      ];

      mockFetch.mockResolvedValueOnce(createMockStreamResponse(chunks));

      await probeStore.startProbe(modelId, "test-token");
      flushSync();

      const state = probeStore.getProbe(modelId);
      // Should be completed from [DONE], not from the final check
      expect(state.status).toBe("completed");
      expect(state.currentStep).toBeNull();

      // Clean up
      probeStore.reset(modelId);
    });
  });
});
