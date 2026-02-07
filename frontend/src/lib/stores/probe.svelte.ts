import type { ModelCapabilities, ProbeStep } from "$lib/api/types";

export interface ProbeState {
  status: "idle" | "probing" | "completed" | "failed";
  currentStep: string | null;
  steps: ProbeStep[];
  capabilities: Partial<ModelCapabilities>;
  error: string | null;
}

function createProbeStore() {
  // eslint-disable-next-line prefer-const -- $state requires let for reactivity
  let probes = $state<Record<string, ProbeState>>({});

  function getProbe(modelId: string): ProbeState {
    return (
      probes[modelId] ?? {
        status: "idle",
        currentStep: null,
        steps: [],
        capabilities: {},
        error: null,
      }
    );
  }

  async function startProbe(modelId: string, token: string): Promise<void> {
    probes[modelId] = {
      status: "probing",
      currentStep: null,
      steps: [],
      capabilities: {},
      error: null,
    };

    try {
      const url = `/api/models/probe/${encodeURIComponent(modelId)}?token=${encodeURIComponent(token)}`;
      const response = await fetch(url, {
        method: "POST",
      });

      if (!response.ok) {
        throw new Error(`Probe failed: ${response.statusText}`);
      }

      const reader = response.body?.getReader();
      if (!reader) throw new Error("No response body");

      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() ?? "";

        for (const line of lines) {
          if (line.startsWith("data: ")) {
            const data = line.slice(6).trim();
            if (data === "[DONE]") {
              probes[modelId] = {
                ...probes[modelId],
                status: "completed",
                currentStep: null,
              };
              return;
            }

            try {
              const step: ProbeStep = JSON.parse(data);
              const current = probes[modelId];

              // Update steps list
              const steps = [...current.steps];
              const existingIdx = steps.findIndex((s) => s.step === step.step);
              if (existingIdx >= 0) {
                steps[existingIdx] = step;
              } else {
                steps.push(step);
              }

              // Update capabilities if a capability was discovered
              const capabilities = { ...current.capabilities };
              if (step.capability && step.value !== undefined) {
                (capabilities as Record<string, unknown>)[step.capability] =
                  step.value;
              }

              probes[modelId] = {
                ...current,
                currentStep:
                  step.status === "running"
                    ? step.step
                    : current.currentStep,
                steps,
                capabilities,
              };
            } catch {
              // Ignore malformed SSE data
            }
          }
        }
      }

      // If we get here without [DONE], mark as completed
      if (probes[modelId]?.status === "probing") {
        probes[modelId] = {
          ...probes[modelId],
          status: "completed",
          currentStep: null,
        };
      }
    } catch (err) {
      probes[modelId] = {
        ...probes[modelId],
        status: "failed",
        currentStep: null,
        error: err instanceof Error ? err.message : String(err),
      };
    }
  }

  function reset(modelId: string): void {
    delete probes[modelId];
  }

  return {
    get probes() {
      return probes;
    },
    getProbe,
    startProbe,
    reset,
  };
}

export const probeStore = createProbeStore();
