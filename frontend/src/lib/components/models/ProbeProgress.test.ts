import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/svelte";
import type { ProbeState } from "$stores/probe.svelte";

import ProbeProgress from "./ProbeProgress.svelte";

function createProbeState(
  overrides: Partial<ProbeState> = {},
): ProbeState {
  return {
    status: "probing",
    currentStep: null,
    steps: [],
    capabilities: {},
    error: null,
    diagnostics: [],
    probeResult: null,
    ...overrides,
  };
}

describe("ProbeProgress", () => {
  it("renders completed step with label", () => {
    render(ProbeProgress, {
      props: {
        probe: createProbeState({
          steps: [{ step: "load_model", status: "completed" }],
        }),
      },
    });
    expect(screen.getByText("Loading model")).toBeInTheDocument();
  });

  it("renders running step with label", () => {
    render(ProbeProgress, {
      props: {
        probe: createProbeState({
          steps: [{ step: "test_thinking", status: "running" }],
        }),
      },
    });
    expect(screen.getByText("Testing thinking")).toBeInTheDocument();
  });

  it("renders failed step with error text", () => {
    render(ProbeProgress, {
      props: {
        probe: createProbeState({
          steps: [
            {
              step: "test_tools",
              status: "failed",
              error: "Model timeout",
            },
          ],
        }),
      },
    });
    expect(screen.getByText("Testing tools")).toBeInTheDocument();
    expect(screen.getByText(/Model timeout/)).toBeInTheDocument();
  });

  it("renders skipped step with label", () => {
    render(ProbeProgress, {
      props: {
        probe: createProbeState({
          steps: [{ step: "cleanup", status: "skipped" }],
        }),
      },
    });
    expect(screen.getByText("Cleaning up")).toBeInTheDocument();
  });

  it("shows probe-level error", () => {
    render(ProbeProgress, {
      props: {
        probe: createProbeState({
          error: "Connection refused",
        }),
      },
    });
    expect(screen.getByText("Connection refused")).toBeInTheDocument();
  });

  it("handles empty steps array", () => {
    const { container } = render(ProbeProgress, {
      props: {
        probe: createProbeState({ steps: [] }),
      },
    });
    expect(container).toBeInTheDocument();
  });

  it("renders multiple steps in order", () => {
    render(ProbeProgress, {
      props: {
        probe: createProbeState({
          steps: [
            { step: "load_model", status: "completed" },
            { step: "check_context", status: "completed" },
            { step: "test_thinking", status: "running" },
            { step: "test_tools", status: "skipped" },
          ],
        }),
      },
    });
    expect(screen.getByText("Loading model")).toBeInTheDocument();
    expect(screen.getByText("Checking context")).toBeInTheDocument();
    expect(screen.getByText("Testing thinking")).toBeInTheDocument();
    expect(screen.getByText("Testing tools")).toBeInTheDocument();
  });

  it("falls back to step name for unknown steps", () => {
    render(ProbeProgress, {
      props: {
        probe: createProbeState({
          steps: [{ step: "custom_step", status: "running" }],
        }),
      },
    });
    expect(screen.getByText("custom_step")).toBeInTheDocument();
  });

  it("shows correct status icons per step", () => {
    const { container } = render(ProbeProgress, {
      props: {
        probe: createProbeState({
          steps: [
            { step: "load_model", status: "completed" },
            { step: "test_thinking", status: "running" },
            { step: "test_tools", status: "failed", error: "err" },
            { step: "cleanup", status: "skipped" },
          ],
        }),
      },
    });
    // Each step renders an svg icon
    const svgs = container.querySelectorAll("svg");
    expect(svgs.length).toBe(4);

    // Running step has animate-spin class
    const spinners = container.querySelectorAll(".animate-spin");
    expect(spinners.length).toBe(1);

    // Green for completed
    const greenIcons = container.querySelectorAll(".text-green-500");
    expect(greenIcons.length).toBe(1);

    // Red for failed
    const redIcons = container.querySelectorAll(".text-red-500");
    // XCircle icon + step text both have red
    expect(redIcons.length).toBeGreaterThanOrEqual(1);
  });
});
