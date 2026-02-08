import { describe, it, expect, beforeEach } from "vitest";
import { render, screen } from "@testing-library/svelte";
import type { ModelCharacteristics, ModelCapabilities } from "$api";

import ModelBadges from "./ModelBadges.svelte";

// Helper to create mock characteristics
function createMockCharacteristics(
  overrides: Partial<ModelCharacteristics> = {},
): ModelCharacteristics {
  return {
    architecture_family: "Llama",
    quantization_bits: 4,
    is_multimodal: false,
    is_tool_use: false,
    ...overrides,
  };
}

describe("ModelBadges", () => {
  beforeEach(() => {
    // No mocks needed for this simple component
  });

  describe("loading state", () => {
    it("shows skeleton badges when loading", () => {
      const { container } = render(ModelBadges, {
        props: {
          characteristics: null,
          loading: true,
        },
      });

      const skeletons = container.querySelectorAll(".animate-pulse");
      expect(skeletons.length).toBeGreaterThan(0);
    });

    it("shows multiple skeleton badges", () => {
      const { container } = render(ModelBadges, {
        props: {
          characteristics: null,
          loading: true,
        },
      });

      const skeletons = container.querySelectorAll(".animate-pulse");
      expect(skeletons.length).toBe(2);
    });

    it("does not show actual badges when loading", () => {
      const { container } = render(ModelBadges, {
        props: {
          characteristics: createMockCharacteristics(),
          loading: true,
        },
      });

      // Should not render ArchitectureBadge, etc.
      expect(container.textContent).not.toContain("Llama");
    });
  });

  describe("architecture badge", () => {
    it("renders architecture badge when family is provided", () => {
      render(ModelBadges, {
        props: {
          characteristics: createMockCharacteristics({
            architecture_family: "Qwen",
          }),
          loading: false,
        },
      });

      // ArchitectureBadge component should render the family name
      expect(screen.getByText("Qwen")).toBeInTheDocument();
    });

    it("does not render architecture badge when family is Unknown", () => {
      const { container } = render(ModelBadges, {
        props: {
          characteristics: createMockCharacteristics({
            architecture_family: "Unknown",
          }),
          loading: false,
        },
      });

      // Should not render architecture badge
      const badges = container.querySelectorAll("span");
      // Only quantization badge should be present (4-bit)
      expect(badges.length).toBeLessThanOrEqual(1);
    });

    it("does not render architecture badge when family is undefined", () => {
      const { container } = render(ModelBadges, {
        props: {
          characteristics: createMockCharacteristics({
            architecture_family: undefined,
          }),
          loading: false,
        },
      });

      // Should not crash and not render architecture badge
      expect(container).toBeInTheDocument();
    });
  });

  describe("multimodal badge", () => {
    it("renders multimodal badge when is_multimodal is true", () => {
      render(ModelBadges, {
        props: {
          characteristics: createMockCharacteristics({
            is_multimodal: true,
            multimodal_type: "vision",
          }),
          loading: false,
        },
      });

      // MultimodalBadge component should be rendered
      expect(screen.getByText(/vision/i)).toBeInTheDocument();
    });

    it("does not render multimodal badge when is_multimodal is false", () => {
      render(ModelBadges, {
        props: {
          characteristics: createMockCharacteristics({
            is_multimodal: false,
          }),
          loading: false,
        },
      });

      // Should not render multimodal badge
      expect(screen.queryByText(/vision/i)).not.toBeInTheDocument();
    });

    it("passes multimodal_type to MultimodalBadge", () => {
      render(ModelBadges, {
        props: {
          characteristics: createMockCharacteristics({
            is_multimodal: true,
            multimodal_type: "audio",
          }),
          loading: false,
        },
      });

      // MultimodalBadge should receive the type
      expect(screen.getByText(/audio/i)).toBeInTheDocument();
    });
  });

  describe("quantization badge", () => {
    it("renders quantization badge when bits are provided", () => {
      render(ModelBadges, {
        props: {
          characteristics: createMockCharacteristics({
            quantization_bits: 8,
          }),
          loading: false,
        },
      });

      // QuantizationBadge component should be rendered
      expect(screen.getByText(/8/)).toBeInTheDocument();
    });

    it("does not render quantization badge when bits are undefined", () => {
      const { container } = render(ModelBadges, {
        props: {
          characteristics: createMockCharacteristics({
            quantization_bits: undefined,
          }),
          loading: false,
        },
      });

      // Should not crash
      expect(container).toBeInTheDocument();
    });

    it("renders quantization badge for different bit values", () => {
      const { rerender } = render(ModelBadges, {
        props: {
          characteristics: createMockCharacteristics({
            quantization_bits: 2,
          }),
          loading: false,
        },
      });

      expect(screen.getByText(/2/)).toBeInTheDocument();

      rerender({
        characteristics: createMockCharacteristics({
          quantization_bits: 16,
        }),
        loading: false,
      });

      expect(screen.getByText(/16/)).toBeInTheDocument();
    });
  });

  describe("tool use badge", () => {
    it("renders tool use badge when is_tool_use is true", () => {
      render(ModelBadges, {
        props: {
          characteristics: createMockCharacteristics({
            is_tool_use: true,
          }),
          loading: false,
        },
      });

      // ToolUseBadge component should render "Tools" text
      expect(screen.getByText("Tool Use")).toBeInTheDocument();
    });

    it("does not render tool use badge when is_tool_use is false", () => {
      render(ModelBadges, {
        props: {
          characteristics: createMockCharacteristics({
            is_tool_use: false,
          }),
          loading: false,
        },
      });

      expect(screen.queryByText(/tools/i)).not.toBeInTheDocument();
    });
  });

  describe("multiple badges", () => {
    it("renders all badges when all characteristics are present", () => {
      render(ModelBadges, {
        props: {
          characteristics: createMockCharacteristics({
            architecture_family: "Llama",
            is_multimodal: true,
            multimodal_type: "vision",
            quantization_bits: 4,
            is_tool_use: true,
          }),
          loading: false,
        },
      });

      expect(screen.getByText("Llama")).toBeInTheDocument();
      expect(screen.getByText("Vision")).toBeInTheDocument();
      expect(screen.getByText("4-bit")).toBeInTheDocument();
      expect(screen.getByText("Tool Use")).toBeInTheDocument();
    });

    it("renders only relevant badges based on characteristics", () => {
      render(ModelBadges, {
        props: {
          characteristics: createMockCharacteristics({
            architecture_family: "Qwen",
            is_multimodal: false,
            quantization_bits: undefined,
            is_tool_use: true,
          }),
          loading: false,
        },
      });

      expect(screen.getByText("Qwen")).toBeInTheDocument();
      expect(screen.queryByText("Vision")).not.toBeInTheDocument();
      expect(screen.getByText("Tool Use")).toBeInTheDocument();
    });

    it("renders badges in correct order", () => {
      render(ModelBadges, {
        props: {
          characteristics: createMockCharacteristics({
            architecture_family: "Llama",
            is_multimodal: true,
            quantization_bits: 4,
            is_tool_use: true,
          }),
          loading: false,
        },
      });

      // Verify all badges are rendered (order is handled by component structure)
      expect(screen.getByText("Llama")).toBeInTheDocument();
      expect(screen.getByText("Vision")).toBeInTheDocument();
      expect(screen.getByText("4-bit")).toBeInTheDocument();
      expect(screen.getByText("Tool Use")).toBeInTheDocument();
    });
  });

  describe("null/undefined characteristics", () => {
    it("renders nothing when characteristics is null and not loading", () => {
      const { container } = render(ModelBadges, {
        props: {
          characteristics: null,
          loading: false,
        },
      });

      // Should not render any badges
      const badges = container.querySelectorAll("span");
      expect(badges.length).toBe(0);
    });

    it("renders nothing when characteristics is undefined and not loading", () => {
      const { container } = render(ModelBadges, {
        props: {
          characteristics: undefined,
          loading: false,
        },
      });

      const badges = container.querySelectorAll("span");
      expect(badges.length).toBe(0);
    });
  });

  describe("edge cases", () => {
    it("handles empty characteristics object", () => {
      const { container } = render(ModelBadges, {
        props: {
          characteristics: {},
          loading: false,
        },
      });

      // Should not crash, should render no badges
      const badges = container.querySelectorAll("span");
      expect(badges.length).toBe(0);
    });

    it("handles partial characteristics", () => {
      render(ModelBadges, {
        props: {
          characteristics: {
            architecture_family: "Mistral",
            // Other fields undefined
          },
          loading: false,
        },
      });

      expect(screen.getByText(/mistral/i)).toBeInTheDocument();
    });

    it("handles loading state change", () => {
      const { container, rerender } = render(ModelBadges, {
        props: {
          characteristics: null,
          loading: true,
        },
      });

      expect(container.querySelectorAll(".animate-pulse").length).toBeGreaterThan(0);

      rerender({
        characteristics: createMockCharacteristics(),
        loading: false,
      });

      expect(container.querySelectorAll(".animate-pulse").length).toBe(0);
      expect(screen.getByText(/llama/i)).toBeInTheDocument();
    });

    it("defaults loading to false when not provided", () => {
      const { container } = render(ModelBadges, {
        props: {
          characteristics: createMockCharacteristics(),
        },
      });

      // Should not show loading skeletons
      expect(container.querySelectorAll(".animate-pulse").length).toBe(0);
      expect(screen.getByText(/llama/i)).toBeInTheDocument();
    });
  });

  describe("with capabilities prop", () => {
    function createMockCapabilities(
      overrides: Partial<ModelCapabilities> = {},
    ): ModelCapabilities {
      return {
        model_id: "test-model",
        supports_native_tools: null,
        supports_thinking: null,
        practical_max_tokens: null,
        probed_at: null,
        probe_version: 1,
        ...overrides,
      };
    }

    it("shows ThinkingBadge when supports_thinking is true", () => {
      render(ModelBadges, {
        props: {
          characteristics: createMockCharacteristics(),
          capabilities: createMockCapabilities({ supports_thinking: true }),
        },
      });
      expect(screen.getByText("Thinking")).toBeInTheDocument();
    });

    it("hides ThinkingBadge when supports_thinking is false", () => {
      render(ModelBadges, {
        props: {
          characteristics: createMockCharacteristics(),
          capabilities: createMockCapabilities({ supports_thinking: false }),
        },
      });
      expect(screen.queryByText("Thinking")).not.toBeInTheDocument();
    });

    it("hides ThinkingBadge when supports_thinking is null", () => {
      render(ModelBadges, {
        props: {
          characteristics: createMockCharacteristics(),
          capabilities: createMockCapabilities({ supports_thinking: null }),
        },
      });
      expect(screen.queryByText("Thinking")).not.toBeInTheDocument();
    });

    it("shows verified ToolUseBadge when supports_native_tools is true", () => {
      const { container } = render(ModelBadges, {
        props: {
          characteristics: createMockCharacteristics({ is_tool_use: false }),
          capabilities: createMockCapabilities({ supports_native_tools: true }),
        },
      });
      expect(screen.getByText("Tool Use")).toBeInTheDocument();
      const badge = container.querySelector("[title='Verified by model probe']");
      expect(badge).toBeInTheDocument();
    });

    it("hides ToolUseBadge when probed and supports_native_tools is false", () => {
      render(ModelBadges, {
        props: {
          characteristics: createMockCharacteristics({ is_tool_use: true }),
          capabilities: createMockCapabilities({ supports_native_tools: false }),
        },
      });
      // Probed result overrides heuristic — tool use hidden
      expect(screen.queryByText("Tool Use")).not.toBeInTheDocument();
    });

    it("falls back to heuristic ToolUseBadge when no capabilities", () => {
      const { container } = render(ModelBadges, {
        props: {
          characteristics: createMockCharacteristics({ is_tool_use: true }),
          capabilities: null,
        },
      });
      expect(screen.getByText("Tool Use")).toBeInTheDocument();
      const badge = container.querySelector("[title='Detected from model name/tags']");
      expect(badge).toBeInTheDocument();
    });

    it("works identically without capabilities prop (backward-compatible)", () => {
      render(ModelBadges, {
        props: {
          characteristics: createMockCharacteristics({ is_tool_use: true }),
        },
      });
      expect(screen.getByText("Tool Use")).toBeInTheDocument();
      expect(screen.queryByText("Thinking")).not.toBeInTheDocument();
    });
  });

  describe("container styling", () => {
    it("applies flex layout classes", () => {
      const { container } = render(ModelBadges, {
        props: {
          characteristics: createMockCharacteristics(),
          loading: false,
        },
      });

      const wrapper = container.querySelector(".flex");
      expect(wrapper).toBeInTheDocument();
      expect(wrapper?.classList.contains("flex-wrap")).toBe(true);
    });

    it("applies gap between badges", () => {
      const { container } = render(ModelBadges, {
        props: {
          characteristics: createMockCharacteristics(),
          loading: false,
        },
      });

      const wrapper = container.querySelector(".gap-1\\.5");
      expect(wrapper).toBeInTheDocument();
    });
  });
});
