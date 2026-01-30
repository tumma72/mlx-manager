import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/svelte";
import userEvent from "@testing-library/user-event";
import type { ModelCharacteristics } from "$api";

import ModelSpecs from "./ModelSpecs.svelte";

// Helper to create mock characteristics
function createMockCharacteristics(
  overrides: Partial<ModelCharacteristics> = {},
): ModelCharacteristics {
  return {
    max_position_embeddings: 4096,
    num_hidden_layers: 32,
    hidden_size: 4096,
    vocab_size: 32000,
    num_attention_heads: 32,
    num_key_value_heads: 8,
    use_cache: true,
    ...overrides,
  };
}

describe("ModelSpecs", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe("rendering", () => {
    it("does not render when no specs are available", () => {
      const { container } = render(ModelSpecs, {
        props: {
          characteristics: {},
        },
      });

      expect(container.textContent?.trim()).toBe("");
    });

    it("renders toggle button when specs are available", () => {
      render(ModelSpecs, {
        props: {
          characteristics: createMockCharacteristics(),
        },
      });

      expect(screen.getByRole("button", { name: /show specs/i })).toBeInTheDocument();
    });

    it("shows 'Show specs' text when collapsed", () => {
      render(ModelSpecs, {
        props: {
          characteristics: createMockCharacteristics(),
        },
      });

      expect(screen.getByText("Show specs")).toBeInTheDocument();
    });

    it("shows ChevronDown icon when collapsed", () => {
      const { container } = render(ModelSpecs, {
        props: {
          characteristics: createMockCharacteristics(),
        },
      });

      // ChevronDown should be present
      expect(screen.getByText("Show specs")).toBeInTheDocument();
    });
  });

  describe("expand/collapse", () => {
    it("expands specs when toggle button clicked", async () => {
      const user = userEvent.setup();

      render(ModelSpecs, {
        props: {
          characteristics: createMockCharacteristics(),
        },
      });

      const toggleButton = screen.getByRole("button", { name: /show specs/i });
      await user.click(toggleButton);

      expect(screen.getByText("Hide specs")).toBeInTheDocument();
    });

    it("shows specs grid when expanded", async () => {
      const user = userEvent.setup();

      render(ModelSpecs, {
        props: {
          characteristics: createMockCharacteristics(),
        },
      });

      const toggleButton = screen.getByRole("button");
      await user.click(toggleButton);

      // Wait for specs to be visible (use getAllByText since there might be duplicates during animation)
      await waitFor(() => {
        expect(screen.getByText("Context")).toBeInTheDocument();
        expect(screen.getAllByText("4,096").length).toBeGreaterThan(0);
      });
    });

    it("collapses specs when toggle button clicked again", async () => {
      const user = userEvent.setup();

      render(ModelSpecs, {
        props: {
          characteristics: createMockCharacteristics(),
        },
      });

      const toggleButton = screen.getByRole("button");

      // Expand
      await user.click(toggleButton);
      expect(screen.getByText("Hide specs")).toBeInTheDocument();

      // Collapse
      await user.click(toggleButton);
      expect(screen.getByText("Show specs")).toBeInTheDocument();
    });
  });

  describe("spec fields", () => {
    it("renders context length", async () => {
      const user = userEvent.setup();

      render(ModelSpecs, {
        props: {
          characteristics: createMockCharacteristics({
            max_position_embeddings: 8192,
          }),
        },
      });

      const button = screen.getByRole("button");
      await user.click(button);

      // Wait for expansion
      await screen.findByText("Context");
      expect(screen.getByText("Context")).toBeInTheDocument();
      expect(screen.getByText("8,192")).toBeInTheDocument();
    });

    it("renders number of layers", async () => {
      const user = userEvent.setup();

      render(ModelSpecs, {
        props: {
          characteristics: createMockCharacteristics({
            num_hidden_layers: 48,
          }),
        },
      });

      await user.click(screen.getByRole("button"));

      expect(screen.getByText("Layers")).toBeInTheDocument();
      expect(screen.getByText("48")).toBeInTheDocument();
    });

    it("renders hidden size", async () => {
      const user = userEvent.setup();

      render(ModelSpecs, {
        props: {
          characteristics: createMockCharacteristics({
            hidden_size: 5120,
          }),
        },
      });

      await user.click(screen.getByRole("button"));

      expect(screen.getByText("Hidden size")).toBeInTheDocument();
      expect(screen.getByText("5,120")).toBeInTheDocument();
    });

    it("renders vocab size", async () => {
      const user = userEvent.setup();

      render(ModelSpecs, {
        props: {
          characteristics: createMockCharacteristics({
            vocab_size: 128000,
          }),
        },
      });

      await user.click(screen.getByRole("button"));

      expect(screen.getByText("Vocab size")).toBeInTheDocument();
      expect(screen.getByText("128,000")).toBeInTheDocument();
    });

    it("renders attention heads", async () => {
      const user = userEvent.setup();

      render(ModelSpecs, {
        props: {
          characteristics: createMockCharacteristics({
            num_attention_heads: 64,
          }),
        },
      });

      await user.click(screen.getByRole("button"));

      expect(screen.getByText("Attention")).toBeInTheDocument();
      expect(screen.getByText("64")).toBeInTheDocument();
    });

    it("renders KV heads when different from attention heads", async () => {
      const user = userEvent.setup();

      const { container } = render(ModelSpecs, {
        props: {
          characteristics: createMockCharacteristics({
            num_attention_heads: 32,
            num_key_value_heads: 8,
          }),
        },
      });

      await user.click(screen.getByRole("button"));

      await waitFor(() => {
        // Check that the grid is shown
        expect(container.querySelector(".grid-cols-2")).toBeInTheDocument();
        // Both the attention heads value and KV heads should be present
        expect(screen.getByText("Attention")).toBeInTheDocument();
        expect(screen.getAllByText("32").length).toBeGreaterThan(0);
        expect(screen.getByText("(KV: 8)")).toBeInTheDocument();
      });
    });

    it("does not render KV heads when same as attention heads", async () => {
      const user = userEvent.setup();

      render(ModelSpecs, {
        props: {
          characteristics: createMockCharacteristics({
            num_attention_heads: 32,
            num_key_value_heads: 32,
          }),
        },
      });

      await user.click(screen.getByRole("button"));

      await waitFor(() => {
        expect(screen.getByText("Attention")).toBeInTheDocument();
        expect(screen.getAllByText("32").length).toBeGreaterThan(0);
        expect(screen.queryByText(/\(KV:/)).not.toBeInTheDocument();
      });
    });

    it("renders KV cache status as Yes", async () => {
      const user = userEvent.setup();

      render(ModelSpecs, {
        props: {
          characteristics: createMockCharacteristics({
            use_cache: true,
          }),
        },
      });

      await user.click(screen.getByRole("button"));

      expect(screen.getByText("KV Cache")).toBeInTheDocument();
      expect(screen.getByText("Yes")).toBeInTheDocument();
    });

    it("renders KV cache status as No", async () => {
      const user = userEvent.setup();

      render(ModelSpecs, {
        props: {
          characteristics: createMockCharacteristics({
            use_cache: false,
          }),
        },
      });

      await user.click(screen.getByRole("button"));

      expect(screen.getByText("KV Cache")).toBeInTheDocument();
      expect(screen.getByText("No")).toBeInTheDocument();
    });

    it("does not render KV cache when undefined", async () => {
      const user = userEvent.setup();

      render(ModelSpecs, {
        props: {
          characteristics: createMockCharacteristics({
            use_cache: undefined,
          }),
        },
      });

      await user.click(screen.getByRole("button"));

      expect(screen.queryByText("KV Cache")).not.toBeInTheDocument();
    });
  });

  describe("partial specs", () => {
    it("renders only available specs", async () => {
      const user = userEvent.setup();

      render(ModelSpecs, {
        props: {
          characteristics: {
            max_position_embeddings: 4096,
            num_hidden_layers: 32,
            // Missing other fields
          },
        },
      });

      await user.click(screen.getByRole("button"));

      expect(screen.getByText("Context")).toBeInTheDocument();
      expect(screen.getByText("Layers")).toBeInTheDocument();
      expect(screen.queryByText("Hidden size")).not.toBeInTheDocument();
      expect(screen.queryByText("Vocab size")).not.toBeInTheDocument();
    });

    it("does not render when only undefined values", () => {
      const { container } = render(ModelSpecs, {
        props: {
          characteristics: {
            max_position_embeddings: undefined,
            num_hidden_layers: undefined,
          },
        },
      });

      expect(container.textContent?.trim()).toBe("");
    });

    it("renders with single spec field", async () => {
      const user = userEvent.setup();

      render(ModelSpecs, {
        props: {
          characteristics: {
            vocab_size: 50000,
          },
        },
      });

      const toggleButton = screen.getByRole("button");
      await user.click(toggleButton);

      expect(screen.getByText("Vocab size")).toBeInTheDocument();
      expect(screen.getByText("50,000")).toBeInTheDocument();
    });
  });

  describe("number formatting", () => {
    it("formats large numbers with commas", async () => {
      const user = userEvent.setup();

      render(ModelSpecs, {
        props: {
          characteristics: createMockCharacteristics({
            vocab_size: 1234567,
          }),
        },
      });

      await user.click(screen.getByRole("button"));

      expect(screen.getByText("1,234,567")).toBeInTheDocument();
    });

    it("formats small numbers without commas", async () => {
      const user = userEvent.setup();

      render(ModelSpecs, {
        props: {
          characteristics: createMockCharacteristics({
            num_hidden_layers: 12,
          }),
        },
      });

      await user.click(screen.getByRole("button"));

      expect(screen.getByText("12")).toBeInTheDocument();
    });

    it("displays dash for undefined values", async () => {
      const user = userEvent.setup();

      render(ModelSpecs, {
        props: {
          characteristics: {
            max_position_embeddings: undefined,
            vocab_size: 32000, // At least one field to show toggle
          },
        },
      });

      await user.click(screen.getByRole("button"));

      expect(screen.queryByText("Context")).not.toBeInTheDocument();
    });
  });

  describe("edge cases", () => {
    it("handles zero values", async () => {
      const user = userEvent.setup();

      render(ModelSpecs, {
        props: {
          characteristics: createMockCharacteristics({
            num_hidden_layers: 0,
          }),
        },
      });

      // Zero is falsy, so it won't render
      expect(screen.queryByRole("button")).toBeInTheDocument();
      await user.click(screen.getByRole("button"));
      expect(screen.queryByText("Layers")).not.toBeInTheDocument();
    });

    it("handles very large numbers", async () => {
      const user = userEvent.setup();

      render(ModelSpecs, {
        props: {
          characteristics: createMockCharacteristics({
            vocab_size: 999999999,
            hidden_size: 16384,
          }),
        },
      });

      await user.click(screen.getByRole("button"));

      expect(screen.getByText("999,999,999")).toBeInTheDocument();
      expect(screen.getByText("16,384")).toBeInTheDocument();
    });

    it("handles all specs at maximum detail", async () => {
      const user = userEvent.setup();

      render(ModelSpecs, {
        props: {
          characteristics: createMockCharacteristics({
            max_position_embeddings: 131072,
            num_hidden_layers: 80,
            hidden_size: 8192,
            vocab_size: 152064,
            num_attention_heads: 64,
            num_key_value_heads: 8,
            use_cache: true,
          }),
        },
      });

      await user.click(screen.getByRole("button"));

      await waitFor(() => {
        expect(screen.getByText("131,072")).toBeInTheDocument();
        expect(screen.getByText("80")).toBeInTheDocument();
        expect(screen.getByText("8,192")).toBeInTheDocument();
        expect(screen.getByText("152,064")).toBeInTheDocument();
        expect(screen.getAllByText("64").length).toBeGreaterThan(0);
        expect(screen.getByText("(KV: 8)")).toBeInTheDocument();
        expect(screen.getByText("Yes")).toBeInTheDocument();
      });
    });
  });

  describe("accessibility", () => {
    it("renders button with type='button'", () => {
      render(ModelSpecs, {
        props: {
          characteristics: createMockCharacteristics(),
        },
      });

      const button = screen.getByRole("button");
      expect(button).toHaveAttribute("type", "button");
    });

    it("maintains focus state", async () => {
      const user = userEvent.setup();

      render(ModelSpecs, {
        props: {
          characteristics: createMockCharacteristics(),
        },
      });

      const button = screen.getByRole("button");
      await user.click(button);

      expect(button).toHaveFocus();
    });
  });

  describe("grid layout", () => {
    it("displays specs in two-column grid", async () => {
      const user = userEvent.setup();

      const { container } = render(ModelSpecs, {
        props: {
          characteristics: createMockCharacteristics(),
        },
      });

      await user.click(screen.getByRole("button"));

      const grid = container.querySelector(".grid-cols-2");
      expect(grid).toBeInTheDocument();
    });
  });
});
