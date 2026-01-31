import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/svelte";
import userEvent from "@testing-library/user-event";
import type { FilterState } from "./filter-types";
import { createEmptyFilters } from "./filter-types";

import FilterChips from "./FilterChips.svelte";

describe("FilterChips", () => {
  let onRemove: (type: "multimodal" | "architecture" | "quantization", value?: string | number) => void;

  beforeEach(() => {
    vi.clearAllMocks();
    onRemove = vi.fn();
  });

  describe("rendering", () => {
    it("renders nothing when no filters are active", () => {
      const filters = createEmptyFilters();
      const { container } = render(FilterChips, {
        props: { filters, onRemove },
      });

      expect(container.textContent?.trim()).toBe("");
    });

    it("renders architecture filter chips", () => {
      const filters: FilterState = {
        architectures: ["Llama", "Qwen"],
        multimodal: null,
        quantization: [],
      };

      render(FilterChips, { props: { filters, onRemove } });

      expect(screen.getByText("Llama")).toBeInTheDocument();
      expect(screen.getByText("Qwen")).toBeInTheDocument();
    });

    it("renders multimodal filter chip when true", () => {
      const filters: FilterState = {
        architectures: [],
        multimodal: true,
        quantization: [],
      };

      render(FilterChips, { props: { filters, onRemove } });

      expect(screen.getByText("Multimodal")).toBeInTheDocument();
    });

    it("renders text-only filter chip when false", () => {
      const filters: FilterState = {
        architectures: [],
        multimodal: false,
        quantization: [],
      };

      render(FilterChips, { props: { filters, onRemove } });

      expect(screen.getByText("Text-only")).toBeInTheDocument();
    });

    it("does not render multimodal chip when null", () => {
      const filters: FilterState = {
        architectures: [],
        multimodal: null,
        quantization: [],
      };

      render(FilterChips, { props: { filters, onRemove } });

      expect(screen.queryByText("Multimodal")).not.toBeInTheDocument();
      expect(screen.queryByText("Text-only")).not.toBeInTheDocument();
    });

    it("renders quantization filter chips", () => {
      const filters: FilterState = {
        architectures: [],
        multimodal: null,
        quantization: [4, 8],
      };

      render(FilterChips, { props: { filters, onRemove } });

      expect(screen.getByText("4-bit")).toBeInTheDocument();
      expect(screen.getByText("8-bit")).toBeInTheDocument();
    });

    it("renders fp16 label for 16-bit quantization", () => {
      const filters: FilterState = {
        architectures: [],
        multimodal: null,
        quantization: [16],
      };

      render(FilterChips, { props: { filters, onRemove } });

      expect(screen.getByText("fp16")).toBeInTheDocument();
      expect(screen.queryByText("16-bit")).not.toBeInTheDocument();
    });

    it("renders all filter types together", () => {
      const filters: FilterState = {
        architectures: ["Llama", "Qwen"],
        multimodal: true,
        quantization: [4, 8],
      };

      render(FilterChips, { props: { filters, onRemove } });

      expect(screen.getByText("Llama")).toBeInTheDocument();
      expect(screen.getByText("Qwen")).toBeInTheDocument();
      expect(screen.getByText("Multimodal")).toBeInTheDocument();
      expect(screen.getByText("4-bit")).toBeInTheDocument();
      expect(screen.getByText("8-bit")).toBeInTheDocument();
    });

    it("renders close buttons on all chips", () => {
      const filters: FilterState = {
        architectures: ["Llama"],
        multimodal: true,
        quantization: [4],
      };

      const { container } = render(FilterChips, { props: { filters, onRemove } });

      // Each chip should have a close button
      const closeButtons = container.querySelectorAll("button");
      expect(closeButtons.length).toBe(3); // Llama, Multimodal, 4-bit
    });
  });

  describe("remove actions", () => {
    it("calls onRemove with architecture type when architecture chip removed", async () => {
      const user = userEvent.setup();
      const filters: FilterState = {
        architectures: ["Llama", "Qwen"],
        multimodal: null,
        quantization: [],
      };

      const { container } = render(FilterChips, { props: { filters, onRemove } });

      // Find all buttons - should have 2 (one for each architecture)
      const buttons = container.querySelectorAll("button");
      expect(buttons.length).toBe(2);

      // Click the first button (for Llama)
      await user.click(buttons[0]);

      expect(onRemove).toHaveBeenCalledWith("architecture", "Llama");
    });

    it("calls onRemove with multimodal type when multimodal chip removed", async () => {
      const user = userEvent.setup();
      const filters: FilterState = {
        architectures: [],
        multimodal: true,
        quantization: [],
      };

      const { container } = render(FilterChips, { props: { filters, onRemove } });

      // Should have 1 button for multimodal
      const buttons = container.querySelectorAll("button");
      expect(buttons.length).toBe(1);

      await user.click(buttons[0]);

      expect(onRemove).toHaveBeenCalledWith("multimodal");
    });

    it("calls onRemove with multimodal type when text-only chip removed", async () => {
      const user = userEvent.setup();
      const filters: FilterState = {
        architectures: [],
        multimodal: false,
        quantization: [],
      };

      const { container } = render(FilterChips, { props: { filters, onRemove } });

      // Should have 1 button for text-only
      const buttons = container.querySelectorAll("button");
      expect(buttons.length).toBe(1);

      await user.click(buttons[0]);

      expect(onRemove).toHaveBeenCalledWith("multimodal");
    });

    it("calls onRemove with quantization type when quantization chip removed", async () => {
      const user = userEvent.setup();
      const filters: FilterState = {
        architectures: [],
        multimodal: null,
        quantization: [4, 8],
      };

      const { container } = render(FilterChips, { props: { filters, onRemove } });

      // Should have 2 buttons (one for each quantization level)
      const buttons = container.querySelectorAll("button");
      expect(buttons.length).toBe(2);

      // First button is for 4-bit
      await user.click(buttons[0]);

      expect(onRemove).toHaveBeenCalledWith("quantization", 4);
    });

    it("calls onRemove with correct value for fp16", async () => {
      const user = userEvent.setup();
      const filters: FilterState = {
        architectures: [],
        multimodal: null,
        quantization: [16],
      };

      const { container } = render(FilterChips, { props: { filters, onRemove } });

      // Should have 1 button for fp16
      const buttons = container.querySelectorAll("button");
      expect(buttons.length).toBe(1);

      await user.click(buttons[0]);

      expect(onRemove).toHaveBeenCalledWith("quantization", 16);
    });
  });

  describe("multiple chips interaction", () => {
    it("allows removing multiple architecture chips independently", async () => {
      const user = userEvent.setup();
      const filters: FilterState = {
        architectures: ["Llama", "Qwen", "Mistral"],
        multimodal: null,
        quantization: [],
      };

      const { container } = render(FilterChips, { props: { filters, onRemove } });

      // Should have 3 buttons (one for each architecture)
      const buttons = container.querySelectorAll("button");
      expect(buttons.length).toBe(3);

      // Remove Qwen (second button)
      await user.click(buttons[1]);
      expect(onRemove).toHaveBeenCalledWith("architecture", "Qwen");

      // Remove Mistral (third button)
      await user.click(buttons[2]);
      expect(onRemove).toHaveBeenCalledWith("architecture", "Mistral");

      expect(onRemove).toHaveBeenCalledTimes(2);
    });

    it("allows removing multiple quantization chips independently", async () => {
      const user = userEvent.setup();
      const filters: FilterState = {
        architectures: [],
        multimodal: null,
        quantization: [2, 4, 8],
      };

      const { container } = render(FilterChips, { props: { filters, onRemove } });

      // Should have 3 buttons (one for each quantization level)
      const buttons = container.querySelectorAll("button");
      expect(buttons.length).toBe(3);

      // Remove 4-bit (second button)
      await user.click(buttons[1]);
      expect(onRemove).toHaveBeenCalledWith("quantization", 4);

      // Remove 8-bit (third button)
      await user.click(buttons[2]);
      expect(onRemove).toHaveBeenCalledWith("quantization", 8);

      expect(onRemove).toHaveBeenCalledTimes(2);
    });

    it("allows removing different types of filters", async () => {
      const user = userEvent.setup();
      const filters: FilterState = {
        architectures: ["Llama"],
        multimodal: true,
        quantization: [4],
      };

      const { container } = render(FilterChips, { props: { filters, onRemove } });

      // Should have 3 buttons (architecture, multimodal, quantization)
      const buttons = container.querySelectorAll("button");
      expect(buttons.length).toBe(3);

      // Remove architecture (first button - Llama)
      await user.click(buttons[0]);
      expect(onRemove).toHaveBeenCalledWith("architecture", "Llama");

      // Remove multimodal (second button - Multimodal)
      await user.click(buttons[1]);
      expect(onRemove).toHaveBeenCalledWith("multimodal");

      // Remove quantization (third button - 4-bit)
      await user.click(buttons[2]);
      expect(onRemove).toHaveBeenCalledWith("quantization", 4);

      expect(onRemove).toHaveBeenCalledTimes(3);
    });
  });

  describe("edge cases", () => {
    it("handles empty architectures array", () => {
      const filters: FilterState = {
        architectures: [],
        multimodal: true,
        quantization: [4],
      };

      render(FilterChips, { props: { filters, onRemove } });

      // Should render only multimodal and quantization chips
      expect(screen.getByText("Multimodal")).toBeInTheDocument();
      expect(screen.getByText("4-bit")).toBeInTheDocument();
    });

    it("handles empty quantization array", () => {
      const filters: FilterState = {
        architectures: ["Llama"],
        multimodal: true,
        quantization: [],
      };

      render(FilterChips, { props: { filters, onRemove } });

      expect(screen.getByText("Llama")).toBeInTheDocument();
      expect(screen.getByText("Multimodal")).toBeInTheDocument();
      expect(screen.queryByText(/bit/)).not.toBeInTheDocument();
    });

    it("handles single architecture", () => {
      const filters: FilterState = {
        architectures: ["Llama"],
        multimodal: null,
        quantization: [],
      };

      render(FilterChips, { props: { filters, onRemove } });

      expect(screen.getByText("Llama")).toBeInTheDocument();
    });

    it("handles single quantization", () => {
      const filters: FilterState = {
        architectures: [],
        multimodal: null,
        quantization: [4],
      };

      render(FilterChips, { props: { filters, onRemove } });

      expect(screen.getByText("4-bit")).toBeInTheDocument();
    });

    it("handles many architectures", () => {
      const filters: FilterState = {
        architectures: [
          "Llama",
          "Qwen",
          "Mistral",
          "Gemma",
          "Phi",
          "DeepSeek",
          "StarCoder",
        ],
        multimodal: null,
        quantization: [],
      };

      render(FilterChips, { props: { filters, onRemove } });

      filters.architectures.forEach((arch) => {
        expect(screen.getByText(arch)).toBeInTheDocument();
      });
    });

    it("handles all quantization levels", () => {
      const filters: FilterState = {
        architectures: [],
        multimodal: null,
        quantization: [2, 3, 4, 8, 16],
      };

      render(FilterChips, { props: { filters, onRemove } });

      expect(screen.getByText("2-bit")).toBeInTheDocument();
      expect(screen.getByText("3-bit")).toBeInTheDocument();
      expect(screen.getByText("4-bit")).toBeInTheDocument();
      expect(screen.getByText("8-bit")).toBeInTheDocument();
      expect(screen.getByText("fp16")).toBeInTheDocument();
    });
  });

  describe("rendering order", () => {
    it("renders chips in the order: architectures, multimodal, quantization", () => {
      const filters: FilterState = {
        architectures: ["Llama", "Qwen"],
        multimodal: true,
        quantization: [4, 8],
      };

      render(FilterChips, { props: { filters, onRemove } });

      // Check rendering order by text content
      expect(screen.getByText("Llama")).toBeInTheDocument();
      expect(screen.getByText("Qwen")).toBeInTheDocument();
      expect(screen.getByText("Multimodal")).toBeInTheDocument();
      expect(screen.getByText("4-bit")).toBeInTheDocument();
      expect(screen.getByText("8-bit")).toBeInTheDocument();

      // All expected chips are rendered
    });
  });
});
