import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/svelte";
import userEvent from "@testing-library/user-event";
import type { FilterState } from "./filter-types";
import {
  ARCHITECTURE_OPTIONS,
  QUANTIZATION_OPTIONS,
  createEmptyFilters,
} from "./filter-types";

import FilterModal from "./FilterModal.svelte";

describe("FilterModal", () => {
  let defaultFilters: FilterState;

  beforeEach(() => {
    vi.clearAllMocks();
    defaultFilters = createEmptyFilters();
  });

  describe("rendering", () => {
    it("does not render when closed", () => {
      render(FilterModal, {
        props: {
          open: false,
          filters: defaultFilters,
        },
      });

      expect(screen.queryByText("Filter Models")).not.toBeInTheDocument();
    });

    it("renders when open", () => {
      render(FilterModal, {
        props: {
          open: true,
          filters: defaultFilters,
        },
      });

      expect(screen.getByText("Filter Models")).toBeInTheDocument();
    });

    it("renders all architecture options", () => {
      render(FilterModal, {
        props: {
          open: true,
          filters: defaultFilters,
        },
      });

      ARCHITECTURE_OPTIONS.forEach((arch) => {
        expect(screen.getByText(arch)).toBeInTheDocument();
      });
    });

    it("renders all quantization options", () => {
      render(FilterModal, {
        props: {
          open: true,
          filters: defaultFilters,
        },
      });

      QUANTIZATION_OPTIONS.forEach((bits) => {
        const label = bits === 16 ? "fp16" : `${bits}-bit`;
        expect(screen.getByText(label)).toBeInTheDocument();
      });
    });

    it("renders multimodal capability options", () => {
      render(FilterModal, {
        props: {
          open: true,
          filters: defaultFilters,
        },
      });

      expect(screen.getByText("Any")).toBeInTheDocument();
      expect(screen.getByText("Text-only")).toBeInTheDocument();
      expect(screen.getByText("Multimodal (Vision)")).toBeInTheDocument();
    });

    it("renders Clear All and Apply buttons", () => {
      render(FilterModal, {
        props: {
          open: true,
          filters: defaultFilters,
        },
      });

      expect(screen.getByRole("button", { name: /clear all/i })).toBeInTheDocument();
      expect(screen.getByRole("button", { name: /apply/i })).toBeInTheDocument();
    });

    it("renders close button", () => {
      render(FilterModal, {
        props: {
          open: true,
          filters: defaultFilters,
        },
      });

      expect(screen.getByRole("button", { name: /close/i })).toBeInTheDocument();
    });
  });

  describe("architecture filtering", () => {
    it("shows checked state for selected architectures", () => {
      render(FilterModal, {
        props: {
          open: true,
          filters: {
            ...defaultFilters,
            architectures: ["Llama", "Qwen"],
          },
        },
      });

      const llamaCheckbox = screen.getByRole("checkbox", { name: /llama/i });
      const qwenCheckbox = screen.getByRole("checkbox", { name: /qwen/i });
      const mistralCheckbox = screen.getByRole("checkbox", { name: /mistral/i });

      expect(llamaCheckbox).toBeChecked();
      expect(qwenCheckbox).toBeChecked();
      expect(mistralCheckbox).not.toBeChecked();
    });

    it("toggles architecture selection on click", async () => {
      const user = userEvent.setup();
      render(FilterModal, {
        props: {
          open: true,
          filters: defaultFilters,
        },
      });

      const llamaCheckbox = screen.getByRole("checkbox", { name: /llama/i });
      expect(llamaCheckbox).not.toBeChecked();

      await user.click(llamaCheckbox);
      expect(llamaCheckbox).toBeChecked();

      await user.click(llamaCheckbox);
      expect(llamaCheckbox).not.toBeChecked();
    });

    it("allows multiple architecture selections", async () => {
      const user = userEvent.setup();
      render(FilterModal, {
        props: {
          open: true,
          filters: defaultFilters,
        },
      });

      const llamaCheckbox = screen.getByRole("checkbox", { name: /llama/i });
      const qwenCheckbox = screen.getByRole("checkbox", { name: /qwen/i });

      await user.click(llamaCheckbox);
      await user.click(qwenCheckbox);

      expect(llamaCheckbox).toBeChecked();
      expect(qwenCheckbox).toBeChecked();
    });
  });

  describe("multimodal filtering", () => {
    it("shows 'Any' as selected by default", () => {
      render(FilterModal, {
        props: {
          open: true,
          filters: { ...defaultFilters, multimodal: null },
        },
      });

      const anyRadio = screen.getByRole("radio", { name: /any/i });
      expect(anyRadio).toBeChecked();
    });

    it("shows 'Text-only' as selected when multimodal is false", () => {
      render(FilterModal, {
        props: {
          open: true,
          filters: { ...defaultFilters, multimodal: false },
        },
      });

      const textOnlyRadio = screen.getByRole("radio", { name: /text-only/i });
      expect(textOnlyRadio).toBeChecked();
    });

    it("shows 'Multimodal' as selected when multimodal is true", () => {
      render(FilterModal, {
        props: {
          open: true,
          filters: { ...defaultFilters, multimodal: true },
        },
      });

      const multimodalRadio = screen.getByRole("radio", {
        name: /multimodal \(vision\)/i,
      });
      expect(multimodalRadio).toBeChecked();
    });

    it("switches multimodal selection on click", async () => {
      const user = userEvent.setup();
      render(FilterModal, {
        props: {
          open: true,
          filters: defaultFilters,
        },
      });

      const textOnlyRadio = screen.getByRole("radio", { name: /text-only/i });
      await user.click(textOnlyRadio);
      expect(textOnlyRadio).toBeChecked();

      const multimodalRadio = screen.getByRole("radio", {
        name: /multimodal \(vision\)/i,
      });
      await user.click(multimodalRadio);
      expect(multimodalRadio).toBeChecked();
      expect(textOnlyRadio).not.toBeChecked();
    });
  });

  describe("quantization filtering", () => {
    it("shows checked state for selected quantization levels", () => {
      render(FilterModal, {
        props: {
          open: true,
          filters: {
            ...defaultFilters,
            quantization: [4, 8],
          },
        },
      });

      const fourBitCheckbox = screen.getByRole("checkbox", { name: /4-bit/i });
      const eightBitCheckbox = screen.getByRole("checkbox", { name: /8-bit/i });
      const twoBitCheckbox = screen.getByRole("checkbox", { name: /2-bit/i });

      expect(fourBitCheckbox).toBeChecked();
      expect(eightBitCheckbox).toBeChecked();
      expect(twoBitCheckbox).not.toBeChecked();
    });

    it("toggles quantization selection on click", async () => {
      const user = userEvent.setup();
      render(FilterModal, {
        props: {
          open: true,
          filters: defaultFilters,
        },
      });

      const fourBitCheckbox = screen.getByRole("checkbox", { name: /4-bit/i });
      expect(fourBitCheckbox).not.toBeChecked();

      await user.click(fourBitCheckbox);
      expect(fourBitCheckbox).toBeChecked();

      await user.click(fourBitCheckbox);
      expect(fourBitCheckbox).not.toBeChecked();
    });

    it("shows fp16 label for 16-bit quantization", () => {
      render(FilterModal, {
        props: {
          open: true,
          filters: defaultFilters,
        },
      });

      expect(screen.getByText("fp16")).toBeInTheDocument();
      expect(screen.getByRole("checkbox", { name: /fp16/i })).toBeInTheDocument();
    });
  });

  describe("apply action", () => {
    it("calls onApply with updated filters when Apply clicked", async () => {
      const user = userEvent.setup();
      const onApply = vi.fn();
      const open = true;

      render(FilterModal, {
        props: {
          open,
          filters: defaultFilters,
          onApply,
        },
      });

      // Modify filters
      const llamaCheckbox = screen.getByRole("checkbox", { name: /llama/i });
      await user.click(llamaCheckbox);

      const fourBitCheckbox = screen.getByRole("checkbox", { name: /4-bit/i });
      await user.click(fourBitCheckbox);

      // Apply
      const applyButton = screen.getByRole("button", { name: /apply/i });
      await user.click(applyButton);

      expect(onApply).toHaveBeenCalledWith({
        architectures: ["Llama"],
        multimodal: null,
        quantization: [4],
      });
    });

    it("closes modal when Apply clicked", async () => {
      const user = userEvent.setup();
      render(FilterModal, {
        props: {
          open: true,
          filters: defaultFilters,
        },
      });

      const applyButton = screen.getByRole("button", { name: /apply/i });
      await user.click(applyButton);

      // Modal should close (dialog content should be removed)
      expect(screen.queryByText("Filter Models")).not.toBeInTheDocument();
    });

    it("applies complex filter combinations", async () => {
      const user = userEvent.setup();
      const onApply = vi.fn();

      render(FilterModal, {
        props: {
          open: true,
          filters: defaultFilters,
          onApply,
        },
      });

      // Select multiple architectures
      await user.click(screen.getByRole("checkbox", { name: /llama/i }));
      await user.click(screen.getByRole("checkbox", { name: /qwen/i }));

      // Select multimodal
      await user.click(screen.getByRole("radio", { name: /multimodal/i }));

      // Select multiple quantizations
      await user.click(screen.getByRole("checkbox", { name: /4-bit/i }));
      await user.click(screen.getByRole("checkbox", { name: /8-bit/i }));

      await user.click(screen.getByRole("button", { name: /apply/i }));

      expect(onApply).toHaveBeenCalledWith({
        architectures: ["Llama", "Qwen"],
        multimodal: true,
        quantization: [4, 8],
      });
    });
  });

  describe("clear all action", () => {
    it("clears all filters when Clear All clicked", async () => {
      const user = userEvent.setup();
      render(FilterModal, {
        props: {
          open: true,
          filters: {
            architectures: ["Llama", "Qwen"],
            multimodal: true,
            quantization: [4, 8],
          },
        },
      });

      // Verify initial state
      expect(screen.getByRole("checkbox", { name: /llama/i })).toBeChecked();
      expect(screen.getByRole("checkbox", { name: /4-bit/i })).toBeChecked();

      // Clear all
      const clearButton = screen.getByRole("button", { name: /clear all/i });
      await user.click(clearButton);

      // All filters should be cleared
      expect(screen.getByRole("checkbox", { name: /llama/i })).not.toBeChecked();
      expect(screen.getByRole("checkbox", { name: /qwen/i })).not.toBeChecked();
      expect(screen.getByRole("checkbox", { name: /4-bit/i })).not.toBeChecked();
      expect(screen.getByRole("checkbox", { name: /8-bit/i })).not.toBeChecked();
      expect(screen.getByRole("radio", { name: /any/i })).toBeChecked();
    });
  });

  describe("modal lifecycle", () => {
    it("resets local filters when modal opens", async () => {
      const user = userEvent.setup();
      const open = false;

      const { rerender } = render(FilterModal, {
        props: {
          open,
          filters: defaultFilters,
        },
      });

      // Open modal
      await rerender({ open: true, filters: defaultFilters });

      // Make changes
      const llamaCheckbox = screen.getByRole("checkbox", { name: /llama/i });
      await user.click(llamaCheckbox);
      expect(llamaCheckbox).toBeChecked();

      // Close modal without applying
      await rerender({ open: false, filters: defaultFilters });

      // Reopen modal
      await rerender({ open: true, filters: defaultFilters });

      // Changes should be reset
      const resetCheckbox = screen.getByRole("checkbox", { name: /llama/i });
      expect(resetCheckbox).not.toBeChecked();
    });

    it("preserves original filters when closed without applying", async () => {
      const user = userEvent.setup();
      const onApply = vi.fn();
      const originalFilters: FilterState = {
        architectures: ["Llama"],
        multimodal: false,
        quantization: [4],
      };

      const { rerender } = render(FilterModal, {
        props: {
          open: true,
          filters: originalFilters,
          onApply,
        },
      });

      // Make changes
      await user.click(screen.getByRole("checkbox", { name: /qwen/i }));

      // Close without applying
      await rerender({ open: false, filters: originalFilters, onApply });

      // onApply should not have been called
      expect(onApply).not.toHaveBeenCalled();
    });
  });

  describe("edge cases", () => {
    it("handles all architectures selected", async () => {
      const user = userEvent.setup();
      const onApply = vi.fn();

      render(FilterModal, {
        props: {
          open: true,
          filters: defaultFilters,
          onApply,
        },
      });

      // Select all architectures
      for (const arch of ARCHITECTURE_OPTIONS) {
        await user.click(screen.getByRole("checkbox", { name: new RegExp(arch, "i") }));
      }

      await user.click(screen.getByRole("button", { name: /apply/i }));

      expect(onApply).toHaveBeenCalledWith({
        architectures: ARCHITECTURE_OPTIONS,
        multimodal: null,
        quantization: [],
      });
    });

    it("handles all quantizations selected", async () => {
      const user = userEvent.setup();
      const onApply = vi.fn();

      render(FilterModal, {
        props: {
          open: true,
          filters: defaultFilters,
          onApply,
        },
      });

      // Select all quantizations
      for (const bits of QUANTIZATION_OPTIONS) {
        const label = bits === 16 ? "fp16" : `${bits}-bit`;
        await user.click(screen.getByRole("checkbox", { name: new RegExp(label, "i") }));
      }

      await user.click(screen.getByRole("button", { name: /apply/i }));

      expect(onApply).toHaveBeenCalledWith({
        architectures: [],
        multimodal: null,
        quantization: QUANTIZATION_OPTIONS,
      });
    });

    it("handles closing via close button", async () => {
      const user = userEvent.setup();
      render(FilterModal, {
        props: {
          open: true,
          filters: defaultFilters,
        },
      });

      const closeButton = screen.getByRole("button", { name: /close/i });
      await user.click(closeButton);

      expect(screen.queryByText("Filter Models")).not.toBeInTheDocument();
    });
  });
});
