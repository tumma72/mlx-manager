import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/svelte";
import userEvent from "@testing-library/user-event";
import RuleCard from "./RuleCard.svelte";
import type { BackendMapping } from "$lib/api/types";

function createMockRule(
  overrides: Partial<BackendMapping> = {},
): BackendMapping {
  return {
    id: 1,
    pattern_type: "prefix",
    model_pattern: "gpt-",
    backend_type: "openai",
    backend_model: null,
    fallback_backend: null,
    priority: 1,
    enabled: true,
    ...overrides,
  };
}

describe("RuleCard", () => {
  let mockOnDelete: () => void;
  let mockOnEdit: () => void;

  beforeEach(() => {
    mockOnDelete = vi.fn();
    mockOnEdit = vi.fn();
  });

  describe("rendering", () => {
    it("renders pattern type badge", () => {
      render(RuleCard, {
        props: {
          rule: createMockRule({ pattern_type: "prefix" }),
          onDelete: mockOnDelete,
          onEdit: mockOnEdit,
        },
      });

      expect(screen.getByText("prefix")).toBeInTheDocument();
    });

    it("renders model pattern", () => {
      render(RuleCard, {
        props: {
          rule: createMockRule({ model_pattern: "gpt-4" }),
          onDelete: mockOnDelete,
          onEdit: mockOnEdit,
        },
      });

      expect(screen.getByText("gpt-4")).toBeInTheDocument();
    });

    it("renders backend type in routing info", () => {
      render(RuleCard, {
        props: {
          rule: createMockRule({ backend_type: "openai" }),
          onDelete: mockOnDelete,
          onEdit: mockOnEdit,
        },
      });

      expect(screen.getByText(/OpenAI/)).toBeInTheDocument();
    });

    it("renders local backend", () => {
      render(RuleCard, {
        props: {
          rule: createMockRule({ backend_type: "local" }),
          onDelete: mockOnDelete,
          onEdit: mockOnEdit,
        },
      });

      expect(screen.getByText(/Local/)).toBeInTheDocument();
    });

    it("renders anthropic backend", () => {
      render(RuleCard, {
        props: {
          rule: createMockRule({ backend_type: "anthropic" }),
          onDelete: mockOnDelete,
          onEdit: mockOnEdit,
        },
      });

      expect(screen.getByText(/Anthropic/)).toBeInTheDocument();
    });

    it("renders drag handle", () => {
      const { container } = render(RuleCard, {
        props: {
          rule: createMockRule(),
          onDelete: mockOnDelete,
          onEdit: mockOnEdit,
        },
      });

      const dragHandle = container.querySelector(".sortable-handle");
      expect(dragHandle).toBeInTheDocument();
    });

    it("renders delete button", () => {
      render(RuleCard, {
        props: {
          rule: createMockRule(),
          onDelete: mockOnDelete,
          onEdit: mockOnEdit,
        },
      });

      expect(screen.getByTitle("Delete rule")).toBeInTheDocument();
    });

    it("renders edit button", () => {
      render(RuleCard, {
        props: {
          rule: createMockRule(),
          onDelete: mockOnDelete,
          onEdit: mockOnEdit,
        },
      });

      expect(screen.getByTitle("Edit rule")).toBeInTheDocument();
    });
  });

  describe("pattern type badges", () => {
    it("applies blue color for exact match", () => {
      render(RuleCard, {
        props: {
          rule: createMockRule({ pattern_type: "exact" }),
          onDelete: mockOnDelete,
          onEdit: mockOnEdit,
        },
      });

      const badge = screen.getByText("exact");
      expect(badge.className).toContain("bg-blue-100");
    });

    it("applies green color for prefix match", () => {
      render(RuleCard, {
        props: {
          rule: createMockRule({ pattern_type: "prefix" }),
          onDelete: mockOnDelete,
          onEdit: mockOnEdit,
        },
      });

      const badge = screen.getByText("prefix");
      expect(badge.className).toContain("bg-green-100");
    });

    it("applies purple color for regex", () => {
      render(RuleCard, {
        props: {
          rule: createMockRule({ pattern_type: "regex" }),
          onDelete: mockOnDelete,
          onEdit: mockOnEdit,
        },
      });

      const badge = screen.getByText("regex");
      expect(badge.className).toContain("bg-purple-100");
    });
  });

  describe("backend model display", () => {
    it("shows backend model when specified", () => {
      render(RuleCard, {
        props: {
          rule: createMockRule({ backend_model: "gpt-4-turbo" }),
          onDelete: mockOnDelete,
          onEdit: mockOnEdit,
        },
      });

      expect(screen.getByText(/as gpt-4-turbo/)).toBeInTheDocument();
    });

    it("does not show backend model when null", () => {
      render(RuleCard, {
        props: {
          rule: createMockRule({ backend_model: null }),
          onDelete: mockOnDelete,
          onEdit: mockOnEdit,
        },
      });

      expect(screen.queryByText(/as /)).not.toBeInTheDocument();
    });
  });

  describe("fallback backend display", () => {
    it("shows fallback backend when specified", () => {
      render(RuleCard, {
        props: {
          rule: createMockRule({ fallback_backend: "local" }),
          onDelete: mockOnDelete,
          onEdit: mockOnEdit,
        },
      });

      expect(screen.getByText(/fallback: Local/)).toBeInTheDocument();
    });

    it("does not show fallback when null", () => {
      render(RuleCard, {
        props: {
          rule: createMockRule({ fallback_backend: null }),
          onDelete: mockOnDelete,
          onEdit: mockOnEdit,
        },
      });

      expect(screen.queryByText(/fallback/)).not.toBeInTheDocument();
    });
  });

  describe("warning display", () => {
    it("shows warning badge when hasWarning is true", () => {
      render(RuleCard, {
        props: {
          rule: createMockRule(),
          hasWarning: true,
          onDelete: mockOnDelete,
          onEdit: mockOnEdit,
        },
      });

      expect(screen.getByText("Unconfigured")).toBeInTheDocument();
    });

    it("does not show warning when hasWarning is false", () => {
      render(RuleCard, {
        props: {
          rule: createMockRule(),
          hasWarning: false,
          onDelete: mockOnDelete,
          onEdit: mockOnEdit,
        },
      });

      expect(screen.queryByText("Unconfigured")).not.toBeInTheDocument();
    });

    it("does not show warning by default", () => {
      render(RuleCard, {
        props: {
          rule: createMockRule(),
          onDelete: mockOnDelete,
          onEdit: mockOnEdit,
        },
      });

      expect(screen.queryByText("Unconfigured")).not.toBeInTheDocument();
    });
  });

  describe("enabled/disabled state", () => {
    it("applies opacity when rule is disabled", () => {
      const { container } = render(RuleCard, {
        props: {
          rule: createMockRule({ enabled: false }),
          onDelete: mockOnDelete,
          onEdit: mockOnEdit,
        },
      });

      const card = container.querySelector('[data-rule-id="1"]');
      expect(card?.className).toContain("opacity-50");
    });

    it("does not apply opacity when rule is enabled", () => {
      const { container } = render(RuleCard, {
        props: {
          rule: createMockRule({ enabled: true }),
          onDelete: mockOnDelete,
          onEdit: mockOnEdit,
        },
      });

      const card = container.querySelector('[data-rule-id="1"]');
      expect(card?.className).not.toContain("opacity-50");
    });
  });

  describe("delete action", () => {
    it("calls onDelete when delete button clicked", async () => {
      const user = userEvent.setup();
      render(RuleCard, {
        props: {
          rule: createMockRule({ id: 42 }),
          onDelete: mockOnDelete,
          onEdit: mockOnEdit,
        },
      });

      await user.click(screen.getByTitle("Delete rule"));

      expect(mockOnDelete).toHaveBeenCalled();
    });
  });

  describe("edit action", () => {
    it("calls onEdit when edit button clicked", async () => {
      const user = userEvent.setup();
      render(RuleCard, {
        props: {
          rule: createMockRule({ id: 42 }),
          onDelete: mockOnDelete,
          onEdit: mockOnEdit,
        },
      });

      await user.click(screen.getByTitle("Edit rule"));

      expect(mockOnEdit).toHaveBeenCalled();
    });
  });

  describe("data attributes", () => {
    it("sets data-rule-id attribute", () => {
      const { container } = render(RuleCard, {
        props: {
          rule: createMockRule({ id: 99 }),
          onDelete: mockOnDelete,
          onEdit: mockOnEdit,
        },
      });

      const card = container.querySelector('[data-rule-id="99"]');
      expect(card).toBeInTheDocument();
    });
  });

  describe("complex rules", () => {
    it("renders rule with all fields", () => {
      const rule = createMockRule({
        pattern_type: "regex",
        model_pattern: "^claude-.*$",
        backend_type: "anthropic",
        backend_model: "claude-3-opus-20240229",
        fallback_backend: "local",
      });

      render(RuleCard, {
        props: { rule, onDelete: mockOnDelete, onEdit: mockOnEdit },
      });

      expect(screen.getByText("regex")).toBeInTheDocument();
      expect(screen.getByText("^claude-.*$")).toBeInTheDocument();
      expect(screen.getByText(/Anthropic/)).toBeInTheDocument();
      expect(
        screen.getByText(/as claude-3-opus-20240229/),
      ).toBeInTheDocument();
      expect(screen.getByText(/fallback: Local/)).toBeInTheDocument();
    });

    it("renders rule with warning and fallback", () => {
      const rule = createMockRule({
        backend_type: "openai",
        fallback_backend: "local",
      });

      render(RuleCard, {
        props: { rule, hasWarning: true, onDelete: mockOnDelete, onEdit: mockOnEdit },
      });

      expect(screen.getByText("Unconfigured")).toBeInTheDocument();
      expect(screen.getByText(/fallback: Local/)).toBeInTheDocument();
    });
  });
});
