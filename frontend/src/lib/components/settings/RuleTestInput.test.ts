import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/svelte";
import userEvent from "@testing-library/user-event";
import RuleTestInput from "./RuleTestInput.svelte";
import type { BackendMapping } from "$lib/api/types";
import { settings } from "$lib/api/client";

// Mock API
vi.mock("$lib/api/client", () => ({
  settings: {
    testRule: vi.fn(),
  },
}));

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
    created_at: "2024-01-01",
    updated_at: "2024-01-01",
    ...overrides,
  };
}

describe("RuleTestInput", () => {
  let mockRules: BackendMapping[];

  beforeEach(() => {
    mockRules = [
      createMockRule({
        id: 1,
        pattern_type: "prefix",
        model_pattern: "gpt-",
        backend_type: "openai",
      }),
      createMockRule({
        id: 2,
        pattern_type: "exact",
        model_pattern: "claude-3-opus",
        backend_type: "anthropic",
      }),
    ];

    vi.mocked(settings.testRule).mockResolvedValue({
      backend_type: "openai",
      matched_rule_id: 1,
    });
  });

  describe("rendering", () => {
    it("renders test input title", () => {
      render(RuleTestInput, { props: { rules: mockRules } });

      expect(screen.getByText("Test Model Routing")).toBeInTheDocument();
    });

    it("renders input field", () => {
      render(RuleTestInput, { props: { rules: mockRules } });

      expect(
        screen.getByPlaceholderText(/Enter model name to test/),
      ).toBeInTheDocument();
    });

    it("renders Test button", () => {
      render(RuleTestInput, { props: { rules: mockRules } });

      expect(
        screen.getByRole("button", { name: /Test/ }),
      ).toBeInTheDocument();
    });
  });

  describe("test action", () => {
    it("tests model name when button clicked", async () => {
      const user = userEvent.setup();
      render(RuleTestInput, { props: { rules: mockRules } });

      await user.type(
        screen.getByPlaceholderText(/Enter model name to test/),
        "gpt-4",
      );
      await user.click(screen.getByRole("button", { name: /Test/ }));

      await waitFor(() => {
        expect(settings.testRule).toHaveBeenCalledWith("gpt-4");
      });
    });

    it("tests model name when Enter pressed", async () => {
      const user = userEvent.setup();
      render(RuleTestInput, { props: { rules: mockRules } });

      const input = screen.getByPlaceholderText(/Enter model name to test/);
      await user.type(input, "gpt-4{Enter}");

      await waitFor(() => {
        expect(settings.testRule).toHaveBeenCalledWith("gpt-4");
      });
    });

    it("trims whitespace from model name", async () => {
      const user = userEvent.setup();
      render(RuleTestInput, { props: { rules: mockRules } });

      await user.type(
        screen.getByPlaceholderText(/Enter model name to test/),
        "  gpt-4  ",
      );
      await user.click(screen.getByRole("button", { name: /Test/ }));

      await waitFor(() => {
        expect(settings.testRule).toHaveBeenCalledWith("gpt-4");
      });
    });

    it("does not test when input is empty", async () => {
      const user = userEvent.setup();
      render(RuleTestInput, { props: { rules: mockRules } });

      await user.click(screen.getByRole("button", { name: /Test/ }));

      expect(settings.testRule).not.toHaveBeenCalled();
    });

    it("disables Test button when input is empty", () => {
      render(RuleTestInput, { props: { rules: mockRules } });

      expect(screen.getByRole("button", { name: /Test/ })).toBeDisabled();
    });

    it("enables Test button when input has value", async () => {
      const user = userEvent.setup();
      render(RuleTestInput, { props: { rules: mockRules } });

      await user.type(
        screen.getByPlaceholderText(/Enter model name to test/),
        "gpt-4",
      );

      expect(screen.getByRole("button", { name: /Test/ })).toBeEnabled();
    });

    it("shows loading state during test", async () => {
      const user = userEvent.setup();
      let resolveTest: () => void;
      const testPromise = new Promise<void>((resolve) => {
        resolveTest = resolve;
      });
      vi.mocked(settings.testRule).mockReturnValue(testPromise as any);

      render(RuleTestInput, { props: { rules: mockRules } });

      await user.type(
        screen.getByPlaceholderText(/Enter model name to test/),
        "gpt-4",
      );
      await user.click(screen.getByRole("button", { name: /Test/ }));

      expect(screen.getByRole("button", { name: /Test/ })).toBeDisabled();

      resolveTest!();
      await testPromise;
    });
  });

  describe("test results - matched rule", () => {
    it("displays matched rule info", async () => {
      const user = userEvent.setup();
      render(RuleTestInput, { props: { rules: mockRules } });

      await user.type(
        screen.getByPlaceholderText(/Enter model name to test/),
        "gpt-4",
      );
      await user.click(screen.getByRole("button", { name: /Test/ }));

      await waitFor(() => {
        expect(screen.getByText("Matched:")).toBeInTheDocument();
        expect(screen.getByText("gpt-")).toBeInTheDocument();
        expect(screen.getByText("OpenAI")).toBeInTheDocument();
      });
    });

    it("displays Local (MLX) for local backend", async () => {
      const user = userEvent.setup();
      vi.mocked(settings.testRule).mockResolvedValue({
        backend_type: "local",
        matched_rule_id: 1,
      });

      render(RuleTestInput, { props: { rules: mockRules } });

      await user.type(
        screen.getByPlaceholderText(/Enter model name to test/),
        "local-model",
      );
      await user.click(screen.getByRole("button", { name: /Test/ }));

      await waitFor(() => {
        expect(screen.getByText("Local (MLX)")).toBeInTheDocument();
      });
    });

    it("displays Anthropic for anthropic backend", async () => {
      const user = userEvent.setup();
      vi.mocked(settings.testRule).mockResolvedValue({
        backend_type: "anthropic",
        matched_rule_id: 2,
      });

      render(RuleTestInput, { props: { rules: mockRules } });

      await user.type(
        screen.getByPlaceholderText(/Enter model name to test/),
        "claude-3-opus",
      );
      await user.click(screen.getByRole("button", { name: /Test/ }));

      await waitFor(() => {
        expect(screen.getByText("Anthropic")).toBeInTheDocument();
      });
    });
  });

  describe("test results - no match", () => {
    it("displays default routing when no rule matches", async () => {
      const user = userEvent.setup();
      vi.mocked(settings.testRule).mockResolvedValue({
        backend_type: "local",
        matched_rule_id: null,
      });

      render(RuleTestInput, { props: { rules: mockRules } });

      await user.type(
        screen.getByPlaceholderText(/Enter model name to test/),
        "unknown-model",
      );
      await user.click(screen.getByRole("button", { name: /Test/ }));

      await waitFor(() => {
        expect(
          screen.getByText(/No rule matched. Routes to/),
        ).toBeInTheDocument();
        expect(screen.getByText("Local (MLX)")).toBeInTheDocument();
      });
    });
  });

  describe("error handling", () => {
    it("displays error message on test failure", async () => {
      const user = userEvent.setup();
      vi.mocked(settings.testRule).mockRejectedValue(
        new Error("API error"),
      );

      render(RuleTestInput, { props: { rules: mockRules } });

      await user.type(
        screen.getByPlaceholderText(/Enter model name to test/),
        "gpt-4",
      );
      await user.click(screen.getByRole("button", { name: /Test/ }));

      await waitFor(() => {
        expect(screen.getByText("API error")).toBeInTheDocument();
      });
    });

    it("displays generic error for non-Error failures", async () => {
      const user = userEvent.setup();
      vi.mocked(settings.testRule).mockRejectedValue("string error");

      render(RuleTestInput, { props: { rules: mockRules } });

      await user.type(
        screen.getByPlaceholderText(/Enter model name to test/),
        "gpt-4",
      );
      await user.click(screen.getByRole("button", { name: /Test/ }));

      await waitFor(() => {
        expect(screen.getByText("Test failed")).toBeInTheDocument();
      });
    });

    it("clears previous results before new test", async () => {
      const user = userEvent.setup();
      render(RuleTestInput, { props: { rules: mockRules } });

      // First test
      await user.type(
        screen.getByPlaceholderText(/Enter model name to test/),
        "gpt-4",
      );
      await user.click(screen.getByRole("button", { name: /Test/ }));

      await waitFor(() => {
        expect(screen.getByText("Matched:")).toBeInTheDocument();
      });

      // Second test - clear input and type new value
      const input = screen.getByPlaceholderText(/Enter model name to test/);
      await user.clear(input);
      await user.type(input, "gpt-3.5");
      await user.click(screen.getByRole("button", { name: /Test/ }));

      // Should still show result after second test
      await waitFor(() => {
        expect(screen.getByText("Matched:")).toBeInTheDocument();
      });
    });
  });

  describe("empty rules list", () => {
    it("handles empty rules array", async () => {
      const user = userEvent.setup();
      vi.mocked(settings.testRule).mockResolvedValue({
        backend_type: "local",
        matched_rule_id: null,
      });

      render(RuleTestInput, { props: { rules: [] } });

      await user.type(
        screen.getByPlaceholderText(/Enter model name to test/),
        "any-model",
      );
      await user.click(screen.getByRole("button", { name: /Test/ }));

      await waitFor(() => {
        expect(screen.getByText(/No rule matched/)).toBeInTheDocument();
      });
    });
  });
});
