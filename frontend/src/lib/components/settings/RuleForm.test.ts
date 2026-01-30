import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/svelte";
import userEvent from "@testing-library/user-event";
import RuleForm from "./RuleForm.svelte";
import type { BackendType } from "$lib/api/types";
import { settings } from "$lib/api/client";

// Mock API
vi.mock("$lib/api/client", () => ({
  settings: {
    createRule: vi.fn(),
  },
}));

describe("RuleForm", () => {
  let mockOnSave: ReturnType<typeof vi.fn>;
  let configuredProviders: BackendType[];

  beforeEach(() => {
    mockOnSave = vi.fn();
    configuredProviders = ["local", "openai"];
    vi.mocked(settings.createRule).mockResolvedValue({
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
    });
  });

  describe("rendering", () => {
    it("renders form title", () => {
      render(RuleForm, {
        props: { onSave: mockOnSave, configuredProviders },
      });

      expect(screen.getByText("Add Routing Rule")).toBeInTheDocument();
    });

    it("renders all required fields", () => {
      render(RuleForm, {
        props: { onSave: mockOnSave, configuredProviders },
      });

      expect(screen.getByLabelText(/Pattern Type/)).toBeInTheDocument();
      expect(screen.getByLabelText(/^Pattern$/)).toBeInTheDocument();
      expect(screen.getByLabelText(/Route to Backend/)).toBeInTheDocument();
    });

    it("renders optional fields", () => {
      render(RuleForm, {
        props: { onSave: mockOnSave, configuredProviders },
      });

      expect(screen.getByLabelText(/Backend Model/)).toBeInTheDocument();
      expect(screen.getByLabelText(/Fallback Backend/)).toBeInTheDocument();
    });

    it("renders Add Rule button", () => {
      render(RuleForm, {
        props: { onSave: mockOnSave, configuredProviders },
      });

      expect(
        screen.getByRole("button", { name: /Add Rule/ }),
      ).toBeInTheDocument();
    });
  });

  describe("pattern type selection", () => {
    it("defaults to prefix pattern type", () => {
      render(RuleForm, {
        props: { onSave: mockOnSave, configuredProviders },
      });

      const select = screen.getByLabelText(
        /Pattern Type/,
      ) as HTMLSelectElement;
      expect(select.value).toBe("prefix");
    });

    it("shows prefix placeholder for prefix type", () => {
      render(RuleForm, {
        props: { onSave: mockOnSave, configuredProviders },
      });

      const input = screen.getByPlaceholderText("gpt-");
      expect(input).toBeInTheDocument();
    });

    it("updates placeholder when changing to exact match", async () => {
      const user = userEvent.setup();
      render(RuleForm, {
        props: { onSave: mockOnSave, configuredProviders },
      });

      const select = screen.getByLabelText(
        /Pattern Type/,
      ) as HTMLSelectElement;
      await user.selectOptions(select, "exact");

      expect(screen.getByPlaceholderText("gpt-4-turbo")).toBeInTheDocument();
    });

    it("updates placeholder when changing to regex", async () => {
      const user = userEvent.setup();
      render(RuleForm, {
        props: { onSave: mockOnSave, configuredProviders },
      });

      const select = screen.getByLabelText(
        /Pattern Type/,
      ) as HTMLSelectElement;
      await user.selectOptions(select, "regex");

      expect(
        screen.getByPlaceholderText("^claude-3-.*$"),
      ).toBeInTheDocument();
    });
  });

  describe("backend selection", () => {
    it("defaults to local backend", () => {
      render(RuleForm, {
        props: { onSave: mockOnSave, configuredProviders },
      });

      const select = screen.getByLabelText(
        /Route to Backend/,
      ) as HTMLSelectElement;
      expect(select.value).toBe("local");
    });

    it("shows all backend options", () => {
      render(RuleForm, {
        props: { onSave: mockOnSave, configuredProviders },
      });

      const select = screen.getByLabelText(
        /Route to Backend/,
      ) as HTMLSelectElement;
      const options = Array.from(select.options).map((opt) => opt.value);

      expect(options).toContain("local");
      expect(options).toContain("openai");
      expect(options).toContain("anthropic");
    });

    it("does not show warning for configured provider", () => {
      render(RuleForm, {
        props: { onSave: mockOnSave, configuredProviders: ["local", "openai"] },
      });

      expect(
        screen.queryByText(/Provider not configured/),
      ).not.toBeInTheDocument();
    });

    it("shows warning when selecting unconfigured provider", async () => {
      const user = userEvent.setup();
      render(RuleForm, {
        props: { onSave: mockOnSave, configuredProviders: ["local"] },
      });

      const select = screen.getByLabelText(
        /Route to Backend/,
      ) as HTMLSelectElement;
      await user.selectOptions(select, "anthropic");

      expect(
        screen.getByText(/Provider not configured/),
      ).toBeInTheDocument();
    });

    it("does not show warning for local backend", () => {
      render(RuleForm, {
        props: { onSave: mockOnSave, configuredProviders: [] },
      });

      expect(
        screen.queryByText(/Provider not configured/),
      ).not.toBeInTheDocument();
    });

    it("updates warning reactively when configuredProviders changes", async () => {
      const { rerender } = render(RuleForm, {
        props: { onSave: mockOnSave, configuredProviders: ["local"] },
      });

      const user = userEvent.setup();
      const select = screen.getByLabelText(
        /Route to Backend/,
      ) as HTMLSelectElement;
      await user.selectOptions(select, "openai");

      // Should show warning initially
      expect(
        screen.getByText(/Provider not configured/),
      ).toBeInTheDocument();

      // Update configured providers
      await rerender({
        onSave: mockOnSave,
        configuredProviders: ["local", "openai"],
      });

      // Warning should disappear
      expect(
        screen.queryByText(/Provider not configured/),
      ).not.toBeInTheDocument();
    });
  });

  describe("fallback backend selection", () => {
    it("defaults to no fallback", () => {
      render(RuleForm, {
        props: { onSave: mockOnSave, configuredProviders },
      });

      const select = screen.getByLabelText(
        /Fallback Backend/,
      ) as HTMLSelectElement;
      expect(select.value).toBe("");
    });

    it("excludes primary backend from fallback options", async () => {
      const user = userEvent.setup();
      render(RuleForm, {
        props: { onSave: mockOnSave, configuredProviders },
      });

      // Select openai as primary
      const backendSelect = screen.getByLabelText(
        /Route to Backend/,
      ) as HTMLSelectElement;
      await user.selectOptions(backendSelect, "openai");

      // Check fallback options
      const fallbackSelect = screen.getByLabelText(
        /Fallback Backend/,
      ) as HTMLSelectElement;
      const options = Array.from(fallbackSelect.options).map(
        (opt) => opt.value,
      );

      expect(options).not.toContain("openai");
      expect(options).toContain("local");
      expect(options).toContain("anthropic");
    });

    it("includes No fallback option", () => {
      render(RuleForm, {
        props: { onSave: mockOnSave, configuredProviders },
      });

      expect(screen.getByText("No fallback")).toBeInTheDocument();
    });
  });

  describe("form submission", () => {
    it("submits minimal rule", async () => {
      const user = userEvent.setup();
      render(RuleForm, {
        props: { onSave: mockOnSave, configuredProviders },
      });

      await user.type(screen.getByLabelText(/^Pattern$/), "gpt-4");
      await user.click(screen.getByRole("button", { name: /Add Rule/ }));

      await waitFor(() => {
        expect(settings.createRule).toHaveBeenCalledWith({
          pattern_type: "prefix",
          model_pattern: "gpt-4",
          backend_type: "local",
          backend_model: undefined,
          fallback_backend: undefined,
        });
      });
    });

    it("submits rule with all fields", async () => {
      const user = userEvent.setup();
      render(RuleForm, {
        props: { onSave: mockOnSave, configuredProviders },
      });

      await user.selectOptions(
        screen.getByLabelText(/Pattern Type/),
        "exact",
      );
      await user.type(screen.getByLabelText(/^Pattern$/), "gpt-4-turbo");
      await user.selectOptions(
        screen.getByLabelText(/Route to Backend/),
        "openai",
      );
      await user.type(
        screen.getByLabelText(/Backend Model/),
        "gpt-4-1106-preview",
      );
      await user.selectOptions(
        screen.getByLabelText(/Fallback Backend/),
        "local",
      );

      await user.click(screen.getByRole("button", { name: /Add Rule/ }));

      await waitFor(() => {
        expect(settings.createRule).toHaveBeenCalledWith({
          pattern_type: "exact",
          model_pattern: "gpt-4-turbo",
          backend_type: "openai",
          backend_model: "gpt-4-1106-preview",
          fallback_backend: "local",
        });
      });
    });

    it("trims whitespace from pattern", async () => {
      const user = userEvent.setup();
      render(RuleForm, {
        props: { onSave: mockOnSave, configuredProviders },
      });

      await user.type(screen.getByLabelText(/^Pattern$/), "  gpt-4  ");
      await user.click(screen.getByRole("button", { name: /Add Rule/ }));

      await waitFor(() => {
        expect(settings.createRule).toHaveBeenCalledWith(
          expect.objectContaining({
            model_pattern: "gpt-4",
          }),
        );
      });
    });

    it("trims whitespace from backend model", async () => {
      const user = userEvent.setup();
      render(RuleForm, {
        props: { onSave: mockOnSave, configuredProviders },
      });

      await user.type(screen.getByLabelText(/^Pattern$/), "gpt");
      await user.type(
        screen.getByLabelText(/Backend Model/),
        "  model-name  ",
      );
      await user.click(screen.getByRole("button", { name: /Add Rule/ }));

      await waitFor(() => {
        expect(settings.createRule).toHaveBeenCalledWith(
          expect.objectContaining({
            backend_model: "model-name",
          }),
        );
      });
    });

    it("shows loading state during submission", async () => {
      const user = userEvent.setup();
      let resolveCreate: () => void;
      const createPromise = new Promise<void>((resolve) => {
        resolveCreate = resolve;
      });
      vi.mocked(settings.createRule).mockReturnValue(
        createPromise as any,
      );

      render(RuleForm, {
        props: { onSave: mockOnSave, configuredProviders },
      });

      await user.type(screen.getByLabelText(/^Pattern$/), "gpt-4");
      await user.click(screen.getByRole("button", { name: /Add Rule/ }));

      expect(
        screen.getByRole("button", { name: /Add Rule/ }),
      ).toBeDisabled();

      resolveCreate!();
      await createPromise;
    });

    it("calls onSave after successful submission", async () => {
      const user = userEvent.setup();
      render(RuleForm, {
        props: { onSave: mockOnSave, configuredProviders },
      });

      await user.type(screen.getByLabelText(/^Pattern$/), "gpt-4");
      await user.click(screen.getByRole("button", { name: /Add Rule/ }));

      await waitFor(() => {
        expect(mockOnSave).toHaveBeenCalled();
      });
    });

    it("resets form after successful submission", async () => {
      const user = userEvent.setup();
      render(RuleForm, {
        props: { onSave: mockOnSave, configuredProviders },
      });

      await user.type(screen.getByLabelText(/^Pattern$/), "gpt-4");
      await user.type(screen.getByLabelText(/Backend Model/), "model-name");
      await user.click(screen.getByRole("button", { name: /Add Rule/ }));

      await waitFor(() => {
        expect(
          (screen.getByLabelText(/^Pattern$/) as HTMLInputElement).value,
        ).toBe("");
        expect(
          (screen.getByLabelText(/Backend Model/) as HTMLInputElement).value,
        ).toBe("");
      });
    });

    it("displays error message on submission failure", async () => {
      const user = userEvent.setup();
      vi.mocked(settings.createRule).mockRejectedValue(
        new Error("Validation error"),
      );

      render(RuleForm, {
        props: { onSave: mockOnSave, configuredProviders },
      });

      await user.type(screen.getByLabelText(/^Pattern$/), "gpt-4");
      await user.click(screen.getByRole("button", { name: /Add Rule/ }));

      await waitFor(() => {
        expect(screen.getByText("Validation error")).toBeInTheDocument();
      });
    });

    it("displays generic error for non-Error failures", async () => {
      const user = userEvent.setup();
      vi.mocked(settings.createRule).mockRejectedValue("string error");

      render(RuleForm, {
        props: { onSave: mockOnSave, configuredProviders },
      });

      await user.type(screen.getByLabelText(/^Pattern$/), "gpt-4");
      await user.click(screen.getByRole("button", { name: /Add Rule/ }));

      await waitFor(() => {
        expect(screen.getByText("Failed to create rule")).toBeInTheDocument();
      });
    });
  });

  describe("validation", () => {
    it("prevents submission with empty pattern via required attribute", () => {
      render(RuleForm, {
        props: { onSave: mockOnSave, configuredProviders },
      });

      // Input is required, so HTML5 validation should prevent submission
      const input = screen.getByLabelText(/^Pattern$/);
      expect(input).toHaveAttribute("required");
    });

    it("validates pattern is not whitespace-only", async () => {
      const user = userEvent.setup();
      render(RuleForm, {
        props: { onSave: mockOnSave, configuredProviders },
      });

      await user.type(screen.getByLabelText(/^Pattern$/), "   ");
      await user.click(screen.getByRole("button", { name: /Add Rule/ }));

      // Whitespace-only should be trimmed and fail validation
      await waitFor(() => {
        // Should not call API since pattern becomes empty after trim
        expect(settings.createRule).not.toHaveBeenCalled();
      });
    });

    it("accepts valid pattern", async () => {
      const user = userEvent.setup();
      render(RuleForm, {
        props: { onSave: mockOnSave, configuredProviders },
      });

      await user.type(screen.getByLabelText(/^Pattern$/), "gpt-4");
      await user.click(screen.getByRole("button", { name: /Add Rule/ }));

      await waitFor(() => {
        expect(settings.createRule).toHaveBeenCalled();
      });
    });
  });

  describe("empty backend model handling", () => {
    it("sends undefined for empty backend model", async () => {
      const user = userEvent.setup();
      render(RuleForm, {
        props: { onSave: mockOnSave, configuredProviders },
      });

      await user.type(screen.getByLabelText(/^Pattern$/), "gpt-4");
      // Leave backend model empty
      await user.click(screen.getByRole("button", { name: /Add Rule/ }));

      await waitFor(() => {
        expect(settings.createRule).toHaveBeenCalledWith(
          expect.objectContaining({
            backend_model: undefined,
          }),
        );
      });
    });

    it("sends undefined for whitespace-only backend model", async () => {
      const user = userEvent.setup();
      render(RuleForm, {
        props: { onSave: mockOnSave, configuredProviders },
      });

      await user.type(screen.getByLabelText(/^Pattern$/), "gpt-4");
      await user.type(screen.getByLabelText(/Backend Model/), "   ");
      await user.click(screen.getByRole("button", { name: /Add Rule/ }));

      await waitFor(() => {
        expect(settings.createRule).toHaveBeenCalledWith(
          expect.objectContaining({
            backend_model: undefined,
          }),
        );
      });
    });
  });
});
