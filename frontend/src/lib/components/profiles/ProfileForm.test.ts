import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/svelte";
import userEvent from "@testing-library/user-event";
import ProfileForm from "./ProfileForm.svelte";
import type { ServerProfile, ParserOptions, ServerProfileCreate, ServerProfileUpdate } from "$api";
import { models as modelsApi, system as systemApi } from "$api";

// Mock API modules
vi.mock("$api", () => ({
  models: {
    detectOptions: vi.fn(),
  },
  system: {
    parserOptions: vi.fn(),
  },
}));

// Helper to create mock profile
function createMockProfile(
  overrides: Partial<ServerProfile> = {},
): ServerProfile {
  return {
    id: 1,
    name: "Test Profile",
    description: null,
    model_path: "mlx-community/test-model",
    model_type: "lm",
    port: 10240,
    host: "127.0.0.1",
    context_length: null,
    max_concurrency: 1,
    queue_timeout: 300,
    queue_size: 100,
    tool_call_parser: null,
    reasoning_parser: null,
    message_converter: null,
    enable_auto_tool_choice: false,
    trust_remote_code: false,
    chat_template_file: null,
    log_level: "INFO",
    log_file: null,
    no_log_file: false,
    auto_start: false,
    system_prompt: null,
    launchd_installed: false,
    created_at: "2024-01-01T00:00:00Z",
    updated_at: "2024-01-01T00:00:00Z",
    ...overrides,
  };
}

// Helper to create mock parser options
function createMockParserOptions(): ParserOptions {
  return {
    tool_call_parsers: ["qwen3", "minimax", "glm"],
    reasoning_parsers: ["deepseek-r1"],
    message_converters: ["qwen3", "glm"],
  };
}

describe("ProfileForm", () => {
  let mockOnSubmit: (data: ServerProfileCreate | ServerProfileUpdate) => Promise<void>;
  let mockOnCancel: () => void;

  beforeEach(() => {
    mockOnSubmit = vi.fn().mockResolvedValue(undefined);
    mockOnCancel = vi.fn();

    // Setup default API mock responses
    vi.mocked(systemApi.parserOptions).mockResolvedValue(
      createMockParserOptions(),
    );
    vi.mocked(modelsApi.detectOptions).mockResolvedValue({
      model_family: "qwen",
      recommended_options: {
        tool_call_parser: "qwen3",
        reasoning_parser: undefined,
        message_converter: "qwen3",
      },
      is_downloaded: true,
    });
  });

  describe("create mode rendering", () => {
    it("renders create profile title", async () => {
      render(ProfileForm, {
        props: {
          nextPort: 10240,
          onSubmit: mockOnSubmit,
          onCancel: mockOnCancel,
        },
      });

      // Wait for parser options to load, then verify title exists
      await waitFor(() => {
        const titles = screen.getAllByText("Create Profile");
        expect(titles.length).toBeGreaterThan(0);
      });
    });

    it("renders all required fields", () => {
      render(ProfileForm, {
        props: {
          nextPort: 10240,
          onSubmit: mockOnSubmit,
          onCancel: mockOnCancel,
        },
      });

      expect(screen.getByLabelText(/Name/)).toBeInTheDocument();
      expect(screen.getByLabelText(/Port/)).toBeInTheDocument();
      expect(screen.getByLabelText(/Model Path/)).toBeInTheDocument();
    });

    it("renders optional fields", () => {
      render(ProfileForm, {
        props: {
          nextPort: 10240,
          onSubmit: mockOnSubmit,
          onCancel: mockOnCancel,
        },
      });

      expect(screen.getByLabelText(/Description/)).toBeInTheDocument();
      expect(screen.getByLabelText(/System Prompt/)).toBeInTheDocument();
      expect(screen.getByLabelText(/Model Type/)).toBeInTheDocument();
    });

    it("renders submit button with Create Profile text", () => {
      render(ProfileForm, {
        props: {
          nextPort: 10240,
          onSubmit: mockOnSubmit,
          onCancel: mockOnCancel,
        },
      });

      expect(
        screen.getByRole("button", { name: "Create Profile" }),
      ).toBeInTheDocument();
    });

    it("renders cancel button", () => {
      render(ProfileForm, {
        props: {
          nextPort: 10240,
          onSubmit: mockOnSubmit,
          onCancel: mockOnCancel,
        },
      });

      expect(
        screen.getByRole("button", { name: "Cancel" }),
      ).toBeInTheDocument();
    });

    it("initializes port with nextPort prop", () => {
      render(ProfileForm, {
        props: {
          nextPort: 12345,
          onSubmit: mockOnSubmit,
          onCancel: mockOnCancel,
        },
      });

      const portInput = screen.getByLabelText(/Port/) as HTMLInputElement;
      expect(portInput.value).toBe("12345");
    });

    it("initializes model path with initialModelPath prop", () => {
      render(ProfileForm, {
        props: {
          nextPort: 10240,
          initialModelPath: "mlx-community/initial-model",
          onSubmit: mockOnSubmit,
          onCancel: mockOnCancel,
        },
      });

      const modelPathInput = screen.getByLabelText(
        /Model Path/,
      ) as HTMLInputElement;
      expect(modelPathInput.value).toBe("mlx-community/initial-model");
    });
  });

  describe("edit mode rendering", () => {
    it("renders edit profile title", () => {
      render(ProfileForm, {
        props: {
          profile: createMockProfile(),
          nextPort: 10240,
          onSubmit: mockOnSubmit,
          onCancel: mockOnCancel,
        },
      });

      expect(screen.getByText("Edit Profile")).toBeInTheDocument();
    });

    it("renders submit button with Update Profile text", () => {
      render(ProfileForm, {
        props: {
          profile: createMockProfile(),
          nextPort: 10240,
          onSubmit: mockOnSubmit,
          onCancel: mockOnCancel,
        },
      });

      expect(
        screen.getByRole("button", { name: "Update Profile" }),
      ).toBeInTheDocument();
    });

    it("populates form with profile data", () => {
      const profile = createMockProfile({
        name: "Existing Profile",
        description: "Test description",
        model_path: "mlx-community/existing-model",
        port: 11111,
      });

      render(ProfileForm, {
        props: {
          profile,
          nextPort: 10240,
          onSubmit: mockOnSubmit,
          onCancel: mockOnCancel,
        },
      });

      expect(
        (screen.getByLabelText(/Name/) as HTMLInputElement).value,
      ).toBe("Existing Profile");
      expect(
        (screen.getByLabelText(/Description/) as HTMLTextAreaElement).value,
      ).toBe("Test description");
      expect(
        (screen.getByLabelText(/Model Path/) as HTMLInputElement).value,
      ).toBe("mlx-community/existing-model");
      expect((screen.getByLabelText(/Port/) as HTMLInputElement).value).toBe(
        "11111",
      );
    });

    it("populates system prompt", () => {
      const profile = createMockProfile({
        system_prompt: "You are a helpful assistant",
      });

      render(ProfileForm, {
        props: {
          profile,
          nextPort: 10240,
          onSubmit: mockOnSubmit,
          onCancel: mockOnCancel,
        },
      });

      expect(
        (screen.getByLabelText(/System Prompt/) as HTMLTextAreaElement).value,
      ).toBe("You are a helpful assistant");
    });
  });

  describe("advanced options", () => {
    it("hides advanced options by default", () => {
      render(ProfileForm, {
        props: {
          nextPort: 10240,
          onSubmit: mockOnSubmit,
          onCancel: mockOnCancel,
        },
      });

      expect(screen.queryByLabelText(/Host/)).not.toBeInTheDocument();
      expect(
        screen.queryByLabelText(/Max Concurrency/),
      ).not.toBeInTheDocument();
    });

    it("shows advanced options when toggle clicked", async () => {
      const user = userEvent.setup();
      render(ProfileForm, {
        props: {
          nextPort: 10240,
          onSubmit: mockOnSubmit,
          onCancel: mockOnCancel,
        },
      });

      const toggleButton = screen.getByText("Show Advanced Options");
      await user.click(toggleButton);

      expect(screen.getByLabelText(/Host/)).toBeInTheDocument();
      expect(screen.getByLabelText(/Max Concurrency/)).toBeInTheDocument();
      expect(screen.getByLabelText(/Queue Timeout/)).toBeInTheDocument();
      expect(screen.getByLabelText(/Queue Size/)).toBeInTheDocument();
    });

    it("hides advanced options when toggle clicked again", async () => {
      const user = userEvent.setup();
      render(ProfileForm, {
        props: {
          nextPort: 10240,
          onSubmit: mockOnSubmit,
          onCancel: mockOnCancel,
        },
      });

      const toggleButton = screen.getByText("Show Advanced Options");
      await user.click(toggleButton);
      await user.click(screen.getByText("Hide Advanced Options"));

      expect(screen.queryByLabelText(/Host/)).not.toBeInTheDocument();
    });

    it("renders parser options in advanced section", async () => {
      const user = userEvent.setup();
      render(ProfileForm, {
        props: {
          nextPort: 10240,
          onSubmit: mockOnSubmit,
          onCancel: mockOnCancel,
        },
      });

      await user.click(screen.getByText("Show Advanced Options"));

      await waitFor(() => {
        expect(
          screen.getByLabelText(/Tool Call Parser/),
        ).toBeInTheDocument();
        expect(
          screen.getByLabelText(/Reasoning Parser/),
        ).toBeInTheDocument();
        expect(
          screen.getByLabelText(/Message Converter/),
        ).toBeInTheDocument();
      });
    });

    it("renders auto-start checkbox in advanced section", async () => {
      const user = userEvent.setup();
      render(ProfileForm, {
        props: {
          nextPort: 10240,
          onSubmit: mockOnSubmit,
          onCancel: mockOnCancel,
        },
      });

      await user.click(screen.getByText("Show Advanced Options"));

      expect(
        screen.getByLabelText(/Start on Login/),
      ).toBeInTheDocument();
    });
  });

  describe("form submission", () => {
    it("submits form with required fields only", async () => {
      const user = userEvent.setup();
      render(ProfileForm, {
        props: {
          nextPort: 10240,
          onSubmit: mockOnSubmit,
          onCancel: mockOnCancel,
        },
      });

      await user.type(screen.getByLabelText(/Name/), "New Profile");
      await user.type(
        screen.getByLabelText(/Model Path/),
        "mlx-community/model",
      );

      await user.click(screen.getByRole("button", { name: "Create Profile" }));

      await waitFor(() => {
        expect(mockOnSubmit).toHaveBeenCalledWith({
          name: "New Profile",
          description: undefined,
          model_path: "mlx-community/model",
          model_type: "lm",
          port: 10240,
          host: "127.0.0.1",
          max_concurrency: 1,
          queue_timeout: 300,
          queue_size: 100,
          auto_start: false,
          tool_call_parser: undefined,
          reasoning_parser: undefined,
          message_converter: undefined,
          system_prompt: undefined,
        });
      });
    });

    it("submits form with all fields filled", async () => {
      const user = userEvent.setup();
      render(ProfileForm, {
        props: {
          nextPort: 10240,
          onSubmit: mockOnSubmit,
          onCancel: mockOnCancel,
        },
      });

      await user.type(screen.getByLabelText(/Name/), "Full Profile");
      await user.type(screen.getByLabelText(/Description/), "A description");
      await user.type(
        screen.getByLabelText(/System Prompt/),
        "Be helpful",
      );
      await user.type(
        screen.getByLabelText(/Model Path/),
        "mlx-community/model",
      );

      await user.click(screen.getByText("Show Advanced Options"));
      await user.clear(screen.getByLabelText(/Host/));
      await user.type(screen.getByLabelText(/Host/), "0.0.0.0");

      await user.click(screen.getByRole("button", { name: "Create Profile" }));

      await waitFor(() => {
        expect(mockOnSubmit).toHaveBeenCalledWith(
          expect.objectContaining({
            name: "Full Profile",
            description: "A description",
            system_prompt: "Be helpful",
            model_path: "mlx-community/model",
            host: "0.0.0.0",
          }),
        );
      });
    });

    it("shows loading state during submission", async () => {
      const user = userEvent.setup();
      let resolveSubmit: () => void;
      const submitPromise = new Promise<void>((resolve) => {
        resolveSubmit = resolve;
      });
      const slowSubmit = vi.fn().mockReturnValue(submitPromise);

      render(ProfileForm, {
        props: {
          nextPort: 10240,
          onSubmit: slowSubmit,
          onCancel: mockOnCancel,
        },
      });

      await user.type(screen.getByLabelText(/Name/), "Test");
      await user.type(
        screen.getByLabelText(/Model Path/),
        "mlx-community/model",
      );
      await user.click(screen.getByRole("button", { name: "Create Profile" }));

      expect(screen.getByText("Saving...")).toBeInTheDocument();

      resolveSubmit!();
      await submitPromise;
    });

    it("disables buttons during submission", async () => {
      const user = userEvent.setup();
      let resolveSubmit: () => void;
      const submitPromise = new Promise<void>((resolve) => {
        resolveSubmit = resolve;
      });
      const slowSubmit = vi.fn().mockReturnValue(submitPromise);

      render(ProfileForm, {
        props: {
          nextPort: 10240,
          onSubmit: slowSubmit,
          onCancel: mockOnCancel,
        },
      });

      await user.type(screen.getByLabelText(/Name/), "Test");
      await user.type(
        screen.getByLabelText(/Model Path/),
        "mlx-community/model",
      );
      await user.click(screen.getByRole("button", { name: "Create Profile" }));

      expect(screen.getByRole("button", { name: "Saving..." })).toBeDisabled();
      expect(screen.getByRole("button", { name: "Cancel" })).toBeDisabled();

      resolveSubmit!();
      await submitPromise;
    });

    it("displays error message on submission failure", async () => {
      const user = userEvent.setup();
      const failingSubmit = vi
        .fn()
        .mockRejectedValue(new Error("Network error"));

      render(ProfileForm, {
        props: {
          nextPort: 10240,
          onSubmit: failingSubmit,
          onCancel: mockOnCancel,
        },
      });

      await user.type(screen.getByLabelText(/Name/), "Test");
      await user.type(
        screen.getByLabelText(/Model Path/),
        "mlx-community/model",
      );
      await user.click(screen.getByRole("button", { name: "Create Profile" }));

      await waitFor(() => {
        expect(screen.getByText("Network error")).toBeInTheDocument();
      });
    });

    it("displays generic error for non-Error failures", async () => {
      const user = userEvent.setup();
      const failingSubmit = vi.fn().mockRejectedValue("string error");

      render(ProfileForm, {
        props: {
          nextPort: 10240,
          onSubmit: failingSubmit,
          onCancel: mockOnCancel,
        },
      });

      await user.type(screen.getByLabelText(/Name/), "Test");
      await user.type(
        screen.getByLabelText(/Model Path/),
        "mlx-community/model",
      );
      await user.click(screen.getByRole("button", { name: "Create Profile" }));

      await waitFor(() => {
        expect(screen.getByText("Failed to save profile")).toBeInTheDocument();
      });
    });
  });

  describe("cancel action", () => {
    it("calls onCancel when cancel button clicked", async () => {
      const user = userEvent.setup();
      render(ProfileForm, {
        props: {
          nextPort: 10240,
          onSubmit: mockOnSubmit,
          onCancel: mockOnCancel,
        },
      });

      await user.click(screen.getByRole("button", { name: "Cancel" }));

      expect(mockOnCancel).toHaveBeenCalled();
    });
  });

  describe("system prompt character count", () => {
    it("displays character count when typing in system prompt", async () => {
      const user = userEvent.setup();
      render(ProfileForm, {
        props: {
          nextPort: 10240,
          onSubmit: mockOnSubmit,
          onCancel: mockOnCancel,
        },
      });

      const prompt = "You are helpful";
      await user.type(screen.getByLabelText(/System Prompt/), prompt);

      expect(screen.getByText(/15 chars/)).toBeInTheDocument();
    });

    it("shows warning for long system prompts", async () => {
      const user = userEvent.setup();
      render(ProfileForm, {
        props: {
          nextPort: 10240,
          onSubmit: mockOnSubmit,
          onCancel: mockOnCancel,
        },
      });

      const longPrompt = "a".repeat(2001);
      await user.type(screen.getByLabelText(/System Prompt/), longPrompt);

      expect(
        screen.getByText(/long prompt may affect performance/),
      ).toBeInTheDocument();
    });
  });

  describe("model detection", () => {
    it("auto-detects model options for new profiles", async () => {
      const user = userEvent.setup();
      render(ProfileForm, {
        props: {
          nextPort: 10240,
          onSubmit: mockOnSubmit,
          onCancel: mockOnCancel,
        },
      });

      await user.type(
        screen.getByLabelText(/Model Path/),
        "mlx-community/qwen-model",
      );

      // Wait for debounced detection
      await waitFor(
        () => {
          expect(modelsApi.detectOptions).toHaveBeenCalledWith(
            "mlx-community/qwen-model",
          );
        },
        { timeout: 1000 },
      );
    });

    it("displays detected model family", async () => {
      const user = userEvent.setup();
      render(ProfileForm, {
        props: {
          nextPort: 10240,
          onSubmit: mockOnSubmit,
          onCancel: mockOnCancel,
        },
      });

      await user.click(screen.getByText("Show Advanced Options"));
      await user.type(
        screen.getByLabelText(/Model Path/),
        "mlx-community/qwen-model",
      );

      await waitFor(
        () => {
          expect(screen.getByText(/Detected: qwen/)).toBeInTheDocument();
        },
        { timeout: 1000 },
      );
    });

    it("does not auto-detect for existing profiles", async () => {
      render(ProfileForm, {
        props: {
          profile: createMockProfile({
            model_path: "mlx-community/existing",
          }),
          nextPort: 10240,
          onSubmit: mockOnSubmit,
          onCancel: mockOnCancel,
        },
      });

      // Should not trigger detection on mount
      expect(modelsApi.detectOptions).not.toHaveBeenCalled();
    });

    it("handles detection failure gracefully", async () => {
      const user = userEvent.setup();
      vi.mocked(modelsApi.detectOptions).mockRejectedValue(
        new Error("Detection failed"),
      );

      render(ProfileForm, {
        props: {
          nextPort: 10240,
          onSubmit: mockOnSubmit,
          onCancel: mockOnCancel,
        },
      });

      await user.type(
        screen.getByLabelText(/Model Path/),
        "mlx-community/unknown",
      );

      // Should not crash - user can still fill form manually
      await waitFor(() => {
        expect(screen.getByLabelText(/Model Path/)).toHaveValue(
          "mlx-community/unknown",
        );
      });
    });
  });

  describe("parser options loading", () => {
    it("loads parser options on mount", async () => {
      render(ProfileForm, {
        props: {
          nextPort: 10240,
          onSubmit: mockOnSubmit,
          onCancel: mockOnCancel,
        },
      });

      await waitFor(() => {
        expect(systemApi.parserOptions).toHaveBeenCalled();
      });
    });

    it("shows loading state for parser dropdowns", async () => {
      const user = userEvent.setup();
      vi.mocked(systemApi.parserOptions).mockImplementation(
        () => new Promise(() => {}), // Never resolves
      );

      render(ProfileForm, {
        props: {
          nextPort: 10240,
          onSubmit: mockOnSubmit,
          onCancel: mockOnCancel,
        },
      });

      await user.click(screen.getByText("Show Advanced Options"));

      const toolCallSelect = screen.getByLabelText(/Tool Call Parser/);
      expect(toolCallSelect).toBeDisabled();
    });

    it("handles parser options load failure gracefully", async () => {
      // Suppress expected console.error output
      const consoleSpy = vi.spyOn(console, "error").mockImplementation(() => {});

      const user = userEvent.setup();
      vi.mocked(systemApi.parserOptions).mockRejectedValue(
        new Error("API error"),
      );

      render(ProfileForm, {
        props: {
          nextPort: 10240,
          onSubmit: mockOnSubmit,
          onCancel: mockOnCancel,
        },
      });

      await user.click(screen.getByText("Show Advanced Options"));

      // Should still render form, just with default parser option
      await waitFor(() => {
        expect(screen.getByLabelText(/Tool Call Parser/)).toBeInTheDocument();
      });

      consoleSpy.mockRestore();
    });
  });

  describe("model type selection", () => {
    it("defaults to lm model type", () => {
      render(ProfileForm, {
        props: {
          nextPort: 10240,
          onSubmit: mockOnSubmit,
          onCancel: mockOnCancel,
        },
      });

      const select = screen.getByLabelText(/Model Type/) as HTMLSelectElement;
      expect(select.value).toBe("lm");
    });

    it("allows selecting multimodal type", async () => {
      const user = userEvent.setup();
      render(ProfileForm, {
        props: {
          nextPort: 10240,
          onSubmit: mockOnSubmit,
          onCancel: mockOnCancel,
        },
      });

      const select = screen.getByLabelText(/Model Type/) as HTMLSelectElement;
      await user.selectOptions(select, "multimodal");

      expect(select.value).toBe("multimodal");
    });

    it("maps unsupported model types to lm", () => {
      const profile = createMockProfile({
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        model_type: "whisper" as any, // Unsupported type
      });

      render(ProfileForm, {
        props: {
          profile,
          nextPort: 10240,
          onSubmit: mockOnSubmit,
          onCancel: mockOnCancel,
        },
      });

      const select = screen.getByLabelText(/Model Type/) as HTMLSelectElement;
      expect(select.value).toBe("lm");
    });
  });
});
