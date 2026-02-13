import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/svelte";
import userEvent from "@testing-library/user-event";
import ProfileForm from "./ProfileForm.svelte";
import type {
  ExecutionProfile,
  ExecutionProfileCreate,
  ExecutionProfileUpdate,
  DownloadedModel,
} from "$api";

// Mock the models API
vi.mock("$api", () => ({
  models: {
    listDownloaded: vi.fn(),
  },
}));

// Helper to create mock profile
function createMockProfile(
  overrides: Partial<ExecutionProfile> = {},
): ExecutionProfile {
  return {
    id: 1,
    name: "Test Profile",
    description: null,
    model_id: 1,
    model_repo_id: "mlx-community/test-model",
    model_type: "lm",
    profile_type: "inference",
    auto_start: false,
    launchd_installed: false,
    inference: { temperature: 0.7, max_tokens: 4096, top_p: 1.0 },
    context: { context_length: null, system_prompt: null, enable_tool_injection: false },
    audio: null,
    created_at: "2024-01-01T00:00:00Z",
    updated_at: "2024-01-01T00:00:00Z",
    ...overrides,
  };
}

describe("ProfileForm", () => {
  let mockOnSubmit: (
    data: ExecutionProfileCreate | ExecutionProfileUpdate,
  ) => Promise<void>;
  let mockOnCancel: () => void;

  const mockModels: DownloadedModel[] = [
    {
      id: 1,
      repo_id: "mlx-community/test-model",
      model_type: "text-gen",
      local_path: "/path/to/model",
      size_bytes: 1000000,
      size_gb: 1,
      downloaded_at: "2024-01-01T00:00:00Z",
      last_used_at: null,
      probed_at: null,
      probe_version: null,
      supports_native_tools: null,
      supports_thinking: null,
      tool_format: null,
      practical_max_tokens: null,
      model_family: null,
      tool_parser_id: null,
      thinking_parser_id: null,
      supports_multi_image: null,
      supports_video: null,
      embedding_dimensions: null,
      max_sequence_length: null,
      is_normalized: null,
      supports_tts: null,
      supports_stt: null,
      capabilities: null,
    },
    {
      id: 2,
      repo_id: "mlx-community/audio-model",
      model_type: "audio",
      local_path: "/path/to/audio",
      size_bytes: 2000000,
      size_gb: 2,
      downloaded_at: "2024-01-01T00:00:00Z",
      last_used_at: null,
      probed_at: null,
      probe_version: null,
      supports_native_tools: null,
      supports_thinking: null,
      tool_format: null,
      practical_max_tokens: null,
      model_family: null,
      tool_parser_id: null,
      thinking_parser_id: null,
      supports_multi_image: null,
      supports_video: null,
      embedding_dimensions: null,
      max_sequence_length: null,
      is_normalized: null,
      supports_tts: true,
      supports_stt: true,
      capabilities: null,
    },
  ];

  beforeEach(async () => {
    mockOnSubmit = vi.fn().mockResolvedValue(undefined);
    mockOnCancel = vi.fn();

    // Mock the API call
    const { models } = await import("$api");
    vi.mocked(models.listDownloaded).mockResolvedValue(mockModels);
  });

  describe("create mode rendering", () => {
    it("renders create profile title", async () => {
      render(ProfileForm, {
        props: {
          onSubmit: mockOnSubmit,
          onCancel: mockOnCancel,
        },
      });

      await waitFor(() => {
        const titles = screen.getAllByText("Create Profile");
        expect(titles.length).toBeGreaterThan(0);
      });
    });

    it("renders all required fields", async () => {
      render(ProfileForm, {
        props: {
          onSubmit: mockOnSubmit,
          onCancel: mockOnCancel,
        },
      });

      // Wait for models to load
      await waitFor(() => {
        expect(screen.getByLabelText(/Name/)).toBeInTheDocument();
      });

      expect(screen.getByLabelText(/Model/)).toBeInTheDocument();
    });

    it("renders optional fields", async () => {
      render(ProfileForm, {
        props: {
          onSubmit: mockOnSubmit,
          onCancel: mockOnCancel,
        },
      });

      await waitFor(() => {
        expect(screen.getByLabelText(/Description/)).toBeInTheDocument();
      });
    });

    it("renders generation settings after selecting text-gen model", async () => {
      const user = userEvent.setup();
      render(ProfileForm, {
        props: {
          onSubmit: mockOnSubmit,
          onCancel: mockOnCancel,
        },
      });

      // Wait for models to load
      await waitFor(() => {
        expect(screen.getByLabelText(/Model/)).toBeInTheDocument();
      });

      // Select the text-gen model
      const modelSelect = screen.getByLabelText(/Model/) as HTMLSelectElement;
      await user.selectOptions(modelSelect, "1");

      // Wait for conditional sections to appear
      await waitFor(() => {
        expect(screen.getByText("Generation Settings")).toBeInTheDocument();
      });

      expect(screen.getByLabelText(/Temperature/)).toBeInTheDocument();
      expect(screen.getByLabelText(/Max Tokens/)).toBeInTheDocument();
      expect(screen.getByLabelText(/Top P/)).toBeInTheDocument();
    });

    it("renders submit button with Create Profile text", () => {
      render(ProfileForm, {
        props: {
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
          onSubmit: mockOnSubmit,
          onCancel: mockOnCancel,
        },
      });

      expect(
        screen.getByRole("button", { name: "Cancel" }),
      ).toBeInTheDocument();
    });

    it("initializes model selection with initialModelId prop", async () => {
      render(ProfileForm, {
        props: {
          initialModelId: 1,
          onSubmit: mockOnSubmit,
          onCancel: mockOnCancel,
        },
      });

      // Wait for models to load
      await waitFor(() => {
        const modelSelect = screen.getByLabelText(/Model/) as HTMLSelectElement;
        expect(modelSelect.value).toBe("1");
      });

      // Check that the model info is displayed
      await waitFor(() => {
        expect(screen.getByText(/Type: text-gen/)).toBeInTheDocument();
      });
    });
  });

  describe("edit mode rendering", () => {
    it("renders edit profile title", async () => {
      render(ProfileForm, {
        props: {
          profile: createMockProfile(),
          onSubmit: mockOnSubmit,
          onCancel: mockOnCancel,
        },
      });

      await waitFor(() => {
        expect(screen.getByText("Edit Profile")).toBeInTheDocument();
      });
    });

    it("renders submit button with Update Profile text", () => {
      render(ProfileForm, {
        props: {
          profile: createMockProfile(),
          onSubmit: mockOnSubmit,
          onCancel: mockOnCancel,
        },
      });

      expect(
        screen.getByRole("button", { name: "Update Profile" }),
      ).toBeInTheDocument();
    });

    it("populates form with profile data", async () => {
      const profile = createMockProfile({
        name: "Existing Profile",
        description: "Test description",
        model_id: 1,
      });

      render(ProfileForm, {
        props: {
          profile,
          onSubmit: mockOnSubmit,
          onCancel: mockOnCancel,
        },
      });

      // Wait for models to load and form to populate
      await waitFor(() => {
        expect((screen.getByLabelText(/Name/) as HTMLInputElement).value).toBe(
          "Existing Profile",
        );
      });

      expect(
        (screen.getByLabelText(/Description/) as HTMLTextAreaElement).value,
      ).toBe("Test description");

      const modelSelect = screen.getByLabelText(/Model/) as HTMLSelectElement;
      expect(modelSelect.value).toBe("1");
    });

    it("populates system prompt", async () => {
      const profile = createMockProfile({
        context: { context_length: null, system_prompt: "You are a helpful assistant", enable_tool_injection: false },
        model_id: 1,
      });

      render(ProfileForm, {
        props: {
          profile,
          onSubmit: mockOnSubmit,
          onCancel: mockOnCancel,
        },
      });

      // Wait for models to load and system prompt to appear
      await waitFor(() => {
        const systemPromptTextarea = screen.getByLabelText(
          /System Prompt/,
        ) as HTMLTextAreaElement;
        expect(systemPromptTextarea.value).toBe("You are a helpful assistant");
      });
    });

    it("populates generation settings", async () => {
      const profile = createMockProfile({
        inference: { temperature: 0.5, max_tokens: 2048, top_p: 0.9 },
        model_id: 1,
      });

      render(ProfileForm, {
        props: {
          profile,
          onSubmit: mockOnSubmit,
          onCancel: mockOnCancel,
        },
      });

      // Wait for models to load and generation settings to appear
      await waitFor(() => {
        const maxTokensInput = screen.getByLabelText(
          /Max Tokens/,
        ) as HTMLInputElement;
        expect(maxTokensInput.value).toBe("2048");
      });
    });
  });

  describe("advanced options", () => {
    it("hides advanced options by default", async () => {
      render(ProfileForm, {
        props: {
          onSubmit: mockOnSubmit,
          onCancel: mockOnCancel,
        },
      });

      await waitFor(() => {
        expect(
          screen.queryByLabelText(/Auto-load on startup/),
        ).not.toBeInTheDocument();
      });
    });

    it("shows advanced options when toggle clicked", async () => {
      const user = userEvent.setup();
      render(ProfileForm, {
        props: {
          onSubmit: mockOnSubmit,
          onCancel: mockOnCancel,
        },
      });

      await waitFor(() => {
        expect(screen.getByText("Show Advanced Options")).toBeInTheDocument();
      });

      const toggleButton = screen.getByText("Show Advanced Options");
      await user.click(toggleButton);

      await waitFor(() => {
        expect(
          screen.getByLabelText(/Auto-load on startup/),
        ).toBeInTheDocument();
      });
    });

    it("hides advanced options when toggle clicked again", async () => {
      const user = userEvent.setup();
      render(ProfileForm, {
        props: {
          onSubmit: mockOnSubmit,
          onCancel: mockOnCancel,
        },
      });

      await waitFor(() => {
        expect(screen.getByText("Show Advanced Options")).toBeInTheDocument();
      });

      const toggleButton = screen.getByText("Show Advanced Options");
      await user.click(toggleButton);

      await waitFor(() => {
        expect(screen.getByText("Hide Advanced Options")).toBeInTheDocument();
      });

      await user.click(screen.getByText("Hide Advanced Options"));

      await waitFor(() => {
        expect(
          screen.queryByLabelText(/Auto-load on startup/),
        ).not.toBeInTheDocument();
      });
    });
  });

  describe("form submission", () => {
    it("submits form with required fields only", async () => {
      const user = userEvent.setup();
      render(ProfileForm, {
        props: {
          onSubmit: mockOnSubmit,
          onCancel: mockOnCancel,
        },
      });

      // Wait for models to load
      await waitFor(() => {
        expect(screen.getByLabelText(/Model/)).toBeInTheDocument();
      });

      await user.type(screen.getByLabelText(/Name/), "New Profile");

      // Select the model
      const modelSelect = screen.getByLabelText(/Model/) as HTMLSelectElement;
      await user.selectOptions(modelSelect, "1");

      // Wait for generation settings to appear (they have defaults)
      await waitFor(() => {
        expect(screen.getByText("Generation Settings")).toBeInTheDocument();
      });

      await user.click(screen.getByRole("button", { name: "Create Profile" }));

      await waitFor(() => {
        expect(mockOnSubmit).toHaveBeenCalledWith({
          name: "New Profile",
          description: undefined,
          model_id: 1,
          auto_start: false,
          inference: {
            temperature: null,
            max_tokens: null,
            top_p: null,
          },
          context: {
            context_length: null,
            system_prompt: null,
            enable_tool_injection: false,
          },
          audio: undefined,
        });
      });
    });

    it("submits form with all fields filled", async () => {
      const user = userEvent.setup();
      render(ProfileForm, {
        props: {
          onSubmit: mockOnSubmit,
          onCancel: mockOnCancel,
        },
      });

      // Wait for models to load
      await waitFor(() => {
        expect(screen.getByLabelText(/Model/)).toBeInTheDocument();
      });

      await user.type(screen.getByLabelText(/Name/), "Full Profile");
      await user.type(screen.getByLabelText(/Description/), "A description");

      // Select the model
      const modelSelect = screen.getByLabelText(/Model/) as HTMLSelectElement;
      await user.selectOptions(modelSelect, "1");

      // Wait for system prompt to appear
      await waitFor(() => {
        expect(screen.getByLabelText(/System Prompt/)).toBeInTheDocument();
      });

      await user.type(screen.getByLabelText(/System Prompt/), "Be helpful");

      await user.click(screen.getByRole("button", { name: "Create Profile" }));

      await waitFor(() => {
        expect(mockOnSubmit).toHaveBeenCalledWith(
          expect.objectContaining({
            name: "Full Profile",
            description: "A description",
            model_id: 1,
            context: expect.objectContaining({
              system_prompt: "Be helpful",
            }),
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
          onSubmit: slowSubmit,
          onCancel: mockOnCancel,
        },
      });

      // Wait for models to load
      await waitFor(() => {
        expect(screen.getByLabelText(/Model/)).toBeInTheDocument();
      });

      await user.type(screen.getByLabelText(/Name/), "Test");

      const modelSelect = screen.getByLabelText(/Model/) as HTMLSelectElement;
      await user.selectOptions(modelSelect, "1");

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
          onSubmit: slowSubmit,
          onCancel: mockOnCancel,
        },
      });

      // Wait for models to load
      await waitFor(() => {
        expect(screen.getByLabelText(/Model/)).toBeInTheDocument();
      });

      await user.type(screen.getByLabelText(/Name/), "Test");

      const modelSelect = screen.getByLabelText(/Model/) as HTMLSelectElement;
      await user.selectOptions(modelSelect, "1");

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
          onSubmit: failingSubmit,
          onCancel: mockOnCancel,
        },
      });

      // Wait for models to load
      await waitFor(() => {
        expect(screen.getByLabelText(/Model/)).toBeInTheDocument();
      });

      await user.type(screen.getByLabelText(/Name/), "Test");

      const modelSelect = screen.getByLabelText(/Model/) as HTMLSelectElement;
      await user.selectOptions(modelSelect, "1");

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
          onSubmit: failingSubmit,
          onCancel: mockOnCancel,
        },
      });

      // Wait for models to load
      await waitFor(() => {
        expect(screen.getByLabelText(/Model/)).toBeInTheDocument();
      });

      await user.type(screen.getByLabelText(/Name/), "Test");

      const modelSelect = screen.getByLabelText(/Model/) as HTMLSelectElement;
      await user.selectOptions(modelSelect, "1");

      await user.click(screen.getByRole("button", { name: "Create Profile" }));

      await waitFor(() => {
        expect(screen.getByText("Failed to save profile")).toBeInTheDocument();
      });
    });

    it("disables submit button when no model selected", async () => {
      render(ProfileForm, {
        props: {
          onSubmit: mockOnSubmit,
          onCancel: mockOnCancel,
        },
      });

      // Wait for models to load
      await waitFor(() => {
        expect(screen.getByLabelText(/Model/)).toBeInTheDocument();
      });

      // Submit button should be disabled without model selection
      const submitButton = screen.getByRole("button", {
        name: "Create Profile",
      });
      expect(submitButton).toBeDisabled();
    });
  });

  describe("cancel action", () => {
    it("calls onCancel when cancel button clicked", async () => {
      const user = userEvent.setup();
      render(ProfileForm, {
        props: {
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
          onSubmit: mockOnSubmit,
          onCancel: mockOnCancel,
        },
      });

      // Wait for models to load
      await waitFor(() => {
        expect(screen.getByLabelText(/Model/)).toBeInTheDocument();
      });

      // Select a text-gen model
      const modelSelect = screen.getByLabelText(/Model/) as HTMLSelectElement;
      await user.selectOptions(modelSelect, "1");

      // Wait for system prompt to appear
      await waitFor(() => {
        expect(screen.getByLabelText(/System Prompt/)).toBeInTheDocument();
      });

      const prompt = "You are helpful";
      await user.type(screen.getByLabelText(/System Prompt/), prompt);

      await waitFor(() => {
        expect(screen.getByText(/15 chars/)).toBeInTheDocument();
      });
    });

    it("shows warning for long system prompts", async () => {
      const user = userEvent.setup();
      render(ProfileForm, {
        props: {
          onSubmit: mockOnSubmit,
          onCancel: mockOnCancel,
        },
      });

      // Wait for models to load
      await waitFor(() => {
        expect(screen.getByLabelText(/Model/)).toBeInTheDocument();
      });

      // Select a text-gen model
      const modelSelect = screen.getByLabelText(/Model/) as HTMLSelectElement;
      await user.selectOptions(modelSelect, "1");

      // Wait for system prompt to appear
      await waitFor(() => {
        expect(screen.getByLabelText(/System Prompt/)).toBeInTheDocument();
      });

      const longPrompt = "a".repeat(2001);
      await user.type(screen.getByLabelText(/System Prompt/), longPrompt);

      await waitFor(() => {
        expect(
          screen.getByText(/long prompt may affect performance/),
        ).toBeInTheDocument();
      });
    });
  });

  describe("audio model support", () => {
    it("shows audio settings when audio model selected", async () => {
      const user = userEvent.setup();
      render(ProfileForm, {
        props: {
          onSubmit: mockOnSubmit,
          onCancel: mockOnCancel,
        },
      });

      // Wait for models to load
      await waitFor(() => {
        expect(screen.getByLabelText(/Model/)).toBeInTheDocument();
      });

      // Select the audio model
      const modelSelect = screen.getByLabelText(/Model/) as HTMLSelectElement;
      await user.selectOptions(modelSelect, "2");

      // Wait for audio settings to appear
      await waitFor(() => {
        expect(screen.getByText("Audio Settings")).toBeInTheDocument();
      });

      expect(screen.getByLabelText(/Default Voice/)).toBeInTheDocument();
      expect(screen.getByLabelText(/Speech Speed/)).toBeInTheDocument();
      expect(screen.getByLabelText(/Sample Rate/)).toBeInTheDocument();
      expect(screen.getByLabelText(/STT Language/)).toBeInTheDocument();
    });

    it("hides generation settings for audio models", async () => {
      const user = userEvent.setup();
      render(ProfileForm, {
        props: {
          onSubmit: mockOnSubmit,
          onCancel: mockOnCancel,
        },
      });

      // Wait for models to load
      await waitFor(() => {
        expect(screen.getByLabelText(/Model/)).toBeInTheDocument();
      });

      // Select the audio model
      const modelSelect = screen.getByLabelText(/Model/) as HTMLSelectElement;
      await user.selectOptions(modelSelect, "2");

      // Wait for audio settings to appear
      await waitFor(() => {
        expect(screen.getByText("Audio Settings")).toBeInTheDocument();
      });

      // Generation settings should not be present
      expect(screen.queryByText("Generation Settings")).not.toBeInTheDocument();
      expect(screen.queryByLabelText(/System Prompt/)).not.toBeInTheDocument();
    });
  });

  describe("model type conditional rendering", () => {
    it("shows system prompt for text-gen models", async () => {
      const user = userEvent.setup();
      render(ProfileForm, {
        props: {
          onSubmit: mockOnSubmit,
          onCancel: mockOnCancel,
        },
      });

      // Wait for models to load
      await waitFor(() => {
        expect(screen.getByLabelText(/Model/)).toBeInTheDocument();
      });

      // Select text-gen model
      const modelSelect = screen.getByLabelText(/Model/) as HTMLSelectElement;
      await user.selectOptions(modelSelect, "1");

      await waitFor(() => {
        expect(screen.getByLabelText(/System Prompt/)).toBeInTheDocument();
      });
    });

    it("hides system prompt before model selection", async () => {
      render(ProfileForm, {
        props: {
          onSubmit: mockOnSubmit,
          onCancel: mockOnCancel,
        },
      });

      // Wait for models to load
      await waitFor(() => {
        expect(screen.getByLabelText(/Model/)).toBeInTheDocument();
      });

      // System prompt should not be visible yet
      expect(screen.queryByLabelText(/System Prompt/)).not.toBeInTheDocument();
    });
  });
});
