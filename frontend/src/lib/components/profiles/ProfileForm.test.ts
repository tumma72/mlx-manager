import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/svelte";
import userEvent from "@testing-library/user-event";
import ProfileForm from "./ProfileForm.svelte";
import type {
  ServerProfile,
  ServerProfileCreate,
  ServerProfileUpdate,
} from "$api";

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
    context_length: null,
    auto_start: false,
    system_prompt: null,
    temperature: 0.7,
    max_tokens: 4096,
    top_p: 1.0,
    launchd_installed: false,
    created_at: "2024-01-01T00:00:00Z",
    updated_at: "2024-01-01T00:00:00Z",
    ...overrides,
  };
}

describe("ProfileForm", () => {
  let mockOnSubmit: (
    data: ServerProfileCreate | ServerProfileUpdate,
  ) => Promise<void>;
  let mockOnCancel: () => void;

  beforeEach(() => {
    mockOnSubmit = vi.fn().mockResolvedValue(undefined);
    mockOnCancel = vi.fn();
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

    it("renders all required fields", () => {
      render(ProfileForm, {
        props: {
          onSubmit: mockOnSubmit,
          onCancel: mockOnCancel,
        },
      });

      expect(screen.getByLabelText(/Name/)).toBeInTheDocument();
      expect(screen.getByLabelText(/Model Path/)).toBeInTheDocument();
    });

    it("renders optional fields", () => {
      render(ProfileForm, {
        props: {
          onSubmit: mockOnSubmit,
          onCancel: mockOnCancel,
        },
      });

      expect(screen.getByLabelText(/Description/)).toBeInTheDocument();
      expect(screen.getByLabelText(/System Prompt/)).toBeInTheDocument();
      expect(screen.getByLabelText(/Model Type/)).toBeInTheDocument();
    });

    it("renders generation settings", () => {
      render(ProfileForm, {
        props: {
          onSubmit: mockOnSubmit,
          onCancel: mockOnCancel,
        },
      });

      expect(screen.getByText("Generation Settings")).toBeInTheDocument();
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

    it("initializes model path with initialModelPath prop", () => {
      render(ProfileForm, {
        props: {
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
      });

      render(ProfileForm, {
        props: {
          profile,
          onSubmit: mockOnSubmit,
          onCancel: mockOnCancel,
        },
      });

      expect((screen.getByLabelText(/Name/) as HTMLInputElement).value).toBe(
        "Existing Profile",
      );
      expect(
        (screen.getByLabelText(/Description/) as HTMLTextAreaElement).value,
      ).toBe("Test description");
      expect(
        (screen.getByLabelText(/Model Path/) as HTMLInputElement).value,
      ).toBe("mlx-community/existing-model");
    });

    it("populates system prompt", () => {
      const profile = createMockProfile({
        system_prompt: "You are a helpful assistant",
      });

      render(ProfileForm, {
        props: {
          profile,
          onSubmit: mockOnSubmit,
          onCancel: mockOnCancel,
        },
      });

      expect(
        (screen.getByLabelText(/System Prompt/) as HTMLTextAreaElement).value,
      ).toBe("You are a helpful assistant");
    });

    it("populates generation settings", () => {
      const profile = createMockProfile({
        temperature: 0.5,
        max_tokens: 2048,
        top_p: 0.9,
      });

      render(ProfileForm, {
        props: {
          profile,
          onSubmit: mockOnSubmit,
          onCancel: mockOnCancel,
        },
      });

      expect(
        (screen.getByLabelText(/Max Tokens/) as HTMLInputElement).value,
      ).toBe("2048");
    });
  });

  describe("advanced options", () => {
    it("hides advanced options by default", () => {
      render(ProfileForm, {
        props: {
          onSubmit: mockOnSubmit,
          onCancel: mockOnCancel,
        },
      });

      expect(
        screen.queryByLabelText(/Auto-load on startup/),
      ).not.toBeInTheDocument();
    });

    it("shows advanced options when toggle clicked", async () => {
      const user = userEvent.setup();
      render(ProfileForm, {
        props: {
          onSubmit: mockOnSubmit,
          onCancel: mockOnCancel,
        },
      });

      const toggleButton = screen.getByText("Show Advanced Options");
      await user.click(toggleButton);

      expect(
        screen.getByLabelText(/Auto-load on startup/),
      ).toBeInTheDocument();
    });

    it("hides advanced options when toggle clicked again", async () => {
      const user = userEvent.setup();
      render(ProfileForm, {
        props: {
          onSubmit: mockOnSubmit,
          onCancel: mockOnCancel,
        },
      });

      const toggleButton = screen.getByText("Show Advanced Options");
      await user.click(toggleButton);
      await user.click(screen.getByText("Hide Advanced Options"));

      expect(
        screen.queryByLabelText(/Auto-load on startup/),
      ).not.toBeInTheDocument();
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
          auto_start: false,
          system_prompt: undefined,
          temperature: 0.7,
          max_tokens: 4096,
          top_p: 1.0,
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

      await user.click(screen.getByRole("button", { name: "Create Profile" }));

      await waitFor(() => {
        expect(mockOnSubmit).toHaveBeenCalledWith(
          expect.objectContaining({
            name: "Full Profile",
            description: "A description",
            system_prompt: "Be helpful",
            model_path: "mlx-community/model",
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

      const prompt = "You are helpful";
      await user.type(screen.getByLabelText(/System Prompt/), prompt);

      expect(screen.getByText(/15 chars/)).toBeInTheDocument();
    });

    it("shows warning for long system prompts", async () => {
      const user = userEvent.setup();
      render(ProfileForm, {
        props: {
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

  describe("model type selection", () => {
    it("defaults to lm model type", () => {
      render(ProfileForm, {
        props: {
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
          onSubmit: mockOnSubmit,
          onCancel: mockOnCancel,
        },
      });

      const select = screen.getByLabelText(/Model Type/) as HTMLSelectElement;
      expect(select.value).toBe("lm");
    });
  });
});
