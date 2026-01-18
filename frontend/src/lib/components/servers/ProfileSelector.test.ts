import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/svelte";
import userEvent from "@testing-library/user-event";
import ProfileSelector from "./ProfileSelector.svelte";
import type { ServerProfile } from "$api";

// Helper to create mock profiles
function createMockProfile(
  overrides: Partial<ServerProfile> = {}
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
    max_concurrency: 4,
    queue_timeout: 30,
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
    launchd_installed: false,
    created_at: "2024-01-01T00:00:00Z",
    updated_at: "2024-01-01T00:00:00Z",
    ...overrides,
  };
}

describe("ProfileSelector", () => {
  const mockProfiles: ServerProfile[] = [
    createMockProfile({
      id: 1,
      name: "Profile One",
      model_path: "mlx-community/llama-2",
    }),
    createMockProfile({
      id: 2,
      name: "Profile Two",
      model_path: "mlx-community/mistral",
    }),
    createMockProfile({
      id: 3,
      name: "Another Profile",
      model_path: "mlx-community/phi-2",
    }),
  ];

  let mockOnStart: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    mockOnStart = vi.fn().mockResolvedValue(undefined);
  });

  describe("rendering", () => {
    it("renders input with placeholder", () => {
      render(ProfileSelector, {
        props: { profiles: mockProfiles, onStart: mockOnStart },
      });

      expect(
        screen.getByPlaceholderText("Select profile to start...")
      ).toBeInTheDocument();
    });

    it("renders Start button", () => {
      render(ProfileSelector, {
        props: { profiles: mockProfiles, onStart: mockOnStart },
      });

      expect(screen.getByRole("button", { name: /start/i })).toBeInTheDocument();
    });

    it("renders with disabled state", () => {
      render(ProfileSelector, {
        props: { profiles: mockProfiles, onStart: mockOnStart, disabled: true },
      });

      const input = screen.getByPlaceholderText("Select profile to start...");
      expect(input).toBeDisabled();
    });

    it("renders combobox element", () => {
      render(ProfileSelector, {
        props: { profiles: mockProfiles, onStart: mockOnStart },
      });

      expect(screen.getByRole("combobox")).toBeInTheDocument();
    });

    it("renders trigger button for dropdown", () => {
      const { container } = render(ProfileSelector, {
        props: { profiles: mockProfiles, onStart: mockOnStart },
      });

      // The dropdown trigger button (with chevron icon)
      const triggerButton = container.querySelector("[data-combobox-trigger]");
      expect(triggerButton).toBeInTheDocument();
    });
  });

  describe("Start button state", () => {
    it("Start button is disabled when no profile is selected", () => {
      render(ProfileSelector, {
        props: { profiles: mockProfiles, onStart: mockOnStart },
      });

      const startButton = screen.getByRole("button", { name: /start/i });
      expect(startButton).toBeDisabled();
    });

    it("Start button is disabled when component is disabled", () => {
      render(ProfileSelector, {
        props: { profiles: mockProfiles, onStart: mockOnStart, disabled: true },
      });

      const startButton = screen.getByRole("button", { name: /start/i });
      expect(startButton).toBeDisabled();
    });

    it("Start button disabled state prevents click", async () => {
      const user = userEvent.setup();
      render(ProfileSelector, {
        props: { profiles: mockProfiles, onStart: mockOnStart },
      });

      const startButton = screen.getByRole("button", { name: /start/i });
      await user.click(startButton);

      // Should not call onStart when button is disabled
      expect(mockOnStart).not.toHaveBeenCalled();
    });
  });

  describe("combobox state", () => {
    it("combobox has closed state initially", () => {
      const { container } = render(ProfileSelector, {
        props: { profiles: mockProfiles, onStart: mockOnStart },
      });

      const input = container.querySelector("[data-combobox-input]");
      expect(input?.getAttribute("data-state")).toBe("closed");
    });

    it("combobox has aria-expanded false initially", () => {
      render(ProfileSelector, {
        props: { profiles: mockProfiles, onStart: mockOnStart },
      });

      const combobox = screen.getByRole("combobox");
      expect(combobox.getAttribute("aria-expanded")).toBe("false");
    });

    it("combobox has aria-autocomplete list attribute", () => {
      render(ProfileSelector, {
        props: { profiles: mockProfiles, onStart: mockOnStart },
      });

      const combobox = screen.getByRole("combobox");
      expect(combobox.getAttribute("aria-autocomplete")).toBe("list");
    });
  });

  describe("disabled state", () => {
    it("disables input when component is disabled", () => {
      render(ProfileSelector, {
        props: { profiles: mockProfiles, onStart: mockOnStart, disabled: true },
      });

      const input = screen.getByPlaceholderText("Select profile to start...");
      expect(input).toBeDisabled();
    });

    it("prevents interaction when disabled", async () => {
      const user = userEvent.setup();
      render(ProfileSelector, {
        props: { profiles: mockProfiles, onStart: mockOnStart, disabled: true },
      });

      // Try to type in input - should be ignored
      const input = screen.getByPlaceholderText("Select profile to start...");
      await user.type(input, "test");

      // Input value should not change
      expect(input).toHaveValue("");
    });
  });

  describe("play icon", () => {
    it("renders play icon on start button", () => {
      const { container } = render(ProfileSelector, {
        props: { profiles: mockProfiles, onStart: mockOnStart },
      });

      // Check for the lucide play icon
      const playIcon = container.querySelector(".lucide-play");
      expect(playIcon).toBeInTheDocument();
    });
  });
});
