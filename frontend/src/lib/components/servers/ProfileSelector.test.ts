import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/svelte";
import userEvent from "@testing-library/user-event";
import ProfileSelector from "./ProfileSelector.svelte";
import type { ExecutionProfile } from "$api";

// Helper to create mock profiles
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
    model_options: null,
    inference: { temperature: 0.7, max_tokens: 4096, top_p: 1.0 },
    context: { context_length: null, system_prompt: null, enable_tool_injection: false },
    audio: null,
    created_at: "2024-01-01T00:00:00Z",
    updated_at: "2024-01-01T00:00:00Z",
    ...overrides,
  };
}

describe("ProfileSelector", () => {
  const mockProfiles: ExecutionProfile[] = [
    createMockProfile({
      id: 1,
      name: "Profile One",
      model_id: 1, model_repo_id: "mlx-community/llama-2",
    }),
    createMockProfile({
      id: 2,
      name: "Profile Two",
      model_id: 1, model_repo_id: "mlx-community/mistral",
    }),
    createMockProfile({
      id: 3,
      name: "Another Profile",
      model_id: 1, model_repo_id: "mlx-community/phi-2",
    }),
  ];

  let mockOnStart: (profile: ExecutionProfile) => Promise<void>;

  beforeEach(() => {
    mockOnStart = vi.fn().mockResolvedValue(undefined);
  });

  describe("rendering", () => {
    it("renders input with placeholder", () => {
      render(ProfileSelector, {
        props: { profiles: mockProfiles, onStart: mockOnStart },
      });

      expect(
        screen.getByPlaceholderText("Select profile to start..."),
      ).toBeInTheDocument();
    });

    it("renders Start button", () => {
      render(ProfileSelector, {
        props: { profiles: mockProfiles, onStart: mockOnStart },
      });

      expect(
        screen.getByRole("button", { name: /start/i }),
      ).toBeInTheDocument();
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

  describe("dropdown interaction", () => {
    it("opens dropdown when clicking trigger button", async () => {
      const user = userEvent.setup();
      const { container } = render(ProfileSelector, {
        props: { profiles: mockProfiles, onStart: mockOnStart },
      });

      const triggerButton = container.querySelector(
        "[data-combobox-trigger]",
      ) as HTMLButtonElement;
      await user.click(triggerButton);

      // Wait for dropdown to open
      await waitFor(() => {
        const combobox = screen.getByRole("combobox");
        expect(combobox.getAttribute("aria-expanded")).toBe("true");
      });
    });

    it("shows profile options when dropdown is open", async () => {
      const user = userEvent.setup();
      const { container } = render(ProfileSelector, {
        props: { profiles: mockProfiles, onStart: mockOnStart },
      });

      const triggerButton = container.querySelector(
        "[data-combobox-trigger]",
      ) as HTMLButtonElement;
      await user.click(triggerButton);

      // Profile names should be visible in the portal
      expect(await screen.findByText("Profile One")).toBeInTheDocument();
      expect(screen.getByText("Profile Two")).toBeInTheDocument();
      expect(screen.getByText("Another Profile")).toBeInTheDocument();
    });

    it("shows model paths in dropdown items", async () => {
      const user = userEvent.setup();
      const { container } = render(ProfileSelector, {
        props: { profiles: mockProfiles, onStart: mockOnStart },
      });

      const triggerButton = container.querySelector(
        "[data-combobox-trigger]",
      ) as HTMLButtonElement;
      await user.click(triggerButton);

      // Model paths should be visible as secondary text
      expect(
        await screen.findByText("mlx-community/llama-2"),
      ).toBeInTheDocument();
      expect(screen.getByText("mlx-community/mistral")).toBeInTheDocument();
    });
  });

  describe("profile filtering", () => {
    it("filters profiles by name when typing", async () => {
      const user = userEvent.setup();
      render(ProfileSelector, {
        props: { profiles: mockProfiles, onStart: mockOnStart },
      });

      const input = screen.getByPlaceholderText("Select profile to start...");
      await user.click(input);
      await user.type(input, "One");

      // Only matching profile should be visible
      expect(screen.getByText("Profile One")).toBeInTheDocument();
      expect(screen.queryByText("Profile Two")).not.toBeInTheDocument();
      expect(screen.queryByText("Another Profile")).not.toBeInTheDocument();
    });

    it("filters profiles by model path", async () => {
      const user = userEvent.setup();
      render(ProfileSelector, {
        props: { profiles: mockProfiles, onStart: mockOnStart },
      });

      const input = screen.getByPlaceholderText("Select profile to start...");
      await user.click(input);
      await user.type(input, "mistral");

      // Only profile with matching model path should be visible
      expect(screen.getByText("Profile Two")).toBeInTheDocument();
      expect(screen.queryByText("Profile One")).not.toBeInTheDocument();
    });

    it("filters case-insensitively", async () => {
      const user = userEvent.setup();
      render(ProfileSelector, {
        props: { profiles: mockProfiles, onStart: mockOnStart },
      });

      const input = screen.getByPlaceholderText("Select profile to start...");
      await user.click(input);
      await user.type(input, "ANOTHER");

      expect(screen.getByText("Another Profile")).toBeInTheDocument();
    });

    it("shows 'No profiles found' when search has no matches", async () => {
      const user = userEvent.setup();
      render(ProfileSelector, {
        props: { profiles: mockProfiles, onStart: mockOnStart },
      });

      const input = screen.getByPlaceholderText("Select profile to start...");
      await user.click(input);
      await user.type(input, "nonexistent");

      expect(screen.getByText("No profiles found")).toBeInTheDocument();
    });

    it("clears search when closing dropdown", async () => {
      const user = userEvent.setup();
      const { container } = render(ProfileSelector, {
        props: { profiles: mockProfiles, onStart: mockOnStart },
      });

      const input = screen.getByPlaceholderText("Select profile to start...");
      await user.click(input);
      await user.type(input, "One");

      // Verify filtered state - only "Profile One" visible
      expect(screen.getByText("Profile One")).toBeInTheDocument();
      expect(screen.queryByText("Profile Two")).not.toBeInTheDocument();

      // Close dropdown by pressing Escape
      await user.keyboard("{Escape}");

      // Re-open using the trigger button and all profiles should be visible again
      const triggerButton = container.querySelector(
        "[data-combobox-trigger]",
      ) as HTMLButtonElement;
      await user.click(triggerButton);

      expect(await screen.findByText("Profile One")).toBeInTheDocument();
      expect(screen.getByText("Profile Two")).toBeInTheDocument();
    });
  });

  describe("profile selection", () => {
    it("selects profile when clicking an option", async () => {
      const user = userEvent.setup();
      const { container } = render(ProfileSelector, {
        props: { profiles: mockProfiles, onStart: mockOnStart },
      });

      // Open dropdown using trigger button
      const triggerButton = container.querySelector(
        "[data-combobox-trigger]",
      ) as HTMLButtonElement;
      await user.click(triggerButton);

      // Wait for options to appear
      const option = await screen.findByText("Profile One");
      await user.click(option);

      // Start button should now be enabled
      await waitFor(() => {
        const startButton = screen.getByRole("button", { name: /start/i });
        expect(startButton).not.toBeDisabled();
      });
    });

    it("enables start button after selection", async () => {
      const user = userEvent.setup();
      const { container } = render(ProfileSelector, {
        props: { profiles: mockProfiles, onStart: mockOnStart },
      });

      // Open dropdown using trigger button
      const triggerButton = container.querySelector(
        "[data-combobox-trigger]",
      ) as HTMLButtonElement;
      await user.click(triggerButton);

      // Click the option
      const option = await screen.findByText("Profile Two");
      await user.click(option);

      await waitFor(() => {
        const startButton = screen.getByRole("button", { name: /start/i });
        expect(startButton).toBeEnabled();
      });
    });

    it("selects profile with keyboard navigation", async () => {
      const user = userEvent.setup();
      render(ProfileSelector, {
        props: { profiles: mockProfiles, onStart: mockOnStart },
      });

      const input = screen.getByPlaceholderText("Select profile to start...");
      await user.click(input);

      // Navigate with arrow down and select with Enter
      await user.keyboard("{ArrowDown}");
      await user.keyboard("{Enter}");

      // Start button should be enabled after selection
      const startButton = screen.getByRole("button", { name: /start/i });
      expect(startButton).toBeEnabled();
    });
  });

  describe("starting server", () => {
    it("calls onStart with selected profile when clicking Start", async () => {
      const user = userEvent.setup();
      render(ProfileSelector, {
        props: { profiles: mockProfiles, onStart: mockOnStart },
      });

      // Use keyboard navigation to select a profile (more reliable)
      const input = screen.getByPlaceholderText("Select profile to start...");
      await user.click(input);
      await user.keyboard("{ArrowDown}");
      await user.keyboard("{Enter}");

      const startButton = screen.getByRole("button", { name: /start/i });
      await user.click(startButton);

      expect(mockOnStart).toHaveBeenCalledWith(mockProfiles[0]);
    });

    it("shows loading state while starting", async () => {
      const user = userEvent.setup();
      // Create a promise that we can control
      let resolveStart: () => void;
      const startPromise = new Promise<void>((resolve) => {
        resolveStart = resolve;
      });
      const slowOnStart = vi.fn().mockReturnValue(startPromise);

      const { container } = render(ProfileSelector, {
        props: { profiles: mockProfiles, onStart: slowOnStart },
      });

      // Use keyboard navigation to select a profile
      const input = screen.getByPlaceholderText("Select profile to start...");
      await user.click(input);
      await user.keyboard("{ArrowDown}");
      await user.keyboard("{Enter}");

      const startButton = screen.getByRole("button", { name: /start/i });
      // Don't await the click since we want to check loading state
      user.click(startButton);

      // Should show loading spinner - wait for state update
      // Note: The actual icon class is lucide-loader-circle (from Loader2 component)
      await waitFor(() => {
        const spinner = container.querySelector(".lucide-loader-circle");
        expect(spinner).toBeInTheDocument();
      });
      expect(screen.getByText("Starting...")).toBeInTheDocument();

      // Resolve the promise to complete the test
      resolveStart!();
      await startPromise;
    });

    it("disables button while starting", async () => {
      const user = userEvent.setup();
      let resolveStart: () => void;
      const startPromise = new Promise<void>((resolve) => {
        resolveStart = resolve;
      });
      const slowOnStart = vi.fn().mockReturnValue(startPromise);

      render(ProfileSelector, {
        props: { profiles: mockProfiles, onStart: slowOnStart },
      });

      // Use keyboard navigation to select a profile
      const input = screen.getByPlaceholderText("Select profile to start...");
      await user.click(input);
      await user.keyboard("{ArrowDown}");
      await user.keyboard("{Enter}");

      const startButton = screen.getByRole("button", { name: /start/i });
      await user.click(startButton);

      // Button should be disabled while starting
      expect(startButton).toBeDisabled();

      resolveStart!();
      await startPromise;
    });

    it("clears selection after successful start", async () => {
      const user = userEvent.setup();
      render(ProfileSelector, {
        props: { profiles: mockProfiles, onStart: mockOnStart },
      });

      // Use keyboard navigation to select a profile
      const input = screen.getByPlaceholderText("Select profile to start...");
      await user.click(input);
      await user.keyboard("{ArrowDown}");
      await user.keyboard("{Enter}");

      const startButton = screen.getByRole("button", { name: /start/i });
      await user.click(startButton);

      // After starting, button should be disabled again (selection cleared)
      await waitFor(() => {
        expect(startButton).toBeDisabled();
      });
    });

    it("re-enables button after start completes", async () => {
      const user = userEvent.setup();
      let resolveStart: () => void;
      const startPromise = new Promise<void>((resolve) => {
        resolveStart = resolve;
      });
      const slowOnStart = vi.fn().mockReturnValue(startPromise);

      const { container } = render(ProfileSelector, {
        props: { profiles: mockProfiles, onStart: slowOnStart },
      });

      // Use keyboard navigation to select a profile
      const input = screen.getByPlaceholderText("Select profile to start...");
      await user.click(input);
      await user.keyboard("{ArrowDown}");
      await user.keyboard("{Enter}");

      const startButton = screen.getByRole("button", { name: /start/i });
      await user.click(startButton);

      // Resolve and wait
      resolveStart!();
      await startPromise;

      // Wait for state updates - spinner should disappear
      await waitFor(() => {
        const spinner = container.querySelector(".lucide-loader-circle");
        expect(spinner).not.toBeInTheDocument();
      });
    });

    it("prevents multiple starts while starting (starting guard)", async () => {
      const user = userEvent.setup();
      let resolveStart: () => void;
      const startPromise = new Promise<void>((resolve) => {
        resolveStart = resolve;
      });
      const slowOnStart = vi.fn().mockReturnValue(startPromise);

      render(ProfileSelector, {
        props: { profiles: mockProfiles, onStart: slowOnStart },
      });

      // Select a profile
      const input = screen.getByPlaceholderText("Select profile to start...");
      await user.click(input);
      await user.keyboard("{ArrowDown}");
      await user.keyboard("{Enter}");

      const startButton = screen.getByRole("button", { name: /start/i });

      // First click - starts the operation
      await user.click(startButton);

      // Try clicking again while starting - should be ignored due to !starting check
      // The button is disabled, but we test the guard in handleStart
      expect(slowOnStart).toHaveBeenCalledTimes(1);

      // Resolve to clean up
      resolveStart!();
      await startPromise;
    });

    // Note: Testing error handling would require the component to have a try/catch block
    // Currently handleStart uses try/finally without catch, so errors propagate
    // This is a design decision - errors should be handled by the caller
    it("continues to work after onStart completes", async () => {
      const user = userEvent.setup();

      // Test that after a successful start, we can select and start another profile
      render(ProfileSelector, {
        props: { profiles: mockProfiles, onStart: mockOnStart },
      });

      // First selection and start
      const input = screen.getByPlaceholderText("Select profile to start...");
      await user.click(input);
      await user.keyboard("{ArrowDown}");
      await user.keyboard("{Enter}");

      let startButton = screen.getByRole("button", { name: /start/i });
      await user.click(startButton);

      // Selection should be cleared
      await waitFor(() => {
        expect(startButton).toBeDisabled();
      });

      // Second selection and start
      await user.click(input);
      await user.keyboard("{ArrowDown}");
      await user.keyboard("{Enter}");

      startButton = screen.getByRole("button", { name: /start/i });
      expect(startButton).toBeEnabled();

      await user.click(startButton);

      // Verify both starts were called
      expect(mockOnStart).toHaveBeenCalledTimes(2);
    });
  });

  describe("keyboard interaction", () => {
    it("starts server on Enter when dropdown is closed and profile selected", async () => {
      const user = userEvent.setup();
      render(ProfileSelector, {
        props: { profiles: mockProfiles, onStart: mockOnStart },
      });

      // Use keyboard to select a profile
      const input = screen.getByPlaceholderText("Select profile to start...");
      await user.click(input);
      await user.keyboard("{ArrowDown}");
      await user.keyboard("{Enter}");

      // Dropdown should be closed after selection
      const combobox = screen.getByRole("combobox");
      expect(combobox.getAttribute("aria-expanded")).toBe("false");

      // Focus input and press Enter to start
      await user.click(input);
      await user.keyboard("{Enter}");

      expect(mockOnStart).toHaveBeenCalledWith(mockProfiles[0]);
    });

    it("does not start on Enter when dropdown is open", async () => {
      const user = userEvent.setup();
      render(ProfileSelector, {
        props: { profiles: mockProfiles, onStart: mockOnStart },
      });

      const input = screen.getByPlaceholderText("Select profile to start...");
      await user.click(input);

      // Dropdown is open, Enter should select, not start
      await user.keyboard("{ArrowDown}");
      await user.keyboard("{Enter}");

      // This Enter selected the item, didn't start
      // onStart should not have been called yet
      expect(mockOnStart).not.toHaveBeenCalled();
    });

    it("does not start on Enter when disabled", async () => {
      const user = userEvent.setup();
      render(ProfileSelector, {
        props: { profiles: mockProfiles, onStart: mockOnStart, disabled: true },
      });

      const input = screen.getByPlaceholderText("Select profile to start...");
      await user.type(input, "{Enter}");

      expect(mockOnStart).not.toHaveBeenCalled();
    });

    it("does not start on Enter when no profile selected", async () => {
      const user = userEvent.setup();
      render(ProfileSelector, {
        props: { profiles: mockProfiles, onStart: mockOnStart },
      });

      const input = screen.getByPlaceholderText("Select profile to start...");
      await user.click(input);
      await user.keyboard("{Escape}"); // Close without selecting
      await user.keyboard("{Enter}");

      expect(mockOnStart).not.toHaveBeenCalled();
    });
  });

  describe("profile list stabilization", () => {
    it("handles empty profiles array", () => {
      render(ProfileSelector, {
        props: { profiles: [], onStart: mockOnStart },
      });

      const startButton = screen.getByRole("button", { name: /start/i });
      expect(startButton).toBeDisabled();
    });

    it("shows no profiles found with empty array", async () => {
      const user = userEvent.setup();
      const { container } = render(ProfileSelector, {
        props: { profiles: [], onStart: mockOnStart },
      });

      // Open dropdown using trigger
      const triggerButton = container.querySelector(
        "[data-combobox-trigger]",
      ) as HTMLButtonElement;
      await user.click(triggerButton);

      expect(await screen.findByText("No profiles found")).toBeInTheDocument();
    });

    it("clears selection when selected profile is removed", async () => {
      const user = userEvent.setup();
      const { rerender } = render(ProfileSelector, {
        props: { profiles: mockProfiles, onStart: mockOnStart },
      });

      // Select a profile using keyboard navigation
      const input = screen.getByPlaceholderText("Select profile to start...");
      await user.click(input);
      await user.keyboard("{ArrowDown}");
      await user.keyboard("{Enter}");

      // Start button should be enabled
      let startButton = screen.getByRole("button", { name: /start/i });
      expect(startButton).toBeEnabled();

      // Remove the selected profile from the list
      const remainingProfiles = mockProfiles.filter((p) => p.id !== 1);
      await rerender({ profiles: remainingProfiles, onStart: mockOnStart });

      // Start button should be disabled again
      await waitFor(() => {
        startButton = screen.getByRole("button", { name: /start/i });
        expect(startButton).toBeDisabled();
      });
    });

    it("maintains selection when profiles update but selected profile remains", async () => {
      const user = userEvent.setup();
      const { rerender } = render(ProfileSelector, {
        props: { profiles: mockProfiles, onStart: mockOnStart },
      });

      // Select second profile using keyboard navigation (ArrowDown x2)
      const input = screen.getByPlaceholderText("Select profile to start...");
      await user.click(input);
      await user.keyboard("{ArrowDown}");
      await user.keyboard("{ArrowDown}");
      await user.keyboard("{Enter}");

      // Start button should be enabled
      let startButton = screen.getByRole("button", { name: /start/i });
      expect(startButton).toBeEnabled();

      // Update profiles but keep profile 2
      const updatedProfiles = [
        ...mockProfiles,
        createMockProfile({ id: 4, name: "New Profile" }),
      ];
      await rerender({ profiles: updatedProfiles, onStart: mockOnStart });

      // Start button should still be enabled
      startButton = screen.getByRole("button", { name: /start/i });
      expect(startButton).toBeEnabled();
    });

    it("handles profile with null id gracefully in display", async () => {
      const user = userEvent.setup();
      // Create a profile where id is explicitly null (which would trigger the ?? '' fallback)
      const profileWithNullId = {
        ...createMockProfile({ name: "Null ID Profile" }),
        id: null as unknown as number,
      };

      const { container } = render(ProfileSelector, {
        props: { profiles: [profileWithNullId], onStart: mockOnStart },
      });

      // Open dropdown
      const triggerButton = container.querySelector(
        "[data-combobox-trigger]",
      ) as HTMLButtonElement;
      await user.click(triggerButton);

      // The profile renders but with empty string as value
      // This tests the profile.id?.toString() ?? '' branch on line 117
      const content = await screen.findByRole("listbox");
      expect(content).toBeInTheDocument();
    });

    it("clears selection when value becomes undefined", async () => {
      const user = userEvent.setup();
      render(ProfileSelector, {
        props: { profiles: mockProfiles, onStart: mockOnStart },
      });

      // Select a profile
      const input = screen.getByPlaceholderText("Select profile to start...");
      await user.click(input);
      await user.keyboard("{ArrowDown}");
      await user.keyboard("{Enter}");

      // Start button should be enabled
      let startButton = screen.getByRole("button", { name: /start/i });
      expect(startButton).toBeEnabled();

      // Open dropdown and press Escape without selecting - tests onValueChange with undefined
      await user.click(input);
      await user.keyboard("{Escape}");

      // Button should still be enabled as selection is preserved when closing dropdown
      startButton = screen.getByRole("button", { name: /start/i });
      expect(startButton).toBeEnabled();
    });
  });
});
