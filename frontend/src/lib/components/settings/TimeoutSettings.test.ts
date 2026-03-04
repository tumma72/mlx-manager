import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/svelte";
import userEvent from "@testing-library/user-event";
import TimeoutSettings from "./TimeoutSettings.svelte";
import { settings } from "$lib/api/client";
import type { TimeoutSettings as TimeoutSettingsType } from "$lib/api/types";

// Mock API
vi.mock("$lib/api/client", () => ({
  settings: {
    getTimeoutSettings: vi.fn(),
    updateTimeoutSettings: vi.fn(),
  },
}));

function createMockSettings(
  overrides: Partial<TimeoutSettingsType> = {},
): TimeoutSettingsType {
  return {
    chat_seconds: 900,
    completions_seconds: 600,
    embeddings_seconds: 120,
    ...overrides,
  };
}

/**
 * Helper to get number inputs by their id attribute.
 * The labels contain extra span text, so we use getElementById instead of getByLabelText.
 */
function getNumberInput(container: HTMLElement, id: string): HTMLInputElement {
  const el = container.querySelector(`#${id}`);
  if (!el) throw new Error(`Input #${id} not found`);
  return el as HTMLInputElement;
}

describe("TimeoutSettings", () => {
  beforeEach(() => {
    vi.mocked(settings.getTimeoutSettings).mockResolvedValue(
      createMockSettings(),
    );
    vi.mocked(settings.updateTimeoutSettings).mockResolvedValue(
      createMockSettings(),
    );
  });

  describe("loading state", () => {
    it("shows loading message initially", () => {
      // Use a never-resolving promise so the component stays in loading state
      vi.mocked(settings.getTimeoutSettings).mockReturnValue(
        new Promise(() => {}),
      );

      render(TimeoutSettings);

      expect(
        screen.getByText("Loading timeout settings..."),
      ).toBeInTheDocument();
    });

    it("hides loading message after data loads", async () => {
      render(TimeoutSettings);

      await waitFor(() => {
        expect(
          screen.queryByText("Loading timeout settings..."),
        ).not.toBeInTheDocument();
      });
    });
  });

  describe("successful load", () => {
    it("renders chat completions section", async () => {
      render(TimeoutSettings);

      await waitFor(() => {
        expect(screen.getByText("Chat Completions")).toBeInTheDocument();
      });
    });

    it("renders text completions section", async () => {
      render(TimeoutSettings);

      await waitFor(() => {
        expect(screen.getByText("Text Completions")).toBeInTheDocument();
      });
    });

    it("renders embeddings section", async () => {
      render(TimeoutSettings);

      await waitFor(() => {
        expect(screen.getByText("Embeddings")).toBeInTheDocument();
      });
    });

    it("renders endpoint paths", async () => {
      render(TimeoutSettings);

      await waitFor(() => {
        expect(
          screen.getByText("(/v1/chat/completions)"),
        ).toBeInTheDocument();
        expect(screen.getByText("(/v1/completions)")).toBeInTheDocument();
        expect(screen.getByText("(/v1/embeddings)")).toBeInTheDocument();
      });
    });

    it("renders Save Changes button", async () => {
      render(TimeoutSettings);

      await waitFor(() => {
        expect(
          screen.getByRole("button", { name: /Save Changes/ }),
        ).toBeInTheDocument();
      });
    });

    it("renders Reset to Defaults button", async () => {
      render(TimeoutSettings);

      await waitFor(() => {
        expect(
          screen.getByRole("button", { name: /Reset to Defaults/ }),
        ).toBeInTheDocument();
      });
    });

    it("renders note about restart", async () => {
      render(TimeoutSettings);

      await waitFor(() => {
        expect(
          screen.getByText(
            /Changes are stored but may require restarting/,
          ),
        ).toBeInTheDocument();
      });
    });
  });

  describe("loaded values display", () => {
    it("displays loaded chat timeout value", async () => {
      vi.mocked(settings.getTimeoutSettings).mockResolvedValue(
        createMockSettings({ chat_seconds: 1800 }),
      );

      const { container } = render(TimeoutSettings);

      await waitFor(() => {
        const input = getNumberInput(container, "chat-timeout");
        expect(input.value).toBe("1800");
      });
    });

    it("displays loaded completions timeout value", async () => {
      vi.mocked(settings.getTimeoutSettings).mockResolvedValue(
        createMockSettings({ completions_seconds: 1200 }),
      );

      const { container } = render(TimeoutSettings);

      await waitFor(() => {
        const input = getNumberInput(container, "completions-timeout");
        expect(input.value).toBe("1200");
      });
    });

    it("displays loaded embeddings timeout value", async () => {
      vi.mocked(settings.getTimeoutSettings).mockResolvedValue(
        createMockSettings({ embeddings_seconds: 300 }),
      );

      const { container } = render(TimeoutSettings);

      await waitFor(() => {
        const input = getNumberInput(container, "embeddings-timeout");
        expect(input.value).toBe("300");
      });
    });
  });

  describe("formatSeconds display", () => {
    it("displays seconds for values under 60", async () => {
      vi.mocked(settings.getTimeoutSettings).mockResolvedValue(
        createMockSettings({ embeddings_seconds: 30 }),
      );

      render(TimeoutSettings);

      await waitFor(() => {
        expect(screen.getByText("30s")).toBeInTheDocument();
      });
    });

    it("displays minutes for values under 3600", async () => {
      render(TimeoutSettings);

      await waitFor(() => {
        // 900s = 15 min (chat default)
        expect(screen.getByText("15 min")).toBeInTheDocument();
        // 600s = 10 min (completions default)
        expect(screen.getByText("10 min")).toBeInTheDocument();
        // 120s = 2 min (embeddings default)
        expect(screen.getByText("2 min")).toBeInTheDocument();
      });
    });

    it("displays hours for values 3600 and above", async () => {
      vi.mocked(settings.getTimeoutSettings).mockResolvedValue(
        createMockSettings({ chat_seconds: 7200 }),
      );

      render(TimeoutSettings);

      await waitFor(() => {
        expect(screen.getByText("2.0 hr")).toBeInTheDocument();
      });
    });
  });

  describe("hasChanges detection", () => {
    it("does not show unsaved changes initially", async () => {
      render(TimeoutSettings);

      await waitFor(() => {
        expect(screen.getByText("Chat Completions")).toBeInTheDocument();
      });

      expect(screen.queryByText("Unsaved changes")).not.toBeInTheDocument();
    });

    it("shows unsaved changes after modifying chat timeout", async () => {
      const user = userEvent.setup();
      const { container } = render(TimeoutSettings);

      await waitFor(() => {
        expect(getNumberInput(container, "chat-timeout")).toBeInTheDocument();
      });

      const input = getNumberInput(container, "chat-timeout");
      await user.clear(input);
      await user.type(input, "1200");

      await waitFor(() => {
        expect(screen.getByText("Unsaved changes")).toBeInTheDocument();
      });
    });

    it("shows unsaved changes after modifying completions timeout", async () => {
      const user = userEvent.setup();
      const { container } = render(TimeoutSettings);

      await waitFor(() => {
        expect(
          getNumberInput(container, "completions-timeout"),
        ).toBeInTheDocument();
      });

      const input = getNumberInput(container, "completions-timeout");
      await user.clear(input);
      await user.type(input, "300");

      await waitFor(() => {
        expect(screen.getByText("Unsaved changes")).toBeInTheDocument();
      });
    });

    it("shows unsaved changes after modifying embeddings timeout", async () => {
      const user = userEvent.setup();
      const { container } = render(TimeoutSettings);

      await waitFor(() => {
        expect(
          getNumberInput(container, "embeddings-timeout"),
        ).toBeInTheDocument();
      });

      const input = getNumberInput(container, "embeddings-timeout");
      await user.clear(input);
      await user.type(input, "60");

      await waitFor(() => {
        expect(screen.getByText("Unsaved changes")).toBeInTheDocument();
      });
    });

    it("hides unsaved changes when value reverted to original", async () => {
      const user = userEvent.setup();
      const { container } = render(TimeoutSettings);

      await waitFor(() => {
        expect(getNumberInput(container, "chat-timeout")).toBeInTheDocument();
      });

      const input = getNumberInput(container, "chat-timeout");
      await user.clear(input);
      await user.type(input, "1200");

      await waitFor(() => {
        expect(screen.getByText("Unsaved changes")).toBeInTheDocument();
      });

      // Revert to original value
      await user.clear(input);
      await user.type(input, "900");

      await waitFor(() => {
        expect(
          screen.queryByText("Unsaved changes"),
        ).not.toBeInTheDocument();
      });
    });
  });

  describe("save button state", () => {
    it("disables Save button when no changes", async () => {
      render(TimeoutSettings);

      await waitFor(() => {
        expect(
          screen.getByRole("button", { name: /Save Changes/ }),
        ).toBeDisabled();
      });
    });

    it("enables Save button when changes exist", async () => {
      const user = userEvent.setup();
      const { container } = render(TimeoutSettings);

      await waitFor(() => {
        expect(getNumberInput(container, "chat-timeout")).toBeInTheDocument();
      });

      const input = getNumberInput(container, "chat-timeout");
      await user.clear(input);
      await user.type(input, "1200");

      await waitFor(() => {
        expect(
          screen.getByRole("button", { name: /Save Changes/ }),
        ).toBeEnabled();
      });
    });
  });

  describe("save action", () => {
    it("calls updateTimeoutSettings with current values", async () => {
      const user = userEvent.setup();
      vi.mocked(settings.updateTimeoutSettings).mockResolvedValue(
        createMockSettings({ chat_seconds: 1200 }),
      );

      const { container } = render(TimeoutSettings);

      await waitFor(() => {
        expect(getNumberInput(container, "chat-timeout")).toBeInTheDocument();
      });

      const input = getNumberInput(container, "chat-timeout");
      await user.clear(input);
      await user.type(input, "1200");

      await user.click(
        screen.getByRole("button", { name: /Save Changes/ }),
      );

      await waitFor(() => {
        expect(settings.updateTimeoutSettings).toHaveBeenCalledWith({
          chat_seconds: 1200,
          completions_seconds: 600,
          embeddings_seconds: 120,
        });
      });
    });

    it("updates currentSettings after successful save", async () => {
      const user = userEvent.setup();
      vi.mocked(settings.updateTimeoutSettings).mockResolvedValue(
        createMockSettings({ chat_seconds: 1200 }),
      );

      const { container } = render(TimeoutSettings);

      await waitFor(() => {
        expect(getNumberInput(container, "chat-timeout")).toBeInTheDocument();
      });

      const input = getNumberInput(container, "chat-timeout");
      await user.clear(input);
      await user.type(input, "1200");

      await user.click(
        screen.getByRole("button", { name: /Save Changes/ }),
      );

      // After save, hasChanges should be false (values match new currentSettings)
      await waitFor(() => {
        expect(
          screen.queryByText("Unsaved changes"),
        ).not.toBeInTheDocument();
      });
    });

    it("shows Saving... text during save", async () => {
      const user = userEvent.setup();
      let resolveSave!: (value: TimeoutSettingsType) => void;
      const savePromise = new Promise<TimeoutSettingsType>((resolve) => {
        resolveSave = resolve;
      });
      vi.mocked(settings.updateTimeoutSettings).mockReturnValue(savePromise);

      const { container } = render(TimeoutSettings);

      await waitFor(() => {
        expect(getNumberInput(container, "chat-timeout")).toBeInTheDocument();
      });

      const input = getNumberInput(container, "chat-timeout");
      await user.clear(input);
      await user.type(input, "1200");

      await user.click(
        screen.getByRole("button", { name: /Save Changes/ }),
      );

      expect(screen.getByText("Saving...")).toBeInTheDocument();

      resolveSave(createMockSettings({ chat_seconds: 1200 }));
      await savePromise;
    });

    it("disables Save button during save", async () => {
      const user = userEvent.setup();
      let resolveSave!: (value: TimeoutSettingsType) => void;
      const savePromise = new Promise<TimeoutSettingsType>((resolve) => {
        resolveSave = resolve;
      });
      vi.mocked(settings.updateTimeoutSettings).mockReturnValue(savePromise);

      const { container } = render(TimeoutSettings);

      await waitFor(() => {
        expect(getNumberInput(container, "chat-timeout")).toBeInTheDocument();
      });

      const input = getNumberInput(container, "chat-timeout");
      await user.clear(input);
      await user.type(input, "1200");

      await user.click(
        screen.getByRole("button", { name: /Save Changes|Saving/ }),
      );

      await waitFor(() => {
        expect(
          screen.getByRole("button", { name: /Saving/ }),
        ).toBeDisabled();
      });

      resolveSave(createMockSettings({ chat_seconds: 1200 }));
      await savePromise;
    });
  });

  describe("save error handling", () => {
    it("displays error on save failure", async () => {
      const user = userEvent.setup();
      vi.mocked(settings.updateTimeoutSettings).mockRejectedValue(
        new Error("Network error"),
      );

      const { container } = render(TimeoutSettings);

      await waitFor(() => {
        expect(getNumberInput(container, "chat-timeout")).toBeInTheDocument();
      });

      const input = getNumberInput(container, "chat-timeout");
      await user.clear(input);
      await user.type(input, "1200");

      await user.click(
        screen.getByRole("button", { name: /Save Changes/ }),
      );

      await waitFor(() => {
        expect(
          screen.getByText("Failed to save timeout settings"),
        ).toBeInTheDocument();
      });
    });
  });

  describe("reset to defaults", () => {
    it("resets chat timeout to 900", async () => {
      const user = userEvent.setup();
      vi.mocked(settings.getTimeoutSettings).mockResolvedValue(
        createMockSettings({ chat_seconds: 1800 }),
      );

      const { container } = render(TimeoutSettings);

      await waitFor(() => {
        expect(getNumberInput(container, "chat-timeout")).toBeInTheDocument();
      });

      await user.click(
        screen.getByRole("button", { name: /Reset to Defaults/ }),
      );

      await waitFor(() => {
        const input = getNumberInput(container, "chat-timeout");
        expect(input.value).toBe("900");
      });
    });

    it("resets completions timeout to 600", async () => {
      const user = userEvent.setup();
      vi.mocked(settings.getTimeoutSettings).mockResolvedValue(
        createMockSettings({ completions_seconds: 1200 }),
      );

      const { container } = render(TimeoutSettings);

      await waitFor(() => {
        expect(
          getNumberInput(container, "completions-timeout"),
        ).toBeInTheDocument();
      });

      await user.click(
        screen.getByRole("button", { name: /Reset to Defaults/ }),
      );

      await waitFor(() => {
        const input = getNumberInput(container, "completions-timeout");
        expect(input.value).toBe("600");
      });
    });

    it("resets embeddings timeout to 120", async () => {
      const user = userEvent.setup();
      vi.mocked(settings.getTimeoutSettings).mockResolvedValue(
        createMockSettings({ embeddings_seconds: 300 }),
      );

      const { container } = render(TimeoutSettings);

      await waitFor(() => {
        expect(
          getNumberInput(container, "embeddings-timeout"),
        ).toBeInTheDocument();
      });

      await user.click(
        screen.getByRole("button", { name: /Reset to Defaults/ }),
      );

      await waitFor(() => {
        const input = getNumberInput(container, "embeddings-timeout");
        expect(input.value).toBe("120");
      });
    });

    it("shows unsaved changes after reset when values differ from server", async () => {
      const user = userEvent.setup();
      vi.mocked(settings.getTimeoutSettings).mockResolvedValue(
        createMockSettings({ chat_seconds: 1800 }),
      );

      const { container } = render(TimeoutSettings);

      await waitFor(() => {
        expect(getNumberInput(container, "chat-timeout")).toBeInTheDocument();
      });

      await user.click(
        screen.getByRole("button", { name: /Reset to Defaults/ }),
      );

      await waitFor(() => {
        expect(screen.getByText("Unsaved changes")).toBeInTheDocument();
      });
    });

    it("does not show unsaved changes when server values match defaults", async () => {
      const user = userEvent.setup();
      const { container } = render(TimeoutSettings);

      await waitFor(() => {
        expect(getNumberInput(container, "chat-timeout")).toBeInTheDocument();
      });

      // Modify a value first
      const input = getNumberInput(container, "chat-timeout");
      await user.clear(input);
      await user.type(input, "1200");

      await waitFor(() => {
        expect(screen.getByText("Unsaved changes")).toBeInTheDocument();
      });

      // Reset to defaults (which happen to match server values)
      await user.click(
        screen.getByRole("button", { name: /Reset to Defaults/ }),
      );

      await waitFor(() => {
        expect(
          screen.queryByText("Unsaved changes"),
        ).not.toBeInTheDocument();
      });
    });
  });

  describe("load error handling", () => {
    it("displays error on load failure", async () => {
      vi.mocked(settings.getTimeoutSettings).mockRejectedValue(
        new Error("Server down"),
      );

      render(TimeoutSettings);

      await waitFor(() => {
        expect(
          screen.getByText("Failed to load timeout settings"),
        ).toBeInTheDocument();
      });
    });

    it("does not render form on load failure", async () => {
      vi.mocked(settings.getTimeoutSettings).mockRejectedValue(
        new Error("Server down"),
      );

      const { container } = render(TimeoutSettings);

      await waitFor(() => {
        expect(
          screen.getByText("Failed to load timeout settings"),
        ).toBeInTheDocument();
      });

      expect(
        container.querySelector("#chat-timeout"),
      ).not.toBeInTheDocument();
      expect(
        screen.queryByRole("button", { name: /Save Changes/ }),
      ).not.toBeInTheDocument();
    });
  });

  describe("slider inputs", () => {
    it("renders chat timeout slider", async () => {
      render(TimeoutSettings);

      await waitFor(() => {
        expect(
          screen.getByLabelText("Chat timeout slider"),
        ).toBeInTheDocument();
      });
    });

    it("renders completions timeout slider", async () => {
      render(TimeoutSettings);

      await waitFor(() => {
        expect(
          screen.getByLabelText("Completions timeout slider"),
        ).toBeInTheDocument();
      });
    });

    it("renders embeddings timeout slider", async () => {
      render(TimeoutSettings);

      await waitFor(() => {
        expect(
          screen.getByLabelText("Embeddings timeout slider"),
        ).toBeInTheDocument();
      });
    });

    it("chat slider has correct min/max/step", async () => {
      render(TimeoutSettings);

      await waitFor(() => {
        const slider = screen.getByLabelText(
          "Chat timeout slider",
        ) as HTMLInputElement;
        expect(slider.min).toBe("60");
        expect(slider.max).toBe("7200");
        expect(slider.step).toBe("60");
      });
    });

    it("completions slider has correct min/max/step", async () => {
      render(TimeoutSettings);

      await waitFor(() => {
        const slider = screen.getByLabelText(
          "Completions timeout slider",
        ) as HTMLInputElement;
        expect(slider.min).toBe("60");
        expect(slider.max).toBe("7200");
        expect(slider.step).toBe("60");
      });
    });

    it("embeddings slider has correct min/max/step", async () => {
      render(TimeoutSettings);

      await waitFor(() => {
        const slider = screen.getByLabelText(
          "Embeddings timeout slider",
        ) as HTMLInputElement;
        expect(slider.min).toBe("30");
        expect(slider.max).toBe("600");
        expect(slider.step).toBe("30");
      });
    });
  });

  describe("default descriptions", () => {
    it("shows default description for chat", async () => {
      render(TimeoutSettings);

      await waitFor(() => {
        expect(
          screen.getByText(/Default: 15 minutes/),
        ).toBeInTheDocument();
      });
    });

    it("shows default description for completions", async () => {
      render(TimeoutSettings);

      await waitFor(() => {
        expect(
          screen.getByText(/Default: 10 minutes/),
        ).toBeInTheDocument();
      });
    });

    it("shows default description for embeddings", async () => {
      render(TimeoutSettings);

      await waitFor(() => {
        expect(
          screen.getByText(/Default: 2 minutes/),
        ).toBeInTheDocument();
      });
    });
  });
});
