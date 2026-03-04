import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor, fireEvent } from "@testing-library/svelte";
import userEvent from "@testing-library/user-event";
import type { ServerPoolConfig, PreloadedProfileInfo } from "$lib/api/types";
import { settings } from "$lib/api/client";

// Mock API client
vi.mock("$lib/api/client", () => ({
  settings: {
    getPoolConfig: vi.fn(),
    updatePoolConfig: vi.fn(),
  },
}));

// Mock systemStore
vi.mock("$stores", () => ({
  systemStore: {
    memory: { total_gb: 32 },
    refreshMemory: vi.fn().mockResolvedValue(undefined),
  },
}));

// Mock SvelteKit navigation
vi.mock("$app/navigation", () => ({
  goto: vi.fn().mockResolvedValue(undefined),
}));

// Mock SvelteKit paths
vi.mock("$app/paths", () => ({
  resolve: vi.fn((path: string) => path),
}));

import ModelPoolSettings from "./ModelPoolSettings.svelte";
import { systemStore } from "$stores";
import { goto } from "$app/navigation";

function createMockConfig(
  overrides: Partial<ServerPoolConfig> = {},
): ServerPoolConfig {
  return {
    memory_limit_mode: "percent",
    memory_limit_value: 80,
    eviction_policy: "lru",
    preloaded_profiles: [],
    ...overrides,
  };
}

function createMockProfile(
  overrides: Partial<PreloadedProfileInfo> = {},
): PreloadedProfileInfo {
  return {
    id: 1,
    name: "Test Profile",
    profile_type: "inference",
    model_repo_id: "mlx-community/Qwen3-0.6B-4bit-DWQ",
    model_name: "Qwen3-0.6B",
    ...overrides,
  };
}

describe("ModelPoolSettings", () => {
  beforeEach(() => {
    vi.mocked(settings.getPoolConfig).mockResolvedValue(createMockConfig());
    vi.mocked(settings.updatePoolConfig).mockResolvedValue(createMockConfig());
    vi.mocked(systemStore.refreshMemory).mockResolvedValue(undefined);
    vi.mocked(goto).mockResolvedValue(undefined);
  });

  describe("loading state", () => {
    it("shows loading spinner on mount", () => {
      // Use a never-resolving promise so loading persists
      vi.mocked(settings.getPoolConfig).mockReturnValue(
        new Promise(() => {}),
      );

      render(ModelPoolSettings);

      expect(screen.getByText("Loading settings...")).toBeInTheDocument();
    });

    it("hides loading spinner after config loads", async () => {
      render(ModelPoolSettings);

      await waitFor(() => {
        expect(
          screen.queryByText("Loading settings..."),
        ).not.toBeInTheDocument();
      });
    });

    it("calls refreshMemory on mount", async () => {
      render(ModelPoolSettings);

      await waitFor(() => {
        expect(vi.mocked(systemStore.refreshMemory)).toHaveBeenCalled();
      });
    });

    it("calls getPoolConfig on mount", async () => {
      render(ModelPoolSettings);

      await waitFor(() => {
        expect(settings.getPoolConfig).toHaveBeenCalled();
      });
    });
  });

  describe("successful load", () => {
    it("renders memory limit heading", async () => {
      render(ModelPoolSettings);

      await waitFor(() => {
        expect(screen.getByText("Memory Limit")).toBeInTheDocument();
      });
    });

    it("displays default memory value as percentage", async () => {
      render(ModelPoolSettings);

      await waitFor(() => {
        expect(screen.getByText("80%")).toBeInTheDocument();
      });
    });

    it("displays total memory info", async () => {
      render(ModelPoolSettings);

      await waitFor(() => {
        expect(screen.getByText(/of 32\.0 GB total/)).toBeInTheDocument();
      });
    });

    it("renders save button", async () => {
      render(ModelPoolSettings);

      await waitFor(() => {
        expect(screen.getByText("Save Changes")).toBeInTheDocument();
      });
    });

    it("renders pre-loaded profiles heading", async () => {
      render(ModelPoolSettings);

      await waitFor(() => {
        expect(screen.getByText("Pre-loaded Profiles")).toBeInTheDocument();
      });
    });

    it("renders advanced options toggle", async () => {
      render(ModelPoolSettings);

      await waitFor(() => {
        expect(screen.getByText("Advanced Options")).toBeInTheDocument();
      });
    });

    it("renders config values from API response", async () => {
      vi.mocked(settings.getPoolConfig).mockResolvedValue(
        createMockConfig({
          memory_limit_mode: "percent",
          memory_limit_value: 60,
          eviction_policy: "lfu",
        }),
      );

      render(ModelPoolSettings);

      await waitFor(() => {
        expect(screen.getByText("60%")).toBeInTheDocument();
      });
    });

    it("renders GB mode from config", async () => {
      vi.mocked(settings.getPoolConfig).mockResolvedValue(
        createMockConfig({
          memory_limit_mode: "gb",
          memory_limit_value: 24,
        }),
      );

      render(ModelPoolSettings);

      await waitFor(() => {
        expect(screen.getByText("24 GB")).toBeInTheDocument();
      });
    });
  });

  describe("error handling on load", () => {
    it("displays error message when getPoolConfig fails", async () => {
      vi.mocked(settings.getPoolConfig).mockRejectedValue(
        new Error("Network error"),
      );

      render(ModelPoolSettings);

      await waitFor(() => {
        expect(screen.getByText("Network error")).toBeInTheDocument();
      });
    });

    it("displays generic error for non-Error exceptions", async () => {
      vi.mocked(settings.getPoolConfig).mockRejectedValue("unknown");

      render(ModelPoolSettings);

      await waitFor(() => {
        expect(
          screen.getByText("Failed to load settings"),
        ).toBeInTheDocument();
      });
    });

    it("hides loading spinner on error", async () => {
      vi.mocked(settings.getPoolConfig).mockRejectedValue(
        new Error("fail"),
      );

      render(ModelPoolSettings);

      await waitFor(() => {
        expect(
          screen.queryByText("Loading settings..."),
        ).not.toBeInTheDocument();
      });
    });
  });

  describe("memory mode toggle", () => {
    it("switches from percent to GB mode", async () => {
      const user = userEvent.setup();
      render(ModelPoolSettings);

      await waitFor(() => {
        expect(screen.getByText("80%")).toBeInTheDocument();
      });

      // Click the toggle button (contains "%" and "GB" text)
      const toggleButton = screen.getByRole("button", { name: /% GB/ });
      await user.click(toggleButton);

      // 80% of 32 GB = 25.6, rounded = 26
      await waitFor(() => {
        expect(screen.getByText("26 GB")).toBeInTheDocument();
      });
    });

    it("switches from GB to percent mode", async () => {
      const user = userEvent.setup();
      vi.mocked(settings.getPoolConfig).mockResolvedValue(
        createMockConfig({
          memory_limit_mode: "gb",
          memory_limit_value: 16,
        }),
      );

      render(ModelPoolSettings);

      await waitFor(() => {
        expect(screen.getByText("16 GB")).toBeInTheDocument();
      });

      const toggleButton = screen.getByRole("button", { name: /% GB/ });
      await user.click(toggleButton);

      // 16 / 32 = 50%, rounded to nearest 5 = 50
      await waitFor(() => {
        expect(screen.getByText("50%")).toBeInTheDocument();
      });
    });

    it("shows 'available' label in GB mode", async () => {
      vi.mocked(settings.getPoolConfig).mockResolvedValue(
        createMockConfig({
          memory_limit_mode: "gb",
          memory_limit_value: 16,
        }),
      );

      render(ModelPoolSettings);

      await waitFor(() => {
        expect(screen.getByText(/of 32\.0 GB available/)).toBeInTheDocument();
      });
    });

    it("shows 'total' label in percent mode", async () => {
      render(ModelPoolSettings);

      await waitFor(() => {
        expect(screen.getByText(/of 32\.0 GB total/)).toBeInTheDocument();
      });
    });

    it("clamps percent value to minimum 10", async () => {
      const user = userEvent.setup();
      vi.mocked(settings.getPoolConfig).mockResolvedValue(
        createMockConfig({
          memory_limit_mode: "gb",
          memory_limit_value: 2,
        }),
      );

      render(ModelPoolSettings);

      await waitFor(() => {
        expect(screen.getByText("2 GB")).toBeInTheDocument();
      });

      const toggleButton = screen.getByRole("button", { name: /% GB/ });
      await user.click(toggleButton);

      // 2 / 32 = 6.25%, rounded to nearest 5 = 5%, but clamped to min 10
      await waitFor(() => {
        // Both the display value and the slider min label show "10%"
        const matches = screen.getAllByText("10%");
        expect(matches.length).toBeGreaterThanOrEqual(2);
      });
    });

    it("clamps GB value to minimum 1", async () => {
      const user = userEvent.setup();
      vi.mocked(settings.getPoolConfig).mockResolvedValue(
        createMockConfig({
          memory_limit_mode: "percent",
          memory_limit_value: 15,
        }),
      );

      render(ModelPoolSettings);

      await waitFor(() => {
        expect(screen.getByText("15%")).toBeInTheDocument();
      });

      const toggleButton = screen.getByRole("button", { name: /% GB/ });
      await user.click(toggleButton);

      // 15% of 32 = 4.8, rounded = 5, clamped to max floor(32) = 32, min 1 => 5
      await waitFor(() => {
        expect(screen.getByText("5 GB")).toBeInTheDocument();
      });
    });
  });

  describe("slider and input synchronization", () => {
    it("renders range input with correct min/max in percent mode", async () => {
      render(ModelPoolSettings);

      await waitFor(() => {
        const slider = screen.getByRole("slider");
        expect(slider).toBeInTheDocument();
        expect(slider).toHaveAttribute("min", "10");
        expect(slider).toHaveAttribute("max", "100");
        expect(slider).toHaveAttribute("step", "5");
      });
    });

    it("renders range input with correct min/max in GB mode", async () => {
      vi.mocked(settings.getPoolConfig).mockResolvedValue(
        createMockConfig({
          memory_limit_mode: "gb",
          memory_limit_value: 16,
        }),
      );

      render(ModelPoolSettings);

      await waitFor(() => {
        const slider = screen.getByRole("slider");
        expect(slider).toBeInTheDocument();
        expect(slider).toHaveAttribute("min", "1");
        expect(slider).toHaveAttribute("max", "32");
        expect(slider).toHaveAttribute("step", "1");
      });
    });

    it("updates display when slider is moved", async () => {
      render(ModelPoolSettings);

      await waitFor(() => {
        expect(screen.getByText("80%")).toBeInTheDocument();
      });

      const slider = screen.getByRole("slider");
      fireEvent.input(slider, { target: { value: "60" } });

      await waitFor(() => {
        expect(screen.getByText("60%")).toBeInTheDocument();
      });
    });

    it("renders slider min/max labels in percent mode", async () => {
      render(ModelPoolSettings);

      await waitFor(() => {
        expect(screen.getByText("10%")).toBeInTheDocument();
        expect(screen.getByText("100%")).toBeInTheDocument();
      });
    });

    it("renders slider min/max labels in GB mode", async () => {
      vi.mocked(settings.getPoolConfig).mockResolvedValue(
        createMockConfig({
          memory_limit_mode: "gb",
          memory_limit_value: 16,
        }),
      );

      render(ModelPoolSettings);

      await waitFor(() => {
        expect(screen.getByText("1 GB")).toBeInTheDocument();
        expect(screen.getByText("32 GB")).toBeInTheDocument();
      });
    });
  });

  describe("eviction policy select", () => {
    it("renders eviction policy select when advanced is toggled", async () => {
      const user = userEvent.setup();
      render(ModelPoolSettings);

      await waitFor(() => {
        expect(screen.getByText("Advanced Options")).toBeInTheDocument();
      });

      await user.click(screen.getByText("Advanced Options"));

      await waitFor(() => {
        expect(screen.getByLabelText("Eviction Policy")).toBeInTheDocument();
      });
    });

    it("shows LRU description by default", async () => {
      const user = userEvent.setup();
      render(ModelPoolSettings);

      await waitFor(() => {
        expect(screen.getByText("Advanced Options")).toBeInTheDocument();
      });

      await user.click(screen.getByText("Advanced Options"));

      await waitFor(() => {
        expect(
          screen.getByText("Evict profiles not used for longest"),
        ).toBeInTheDocument();
      });
    });

    it("shows LFU option in select", async () => {
      const user = userEvent.setup();
      vi.mocked(settings.getPoolConfig).mockResolvedValue(
        createMockConfig({ eviction_policy: "lfu" }),
      );

      render(ModelPoolSettings);

      await waitFor(() => {
        expect(screen.getByText("Advanced Options")).toBeInTheDocument();
      });

      await user.click(screen.getByText("Advanced Options"));

      await waitFor(() => {
        expect(
          screen.getByText("Evict profiles with fewest requests"),
        ).toBeInTheDocument();
      });
    });

    it("changes eviction policy via select", async () => {
      const user = userEvent.setup();
      render(ModelPoolSettings);

      await waitFor(() => {
        expect(screen.getByText("Advanced Options")).toBeInTheDocument();
      });

      await user.click(screen.getByText("Advanced Options"));

      const select = screen.getByLabelText("Eviction Policy");
      await user.selectOptions(select, "ttl");

      await waitFor(() => {
        expect(
          screen.getByText("Evict profiles after idle timeout"),
        ).toBeInTheDocument();
      });
    });
  });

  describe("save button behavior", () => {
    it("calls updatePoolConfig with current form values", async () => {
      const user = userEvent.setup();
      vi.mocked(settings.updatePoolConfig).mockResolvedValue(
        createMockConfig(),
      );

      render(ModelPoolSettings);

      await waitFor(() => {
        expect(screen.getByText("Save Changes")).toBeInTheDocument();
      });

      await user.click(screen.getByText("Save Changes"));

      await waitFor(() => {
        expect(settings.updatePoolConfig).toHaveBeenCalledWith({
          memory_limit_mode: "percent",
          memory_limit_value: 80,
          eviction_policy: "lru",
        });
      });
    });

    it("shows success message after successful save", async () => {
      const user = userEvent.setup();
      vi.mocked(settings.updatePoolConfig).mockResolvedValue(
        createMockConfig(),
      );

      render(ModelPoolSettings);

      await waitFor(() => {
        expect(screen.getByText("Save Changes")).toBeInTheDocument();
      });

      await user.click(screen.getByText("Save Changes"));

      await waitFor(() => {
        expect(
          screen.getByText("Settings saved successfully"),
        ).toBeInTheDocument();
      });
    });

    it("shows saving state during save", async () => {
      const user = userEvent.setup();
      let resolveSave!: (value: ServerPoolConfig) => void;
      const savePromise = new Promise<ServerPoolConfig>((resolve) => {
        resolveSave = resolve;
      });
      vi.mocked(settings.updatePoolConfig).mockReturnValue(savePromise);

      render(ModelPoolSettings);

      await waitFor(() => {
        expect(screen.getByText("Save Changes")).toBeInTheDocument();
      });

      await user.click(screen.getByText("Save Changes"));

      expect(screen.getByText("Saving...")).toBeInTheDocument();

      resolveSave(createMockConfig());
      await savePromise;
    });

    it("shows error message on save failure", async () => {
      const user = userEvent.setup();
      vi.mocked(settings.updatePoolConfig).mockRejectedValue(
        new Error("Server error"),
      );

      render(ModelPoolSettings);

      await waitFor(() => {
        expect(screen.getByText("Save Changes")).toBeInTheDocument();
      });

      await user.click(screen.getByText("Save Changes"));

      await waitFor(() => {
        expect(screen.getByText("Server error")).toBeInTheDocument();
      });
    });

    it("shows generic error for non-Error save failures", async () => {
      const user = userEvent.setup();
      vi.mocked(settings.updatePoolConfig).mockRejectedValue("unknown");

      render(ModelPoolSettings);

      await waitFor(() => {
        expect(screen.getByText("Save Changes")).toBeInTheDocument();
      });

      await user.click(screen.getByText("Save Changes"));

      await waitFor(() => {
        expect(
          screen.getByText("Failed to save settings"),
        ).toBeInTheDocument();
      });
    });

    it("clears error on new save attempt", async () => {
      const user = userEvent.setup();
      // First save fails
      vi.mocked(settings.updatePoolConfig).mockRejectedValueOnce(
        new Error("Server error"),
      );
      // Second save succeeds
      vi.mocked(settings.updatePoolConfig).mockResolvedValueOnce(
        createMockConfig(),
      );

      render(ModelPoolSettings);

      await waitFor(() => {
        expect(screen.getByText("Save Changes")).toBeInTheDocument();
      });

      // First save - fails
      await user.click(screen.getByText("Save Changes"));

      await waitFor(() => {
        expect(screen.getByText("Server error")).toBeInTheDocument();
      });

      // Second save - should clear error
      await user.click(screen.getByText("Save Changes"));

      await waitFor(() => {
        expect(screen.queryByText("Server error")).not.toBeInTheDocument();
      });
    });

    it("updates preloaded profiles from save response", async () => {
      const user = userEvent.setup();
      vi.mocked(settings.updatePoolConfig).mockResolvedValue(
        createMockConfig({
          preloaded_profiles: [
            createMockProfile({ id: 1, name: "Updated Profile" }),
          ],
        }),
      );

      render(ModelPoolSettings);

      await waitFor(() => {
        expect(screen.getByText("Save Changes")).toBeInTheDocument();
      });

      await user.click(screen.getByText("Save Changes"));

      await waitFor(() => {
        expect(screen.getByText("Updated Profile")).toBeInTheDocument();
      });
    });
  });

  describe("pre-loaded profiles display", () => {
    it("shows empty state when no profiles are preloaded", async () => {
      render(ModelPoolSettings);

      await waitFor(() => {
        expect(
          screen.getByText("No profiles configured for auto-loading."),
        ).toBeInTheDocument();
      });
    });

    it("renders preloaded profile names", async () => {
      vi.mocked(settings.getPoolConfig).mockResolvedValue(
        createMockConfig({
          preloaded_profiles: [
            createMockProfile({ name: "My Qwen Profile" }),
            createMockProfile({ id: 2, name: "My GLM Profile" }),
          ],
        }),
      );

      render(ModelPoolSettings);

      await waitFor(() => {
        expect(screen.getByText("My Qwen Profile")).toBeInTheDocument();
        expect(screen.getByText("My GLM Profile")).toBeInTheDocument();
      });
    });

    it("renders profile type badge", async () => {
      vi.mocked(settings.getPoolConfig).mockResolvedValue(
        createMockConfig({
          preloaded_profiles: [
            createMockProfile({ profile_type: "inference" }),
          ],
        }),
      );

      render(ModelPoolSettings);

      await waitFor(() => {
        expect(screen.getByText("inference")).toBeInTheDocument();
      });
    });

    it("renders model name when available", async () => {
      vi.mocked(settings.getPoolConfig).mockResolvedValue(
        createMockConfig({
          preloaded_profiles: [
            createMockProfile({ model_name: "Qwen3-0.6B" }),
          ],
        }),
      );

      render(ModelPoolSettings);

      await waitFor(() => {
        expect(screen.getByText("Qwen3-0.6B")).toBeInTheDocument();
      });
    });

    it("does not render model name when null", async () => {
      vi.mocked(settings.getPoolConfig).mockResolvedValue(
        createMockConfig({
          preloaded_profiles: [
            createMockProfile({ model_name: null }),
          ],
        }),
      );

      render(ModelPoolSettings);

      await waitFor(() => {
        expect(screen.getByText("Test Profile")).toBeInTheDocument();
      });

      // Should not have any model name text
      expect(screen.queryByText("Qwen3-0.6B")).not.toBeInTheDocument();
    });

    it("navigates to profile edit on click", async () => {
      const user = userEvent.setup();
      vi.mocked(settings.getPoolConfig).mockResolvedValue(
        createMockConfig({
          preloaded_profiles: [
            createMockProfile({ id: 42, name: "Clickable Profile" }),
          ],
        }),
      );

      render(ModelPoolSettings);

      await waitFor(() => {
        expect(screen.getByText("Clickable Profile")).toBeInTheDocument();
      });

      await user.click(screen.getByText("Clickable Profile"));

      await waitFor(() => {
        expect(vi.mocked(goto)).toHaveBeenCalledWith("/profiles?edit=42");
      });
    });
  });

  describe("advanced options toggle", () => {
    it("hides eviction policy by default", async () => {
      render(ModelPoolSettings);

      await waitFor(() => {
        expect(screen.getByText("Advanced Options")).toBeInTheDocument();
      });

      expect(
        screen.queryByLabelText("Eviction Policy"),
      ).not.toBeInTheDocument();
    });

    it("shows eviction policy when toggled open", async () => {
      const user = userEvent.setup();
      render(ModelPoolSettings);

      await waitFor(() => {
        expect(screen.getByText("Advanced Options")).toBeInTheDocument();
      });

      await user.click(screen.getByText("Advanced Options"));

      await waitFor(() => {
        expect(screen.getByLabelText("Eviction Policy")).toBeInTheDocument();
      });
    });

    it("hides eviction policy when toggled closed", async () => {
      const user = userEvent.setup();
      render(ModelPoolSettings);

      await waitFor(() => {
        expect(screen.getByText("Advanced Options")).toBeInTheDocument();
      });

      await user.click(screen.getByText("Advanced Options"));

      await waitFor(() => {
        expect(screen.getByLabelText("Eviction Policy")).toBeInTheDocument();
      });

      await user.click(screen.getByText("Advanced Options"));

      await waitFor(() => {
        expect(
          screen.queryByLabelText("Eviction Policy"),
        ).not.toBeInTheDocument();
      });
    });

    it("shows all eviction options in select", async () => {
      const user = userEvent.setup();
      render(ModelPoolSettings);

      await waitFor(() => {
        expect(screen.getByText("Advanced Options")).toBeInTheDocument();
      });

      await user.click(screen.getByText("Advanced Options"));

      await waitFor(() => {
        expect(
          screen.getByText("LRU (Least Recently Used)"),
        ).toBeInTheDocument();
        expect(
          screen.getByText("LFU (Least Frequently Used)"),
        ).toBeInTheDocument();
        expect(
          screen.getByText("TTL (Time To Live)"),
        ).toBeInTheDocument();
      });
    });
  });
});
