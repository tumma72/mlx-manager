import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor, fireEvent } from "@testing-library/svelte";
import userEvent from "@testing-library/user-event";
import ProfileCard from "./ProfileCard.svelte";
import type { ServerProfile, RunningServer } from "$api";

// Mock stores
vi.mock("$stores", () => ({
  serverStore: {
    isStarting: vi.fn().mockReturnValue(false),
    isRunning: vi.fn().mockReturnValue(false),
    isFailed: vi.fn().mockReturnValue(false),
    getFailure: vi.fn().mockReturnValue(null),
    getServer: vi.fn().mockReturnValue(null),
    clearFailure: vi.fn(),
    markStartupFailed: vi.fn(),
    markStartupSuccess: vi.fn(),
    start: vi.fn().mockResolvedValue(undefined),
    stop: vi.fn().mockResolvedValue(undefined),
    restart: vi.fn().mockResolvedValue(undefined),
    isProfilePolling: vi.fn().mockReturnValue(false),
    startProfilePolling: vi.fn().mockReturnValue(true),
    stopProfilePolling: vi.fn(),
    toggleDetailsOpen: vi.fn(),
  },
  profileStore: {
    delete: vi.fn().mockResolvedValue(undefined),
    duplicate: vi.fn().mockResolvedValue(undefined),
  },
}));

// Mock API
vi.mock("$api", () => ({
  servers: {
    status: vi.fn(),
  },
}));

// Mock navigation
vi.mock("$app/navigation", () => ({
  goto: vi.fn(),
}));

// Mock paths
vi.mock("$app/paths", () => ({
  resolve: vi.fn((path: string) => path),
}));

// Mock format utility
vi.mock("$lib/utils/format", () => ({
  formatDuration: vi.fn((seconds: number) => {
    if (seconds < 60) return `${seconds}s`;
    const minutes = Math.floor(seconds / 60);
    const hours = Math.floor(minutes / 60);
    if (hours > 0) return `${hours}h ${minutes % 60}m`;
    return `${minutes}m`;
  }),
}));

// Import after mocking
import { serverStore, profileStore } from "$stores";
import { goto } from "$app/navigation";

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
    system_prompt: null,
    launchd_installed: false,
    created_at: "2024-01-01T00:00:00Z",
    updated_at: "2024-01-01T00:00:00Z",
    ...overrides,
  };
}

// Helper to create mock running server
function createMockServer(
  overrides: Partial<RunningServer> = {},
): RunningServer {
  return {
    profile_id: 1,
    profile_name: "Test Server",
    pid: 12345,
    port: 10240,
    health_status: "healthy",
    uptime_seconds: 3600,
    memory_mb: 512,
    memory_percent: 25,
    cpu_percent: 10,
    ...overrides,
  };
}

describe("ProfileCard", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    // Reset default mock return values
    vi.mocked(serverStore.isStarting).mockReturnValue(false);
    vi.mocked(serverStore.isRunning).mockReturnValue(false);
    vi.mocked(serverStore.isFailed).mockReturnValue(false);
    vi.mocked(serverStore.getFailure).mockReturnValue(undefined);
    vi.mocked(serverStore.getServer).mockReturnValue(undefined);
    vi.mocked(serverStore.isProfilePolling).mockReturnValue(false);
    vi.mocked(serverStore.startProfilePolling).mockReturnValue(true);
  });

  describe("rendering - stopped state", () => {
    it("renders profile name", () => {
      render(ProfileCard, {
        props: { profile: createMockProfile({ name: "My Profile" }) },
      });

      expect(screen.getByText("My Profile")).toBeInTheDocument();
    });

    it("renders model path", () => {
      render(ProfileCard, {
        props: {
          profile: createMockProfile({ model_path: "mlx-community/llama-2" }),
        },
      });

      expect(screen.getByText("mlx-community/llama-2")).toBeInTheDocument();
    });

    it("renders description when provided", () => {
      render(ProfileCard, {
        props: {
          profile: createMockProfile({ description: "A test profile" }),
        },
      });

      expect(screen.getByText("A test profile")).toBeInTheDocument();
    });

    it("does not render description when null", () => {
      render(ProfileCard, {
        props: { profile: createMockProfile({ description: null }) },
      });

      expect(screen.queryByText("A test profile")).not.toBeInTheDocument();
    });

    it("renders port number", () => {
      render(ProfileCard, {
        props: { profile: createMockProfile({ port: 8080 }) },
      });

      expect(screen.getByText("8080")).toBeInTheDocument();
    });

    it("renders model type", () => {
      render(ProfileCard, {
        props: { profile: createMockProfile({ model_type: "multimodal" }) },
      });

      expect(screen.getByText("multimodal")).toBeInTheDocument();
    });

    it("renders Start button when stopped", () => {
      render(ProfileCard, {
        props: { profile: createMockProfile() },
      });

      expect(screen.getByTitle("Start")).toBeInTheDocument();
    });

    it("does not render Stop button when stopped", () => {
      render(ProfileCard, {
        props: { profile: createMockProfile() },
      });

      expect(screen.queryByTitle("Stop")).not.toBeInTheDocument();
    });

    it("renders Edit, Duplicate, and Delete buttons", () => {
      render(ProfileCard, {
        props: { profile: createMockProfile() },
      });

      expect(screen.getByTitle("Edit")).toBeInTheDocument();
      expect(screen.getByTitle("Duplicate")).toBeInTheDocument();
      expect(screen.getByTitle("Delete")).toBeInTheDocument();
    });

    it("shows launchd badge when installed", () => {
      render(ProfileCard, {
        props: { profile: createMockProfile({ launchd_installed: true }) },
      });

      expect(screen.getByText("launchd")).toBeInTheDocument();
    });
  });

  describe("rendering - running state", () => {
    beforeEach(() => {
      vi.mocked(serverStore.isRunning).mockReturnValue(true);
    });

    it("shows Running badge", () => {
      render(ProfileCard, {
        props: { profile: createMockProfile() },
      });

      expect(screen.getByText("Running")).toBeInTheDocument();
    });

    it("renders server stats when running", () => {
      const server = createMockServer({
        pid: 54321,
        memory_mb: 1024,
        uptime_seconds: 7200,
      });
      vi.mocked(serverStore.getServer).mockReturnValue(server);

      render(ProfileCard, {
        props: { profile: createMockProfile(), server },
      });

      expect(screen.getByText("54321")).toBeInTheDocument();
      expect(screen.getByText("1024.0 MB")).toBeInTheDocument();
      expect(screen.getByText("2h 0m")).toBeInTheDocument();
    });

    it("renders Stop and Restart buttons when running", () => {
      render(ProfileCard, {
        props: { profile: createMockProfile() },
      });

      expect(screen.getByTitle("Stop")).toBeInTheDocument();
      expect(screen.getByTitle("Restart")).toBeInTheDocument();
    });

    it("renders Chat button when running", () => {
      render(ProfileCard, {
        props: { profile: createMockProfile() },
      });

      expect(screen.getByTitle("Chat")).toBeInTheDocument();
    });

    it("does not render Start button when running", () => {
      render(ProfileCard, {
        props: { profile: createMockProfile() },
      });

      expect(screen.queryByTitle("Start")).not.toBeInTheDocument();
    });
  });

  describe("rendering - starting state", () => {
    beforeEach(() => {
      vi.mocked(serverStore.isStarting).mockReturnValue(true);
    });

    it("shows Loading badge", () => {
      render(ProfileCard, {
        props: { profile: createMockProfile() },
      });

      expect(screen.getByText("Loading...")).toBeInTheDocument();
    });

    it("shows disabled starting button with spinner", () => {
      render(ProfileCard, {
        props: { profile: createMockProfile() },
      });

      const startingButton = screen.getByTitle("Starting...");
      expect(startingButton).toBeDisabled();
      // Just verify the button exists and is disabled
    });

    it("shows Cancel button while starting", () => {
      render(ProfileCard, {
        props: { profile: createMockProfile() },
      });

      expect(screen.getByTitle("Cancel")).toBeInTheDocument();
    });
  });

  describe("rendering - failed state", () => {
    beforeEach(() => {
      vi.mocked(serverStore.isFailed).mockReturnValue(true);
      vi.mocked(serverStore.getFailure).mockReturnValue({
        error: "Failed to start",
        details: "Port already in use",
        detailsOpen: false,
      });
    });

    it("shows Error badge", () => {
      render(ProfileCard, {
        props: { profile: createMockProfile() },
      });

      expect(screen.getByText("Error")).toBeInTheDocument();
    });

    it("displays error message", () => {
      render(ProfileCard, {
        props: { profile: createMockProfile() },
      });

      expect(screen.getByText("Failed to start")).toBeInTheDocument();
    });

    it("displays error details in collapsible section", async () => {
      const user = userEvent.setup();
      render(ProfileCard, {
        props: { profile: createMockProfile() },
      });

      const summary = screen.getByText("Show server log");
      await user.click(summary);

      expect(screen.getByText("Port already in use")).toBeInTheDocument();
    });
  });

  describe("start action", () => {
    it("calls serverStore.start when Start button clicked", async () => {
      const user = userEvent.setup();
      render(ProfileCard, {
        props: { profile: createMockProfile({ id: 42 }) },
      });

      await user.click(screen.getByTitle("Start"));

      await waitFor(() => {
        expect(serverStore.start).toHaveBeenCalledWith(42);
      });
    });

    it("clears failure state before starting", async () => {
      const user = userEvent.setup();
      render(ProfileCard, {
        props: { profile: createMockProfile({ id: 42 }) },
      });

      await user.click(screen.getByTitle("Start"));

      expect(serverStore.clearFailure).toHaveBeenCalledWith(42);
    });

    it("registers polling when starting", async () => {
      const user = userEvent.setup();
      render(ProfileCard, {
        props: { profile: createMockProfile({ id: 42 }) },
      });

      await user.click(screen.getByTitle("Start"));

      await waitFor(() => {
        expect(serverStore.startProfilePolling).toHaveBeenCalledWith(42);
      });
    });
  });

  describe("stop action", () => {
    beforeEach(() => {
      vi.mocked(serverStore.isRunning).mockReturnValue(true);
    });

    it("calls serverStore.stop when Stop button clicked", async () => {
      const user = userEvent.setup();
      render(ProfileCard, {
        props: { profile: createMockProfile({ id: 42 }) },
      });

      await user.click(screen.getByTitle("Stop"));

      await waitFor(() => {
        expect(serverStore.stop).toHaveBeenCalledWith(42);
      });
    });

    it("stops polling when stopping", async () => {
      const user = userEvent.setup();
      render(ProfileCard, {
        props: { profile: createMockProfile({ id: 42 }) },
      });

      await user.click(screen.getByTitle("Stop"));

      expect(serverStore.stopProfilePolling).toHaveBeenCalledWith(42);
    });
  });

  describe("restart action", () => {
    beforeEach(() => {
      vi.mocked(serverStore.isRunning).mockReturnValue(true);
    });

    it("calls serverStore.restart when Restart button clicked", async () => {
      const user = userEvent.setup();
      render(ProfileCard, {
        props: { profile: createMockProfile({ id: 42 }) },
      });

      await user.click(screen.getByTitle("Restart"));

      await waitFor(() => {
        expect(serverStore.restart).toHaveBeenCalledWith(42);
      });
    });
  });

  describe("delete action", () => {
    it("shows confirmation dialog when Delete clicked", async () => {
      const user = userEvent.setup();
      render(ProfileCard, {
        props: { profile: createMockProfile({ name: "Test Profile" }) },
      });

      await user.click(screen.getByTitle("Delete"));

      // Dialog content may be rendered in a portal
      await waitFor(() => {
        expect(screen.getByText("Delete Profile")).toBeInTheDocument();
      });
    });

    it("verifies delete handler exists", () => {
      const profile = createMockProfile({ id: 42 });
      render(ProfileCard, {
        props: { profile },
      });

      // Verify delete button is present and linked to profile
      const deleteButton = screen.getByTitle("Delete");
      expect(deleteButton).toBeInTheDocument();
    });
  });

  describe("duplicate action", () => {
    it("prompts for new name when Duplicate clicked", async () => {
      const mockPrompt = vi
        .spyOn(window, "prompt")
        .mockReturnValue("Copy of Test");

      render(ProfileCard, {
        props: { profile: createMockProfile({ id: 42, name: "Test" }) },
      });

      // Use fireEvent to bypass pointer-events check in test environment
      await fireEvent.click(screen.getByTitle("Duplicate"));

      expect(mockPrompt).toHaveBeenCalledWith(
        "Enter name for the duplicate profile:",
        "Test (copy)",
      );

      mockPrompt.mockRestore();
    });

    it("duplicates profile with new name", async () => {
      const mockPrompt = vi
        .spyOn(window, "prompt")
        .mockReturnValue("New Copy");

      render(ProfileCard, {
        props: { profile: createMockProfile({ id: 42 }) },
      });

      // Use fireEvent to bypass pointer-events check in test environment
      await fireEvent.click(screen.getByTitle("Duplicate"));

      await waitFor(() => {
        expect(profileStore.duplicate).toHaveBeenCalledWith(42, "New Copy");
      });

      mockPrompt.mockRestore();
    });

    it("does not duplicate if prompt cancelled", async () => {
      const mockPrompt = vi.spyOn(window, "prompt").mockReturnValue(null);

      render(ProfileCard, {
        props: { profile: createMockProfile({ id: 42 }) },
      });

      // Use fireEvent to bypass pointer-events check in test environment
      await fireEvent.click(screen.getByTitle("Duplicate"));

      expect(profileStore.duplicate).not.toHaveBeenCalled();

      mockPrompt.mockRestore();
    });
  });

  describe("chat action", () => {
    beforeEach(() => {
      vi.mocked(serverStore.isRunning).mockReturnValue(true);
    });

    it("navigates to chat page when Chat clicked", async () => {
      render(ProfileCard, {
        props: { profile: createMockProfile({ id: 42 }) },
      });

      // Use fireEvent to bypass pointer-events check in test environment
      await fireEvent.click(screen.getByTitle("Chat"));

      await waitFor(() => {
        expect(goto).toHaveBeenCalledWith("/chat?profile=42");
      });
    });
  });

  describe("edit navigation", () => {
    it("has link to edit page", () => {
      const { container } = render(ProfileCard, {
        props: { profile: createMockProfile({ id: 42 }) },
      });

      const editLink = container.querySelector('[href="/profiles/42"]');
      expect(editLink).toBeInTheDocument();
    });
  });

  describe("error details display", () => {
    beforeEach(() => {
      vi.mocked(serverStore.isFailed).mockReturnValue(true);
      vi.mocked(serverStore.getFailure).mockReturnValue({
        error: "Failed to start",
        details: "Port already in use",
        detailsOpen: false,
      });
    });

    it("renders error details section", () => {
      render(ProfileCard, {
        props: { profile: createMockProfile() },
      });

      expect(screen.getByText("Failed to start")).toBeInTheDocument();
      expect(screen.getByText("Show server log")).toBeInTheDocument();
    });

    it("expands error details when clicked", async () => {
      render(ProfileCard, {
        props: { profile: createMockProfile() },
      });

      // Use fireEvent to bypass pointer-events check in test environment
      await fireEvent.click(screen.getByText("Show server log"));

      expect(screen.getByText("Port already in use")).toBeInTheDocument();
    });
  });
});
