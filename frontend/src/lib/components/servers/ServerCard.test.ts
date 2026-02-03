import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/svelte";
import userEvent from "@testing-library/user-event";
import type { ServerProfile, RunningServer } from "$api";

// Mock the serverStore
vi.mock("$stores", () => ({
  serverStore: {
    start: vi.fn().mockResolvedValue(undefined),
    stop: vi.fn().mockResolvedValue(undefined),
    restart: vi.fn().mockResolvedValue(undefined),
  },
}));

// Mock format utilities
vi.mock("$lib/utils/format", () => ({
  formatDuration: vi.fn((seconds: number) => {
    if (seconds < 60) return `${seconds}s`;
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    if (hours > 0) return `${hours}h ${minutes}m`;
    return `${minutes}m`;
  }),
  formatBytes: vi.fn((bytes: number, decimals = 2) => {
    if (bytes === 0) return "0 Bytes";
    const k = 1024;
    const dm = decimals < 0 ? 0 : decimals;
    const sizes = ["Bytes", "KB", "MB", "GB", "TB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + " " + sizes[i];
  }),
}));

// Import after mocking
import ServerCard from "./ServerCard.svelte";
import { serverStore } from "$stores";

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
    memory_limit_percent: 30,
    cpu_percent: 10,
    ...overrides,
  };
}

describe("ServerCard", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe("rendering without server (stopped state)", () => {
    it("renders profile name", () => {
      render(ServerCard, {
        props: { profile: createMockProfile({ name: "My Profile" }) },
      });

      expect(screen.getByText("My Profile")).toBeInTheDocument();
    });

    it("renders model path", () => {
      render(ServerCard, {
        props: {
          profile: createMockProfile({ model_path: "mlx-community/llama-2" }),
        },
      });

      expect(screen.getByText("mlx-community/llama-2")).toBeInTheDocument();
    });

    it("renders description when provided", () => {
      render(ServerCard, {
        props: {
          profile: createMockProfile({ description: "A test description" }),
        },
      });

      expect(screen.getByText("A test description")).toBeInTheDocument();
    });

    it("does not render description when null", () => {
      render(ServerCard, {
        props: { profile: createMockProfile({ description: null }) },
      });

      // Should not have any description element
      expect(screen.queryByText("A test description")).not.toBeInTheDocument();
    });

    it("renders Start button when server is not running", () => {
      render(ServerCard, {
        props: { profile: createMockProfile() },
      });

      expect(
        screen.getByRole("button", { name: /start/i }),
      ).toBeInTheDocument();
    });

    it("does not render Stop button when server is not running", () => {
      render(ServerCard, {
        props: { profile: createMockProfile() },
      });

      expect(
        screen.queryByRole("button", { name: /stop/i }),
      ).not.toBeInTheDocument();
    });

    it("does not render Restart button when server is not running", () => {
      render(ServerCard, {
        props: { profile: createMockProfile() },
      });

      expect(
        screen.queryByRole("button", { name: /restart/i }),
      ).not.toBeInTheDocument();
    });

    it("renders gray health indicator when not running", () => {
      const { container } = render(ServerCard, {
        props: { profile: createMockProfile() },
      });

      const healthIndicator = container.querySelector(".bg-gray-400");
      expect(healthIndicator).toBeInTheDocument();
    });

    it("does not render server metrics when not running", () => {
      render(ServerCard, {
        props: { profile: createMockProfile() },
      });

      // PID, Memory, Uptime should not be visible
      expect(screen.queryByText(/PID:/)).not.toBeInTheDocument();
      expect(screen.queryByText(/Memory:/)).not.toBeInTheDocument();
      expect(screen.queryByText(/Uptime:/)).not.toBeInTheDocument();
    });
  });

  describe("rendering with server (running state)", () => {
    it("renders profile name", () => {
      render(ServerCard, {
        props: {
          profile: createMockProfile({ name: "Running Profile" }),
          server: createMockServer(),
        },
      });

      expect(screen.getByText("Running Profile")).toBeInTheDocument();
    });

    it("renders Stop button when server is running", () => {
      render(ServerCard, {
        props: {
          profile: createMockProfile(),
          server: createMockServer(),
        },
      });

      expect(screen.getByRole("button", { name: /stop/i })).toBeInTheDocument();
    });

    it("renders Restart button when server is running", () => {
      render(ServerCard, {
        props: {
          profile: createMockProfile(),
          server: createMockServer(),
        },
      });

      expect(
        screen.getByRole("button", { name: /restart/i }),
      ).toBeInTheDocument();
    });

    it("does not render Start button when server is running", () => {
      render(ServerCard, {
        props: {
          profile: createMockProfile(),
          server: createMockServer(),
        },
      });

      // Use exact match to avoid matching "Restart" which contains "start"
      expect(
        screen.queryByRole("button", { name: /^Start$/i }),
      ).not.toBeInTheDocument();
    });

    it("renders PID when server is running", () => {
      render(ServerCard, {
        props: {
          profile: createMockProfile(),
          server: createMockServer({ pid: 54321 }),
        },
      });

      expect(screen.getByText("54321")).toBeInTheDocument();
    });

    it("renders memory usage when server is running", () => {
      render(ServerCard, {
        props: {
          profile: createMockProfile(),
          server: createMockServer({ memory_mb: 1024 }),
        },
      });

      // 1024 MB = 1073741824 bytes = 1 GB
      expect(screen.getByText("1 GB")).toBeInTheDocument();
    });

    it("renders uptime when server is running", () => {
      render(ServerCard, {
        props: {
          profile: createMockProfile(),
          server: createMockServer({ uptime_seconds: 3600 }),
        },
      });

      expect(screen.getByText("1h 0m")).toBeInTheDocument();
    });

    it("renders uptime in seconds for short uptimes", () => {
      render(ServerCard, {
        props: {
          profile: createMockProfile(),
          server: createMockServer({ uptime_seconds: 45 }),
        },
      });

      expect(screen.getByText("45s")).toBeInTheDocument();
    });

    it("renders uptime in minutes", () => {
      render(ServerCard, {
        props: {
          profile: createMockProfile(),
          server: createMockServer({ uptime_seconds: 300 }),
        },
      });

      expect(screen.getByText("5m")).toBeInTheDocument();
    });
  });

  describe("health status indicator", () => {
    it("renders green indicator for healthy status", () => {
      const { container } = render(ServerCard, {
        props: {
          profile: createMockProfile(),
          server: createMockServer({ health_status: "healthy" }),
        },
      });

      const healthIndicator = container.querySelector(".bg-green-500");
      expect(healthIndicator).toBeInTheDocument();
    });

    it("renders yellow indicator for starting status", () => {
      const { container } = render(ServerCard, {
        props: {
          profile: createMockProfile(),
          server: createMockServer({ health_status: "starting" }),
        },
      });

      const healthIndicator = container.querySelector(".bg-yellow-500");
      expect(healthIndicator).toBeInTheDocument();
    });

    it("renders red indicator for unhealthy status", () => {
      const { container } = render(ServerCard, {
        props: {
          profile: createMockProfile(),
          server: createMockServer({ health_status: "unhealthy" }),
        },
      });

      const healthIndicator = container.querySelector(".bg-red-500");
      expect(healthIndicator).toBeInTheDocument();
    });

    it("renders gray indicator for stopped status", () => {
      const { container } = render(ServerCard, {
        props: {
          profile: createMockProfile(),
          server: createMockServer({ health_status: "stopped" }),
        },
      });

      const healthIndicator = container.querySelector(".bg-gray-400");
      expect(healthIndicator).toBeInTheDocument();
    });

    it("renders gray indicator for unknown/unexpected status (default case)", () => {
      const { container } = render(ServerCard, {
        props: {
          profile: createMockProfile(),
          // Force an unexpected value to trigger the default case
          server: createMockServer({
            health_status: "unknown" as RunningServer["health_status"],
          }),
        },
      });

      const healthIndicator = container.querySelector(".bg-gray-400");
      expect(healthIndicator).toBeInTheDocument();
    });
  });

  describe("start action", () => {
    it("calls serverStore.start when start button clicked", async () => {
      const user = userEvent.setup();
      render(ServerCard, {
        props: { profile: createMockProfile({ id: 42 }) },
      });

      const startButton = screen.getByRole("button", { name: /start/i });
      await user.click(startButton);

      expect(serverStore.start).toHaveBeenCalledWith(42);
    });

    it("disables start button while loading", async () => {
      const user = userEvent.setup();
      let resolveStart: () => void;
      vi.mocked(serverStore.start).mockImplementation(
        () =>
          new Promise((resolve) => {
            resolveStart = resolve;
          }),
      );

      render(ServerCard, { props: { profile: createMockProfile() } });

      const startButton = screen.getByRole("button", { name: /start/i });
      await user.click(startButton);

      // Button should be disabled while starting
      expect(startButton).toBeDisabled();

      // Resolve the start
      resolveStart!();
      await waitFor(() => {
        expect(startButton).not.toBeDisabled();
      });
    });

    it("displays error message when start fails with Error", async () => {
      const user = userEvent.setup();
      vi.mocked(serverStore.start).mockRejectedValue(
        new Error("Failed to start: port in use"),
      );

      render(ServerCard, {
        props: { profile: createMockProfile({ id: 42 }) },
      });

      const startButton = screen.getByRole("button", { name: /start/i });
      await user.click(startButton);

      await waitFor(() => {
        expect(
          screen.getByText("Failed to start: port in use"),
        ).toBeInTheDocument();
      });
    });

    it("displays generic error message when start fails with non-Error", async () => {
      const user = userEvent.setup();
      vi.mocked(serverStore.start).mockRejectedValue("string error");

      render(ServerCard, {
        props: { profile: createMockProfile({ id: 42 }) },
      });

      const startButton = screen.getByRole("button", { name: /start/i });
      await user.click(startButton);

      await waitFor(() => {
        expect(screen.getByText("Failed to start server")).toBeInTheDocument();
      });
    });
  });

  describe("stop action", () => {
    it("calls serverStore.stop when stop button clicked", async () => {
      const user = userEvent.setup();
      render(ServerCard, {
        props: {
          profile: createMockProfile({ id: 42 }),
          server: createMockServer({ profile_id: 42 }),
        },
      });

      const stopButton = screen.getByRole("button", { name: /stop/i });
      await user.click(stopButton);

      expect(serverStore.stop).toHaveBeenCalledWith(42);
    });

    it("disables stop button while loading", async () => {
      const user = userEvent.setup();
      let resolveStop: () => void;
      vi.mocked(serverStore.stop).mockImplementation(
        () =>
          new Promise((resolve) => {
            resolveStop = resolve;
          }),
      );

      render(ServerCard, {
        props: {
          profile: createMockProfile(),
          server: createMockServer(),
        },
      });

      const stopButton = screen.getByRole("button", { name: /stop/i });
      await user.click(stopButton);

      // Button should be disabled while stopping
      expect(stopButton).toBeDisabled();

      // Resolve
      resolveStop!();
      await waitFor(() => {
        expect(stopButton).not.toBeDisabled();
      });
    });

    it("disables restart button while stopping", async () => {
      const user = userEvent.setup();
      let resolveStop: () => void;
      vi.mocked(serverStore.stop).mockImplementation(
        () =>
          new Promise((resolve) => {
            resolveStop = resolve;
          }),
      );

      render(ServerCard, {
        props: {
          profile: createMockProfile(),
          server: createMockServer(),
        },
      });

      const stopButton = screen.getByRole("button", { name: /stop/i });
      const restartButton = screen.getByRole("button", { name: /restart/i });

      await user.click(stopButton);

      // Restart button should also be disabled
      expect(restartButton).toBeDisabled();

      resolveStop!();
      await waitFor(() => {
        expect(restartButton).not.toBeDisabled();
      });
    });

    it("displays error message when stop fails with Error", async () => {
      const user = userEvent.setup();
      vi.mocked(serverStore.stop).mockRejectedValue(
        new Error("Process not found"),
      );

      render(ServerCard, {
        props: {
          profile: createMockProfile({ id: 42 }),
          server: createMockServer(),
        },
      });

      const stopButton = screen.getByRole("button", { name: /stop/i });
      await user.click(stopButton);

      await waitFor(() => {
        expect(screen.getByText("Process not found")).toBeInTheDocument();
      });
    });

    it("displays generic error message when stop fails with non-Error", async () => {
      const user = userEvent.setup();
      vi.mocked(serverStore.stop).mockRejectedValue("string error");

      render(ServerCard, {
        props: {
          profile: createMockProfile({ id: 42 }),
          server: createMockServer(),
        },
      });

      const stopButton = screen.getByRole("button", { name: /stop/i });
      await user.click(stopButton);

      await waitFor(() => {
        expect(screen.getByText("Failed to stop server")).toBeInTheDocument();
      });
    });
  });

  describe("restart action", () => {
    it("calls serverStore.restart when restart button clicked", async () => {
      const user = userEvent.setup();
      render(ServerCard, {
        props: {
          profile: createMockProfile({ id: 42 }),
          server: createMockServer({ profile_id: 42 }),
        },
      });

      const restartButton = screen.getByRole("button", { name: /restart/i });
      await user.click(restartButton);

      expect(serverStore.restart).toHaveBeenCalledWith(42);
    });

    it("disables restart button while loading", async () => {
      const user = userEvent.setup();
      let resolveRestart: () => void;
      vi.mocked(serverStore.restart).mockImplementation(
        () =>
          new Promise((resolve) => {
            resolveRestart = resolve;
          }),
      );

      render(ServerCard, {
        props: {
          profile: createMockProfile(),
          server: createMockServer(),
        },
      });

      const restartButton = screen.getByRole("button", { name: /restart/i });
      await user.click(restartButton);

      // Button should be disabled while restarting
      expect(restartButton).toBeDisabled();

      // Resolve
      resolveRestart!();
      await waitFor(() => {
        expect(restartButton).not.toBeDisabled();
      });
    });

    it("disables stop button while restarting", async () => {
      const user = userEvent.setup();
      let resolveRestart: () => void;
      vi.mocked(serverStore.restart).mockImplementation(
        () =>
          new Promise((resolve) => {
            resolveRestart = resolve;
          }),
      );

      render(ServerCard, {
        props: {
          profile: createMockProfile(),
          server: createMockServer(),
        },
      });

      const restartButton = screen.getByRole("button", { name: /restart/i });
      const stopButton = screen.getByRole("button", { name: /stop/i });

      await user.click(restartButton);

      // Stop button should also be disabled
      expect(stopButton).toBeDisabled();

      resolveRestart!();
      await waitFor(() => {
        expect(stopButton).not.toBeDisabled();
      });
    });

    it("displays error message when restart fails with Error", async () => {
      const user = userEvent.setup();
      vi.mocked(serverStore.restart).mockRejectedValue(
        new Error("Restart timeout"),
      );

      render(ServerCard, {
        props: {
          profile: createMockProfile({ id: 42 }),
          server: createMockServer(),
        },
      });

      const restartButton = screen.getByRole("button", { name: /restart/i });
      await user.click(restartButton);

      await waitFor(() => {
        expect(screen.getByText("Restart timeout")).toBeInTheDocument();
      });
    });

    it("displays generic error message when restart fails with non-Error", async () => {
      const user = userEvent.setup();
      vi.mocked(serverStore.restart).mockRejectedValue("string error");

      render(ServerCard, {
        props: {
          profile: createMockProfile({ id: 42 }),
          server: createMockServer(),
        },
      });

      const restartButton = screen.getByRole("button", { name: /restart/i });
      await user.click(restartButton);

      await waitFor(() => {
        expect(
          screen.getByText("Failed to restart server"),
        ).toBeInTheDocument();
      });
    });
  });

  describe("edge cases", () => {
    it("handles zero memory values", () => {
      render(ServerCard, {
        props: {
          profile: createMockProfile(),
          server: createMockServer({ memory_mb: 0 }),
        },
      });

      expect(screen.getByText("0 Bytes")).toBeInTheDocument();
    });

    it("handles decimal memory values", () => {
      render(ServerCard, {
        props: {
          profile: createMockProfile(),
          server: createMockServer({ memory_mb: 512.75 }),
        },
      });

      // 512.75 MB formatted with 2 decimals
      expect(screen.getByText(/512\.75 MB/)).toBeInTheDocument();
    });

    it("handles zero uptime", () => {
      render(ServerCard, {
        props: {
          profile: createMockProfile(),
          server: createMockServer({ uptime_seconds: 0 }),
        },
      });

      expect(screen.getByText("0s")).toBeInTheDocument();
    });

    it("handles long profile names", () => {
      render(ServerCard, {
        props: {
          profile: createMockProfile({
            name: "Very Long Profile Name That Should Be Displayed",
          }),
        },
      });

      expect(
        screen.getByText("Very Long Profile Name That Should Be Displayed"),
      ).toBeInTheDocument();
    });

    it("handles long model paths", () => {
      render(ServerCard, {
        props: {
          profile: createMockProfile({
            model_path:
              "mlx-community/very-long-model-name-that-is-really-quite-long",
          }),
        },
      });

      expect(
        screen.getByText(
          "mlx-community/very-long-model-name-that-is-really-quite-long",
        ),
      ).toBeInTheDocument();
    });

    it("handles large PID numbers", () => {
      render(ServerCard, {
        props: {
          profile: createMockProfile(),
          server: createMockServer({ pid: 999999 }),
        },
      });

      expect(screen.getByText("999999")).toBeInTheDocument();
    });

    it("clears error state on successful action", async () => {
      const user = userEvent.setup();

      // First, cause an error
      vi.mocked(serverStore.start).mockRejectedValueOnce(
        new Error("First error"),
      );

      render(ServerCard, {
        props: { profile: createMockProfile({ id: 42 }) },
      });

      const startButton = screen.getByRole("button", { name: /start/i });
      await user.click(startButton);

      await waitFor(() => {
        expect(screen.getByText("First error")).toBeInTheDocument();
      });

      // Now reset mock to succeed
      vi.mocked(serverStore.start).mockResolvedValueOnce(undefined);

      // Click again
      await user.click(startButton);

      // Error should be cleared
      await waitFor(() => {
        expect(screen.queryByText("First error")).not.toBeInTheDocument();
      });
    });
  });

  describe("loading state interactions", () => {
    it("prevents multiple simultaneous start calls", async () => {
      const user = userEvent.setup();
      let resolveStart: () => void;
      vi.mocked(serverStore.start).mockImplementation(
        () =>
          new Promise((resolve) => {
            resolveStart = resolve;
          }),
      );

      render(ServerCard, { props: { profile: createMockProfile() } });

      const startButton = screen.getByRole("button", { name: /start/i });

      // First click
      await user.click(startButton);

      // Verify start was called once
      expect(serverStore.start).toHaveBeenCalledTimes(1);

      // Button should be disabled, clicking again shouldn't trigger another call
      expect(startButton).toBeDisabled();

      // Resolve
      resolveStart!();
      await waitFor(() => {
        expect(startButton).not.toBeDisabled();
      });
    });

    it("prevents multiple simultaneous stop calls", async () => {
      const user = userEvent.setup();
      let resolveStop: () => void;
      vi.mocked(serverStore.stop).mockImplementation(
        () =>
          new Promise((resolve) => {
            resolveStop = resolve;
          }),
      );

      render(ServerCard, {
        props: {
          profile: createMockProfile(),
          server: createMockServer(),
        },
      });

      const stopButton = screen.getByRole("button", { name: /stop/i });

      // First click
      await user.click(stopButton);

      // Verify stop was called once
      expect(serverStore.stop).toHaveBeenCalledTimes(1);

      // Button should be disabled
      expect(stopButton).toBeDisabled();

      // Resolve
      resolveStop!();
      await waitFor(() => {
        expect(stopButton).not.toBeDisabled();
      });
    });
  });
});
