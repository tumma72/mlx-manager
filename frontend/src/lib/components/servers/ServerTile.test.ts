import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/svelte";
import userEvent from "@testing-library/user-event";
import type { RunningServer } from "$api";

// Mock the serverStore
vi.mock("$stores", () => ({
  serverStore: {
    stop: vi.fn().mockResolvedValue(undefined),
    restart: vi.fn().mockResolvedValue(undefined),
    isRestarting: vi.fn().mockReturnValue(false),
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
import ServerTile from "./ServerTile.svelte";
import { serverStore } from "$stores";
import { goto } from "$app/navigation";

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

describe("ServerTile", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe("rendering", () => {
    it("renders server name", () => {
      render(ServerTile, {
        props: { server: createMockServer({ profile_name: "My Server" }) },
      });

      expect(screen.getByText("My Server")).toBeInTheDocument();
    });

    it("renders Running badge", () => {
      render(ServerTile, { props: { server: createMockServer() } });

      expect(screen.getByText("Running")).toBeInTheDocument();
    });

    it("renders port number", () => {
      render(ServerTile, {
        props: { server: createMockServer({ port: 8080 }) },
      });

      expect(screen.getByText("Port 8080")).toBeInTheDocument();
    });

    it("renders PID", () => {
      render(ServerTile, {
        props: { server: createMockServer({ pid: 54321 }) },
      });

      expect(screen.getByText("PID: 54321")).toBeInTheDocument();
    });

    it("renders memory with formatBytes", () => {
      render(ServerTile, {
        props: { server: createMockServer({ memory_mb: 1024 }) },
      });

      // 1024 MB = 1073741824 bytes = 1 GB
      expect(screen.getByText("Memory: 1 GB")).toBeInTheDocument();
    });

    it("renders uptime", () => {
      render(ServerTile, {
        props: { server: createMockServer({ uptime_seconds: 3600 }) },
      });

      expect(screen.getByText("Uptime: 1h 0m")).toBeInTheDocument();
    });

    it("renders uptime in seconds for short uptimes", () => {
      render(ServerTile, {
        props: { server: createMockServer({ uptime_seconds: 45 }) },
      });

      expect(screen.getByText("Uptime: 45s")).toBeInTheDocument();
    });

    it("renders uptime in minutes", () => {
      render(ServerTile, {
        props: { server: createMockServer({ uptime_seconds: 300 }) },
      });

      expect(screen.getByText("Uptime: 5m")).toBeInTheDocument();
    });
  });

  describe("metric gauges", () => {
    it("renders memory gauge with correct percentage", () => {
      render(ServerTile, {
        props: { server: createMockServer({ memory_percent: 75 }) },
      });

      expect(screen.getByText("75%")).toBeInTheDocument();
      expect(screen.getByText("Memory")).toBeInTheDocument();
    });

    it("renders CPU gauge with correct percentage", () => {
      render(ServerTile, {
        props: { server: createMockServer({ cpu_percent: 50 }) },
      });

      expect(screen.getByText("50%")).toBeInTheDocument();
      expect(screen.getByText("CPU")).toBeInTheDocument();
    });

    it("renders both memory and CPU gauges", () => {
      render(ServerTile, {
        props: {
          server: createMockServer({ memory_percent: 30, cpu_percent: 20 }),
        },
      });

      expect(screen.getByText("30%")).toBeInTheDocument();
      expect(screen.getByText("20%")).toBeInTheDocument();
    });
  });

  describe("action buttons", () => {
    it("renders chat button", () => {
      render(ServerTile, { props: { server: createMockServer() } });

      const chatButton = screen.getByTitle("Open Chat");
      expect(chatButton).toBeInTheDocument();
    });

    it("renders restart button", () => {
      render(ServerTile, { props: { server: createMockServer() } });

      const restartButton = screen.getByTitle("Restart Server");
      expect(restartButton).toBeInTheDocument();
    });

    it("renders stop button", () => {
      render(ServerTile, { props: { server: createMockServer() } });

      const stopButton = screen.getByTitle("Stop Server");
      expect(stopButton).toBeInTheDocument();
    });
  });

  describe("stop action", () => {
    it("calls serverStore.stop when stop button clicked", async () => {
      const user = userEvent.setup();
      render(ServerTile, {
        props: { server: createMockServer({ profile_id: 42 }) },
      });

      const stopButton = screen.getByTitle("Stop Server");
      await user.click(stopButton);

      expect(serverStore.stop).toHaveBeenCalledWith(42);
    });

    it("disables stop button while stopping", async () => {
      const user = userEvent.setup();
      let resolveStop: () => void;
      vi.mocked(serverStore.stop).mockImplementation(
        () =>
          new Promise((resolve) => {
            resolveStop = resolve;
          }),
      );

      render(ServerTile, { props: { server: createMockServer() } });

      const stopButton = screen.getByTitle("Stop Server");
      await user.click(stopButton);

      // Button should be disabled while stopping
      expect(stopButton).toBeDisabled();

      // Resolve the stop
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

      render(ServerTile, { props: { server: createMockServer() } });

      const stopButton = screen.getByTitle("Stop Server");
      const restartButton = screen.getByTitle("Restart Server");

      await user.click(stopButton);

      // Restart button should also be disabled
      expect(restartButton).toBeDisabled();

      resolveStop!();
      await waitFor(() => {
        expect(restartButton).not.toBeDisabled();
      });
    });
  });

  describe("restart action", () => {
    it("calls serverStore.restart when restart button clicked", async () => {
      const user = userEvent.setup();
      render(ServerTile, {
        props: { server: createMockServer({ profile_id: 42 }) },
      });

      const restartButton = screen.getByTitle("Restart Server");
      await user.click(restartButton);

      expect(serverStore.restart).toHaveBeenCalledWith(42);
    });

    it("disables restart button while restarting", async () => {
      const user = userEvent.setup();
      let resolveRestart: () => void;
      vi.mocked(serverStore.restart).mockImplementation(
        () =>
          new Promise((resolve) => {
            resolveRestart = resolve;
          }),
      );

      render(ServerTile, { props: { server: createMockServer() } });

      const restartButton = screen.getByTitle("Restart Server");
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

      render(ServerTile, { props: { server: createMockServer() } });

      const restartButton = screen.getByTitle("Restart Server");
      const stopButton = screen.getByTitle("Stop Server");

      await user.click(restartButton);

      // Stop button should also be disabled
      expect(stopButton).toBeDisabled();

      resolveRestart!();
      await waitFor(() => {
        expect(stopButton).not.toBeDisabled();
      });
    });
  });

  describe("chat action", () => {
    it("navigates to chat page with profile id", async () => {
      const user = userEvent.setup();
      render(ServerTile, {
        props: { server: createMockServer({ profile_id: 42 }) },
      });

      const chatButton = screen.getByTitle("Open Chat");
      await user.click(chatButton);

      expect(goto).toHaveBeenCalledWith("/chat?profile=42");
    });
  });

  describe("edge cases", () => {
    it("handles zero memory values", () => {
      render(ServerTile, {
        props: {
          server: createMockServer({ memory_mb: 0, memory_percent: 0 }),
        },
      });

      expect(screen.getByText("Memory: 0 Bytes")).toBeInTheDocument();
      expect(screen.getByText("0%")).toBeInTheDocument();
    });

    it("handles zero CPU values", () => {
      render(ServerTile, {
        props: { server: createMockServer({ cpu_percent: 0 }) },
      });

      expect(screen.getByText("CPU")).toBeInTheDocument();
    });

    it("handles high CPU values", () => {
      render(ServerTile, {
        props: { server: createMockServer({ cpu_percent: 100 }) },
      });

      expect(screen.getByText("100%")).toBeInTheDocument();
    });

    it("handles decimal memory values", () => {
      render(ServerTile, {
        props: { server: createMockServer({ memory_mb: 512.7 }) },
      });

      // 512.7 MB = 537559654.4 bytes ≈ 512.7 MB (formatted)
      expect(screen.getByText(/Memory: 512\.7 MB/)).toBeInTheDocument();
    });

    it("handles long server names", () => {
      render(ServerTile, {
        props: {
          server: createMockServer({
            profile_name: "Very Long Server Name That Should Be Truncated",
          }),
        },
      });

      expect(
        screen.getByText("Very Long Server Name That Should Be Truncated"),
      ).toBeInTheDocument();
    });
  });
});
