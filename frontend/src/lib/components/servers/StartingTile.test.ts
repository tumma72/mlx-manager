import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/svelte";
import userEvent from "@testing-library/user-event";
import type { ServerProfile } from "$api";
import type { FailedServer } from "$stores/servers.svelte";

// Mock the stores module with inline mock functions
vi.mock("$stores", () => {
  return {
    serverStore: {
      isStarting: vi.fn().mockReturnValue(true),
      isFailed: vi.fn().mockReturnValue(false),
      getFailure: vi.fn().mockReturnValue(undefined),
      startProfilePolling: vi.fn().mockReturnValue(true),
      stopProfilePolling: vi.fn(),
      isProfilePolling: vi.fn().mockReturnValue(false),
      markStartupFailed: vi.fn(),
      markStartupSuccess: vi.fn(),
      toggleDetailsOpen: vi.fn(),
      clearFailure: vi.fn(),
      stop: vi.fn().mockResolvedValue(undefined),
      start: vi.fn().mockResolvedValue(undefined),
    },
  };
});

// Mock the API module
vi.mock("$api", () => ({
  servers: {
    status: vi.fn().mockResolvedValue({ running: false, failed: false }),
  },
}));

// Import after mocking
import StartingTile from "./StartingTile.svelte";
import { serverStore } from "$stores";

// Helper to create mock profile
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

// Helper to create mock failure
function createMockFailure(
  overrides: Partial<FailedServer> = {}
): FailedServer {
  return {
    error: "Server failed to start",
    details: null,
    detailsOpen: false,
    ...overrides,
  };
}

describe("StartingTile", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    vi.clearAllMocks();

    // Reset default mock implementations
    vi.mocked(serverStore.isStarting).mockReturnValue(true);
    vi.mocked(serverStore.isFailed).mockReturnValue(false);
    vi.mocked(serverStore.getFailure).mockReturnValue(undefined);
    vi.mocked(serverStore.startProfilePolling).mockReturnValue(true);
    vi.mocked(serverStore.isProfilePolling).mockReturnValue(false);
    vi.mocked(serverStore.stop).mockResolvedValue(undefined);
    vi.mocked(serverStore.start).mockResolvedValue(undefined);
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  describe("starting state rendering", () => {
    it("renders profile name", () => {
      render(StartingTile, {
        props: { profile: createMockProfile({ name: "My Profile" }) },
      });

      expect(screen.getByText("My Profile")).toBeInTheDocument();
    });

    it("renders Starting badge when starting", () => {
      render(StartingTile, { props: { profile: createMockProfile() } });

      expect(screen.getByText("Starting")).toBeInTheDocument();
    });

    it("renders model path", () => {
      render(StartingTile, {
        props: {
          profile: createMockProfile({ model_path: "mlx-community/llama-2" }),
        },
      });

      expect(screen.getByText("mlx-community/llama-2")).toBeInTheDocument();
    });

    it("renders loading message", () => {
      render(StartingTile, { props: { profile: createMockProfile() } });

      expect(
        screen.getByText("Loading model... this may take a minute")
      ).toBeInTheDocument();
    });

    it("renders cancel button", () => {
      render(StartingTile, { props: { profile: createMockProfile() } });

      expect(screen.getByText("Cancel")).toBeInTheDocument();
    });
  });

  describe("failed state rendering", () => {
    beforeEach(() => {
      vi.mocked(serverStore.isStarting).mockReturnValue(false);
      vi.mocked(serverStore.isFailed).mockReturnValue(true);
      vi.mocked(serverStore.getFailure).mockReturnValue(createMockFailure());
    });

    it("renders Error badge when failed", () => {
      render(StartingTile, { props: { profile: createMockProfile() } });

      expect(screen.getByText("Error")).toBeInTheDocument();
    });

    it("renders error message", () => {
      vi.mocked(serverStore.getFailure).mockReturnValue(
        createMockFailure({ error: "Connection refused" })
      );

      render(StartingTile, { props: { profile: createMockProfile() } });

      expect(screen.getByText("Connection refused")).toBeInTheDocument();
    });

    it("renders Retry button when failed", () => {
      render(StartingTile, { props: { profile: createMockProfile() } });

      expect(screen.getByText("Retry")).toBeInTheDocument();
    });

    it("renders Dismiss button when failed", () => {
      render(StartingTile, { props: { profile: createMockProfile() } });

      expect(screen.getByText("Dismiss")).toBeInTheDocument();
    });

    it("does not render loading message when failed", () => {
      render(StartingTile, { props: { profile: createMockProfile() } });

      expect(
        screen.queryByText("Loading model... this may take a minute")
      ).not.toBeInTheDocument();
    });
  });

  describe("error details", () => {
    beforeEach(() => {
      vi.mocked(serverStore.isStarting).mockReturnValue(false);
      vi.mocked(serverStore.isFailed).mockReturnValue(true);
    });

    it("renders error details toggle when details available", () => {
      vi.mocked(serverStore.getFailure).mockReturnValue(
        createMockFailure({
          error: "Server crashed",
          details: "Traceback (most recent call last):\n  Error details here",
        })
      );

      render(StartingTile, { props: { profile: createMockProfile() } });

      expect(screen.getByText("Show server log")).toBeInTheDocument();
    });

    it("does not render details toggle when no details", () => {
      vi.mocked(serverStore.getFailure).mockReturnValue(
        createMockFailure({ details: null })
      );

      render(StartingTile, { props: { profile: createMockProfile() } });

      expect(screen.queryByText("Show server log")).not.toBeInTheDocument();
    });

    it("shows details content when expanded", () => {
      vi.mocked(serverStore.getFailure).mockReturnValue(
        createMockFailure({
          error: "Server crashed",
          details: "Error stack trace content",
          detailsOpen: true,
        })
      );

      render(StartingTile, { props: { profile: createMockProfile() } });

      expect(screen.getByText("Error stack trace content")).toBeInTheDocument();
    });
  });

  describe("cancel action", () => {
    it("calls serverStore.stop when cancel clicked", async () => {
      const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
      render(StartingTile, {
        props: { profile: createMockProfile({ id: 42 }) },
      });

      const cancelButton = screen.getByText("Cancel");
      await user.click(cancelButton);

      expect(serverStore.stop).toHaveBeenCalledWith(42);
    });

    it("stops profile polling when cancel clicked", async () => {
      const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
      render(StartingTile, {
        props: { profile: createMockProfile({ id: 42 }) },
      });

      const cancelButton = screen.getByText("Cancel");
      await user.click(cancelButton);

      expect(serverStore.stopProfilePolling).toHaveBeenCalledWith(42);
    });
  });

  describe("retry action", () => {
    beforeEach(() => {
      vi.mocked(serverStore.isStarting).mockReturnValue(false);
      vi.mocked(serverStore.isFailed).mockReturnValue(true);
      vi.mocked(serverStore.getFailure).mockReturnValue(createMockFailure());
    });

    it("clears failure and starts server when retry clicked", async () => {
      const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
      render(StartingTile, {
        props: { profile: createMockProfile({ id: 42 }) },
      });

      const retryButton = screen.getByText("Retry");
      await user.click(retryButton);

      expect(serverStore.clearFailure).toHaveBeenCalledWith(42);
      expect(serverStore.start).toHaveBeenCalledWith(42);
    });

    it("starts profile polling when retry clicked", async () => {
      const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
      render(StartingTile, {
        props: { profile: createMockProfile({ id: 42 }) },
      });

      const retryButton = screen.getByText("Retry");
      await user.click(retryButton);

      expect(serverStore.startProfilePolling).toHaveBeenCalledWith(42);
    });

    it("marks failure if start throws error", async () => {
      const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
      vi.mocked(serverStore.start).mockRejectedValue(new Error("Start failed"));

      render(StartingTile, {
        props: { profile: createMockProfile({ id: 42 }) },
      });

      const retryButton = screen.getByText("Retry");
      await user.click(retryButton);

      await waitFor(() => {
        expect(serverStore.markStartupFailed).toHaveBeenCalledWith(
          42,
          "Start failed"
        );
      });
    });
  });

  describe("dismiss action", () => {
    beforeEach(() => {
      vi.mocked(serverStore.isStarting).mockReturnValue(false);
      vi.mocked(serverStore.isFailed).mockReturnValue(true);
      vi.mocked(serverStore.getFailure).mockReturnValue(createMockFailure());
    });

    it("clears failure when dismiss clicked", async () => {
      const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
      render(StartingTile, {
        props: { profile: createMockProfile({ id: 42 }) },
      });

      const dismissButton = screen.getByText("Dismiss");
      await user.click(dismissButton);

      expect(serverStore.clearFailure).toHaveBeenCalledWith(42);
    });
  });

  describe("copy to clipboard", () => {
    beforeEach(() => {
      vi.mocked(serverStore.isStarting).mockReturnValue(false);
      vi.mocked(serverStore.isFailed).mockReturnValue(true);
    });

    it("renders copy button when details are expanded", () => {
      vi.mocked(serverStore.getFailure).mockReturnValue(
        createMockFailure({
          error: "Server crashed",
          details: "Full error details here",
          detailsOpen: true,
        })
      );

      render(StartingTile, { props: { profile: createMockProfile() } });

      const copyButton = screen.getByTitle("Copy to clipboard");
      expect(copyButton).toBeInTheDocument();
    });

    it("does not render copy button when no details", () => {
      vi.mocked(serverStore.getFailure).mockReturnValue(
        createMockFailure({
          error: "Server crashed",
          details: null,
          detailsOpen: false,
        })
      );

      render(StartingTile, { props: { profile: createMockProfile() } });

      expect(screen.queryByTitle("Copy to clipboard")).not.toBeInTheDocument();
    });
  });

  describe("details toggle", () => {
    beforeEach(() => {
      vi.mocked(serverStore.isStarting).mockReturnValue(false);
      vi.mocked(serverStore.isFailed).mockReturnValue(true);
    });

    it("shows details element with correct open state when detailsOpen is true", () => {
      vi.mocked(serverStore.getFailure).mockReturnValue(
        createMockFailure({
          error: "Server crashed",
          details: "Error details",
          detailsOpen: true,
        })
      );

      const { container } = render(StartingTile, {
        props: { profile: createMockProfile({ id: 42 }) },
      });

      const details = container.querySelector("details");
      expect(details).toHaveAttribute("open");
    });

    it("shows details element without open state when detailsOpen is false", () => {
      vi.mocked(serverStore.getFailure).mockReturnValue(
        createMockFailure({
          error: "Server crashed",
          details: "Error details",
          detailsOpen: false,
        })
      );

      const { container } = render(StartingTile, {
        props: { profile: createMockProfile({ id: 42 }) },
      });

      const details = container.querySelector("details");
      expect(details).not.toHaveAttribute("open");
    });
  });

  describe("polling behavior", () => {
    it("starts polling when component mounts for starting profile", () => {
      vi.mocked(serverStore.isStarting).mockReturnValue(true);
      vi.mocked(serverStore.isProfilePolling).mockReturnValue(false);
      vi.mocked(serverStore.startProfilePolling).mockReturnValue(true);

      render(StartingTile, {
        props: { profile: createMockProfile({ id: 42 }) },
      });

      expect(serverStore.startProfilePolling).toHaveBeenCalledWith(42);
    });

    it("does not start polling if already polling", () => {
      vi.mocked(serverStore.isStarting).mockReturnValue(true);
      vi.mocked(serverStore.isProfilePolling).mockReturnValue(true);

      render(StartingTile, {
        props: { profile: createMockProfile({ id: 42 }) },
      });

      expect(serverStore.startProfilePolling).not.toHaveBeenCalled();
    });
  });
});
