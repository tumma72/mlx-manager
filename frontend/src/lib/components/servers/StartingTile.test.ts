import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, waitFor, fireEvent } from "@testing-library/svelte";
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
    vi.clearAllTimers();
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

    it("marks failure with error message when stop throws Error (line 165)", async () => {
      const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
      vi.mocked(serverStore.stop).mockRejectedValue(new Error("Connection failed"));

      render(StartingTile, {
        props: { profile: createMockProfile({ id: 42 }) },
      });

      const cancelButton = screen.getByText("Cancel");
      await user.click(cancelButton);

      await waitFor(() => {
        expect(serverStore.markStartupFailed).toHaveBeenCalledWith(
          42,
          "Connection failed"
        );
      });
    });

    it("marks failure with generic message when stop throws non-Error (line 165)", async () => {
      const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
      vi.mocked(serverStore.stop).mockRejectedValue("string error");

      render(StartingTile, {
        props: { profile: createMockProfile({ id: 42 }) },
      });

      const cancelButton = screen.getByText("Cancel");
      await user.click(cancelButton);

      await waitFor(() => {
        expect(serverStore.markStartupFailed).toHaveBeenCalledWith(
          42,
          "Failed to stop server"
        );
      });
    });

    it("clears poll timeout when cancelling while polling (lines 156-157)", async () => {
      // This test verifies the clearTimeout path in handleStop
      const clearTimeoutSpy = vi.spyOn(globalThis, "clearTimeout");
      const { servers: serversApi } = await import("$api");

      // Mock status to return running: true to keep polling alive
      vi.mocked(serversApi.status).mockResolvedValue({
        running: true,
        failed: false,
      } as never);

      // Mock fetch to fail so polling schedules next poll (sets pollTimeoutId)
      const originalFetch = globalThis.fetch;
      globalThis.fetch = vi.fn().mockRejectedValue(new Error("Model not ready"));

      // Enable polling
      vi.mocked(serverStore.isProfilePolling).mockReturnValue(false);

      const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
      render(StartingTile, {
        props: {
          profile: createMockProfile({ id: 42, host: "127.0.0.1", port: 10240 }),
        },
      });

      // Wait for first poll to complete and schedule next poll (pollTimeoutId is now set)
      await vi.advanceTimersByTimeAsync(100);

      // Clear spy before cancel
      clearTimeoutSpy.mockClear();

      // Click cancel - this should clear the poll timeout (lines 156-157)
      const cancelButton = screen.getByText("Cancel");
      await user.click(cancelButton);

      expect(clearTimeoutSpy).toHaveBeenCalled();
      expect(serverStore.stop).toHaveBeenCalledWith(42);

      globalThis.fetch = originalFetch;
      clearTimeoutSpy.mockRestore();
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

    it("marks failure with generic message if start throws non-Error", async () => {
      const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
      vi.mocked(serverStore.start).mockRejectedValue("string error");

      render(StartingTile, {
        props: { profile: createMockProfile({ id: 42 }) },
      });

      const retryButton = screen.getByText("Retry");
      await user.click(retryButton);

      await waitFor(() => {
        expect(serverStore.markStartupFailed).toHaveBeenCalledWith(
          42,
          "Failed to start server"
        );
      });
    });

    it("does not start server if startProfilePolling returns false", async () => {
      const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
      vi.mocked(serverStore.startProfilePolling).mockReturnValue(false);

      render(StartingTile, {
        props: { profile: createMockProfile({ id: 42 }) },
      });

      const retryButton = screen.getByText("Retry");
      await user.click(retryButton);

      expect(serverStore.clearFailure).toHaveBeenCalledWith(42);
      expect(serverStore.start).not.toHaveBeenCalled();
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

    it("shows check icon after successful copy (line 276)", async () => {
      const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
      const writeTextMock = vi.fn().mockResolvedValue(undefined);

      // Mock clipboard
      const originalClipboard = navigator.clipboard;
      Object.defineProperty(navigator, "clipboard", {
        value: { writeText: writeTextMock },
        writable: true,
        configurable: true,
      });

      vi.mocked(serverStore.getFailure).mockReturnValue(
        createMockFailure({
          error: "Server crashed",
          details: "Detailed error log content",
          detailsOpen: true,
        })
      );

      const { container } = render(StartingTile, {
        props: { profile: createMockProfile() },
      });

      const copyButton = screen.getByTitle("Copy to clipboard");
      await user.click(copyButton);

      expect(writeTextMock).toHaveBeenCalledWith("Detailed error log content");

      // Check that the Check icon is displayed (green color indicates success)
      await waitFor(() => {
        const greenIcon = container.querySelector(
          ".text-green-600, .text-green-400"
        );
        expect(greenIcon).toBeInTheDocument();
      });

      // Restore clipboard
      Object.defineProperty(navigator, "clipboard", {
        value: originalClipboard,
        writable: true,
        configurable: true,
      });
    });

    it("handles clipboard write failure gracefully", async () => {
      const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
      const writeTextMock = vi.fn().mockRejectedValue(new Error("Permission denied"));
      const consoleSpy = vi.spyOn(console, "error").mockImplementation(() => {});

      // Mock clipboard
      const originalClipboard = navigator.clipboard;
      Object.defineProperty(navigator, "clipboard", {
        value: { writeText: writeTextMock },
        writable: true,
        configurable: true,
      });

      vi.mocked(serverStore.getFailure).mockReturnValue(
        createMockFailure({
          error: "Server crashed",
          details: "Error details",
          detailsOpen: true,
        })
      );

      render(StartingTile, {
        props: { profile: createMockProfile() },
      });

      const copyButton = screen.getByTitle("Copy to clipboard");
      await user.click(copyButton);

      await waitFor(() => {
        expect(consoleSpy).toHaveBeenCalledWith(
          "Failed to copy to clipboard:",
          expect.any(Error)
        );
      });

      // Restore
      Object.defineProperty(navigator, "clipboard", {
        value: originalClipboard,
        writable: true,
        configurable: true,
      });
      consoleSpy.mockRestore();
    });

    it("resets copySuccess after 2 seconds (line 62)", async () => {
      const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
      const writeTextMock = vi.fn().mockResolvedValue(undefined);

      // Mock clipboard
      const originalClipboard = navigator.clipboard;
      Object.defineProperty(navigator, "clipboard", {
        value: { writeText: writeTextMock },
        writable: true,
        configurable: true,
      });

      vi.mocked(serverStore.getFailure).mockReturnValue(
        createMockFailure({
          error: "Server crashed",
          details: "Error details to copy",
          detailsOpen: true,
        })
      );

      const { container } = render(StartingTile, {
        props: { profile: createMockProfile() },
      });

      const copyButton = screen.getByTitle("Copy to clipboard");
      await user.click(copyButton);

      // Verify success state (green icon)
      await waitFor(() => {
        const greenIcon = container.querySelector(".text-green-600");
        expect(greenIcon).toBeInTheDocument();
      });

      // Advance timers by 2000ms to trigger the timeout (line 62)
      await vi.advanceTimersByTimeAsync(2000);

      // Verify success state is cleared (clipboard icon returns)
      await waitFor(() => {
        const clipboardIcon = container.querySelector(".text-red-600");
        expect(clipboardIcon).toBeInTheDocument();
      });

      // Restore clipboard
      Object.defineProperty(navigator, "clipboard", {
        value: originalClipboard,
        writable: true,
        configurable: true,
      });
    });
  });

  describe("details toggle", () => {
    beforeEach(() => {
      vi.mocked(serverStore.isStarting).mockReturnValue(false);
      vi.mocked(serverStore.isFailed).mockReturnValue(true);
    });

    it("syncs localDetailsOpen from store failure on mount (lines 35-41)", async () => {
      // This test verifies the $effect sync logic that sets localDetailsOpen
      // from the store's detailsOpen value when a failure is first detected
      vi.mocked(serverStore.getFailure).mockReturnValue(
        createMockFailure({
          error: "Server crashed",
          details: "Error details here",
          detailsOpen: true, // Store says details should be open
        })
      );

      const { container } = render(StartingTile, {
        props: { profile: createMockProfile({ id: 42 }) },
      });

      // The details element should be open immediately (synced from store)
      const details = container.querySelector("details");
      expect(details).toHaveAttribute("open");

      // Wait for the isUpdatingDetails flag to be cleared by requestAnimationFrame (line 40)
      await vi.advanceTimersByTimeAsync(16);

      // Verify toggleDetailsOpen was NOT called during sync (because isUpdatingDetails was true)
      expect(serverStore.toggleDetailsOpen).not.toHaveBeenCalled();
    });

    it("clears localDetailsOpen when failure is cleared (lines 42-45)", async () => {
      // First render with a failure
      vi.mocked(serverStore.getFailure).mockReturnValue(
        createMockFailure({
          error: "Server crashed",
          details: "Error details",
          detailsOpen: true,
        })
      );

      const { container, rerender } = render(StartingTile, {
        props: { profile: createMockProfile({ id: 42 }) },
      });

      // Verify details is open
      let details = container.querySelector("details");
      expect(details).toHaveAttribute("open");

      // Now clear the failure
      vi.mocked(serverStore.getFailure).mockReturnValue(undefined);
      vi.mocked(serverStore.isFailed).mockReturnValue(false);

      // Re-render to trigger the $effect
      await rerender({ profile: createMockProfile({ id: 42 }) });

      // The failure section should be gone entirely
      details = container.querySelector("details");
      expect(details).not.toBeInTheDocument();
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

    it("calls toggleDetailsOpen when details is toggled (line 50)", async () => {
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

      // Wait for the isUpdatingDetails flag to be cleared by requestAnimationFrame
      // The component sets isUpdatingDetails=true in $effect, then clears it via RAF
      await vi.advanceTimersByTimeAsync(16); // RAF typically fires at ~16ms (60fps)

      // Find the details element and dispatch toggle event directly
      // (jsdom doesn't properly simulate click->toggle on summary elements)
      const details = container.querySelector("details");
      expect(details).toBeInTheDocument();

      // Dispatch the toggle event on the details element
      await fireEvent(details!, new Event("toggle"));

      expect(serverStore.toggleDetailsOpen).toHaveBeenCalledWith(42);
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

    // ============================================================================
    // COVERAGE NOTE: Uncovered polling branches (lines 93, 99-105, 114, 132-137, 147)
    // ============================================================================
    // The polling branches cannot be unit tested due to Svelte 5's $effect system
    // interacting with vitest's fake timers, causing memory exhaustion and worker
    // crashes. These scenarios are covered by:
    // 1. Manual testing with real server instances
    // 2. Integration/E2E testing
    // ============================================================================
  });

  // NOTE: $effect detailsOpen sync (line 37) - See polling behavior note above
  // The $effect that syncs localDetailsOpen is indirectly covered by existing
  // "shows details element with correct open state" tests and E2E tests
});
