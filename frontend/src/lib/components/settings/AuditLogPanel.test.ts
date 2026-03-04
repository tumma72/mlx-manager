import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/svelte";
import userEvent from "@testing-library/user-event";
import AuditLogPanel from "./AuditLogPanel.svelte";
import type { AuditLog, AuditStats } from "$lib/api/types";

// Track WebSocket instances for assertions
let mockWsInstances: MockWebSocket[] = [];

class MockWebSocket {
  url: string;
  onopen: ((ev: Event) => void) | null = null;
  onclose: ((ev: CloseEvent) => void) | null = null;
  onmessage: ((ev: MessageEvent) => void) | null = null;
  onerror: ((ev: Event) => void) | null = null;
  close = vi.fn();

  constructor(url: string) {
    this.url = url;
    mockWsInstances.push(this);
  }

  simulateOpen() {
    this.onopen?.(new Event("open"));
  }

  simulateMessage(data: unknown) {
    this.onmessage?.(new MessageEvent("message", { data: JSON.stringify(data) }));
  }

  simulateClose() {
    this.onclose?.({} as CloseEvent);
  }

  simulateError() {
    this.onerror?.(new Event("error"));
  }
}

// Install mock WebSocket globally
vi.stubGlobal("WebSocket", MockWebSocket);

// Mock API
vi.mock("$lib/api/client", () => ({
  auditLogs: {
    list: vi.fn(),
    stats: vi.fn(),
    exportUrl: vi.fn(),
    createWebSocket: vi.fn(),
  },
}));

import { auditLogs } from "$lib/api/client";

function createMockLog(overrides: Partial<AuditLog> = {}): AuditLog {
  return {
    id: 1,
    request_id: "req-abc-123",
    timestamp: "2025-01-15T10:30:00Z",
    model: "mlx-community/Qwen3-0.6B-4bit-DWQ",
    backend_type: "local",
    endpoint: "/v1/chat/completions",
    duration_ms: 1500,
    status: "success",
    prompt_tokens: 50,
    completion_tokens: 100,
    total_tokens: 150,
    error_type: null,
    error_message: null,
    ...overrides,
  };
}

function createMockStats(overrides: Partial<AuditStats> = {}): AuditStats {
  return {
    total_requests: 100,
    by_status: { success: 90, error: 8, timeout: 2 },
    by_backend: { local: 70, openai: 20, anthropic: 10 },
    unique_models: 5,
    ...overrides,
  };
}

describe("AuditLogPanel", () => {
  let mockWindowOpen: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    mockWsInstances = [];
    mockWindowOpen = vi.fn();
    vi.stubGlobal("open", mockWindowOpen);
    // Also mock on window object for the component's window.open call
    Object.defineProperty(window, "open", { value: mockWindowOpen, writable: true });

    vi.mocked(auditLogs.list).mockResolvedValue([]);
    vi.mocked(auditLogs.stats).mockResolvedValue(createMockStats());
    vi.mocked(auditLogs.exportUrl).mockReturnValue("/api/system/audit-logs/export?format=jsonl");
    vi.mocked(auditLogs.createWebSocket).mockReturnValue(new MockWebSocket("ws://localhost/api/system/ws/audit-logs") as unknown as WebSocket);

    // Suppress console.error for expected failures
    vi.spyOn(console, "error").mockImplementation(() => {});
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.restoreAllMocks();
  });

  describe("initial loading", () => {
    it("calls list and stats on mount", async () => {
      render(AuditLogPanel);

      await waitFor(() => {
        expect(auditLogs.list).toHaveBeenCalledWith(
          expect.objectContaining({ limit: 50, offset: 0 }),
        );
        expect(auditLogs.stats).toHaveBeenCalled();
      });
    });

    it("connects WebSocket on mount", async () => {
      render(AuditLogPanel);

      await waitFor(() => {
        expect(auditLogs.createWebSocket).toHaveBeenCalled();
      });
    });

    it("shows empty state when no logs returned", async () => {
      vi.mocked(auditLogs.list).mockResolvedValue([]);

      render(AuditLogPanel);

      await waitFor(() => {
        expect(screen.getByText("No audit logs found")).toBeInTheDocument();
      });
    });
  });

  describe("stats display", () => {
    it("renders total requests", async () => {
      vi.mocked(auditLogs.stats).mockResolvedValue(createMockStats({ total_requests: 250 }));

      render(AuditLogPanel);

      await waitFor(() => {
        expect(screen.getByText("250")).toBeInTheDocument();
      });
      expect(screen.getByText("Total Requests")).toBeInTheDocument();
    });

    it("renders successful count", async () => {
      vi.mocked(auditLogs.stats).mockResolvedValue(
        createMockStats({ by_status: { success: 42, error: 3, timeout: 1 } }),
      );

      render(AuditLogPanel);

      await waitFor(() => {
        expect(screen.getByText("42")).toBeInTheDocument();
      });
      expect(screen.getByText("Successful")).toBeInTheDocument();
    });

    it("renders error count", async () => {
      vi.mocked(auditLogs.stats).mockResolvedValue(
        createMockStats({ by_status: { success: 90, error: 15, timeout: 2 } }),
      );

      render(AuditLogPanel);

      await waitFor(() => {
        expect(screen.getByText("15")).toBeInTheDocument();
      });
      expect(screen.getByText("Errors")).toBeInTheDocument();
    });

    it("renders unique models count", async () => {
      vi.mocked(auditLogs.stats).mockResolvedValue(createMockStats({ unique_models: 7 }));

      render(AuditLogPanel);

      await waitFor(() => {
        expect(screen.getByText("7")).toBeInTheDocument();
      });
      expect(screen.getByText("Models Used")).toBeInTheDocument();
    });

    it("renders zero for missing status counts", async () => {
      vi.mocked(auditLogs.stats).mockResolvedValue(
        createMockStats({ by_status: {} }),
      );

      render(AuditLogPanel);

      await waitFor(() => {
        expect(screen.getByText("Successful")).toBeInTheDocument();
      });
      // success count defaults to 0 via || 0 — the parent container
      // has two children: the count div and the label div
      const successLabel = screen.getByText("Successful");
      const container = successLabel.parentElement!;
      const countDiv = container.querySelector(".text-2xl");
      expect(countDiv?.textContent).toBe("0");
    });
  });

  describe("log table rendering", () => {
    it("renders log entries in table", async () => {
      const logs = [
        createMockLog({
          id: 1,
          model: "gpt-4",
          backend_type: "openai",
          duration_ms: 2500,
          status: "success",
          total_tokens: 200,
        }),
        createMockLog({
          id: 2,
          model: "claude-3",
          backend_type: "anthropic",
          duration_ms: 500,
          status: "error",
          total_tokens: null,
        }),
      ];
      vi.mocked(auditLogs.list).mockResolvedValue(logs);

      render(AuditLogPanel);

      await waitFor(() => {
        expect(screen.getByText("gpt-4")).toBeInTheDocument();
      });
      expect(screen.getByText("claude-3")).toBeInTheDocument();
      expect(screen.getByText("openai")).toBeInTheDocument();
      expect(screen.getByText("anthropic")).toBeInTheDocument();
      expect(screen.getByText("2.5s")).toBeInTheDocument();
      expect(screen.getByText("500ms")).toBeInTheDocument();
      expect(screen.getByText("200")).toBeInTheDocument();
      // null tokens shows "-"
      expect(screen.getByText("-")).toBeInTheDocument();
    });

    it("renders table headers", async () => {
      render(AuditLogPanel);

      await waitFor(() => {
        expect(screen.getByText("Time")).toBeInTheDocument();
      });
      expect(screen.getByText("Model")).toBeInTheDocument();
      expect(screen.getByText("Backend")).toBeInTheDocument();
      expect(screen.getByText("Duration")).toBeInTheDocument();
      expect(screen.getByText("Status")).toBeInTheDocument();
      expect(screen.getByText("Tokens")).toBeInTheDocument();
    });

    it("renders status badges with correct text", async () => {
      const logs = [
        createMockLog({ id: 1, status: "success" }),
        createMockLog({ id: 2, status: "error" }),
        createMockLog({ id: 3, status: "timeout" }),
      ];
      vi.mocked(auditLogs.list).mockResolvedValue(logs);

      render(AuditLogPanel);

      await waitFor(() => {
        expect(screen.getByText("success")).toBeInTheDocument();
      });
      expect(screen.getByText("error")).toBeInTheDocument();
      expect(screen.getByText("timeout")).toBeInTheDocument();
    });
  });

  describe("filter inputs", () => {
    it("reloads logs when model filter changes", async () => {
      const user = userEvent.setup();
      render(AuditLogPanel);

      await waitFor(() => {
        expect(auditLogs.list).toHaveBeenCalled();
      });
      vi.mocked(auditLogs.list).mockClear();

      const modelInput = screen.getByPlaceholderText("Model");
      await user.type(modelInput, "gpt-4");
      // Trigger the onchange event by blurring or pressing Enter
      await user.tab();

      await waitFor(() => {
        expect(auditLogs.list).toHaveBeenCalledWith(
          expect.objectContaining({ model: "gpt-4" }),
        );
      });
    });

    it("reloads logs when backend filter changes", async () => {
      const user = userEvent.setup();
      render(AuditLogPanel);

      await waitFor(() => {
        expect(auditLogs.list).toHaveBeenCalled();
      });
      vi.mocked(auditLogs.list).mockClear();

      const backendSelect = screen.getByDisplayValue("All Backends");
      await user.selectOptions(backendSelect, "openai");

      await waitFor(() => {
        expect(auditLogs.list).toHaveBeenCalledWith(
          expect.objectContaining({ backend_type: "openai" }),
        );
      });
    });

    it("reloads logs when status filter changes", async () => {
      const user = userEvent.setup();
      render(AuditLogPanel);

      await waitFor(() => {
        expect(auditLogs.list).toHaveBeenCalled();
      });
      vi.mocked(auditLogs.list).mockClear();

      const statusSelect = screen.getByDisplayValue("All Status");
      await user.selectOptions(statusSelect, "error");

      await waitFor(() => {
        expect(auditLogs.list).toHaveBeenCalledWith(
          expect.objectContaining({ status: "error" }),
        );
      });
    });
  });

  describe("refresh button", () => {
    it("reloads logs when Refresh clicked", async () => {
      const user = userEvent.setup();
      render(AuditLogPanel);

      await waitFor(() => {
        expect(auditLogs.list).toHaveBeenCalled();
      });
      vi.mocked(auditLogs.list).mockClear();

      await user.click(screen.getByText("Refresh"));

      await waitFor(() => {
        expect(auditLogs.list).toHaveBeenCalled();
      });
    });
  });

  describe("export buttons", () => {
    it("opens JSONL export URL in new tab", async () => {
      const user = userEvent.setup();
      vi.mocked(auditLogs.exportUrl).mockReturnValue(
        "/api/system/audit-logs/export?format=jsonl",
      );

      render(AuditLogPanel);
      await waitFor(() => {
        expect(auditLogs.list).toHaveBeenCalled();
      });

      await user.click(screen.getByText("JSONL"));

      expect(auditLogs.exportUrl).toHaveBeenCalledWith({}, "jsonl");
      expect(mockWindowOpen).toHaveBeenCalledWith(
        "/api/system/audit-logs/export?format=jsonl",
        "_blank",
      );
    });

    it("opens CSV export URL in new tab", async () => {
      const user = userEvent.setup();
      vi.mocked(auditLogs.exportUrl).mockReturnValue(
        "/api/system/audit-logs/export?format=csv",
      );

      render(AuditLogPanel);
      await waitFor(() => {
        expect(auditLogs.list).toHaveBeenCalled();
      });

      await user.click(screen.getByText("CSV"));

      expect(auditLogs.exportUrl).toHaveBeenCalledWith({}, "csv");
      expect(mockWindowOpen).toHaveBeenCalledWith(
        "/api/system/audit-logs/export?format=csv",
        "_blank",
      );
    });

    it("includes active filters in export URL", async () => {
      const user = userEvent.setup();

      render(AuditLogPanel);
      await waitFor(() => {
        expect(auditLogs.list).toHaveBeenCalled();
      });

      // Set a backend filter
      const backendSelect = screen.getByDisplayValue("All Backends");
      await user.selectOptions(backendSelect, "openai");

      await waitFor(() => {
        expect(auditLogs.list).toHaveBeenCalled();
      });

      await user.click(screen.getByText("JSONL"));

      expect(auditLogs.exportUrl).toHaveBeenCalledWith(
        expect.objectContaining({ backend_type: "openai" }),
        "jsonl",
      );
    });
  });

  describe("pagination", () => {
    it("shows Load More when logs >= limit", async () => {
      const fiftyLogs = Array.from({ length: 50 }, (_, i) =>
        createMockLog({ id: i + 1 }),
      );
      vi.mocked(auditLogs.list).mockResolvedValue(fiftyLogs);

      render(AuditLogPanel);

      await waitFor(() => {
        expect(screen.getByText("Load More")).toBeInTheDocument();
      });
    });

    it("does not show Load More when logs < limit", async () => {
      const fewLogs = [createMockLog({ id: 1 }), createMockLog({ id: 2 })];
      vi.mocked(auditLogs.list).mockResolvedValue(fewLogs);

      render(AuditLogPanel);

      await waitFor(() => {
        expect(auditLogs.list).toHaveBeenCalled();
      });
      expect(screen.queryByText("Load More")).not.toBeInTheDocument();
    });

    it("appends logs when Load More clicked", async () => {
      const firstBatch = Array.from({ length: 50 }, (_, i) =>
        createMockLog({ id: i + 1, model: `model-${i}` }),
      );
      const secondBatch = Array.from({ length: 10 }, (_, i) =>
        createMockLog({ id: i + 51, model: `extra-${i}` }),
      );
      vi.mocked(auditLogs.list)
        .mockResolvedValueOnce(firstBatch)
        .mockResolvedValueOnce(secondBatch);

      const user = userEvent.setup();
      render(AuditLogPanel);

      await waitFor(() => {
        expect(screen.getByText("Load More")).toBeInTheDocument();
      });

      await user.click(screen.getByText("Load More"));

      await waitFor(() => {
        expect(auditLogs.list).toHaveBeenCalledTimes(2);
      });
      // Second call should be append (offset = 50)
      expect(auditLogs.list).toHaveBeenLastCalledWith(
        expect.objectContaining({ limit: 50, offset: 50 }),
      );
    });
  });

  describe("WebSocket connection indicator", () => {
    it("shows Offline initially", async () => {
      render(AuditLogPanel);

      await waitFor(() => {
        expect(screen.getByText("Offline")).toBeInTheDocument();
      });
    });

    it("shows Live when WebSocket connects", async () => {
      render(AuditLogPanel);

      await waitFor(() => {
        expect(auditLogs.createWebSocket).toHaveBeenCalled();
      });

      // Get the WebSocket returned by createWebSocket
      const ws = vi.mocked(auditLogs.createWebSocket).mock.results[0].value as MockWebSocket;
      ws.simulateOpen();

      await waitFor(() => {
        expect(screen.getByText("Live")).toBeInTheDocument();
      });
    });

    it("shows Offline when WebSocket closes", async () => {
      render(AuditLogPanel);

      await waitFor(() => {
        expect(auditLogs.createWebSocket).toHaveBeenCalled();
      });

      const ws = vi.mocked(auditLogs.createWebSocket).mock.results[0].value as MockWebSocket;
      ws.simulateOpen();

      await waitFor(() => {
        expect(screen.getByText("Live")).toBeInTheDocument();
      });

      ws.simulateClose();

      await waitFor(() => {
        expect(screen.getByText("Offline")).toBeInTheDocument();
      });
    });

    it("shows Offline when WebSocket errors", async () => {
      render(AuditLogPanel);

      await waitFor(() => {
        expect(auditLogs.createWebSocket).toHaveBeenCalled();
      });

      const ws = vi.mocked(auditLogs.createWebSocket).mock.results[0].value as MockWebSocket;
      ws.simulateOpen();

      await waitFor(() => {
        expect(screen.getByText("Live")).toBeInTheDocument();
      });

      ws.simulateError();

      await waitFor(() => {
        expect(screen.getByText("Offline")).toBeInTheDocument();
      });
    });

    it("prepends new log from WebSocket message", async () => {
      vi.mocked(auditLogs.list).mockResolvedValue([
        createMockLog({ id: 1, model: "existing-model" }),
      ]);

      render(AuditLogPanel);

      await waitFor(() => {
        expect(screen.getByText("existing-model")).toBeInTheDocument();
      });

      const ws = vi.mocked(auditLogs.createWebSocket).mock.results[0].value as MockWebSocket;
      ws.simulateOpen();
      ws.simulateMessage({
        type: "log",
        data: createMockLog({ id: 99, model: "live-model" }),
      });

      await waitFor(() => {
        expect(screen.getByText("live-model")).toBeInTheDocument();
      });
      // Old log should still be there
      expect(screen.getByText("existing-model")).toBeInTheDocument();
    });

    it("refreshes stats when WebSocket log arrives", async () => {
      render(AuditLogPanel);

      await waitFor(() => {
        expect(auditLogs.stats).toHaveBeenCalledTimes(1);
      });

      const ws = vi.mocked(auditLogs.createWebSocket).mock.results[0].value as MockWebSocket;
      ws.simulateOpen();
      ws.simulateMessage({
        type: "log",
        data: createMockLog({ id: 99 }),
      });

      await waitFor(() => {
        expect(auditLogs.stats).toHaveBeenCalledTimes(2);
      });
    });
  });

  describe("helper functions", () => {
    // formatDuration is tested through rendered output
    it("formats duration in milliseconds", async () => {
      vi.mocked(auditLogs.list).mockResolvedValue([
        createMockLog({ id: 1, duration_ms: 450 }),
      ]);

      render(AuditLogPanel);

      await waitFor(() => {
        expect(screen.getByText("450ms")).toBeInTheDocument();
      });
    });

    it("formats duration in seconds", async () => {
      vi.mocked(auditLogs.list).mockResolvedValue([
        createMockLog({ id: 1, duration_ms: 3200 }),
      ]);

      render(AuditLogPanel);

      await waitFor(() => {
        expect(screen.getByText("3.2s")).toBeInTheDocument();
      });
    });

    it("formats duration in minutes", async () => {
      vi.mocked(auditLogs.list).mockResolvedValue([
        createMockLog({ id: 1, duration_ms: 120000 }),
      ]);

      render(AuditLogPanel);

      await waitFor(() => {
        expect(screen.getByText("2.0m")).toBeInTheDocument();
      });
    });

    it("formats timestamp as locale string", async () => {
      vi.mocked(auditLogs.list).mockResolvedValue([
        createMockLog({ id: 1, timestamp: "2025-06-15T14:30:00Z" }),
      ]);

      render(AuditLogPanel);

      const expected = new Date("2025-06-15T14:30:00Z").toLocaleString();
      await waitFor(() => {
        expect(screen.getByText(expected)).toBeInTheDocument();
      });
    });

    it("applies green color class for success status", async () => {
      vi.mocked(auditLogs.list).mockResolvedValue([
        createMockLog({ id: 1, status: "success" }),
      ]);

      render(AuditLogPanel);

      await waitFor(() => {
        const badge = screen.getByText("success");
        expect(badge.className).toContain("bg-green-100");
      });
    });

    it("applies red color class for error status", async () => {
      vi.mocked(auditLogs.list).mockResolvedValue([
        createMockLog({ id: 1, status: "error" }),
      ]);

      render(AuditLogPanel);

      await waitFor(() => {
        const badge = screen.getByText("error");
        expect(badge.className).toContain("bg-red-100");
      });
    });

    it("applies yellow color class for timeout status", async () => {
      vi.mocked(auditLogs.list).mockResolvedValue([
        createMockLog({ id: 1, status: "timeout" }),
      ]);

      render(AuditLogPanel);

      await waitFor(() => {
        const badge = screen.getByText("timeout");
        expect(badge.className).toContain("bg-yellow-100");
      });
    });

    it("applies gray color class for unknown status", async () => {
      vi.mocked(auditLogs.list).mockResolvedValue([
        createMockLog({ id: 1, status: "pending" as AuditLog["status"] }),
      ]);

      render(AuditLogPanel);

      await waitFor(() => {
        const badge = screen.getByText("pending");
        expect(badge.className).toContain("bg-gray-100");
      });
    });
  });

  describe("error handling", () => {
    it("handles list API failure gracefully", async () => {
      vi.mocked(auditLogs.list).mockRejectedValue(new Error("Network error"));

      render(AuditLogPanel);

      await waitFor(() => {
        expect(console.error).toHaveBeenCalledWith(
          "Failed to load logs:",
          expect.any(Error),
        );
      });
      // Should show empty state rather than crashing
      expect(screen.getByText("No audit logs found")).toBeInTheDocument();
    });

    it("handles stats API failure gracefully", async () => {
      vi.mocked(auditLogs.stats).mockRejectedValue(new Error("Stats error"));

      render(AuditLogPanel);

      await waitFor(() => {
        expect(console.error).toHaveBeenCalledWith(
          "Failed to load stats:",
          expect.any(Error),
        );
      });
      // Stats section should not render when stats is null
      expect(screen.queryByText("Total Requests")).not.toBeInTheDocument();
    });

    it("handles WebSocket creation failure gracefully", async () => {
      vi.mocked(auditLogs.createWebSocket).mockImplementation(() => {
        throw new Error("WebSocket not supported");
      });

      render(AuditLogPanel);

      await waitFor(() => {
        expect(console.error).toHaveBeenCalledWith(
          "WebSocket error:",
          expect.any(Error),
        );
      });
      // Component should still render
      expect(screen.getByText("Offline")).toBeInTheDocument();
    });
  });
});
