import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/svelte";
import type { DownloadState } from "$lib/stores/downloads.svelte";

// Mock format utilities
vi.mock("$lib/utils/format", () => ({
  formatBytes: vi.fn((bytes: number) => {
    if (bytes === 0) return "0 Bytes";
    const k = 1024;
    const sizes = ["Bytes", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return `${Math.round(bytes / Math.pow(k, i))} ${sizes[i]}`;
  }),
}));

import DownloadProgressTile from "./DownloadProgressTile.svelte";

// Helper to create mock download state
function createMockDownload(
  overrides: Partial<DownloadState> = {},
): DownloadState {
  return {
    model_id: "mlx-community/test-model",
    task_id: "task-123",
    status: "downloading",
    progress: 50,
    downloaded_bytes: 500000000, // 500 MB
    total_bytes: 1000000000, // 1 GB
    ...overrides,
  };
}

describe("DownloadProgressTile", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe("basic rendering", () => {
    it("renders short model name", () => {
      render(DownloadProgressTile, {
        props: {
          download: createMockDownload({
            model_id: "mlx-community/llama-2-7b",
          }),
        },
      });

      expect(screen.getByText("llama-2-7b")).toBeInTheDocument();
    });

    it("shows full model ID in title attribute", () => {
      render(DownloadProgressTile, {
        props: {
          download: createMockDownload({
            model_id: "mlx-community/llama-2-7b-4bit",
          }),
        },
      });

      const element = screen.getByTitle("mlx-community/llama-2-7b-4bit");
      expect(element).toBeInTheDocument();
    });

    it("extracts model name from path correctly", () => {
      render(DownloadProgressTile, {
        props: {
          download: createMockDownload({
            model_id: "organization/sub-org/model-name",
          }),
        },
      });

      expect(screen.getByText("model-name")).toBeInTheDocument();
    });
  });

  describe("status display", () => {
    it("shows 'Preparing...' for pending status", () => {
      render(DownloadProgressTile, {
        props: {
          download: createMockDownload({ status: "pending", progress: 0 }),
        },
      });

      expect(screen.getByText("Preparing...")).toBeInTheDocument();
    });

    it("shows 'Preparing...' for starting status", () => {
      render(DownloadProgressTile, {
        props: {
          download: createMockDownload({ status: "starting", progress: 0 }),
        },
      });

      expect(screen.getByText("Preparing...")).toBeInTheDocument();
    });

    it("shows progress percentage for downloading status", () => {
      render(DownloadProgressTile, {
        props: {
          download: createMockDownload({ status: "downloading", progress: 75 }),
        },
      });

      expect(screen.getByText("75%")).toBeInTheDocument();
    });

    it("shows 'Starting...' when downloading with 0 progress", () => {
      render(DownloadProgressTile, {
        props: {
          download: createMockDownload({ status: "downloading", progress: 0 }),
        },
      });

      expect(screen.getByText("Starting...")).toBeInTheDocument();
    });

    it("shows 'Complete' for completed status", () => {
      render(DownloadProgressTile, {
        props: {
          download: createMockDownload({ status: "completed", progress: 100 }),
        },
      });

      expect(screen.getByText("Complete")).toBeInTheDocument();
    });

    it("shows 'Failed' for failed status", () => {
      render(DownloadProgressTile, {
        props: {
          download: createMockDownload({ status: "failed", progress: 50 }),
        },
      });

      expect(screen.getByText("Failed")).toBeInTheDocument();
    });
  });

  describe("status icons", () => {
    it("shows CheckCircle icon for completed status", () => {
      const { container } = render(DownloadProgressTile, {
        props: {
          download: createMockDownload({ status: "completed" }),
        },
      });

      // CheckCircle should have green color
      const icon = container.querySelector(".text-green-500");
      expect(icon).toBeInTheDocument();
    });

    it("shows XCircle icon for failed status", () => {
      const { container } = render(DownloadProgressTile, {
        props: {
          download: createMockDownload({ status: "failed" }),
        },
      });

      // XCircle should have red color
      const icon = container.querySelector(".text-red-500");
      expect(icon).toBeInTheDocument();
    });

    it("shows spinning Loader2 icon for pending status", () => {
      const { container } = render(DownloadProgressTile, {
        props: {
          download: createMockDownload({ status: "pending" }),
        },
      });

      // Loader2 should have animate-spin class
      const icon = container.querySelector(".animate-spin");
      expect(icon).toBeInTheDocument();
    });

    it("shows spinning Loader2 icon for starting status", () => {
      const { container } = render(DownloadProgressTile, {
        props: {
          download: createMockDownload({ status: "starting" }),
        },
      });

      const icon = container.querySelector(".animate-spin");
      expect(icon).toBeInTheDocument();
    });

    it("shows Download icon for downloading status", () => {
      const { container } = render(DownloadProgressTile, {
        props: {
          download: createMockDownload({ status: "downloading", progress: 50 }),
        },
      });

      // Download icon should have primary color (not spinning)
      const icon = container.querySelector(".text-primary");
      expect(icon).toBeInTheDocument();
      expect(icon?.classList.contains("animate-spin")).toBe(false);
    });
  });

  describe("progress bar", () => {
    it("shows progress bar for downloading status", () => {
      const { container } = render(DownloadProgressTile, {
        props: {
          download: createMockDownload({
            status: "downloading",
            progress: 60,
          }),
        },
      });

      const progressBar = container.querySelector(".bg-primary");
      expect(progressBar).toBeInTheDocument();
    });

    it("sets correct width for progress bar", () => {
      const { container } = render(DownloadProgressTile, {
        props: {
          download: createMockDownload({
            status: "downloading",
            progress: 75,
          }),
        },
      });

      const progressBar = container.querySelector(".bg-primary");
      expect(progressBar).toHaveAttribute("style", "width: 75%;");
    });

    it("shows progress bar for starting status", () => {
      const { container } = render(DownloadProgressTile, {
        props: {
          download: createMockDownload({ status: "starting" }),
        },
      });

      const progressBar = container.querySelector(".bg-primary");
      expect(progressBar).toBeInTheDocument();
    });

    it("shows progress bar for pending status", () => {
      const { container } = render(DownloadProgressTile, {
        props: {
          download: createMockDownload({ status: "pending" }),
        },
      });

      const progressBar = container.querySelector(".bg-primary");
      expect(progressBar).toBeInTheDocument();
    });

    it("does not show progress bar for completed status", () => {
      const { container } = render(DownloadProgressTile, {
        props: {
          download: createMockDownload({ status: "completed" }),
        },
      });

      const progressBars = container.querySelectorAll(".bg-muted");
      expect(progressBars.length).toBe(0);
    });

    it("does not show progress bar for failed status", () => {
      const { container } = render(DownloadProgressTile, {
        props: {
          download: createMockDownload({ status: "failed" }),
        },
      });

      const progressBars = container.querySelectorAll(".bg-muted");
      expect(progressBars.length).toBe(0);
    });
  });

  describe("byte counts display", () => {
    it("shows downloaded and total bytes when total_bytes > 0", () => {
      render(DownloadProgressTile, {
        props: {
          download: createMockDownload({
            status: "downloading",
            downloaded_bytes: 500000000, // ~500 MB
            total_bytes: 1000000000, // ~1 GB
          }),
        },
      });

      // formatBytes returns approximate values, check for pattern
      expect(screen.getByText(/MB/)).toBeInTheDocument();
      const byteDisplay = screen.getByText(/\d+ MB \/ \d+ MB/);
      expect(byteDisplay).toBeInTheDocument();
    });

    it("does not show byte counts when total_bytes is 0", () => {
      render(DownloadProgressTile, {
        props: {
          download: createMockDownload({
            status: "downloading",
            downloaded_bytes: 0,
            total_bytes: 0,
          }),
        },
      });

      expect(screen.queryByText(/MB/)).not.toBeInTheDocument();
    });

    it("formats bytes correctly for small downloads", () => {
      render(DownloadProgressTile, {
        props: {
          download: createMockDownload({
            status: "downloading",
            downloaded_bytes: 512000, // ~512 KB
            total_bytes: 1024000, // ~1 MB
          }),
        },
      });

      // Check for KB pattern in the display
      expect(screen.getByText(/KB/)).toBeInTheDocument();
      const byteDisplay = screen.getByText(/\d+ KB \/ \d+ KB/);
      expect(byteDisplay).toBeInTheDocument();
    });

    it("shows byte counts only during active download states", () => {
      const { rerender } = render(DownloadProgressTile, {
        props: {
          download: createMockDownload({
            status: "downloading",
            downloaded_bytes: 500000000,
            total_bytes: 1000000000,
          }),
        },
      });

      // Check for byte pattern during download
      expect(screen.getByText(/\d+ MB \/ \d+ MB/)).toBeInTheDocument();

      // Change to completed
      rerender({
        download: createMockDownload({
          status: "completed",
          downloaded_bytes: 1000000000,
          total_bytes: 1000000000,
        }),
      });

      expect(screen.queryByText(/\d+ MB \/ \d+ MB/)).not.toBeInTheDocument();
    });
  });

  describe("error display", () => {
    it("shows error message when download fails", () => {
      render(DownloadProgressTile, {
        props: {
          download: createMockDownload({
            status: "failed",
            error: "Network connection lost",
          }),
        },
      });

      expect(screen.getByText("Network connection lost")).toBeInTheDocument();
    });

    it("shows full error in title attribute for truncated text", () => {
      render(DownloadProgressTile, {
        props: {
          download: createMockDownload({
            status: "failed",
            error: "Very long error message that should be truncated in the UI",
          }),
        },
      });

      const errorElement = screen.getByTitle(
        "Very long error message that should be truncated in the UI",
      );
      expect(errorElement).toBeInTheDocument();
    });

    it("does not show error message when no error", () => {
      render(DownloadProgressTile, {
        props: {
          download: createMockDownload({
            status: "downloading",
            error: undefined,
          }),
        },
      });

      const errorElements = screen
        .queryAllByText(/error/i)
        .filter((el) => el.classList.contains("text-red-500"));
      expect(errorElements.length).toBe(0);
    });

    it("shows error with red styling", () => {
      const { container } = render(DownloadProgressTile, {
        props: {
          download: createMockDownload({
            status: "failed",
            error: "Download failed",
          }),
        },
      });

      const errorElement = screen.getByText("Download failed");
      expect(errorElement.classList.contains("text-red-500")).toBe(true);
    });
  });

  describe("edge cases", () => {
    it("handles 0% progress", () => {
      const { container } = render(DownloadProgressTile, {
        props: {
          download: createMockDownload({
            status: "downloading",
            progress: 0,
          }),
        },
      });

      const progressBar = container.querySelector(".bg-primary");
      expect(progressBar).toHaveAttribute("style", "width: 0%;");
    });

    it("handles 100% progress", () => {
      const { container } = render(DownloadProgressTile, {
        props: {
          download: createMockDownload({
            status: "downloading",
            progress: 100,
          }),
        },
      });

      const progressBar = container.querySelector(".bg-primary");
      expect(progressBar).toHaveAttribute("style", "width: 100%;");
    });

    it("handles model ID without slash", () => {
      render(DownloadProgressTile, {
        props: {
          download: createMockDownload({
            model_id: "simple-model-name",
          }),
        },
      });

      expect(screen.getByText("simple-model-name")).toBeInTheDocument();
    });

    it("handles very long model names", () => {
      render(DownloadProgressTile, {
        props: {
          download: createMockDownload({
            model_id:
              "mlx-community/very-very-very-long-model-name-that-should-truncate",
          }),
        },
      });

      expect(
        screen.getByText(
          "very-very-very-long-model-name-that-should-truncate",
        ),
      ).toBeInTheDocument();
    });

    it("handles zero byte downloads", () => {
      render(DownloadProgressTile, {
        props: {
          download: createMockDownload({
            downloaded_bytes: 0,
            total_bytes: 0,
          }),
        },
      });

      // Should not show byte counts
      expect(screen.queryByText(/0 Bytes/)).not.toBeInTheDocument();
    });

    it("handles large file sizes", () => {
      render(DownloadProgressTile, {
        props: {
          download: createMockDownload({
            status: "downloading",
            downloaded_bytes: 5000000000, // ~5 GB
            total_bytes: 10000000000, // ~10 GB
          }),
        },
      });

      expect(screen.getByText(/GB/)).toBeInTheDocument();
    });
  });

  describe("status transitions", () => {
    it("updates display when status changes", () => {
      const { rerender } = render(DownloadProgressTile, {
        props: {
          download: createMockDownload({
            status: "pending",
            progress: 0,
          }),
        },
      });

      expect(screen.getByText("Preparing...")).toBeInTheDocument();

      rerender({
        download: createMockDownload({
          status: "downloading",
          progress: 50,
        }),
      });

      expect(screen.getByText("50%")).toBeInTheDocument();

      rerender({
        download: createMockDownload({
          status: "completed",
          progress: 100,
        }),
      });

      expect(screen.getByText("Complete")).toBeInTheDocument();
    });

    it("updates progress bar when progress changes", () => {
      const { container, rerender } = render(DownloadProgressTile, {
        props: {
          download: createMockDownload({
            status: "downloading",
            progress: 25,
          }),
        },
      });

      let progressBar = container.querySelector(".bg-primary");
      expect(progressBar).toHaveAttribute("style", "width: 25%;");

      rerender({
        download: createMockDownload({
          status: "downloading",
          progress: 75,
        }),
      });

      progressBar = container.querySelector(".bg-primary");
      expect(progressBar).toHaveAttribute("style", "width: 75%;");
    });
  });
});
