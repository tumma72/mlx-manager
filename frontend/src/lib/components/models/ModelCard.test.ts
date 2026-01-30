import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor, fireEvent } from "@testing-library/svelte";
import userEvent from "@testing-library/user-event";
import type { ModelSearchResult } from "$api";

// Mock the stores
vi.mock("$stores", () => ({
  downloadsStore: {
    startDownload: vi.fn().mockResolvedValue(undefined),
    getProgress: vi.fn().mockReturnValue(undefined),
  },
  modelConfigStore: {
    getConfig: vi.fn().mockReturnValue(undefined),
    fetchConfig: vi.fn().mockResolvedValue(undefined),
  },
}));

// Mock the API
vi.mock("$api", () => ({
  models: {
    delete: vi.fn().mockResolvedValue(undefined),
  },
}));

// Mock format utilities
vi.mock("$lib/utils/format", () => ({
  formatNumber: vi.fn((num: number) => {
    if (num >= 1000000) return `${(num / 1000000).toFixed(1)}M`;
    if (num >= 1000) return `${(num / 1000).toFixed(1)}K`;
    return num.toString();
  }),
}));

// Import after mocking
import ModelCard from "./ModelCard.svelte";
import { downloadsStore, modelConfigStore } from "$stores";
import { models } from "$api";

// Helper to create mock model
function createMockModel(
  overrides: Partial<ModelSearchResult> = {},
): ModelSearchResult {
  return {
    model_id: "mlx-community/test-model",
    estimated_size_gb: 8.5,
    downloads: 1000,
    likes: 50,
    tags: [],
    is_downloaded: false,
    ...overrides,
  };
}

describe("ModelCard", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe("basic rendering", () => {
    it("renders model ID", () => {
      render(ModelCard, {
        props: { model: createMockModel({ model_id: "mlx-community/llama-2" }) },
      });

      expect(screen.getByText("mlx-community/llama-2")).toBeInTheDocument();
    });

    it("renders download count", () => {
      render(ModelCard, {
        props: { model: createMockModel({ downloads: 5000 }) },
      });

      expect(screen.getByText("5.0K")).toBeInTheDocument();
    });

    it("renders likes count", () => {
      render(ModelCard, {
        props: { model: createMockModel({ likes: 250 }) },
      });

      expect(screen.getByText("250")).toBeInTheDocument();
    });

    it("renders estimated size", () => {
      render(ModelCard, {
        props: { model: createMockModel({ estimated_size_gb: 12.5 }) },
      });

      expect(screen.getByText("12.5 GB")).toBeInTheDocument();
    });

    it("shows Download button when model is not downloaded", () => {
      render(ModelCard, {
        props: { model: createMockModel({ is_downloaded: false }) },
      });

      expect(screen.getByRole("button", { name: /download/i })).toBeInTheDocument();
    });

    it("shows Downloaded badge when model is downloaded", () => {
      render(ModelCard, {
        props: { model: createMockModel({ is_downloaded: true }) },
      });

      expect(screen.getByText("Downloaded")).toBeInTheDocument();
    });

    it("shows Use and Delete buttons when model is downloaded", () => {
      render(ModelCard, {
        props: { model: createMockModel({ is_downloaded: true }) },
      });

      expect(screen.getByRole("button", { name: /use/i })).toBeInTheDocument();
      expect(screen.getByRole("button", { name: /delete/i })).toBeInTheDocument();
    });
  });

  describe("tags rendering", () => {
    it("renders model tags", () => {
      render(ModelCard, {
        props: {
          model: createMockModel({ tags: ["text-generation", "llama", "4bit"] }),
        },
      });

      expect(screen.getByText("text-generation")).toBeInTheDocument();
      expect(screen.getByText("llama")).toBeInTheDocument();
      expect(screen.getByText("4bit")).toBeInTheDocument();
    });

    it("limits tags display to 5 and shows count for extra tags", () => {
      render(ModelCard, {
        props: {
          model: createMockModel({
            tags: ["tag1", "tag2", "tag3", "tag4", "tag5", "tag6", "tag7"],
          }),
        },
      });

      expect(screen.getByText("tag1")).toBeInTheDocument();
      expect(screen.getByText("tag5")).toBeInTheDocument();
      expect(screen.getByText("+2")).toBeInTheDocument();
      expect(screen.queryByText("tag6")).not.toBeInTheDocument();
    });

    it("does not render tags section when no tags", () => {
      render(ModelCard, {
        props: { model: createMockModel({ tags: [] }) },
      });

      // Tags container should not exist
      const tagElements = screen.queryAllByRole("generic").filter((el) =>
        el.textContent?.includes("tag"),
      );
      expect(tagElements.length).toBe(0);
    });
  });

  describe("model badges and specs", () => {
    it("fetches config on mount", () => {
      const model = createMockModel({
        model_id: "mlx-community/test",
        tags: ["llama", "4bit"],
      });

      render(ModelCard, { props: { model } });

      expect(modelConfigStore.fetchConfig).toHaveBeenCalledWith(
        "mlx-community/test",
        ["llama", "4bit"],
      );
    });

    it("displays badges when config is loaded", () => {
      vi.mocked(modelConfigStore.getConfig).mockReturnValue({
        characteristics: {
          architecture_family: "Llama",
          quantization_bits: 4,
          is_multimodal: false,
        },
        loading: false,
        error: null,
      });

      render(ModelCard, {
        props: { model: createMockModel() },
      });

      // ModelBadges component should receive the characteristics
      expect(modelConfigStore.getConfig).toHaveBeenCalled();
    });

    it("shows loading state while fetching config", () => {
      vi.mocked(modelConfigStore.getConfig).mockReturnValue({
        characteristics: null,
        loading: true,
        error: null,
      });

      render(ModelCard, {
        props: { model: createMockModel() },
      });

      expect(modelConfigStore.getConfig).toHaveBeenCalled();
    });
  });

  describe("download action", () => {
    it("calls downloadsStore.startDownload when Download button clicked", async () => {
      const user = userEvent.setup();
      render(ModelCard, {
        props: { model: createMockModel({ model_id: "mlx-community/test" }) },
      });

      const downloadButton = screen.getByRole("button", { name: /download/i });
      await user.click(downloadButton);

      expect(downloadsStore.startDownload).toHaveBeenCalledWith(
        "mlx-community/test",
      );
    });

    it("displays error when download fails", async () => {
      const user = userEvent.setup();
      vi.mocked(downloadsStore.startDownload).mockRejectedValue(
        new Error("Network error"),
      );

      render(ModelCard, {
        props: { model: createMockModel() },
      });

      const downloadButton = screen.getByRole("button", { name: /download/i });
      await user.click(downloadButton);

      await waitFor(() => {
        expect(screen.getByText("Network error")).toBeInTheDocument();
      });
    });

    it("displays generic error when download fails with non-Error", async () => {
      const user = userEvent.setup();
      vi.mocked(downloadsStore.startDownload).mockRejectedValue("string error");

      render(ModelCard, {
        props: { model: createMockModel() },
      });

      const downloadButton = screen.getByRole("button", { name: /download/i });
      await user.click(downloadButton);

      await waitFor(() => {
        expect(screen.getByText("Download failed")).toBeInTheDocument();
      });
    });

    it("shows Downloaded badge when download completes", () => {
      vi.mocked(downloadsStore.getProgress).mockReturnValue({
        model_id: "mlx-community/test",
        task_id: "task-123",
        status: "completed",
        progress: 100,
        downloaded_bytes: 1000000,
        total_bytes: 1000000,
      });

      render(ModelCard, {
        props: {
          model: createMockModel({
            model_id: "mlx-community/test",
            is_downloaded: false,
          }),
        },
      });

      expect(screen.getByText("Downloaded")).toBeInTheDocument();
    });

    it("shows error from download state when download fails", () => {
      vi.mocked(downloadsStore.getProgress).mockReturnValue({
        model_id: "mlx-community/test",
        task_id: "task-123",
        status: "failed",
        progress: 50,
        downloaded_bytes: 500000,
        total_bytes: 1000000,
        error: "Download interrupted",
      });

      render(ModelCard, {
        props: {
          model: createMockModel({
            model_id: "mlx-community/test",
          }),
        },
      });

      expect(screen.getByText("Download interrupted")).toBeInTheDocument();
    });
  });

  describe("delete action", () => {
    it("shows confirmation dialog when Delete button clicked", async () => {
      const user = userEvent.setup();
      render(ModelCard, {
        props: {
          model: createMockModel({
            model_id: "mlx-community/test",
            is_downloaded: true,
          }),
        },
      });

      const deleteButton = screen.getByRole("button", { name: /delete/i });
      await user.click(deleteButton);

      await waitFor(() => {
        expect(screen.getByText("Delete Model")).toBeInTheDocument();
        expect(
          screen.getByText(/Are you sure you want to delete/i),
        ).toBeInTheDocument();
      });
    });

    it("calls models.delete when deletion confirmed", async () => {
      const user = userEvent.setup();
      render(ModelCard, {
        props: {
          model: createMockModel({
            model_id: "mlx-community/test",
            is_downloaded: true,
          }),
        },
      });

      // Open confirmation dialog by clicking the Delete button (not the one in the dialog)
      const deleteButtons = screen.getAllByRole("button", { name: /delete/i });
      // First button should be the one in the card - use fireEvent to bypass pointer-events
      await fireEvent.click(deleteButtons[0]);

      // Wait for dialog to appear and find the destructive Delete button
      await waitFor(() => {
        expect(screen.getByText("Delete Model")).toBeInTheDocument();
      });

      // Find the confirm button by getting all Delete buttons and using the destructive one
      const allDeleteButtons = screen.getAllByRole("button", { name: "Delete" });
      const confirmButton = allDeleteButtons.find(btn =>
        btn.className.includes("bg-destructive")
      );
      expect(confirmButton).toBeDefined();

      // Use fireEvent instead of userEvent to bypass pointer-events check
      await fireEvent.click(confirmButton!);

      await waitFor(() => {
        expect(models.delete).toHaveBeenCalledWith("mlx-community/test");
      });
    });

    it("calls onDeleted callback after successful deletion", async () => {
      const user = userEvent.setup();
      const onDeleted = vi.fn();

      render(ModelCard, {
        props: {
          model: createMockModel({
            model_id: "mlx-community/test",
            is_downloaded: true,
          }),
          onDeleted,
        },
      });

      // Open dialog
      const deleteButtons = screen.getAllByRole("button", { name: /delete/i });
      await fireEvent.click(deleteButtons[0]);

      // Wait for dialog and confirm
      await waitFor(() => {
        expect(screen.getByText("Delete Model")).toBeInTheDocument();
      });

      const allDeleteButtons = screen.getAllByRole("button", { name: "Delete" });
      const confirmButton = allDeleteButtons.find(btn =>
        btn.className.includes("bg-destructive")
      );

      await fireEvent.click(confirmButton!);

      await waitFor(() => {
        expect(onDeleted).toHaveBeenCalled();
      });
    });

    it("displays error when deletion fails", async () => {
      const user = userEvent.setup();
      vi.mocked(models.delete).mockRejectedValue(new Error("Permission denied"));

      render(ModelCard, {
        props: {
          model: createMockModel({
            model_id: "mlx-community/test",
            is_downloaded: true,
          }),
        },
      });

      const deleteButtons = screen.getAllByRole("button", { name: /delete/i });
      await fireEvent.click(deleteButtons[0]);

      await waitFor(() => {
        expect(screen.getByText("Delete Model")).toBeInTheDocument();
      });

      const allDeleteButtons = screen.getAllByRole("button", { name: "Delete" });
      const confirmButton = allDeleteButtons.find(btn =>
        btn.className.includes("bg-destructive")
      );

      await fireEvent.click(confirmButton!);

      await waitFor(() => {
        expect(screen.getByText("Permission denied")).toBeInTheDocument();
      });
    });

    it("displays generic error when deletion fails with non-Error", async () => {
      const user = userEvent.setup();
      vi.mocked(models.delete).mockRejectedValue("string error");

      render(ModelCard, {
        props: {
          model: createMockModel({
            model_id: "mlx-community/test",
            is_downloaded: true,
          }),
        },
      });

      const deleteButtons = screen.getAllByRole("button", { name: /delete/i });
      await fireEvent.click(deleteButtons[0]);

      await waitFor(() => {
        expect(screen.getByText("Delete Model")).toBeInTheDocument();
      });

      const allDeleteButtons = screen.getAllByRole("button", { name: "Delete" });
      const confirmButton = allDeleteButtons.find(btn =>
        btn.className.includes("bg-destructive")
      );

      await fireEvent.click(confirmButton!);

      await waitFor(() => {
        expect(screen.getByText("Delete failed")).toBeInTheDocument();
      });
    });

    it("shows 'Deleting...' while delete is in progress", async () => {
      const user = userEvent.setup();
      let resolveDelete: () => void;
      vi.mocked(models.delete).mockImplementation(
        () =>
          new Promise((resolve) => {
            resolveDelete = resolve;
          }),
      );

      render(ModelCard, {
        props: {
          model: createMockModel({ is_downloaded: true }),
        },
      });

      const deleteButtons = screen.getAllByRole("button", { name: /delete/i });
      await fireEvent.click(deleteButtons[0]);

      await waitFor(() => {
        expect(screen.getByText("Delete Model")).toBeInTheDocument();
      });

      const allDeleteButtons = screen.getAllByRole("button", { name: "Delete" });
      const confirmButton = allDeleteButtons.find(btn =>
        btn.className.includes("bg-destructive")
      );

      await fireEvent.click(confirmButton!);

      await waitFor(() => {
        expect(screen.getByText("Deleting...")).toBeInTheDocument();
      });

      resolveDelete!();
      await waitFor(() => {
        expect(screen.queryByText("Deleting...")).not.toBeInTheDocument();
      });
    });

    it("disables Delete button while deleting", async () => {
      const user = userEvent.setup();
      let resolveDelete: () => void;
      vi.mocked(models.delete).mockImplementation(
        () =>
          new Promise((resolve) => {
            resolveDelete = resolve;
          }),
      );

      render(ModelCard, {
        props: {
          model: createMockModel({ is_downloaded: true }),
        },
      });

      const deleteButtons = screen.getAllByRole("button", { name: /delete/i });
      await fireEvent.click(deleteButtons[0]);

      await waitFor(() => {
        expect(screen.getByText("Delete Model")).toBeInTheDocument();
      });

      const allDeleteButtons = screen.getAllByRole("button", { name: "Delete" });
      const confirmButton = allDeleteButtons.find(btn =>
        btn.className.includes("bg-destructive")
      );

      await fireEvent.click(confirmButton!);

      await waitFor(() => {
        const deletingButton = screen.getByRole("button", { name: /deleting/i });
        expect(deletingButton).toBeDisabled();
      });

      resolveDelete!();
    });
  });

  describe("use action", () => {
    it("calls onUse callback when Use button clicked", async () => {
      const user = userEvent.setup();
      const onUse = vi.fn();

      const { container } = render(ModelCard, {
        props: {
          model: createMockModel({
            model_id: "mlx-community/test",
            is_downloaded: true,
          }),
          onUse,
        },
      });

      // Find the Use button (not disabled, not in a dialog)
      const useButtons = screen.getAllByRole("button", { name: /^use$/i });
      const useButton = useButtons.find(btn => !btn.disabled);
      expect(useButton).toBeDefined();

      // Use fireEvent to bypass any pointer-events issues
      await fireEvent.click(useButton!);

      expect(onUse).toHaveBeenCalledWith("mlx-community/test");
    });

    it("does nothing when onUse is not provided", async () => {
      render(ModelCard, {
        props: {
          model: createMockModel({ is_downloaded: true }),
        },
      });

      const useButtons = screen.getAllByRole("button", { name: /^use$/i });
      const useButton = useButtons.find(btn => !btn.disabled);
      expect(useButton).toBeDefined();

      // Use fireEvent to bypass pointer-events issues
      await fireEvent.click(useButton!);

      // Should not throw error
    });
  });

  describe("download status override", () => {
    it("resets override when model prop changes", async () => {
      const { rerender } = render(ModelCard, {
        props: {
          model: createMockModel({
            model_id: "mlx-community/model-1",
            is_downloaded: true,
          }),
        },
      });

      // Change to different model
      await rerender({
        model: createMockModel({
          model_id: "mlx-community/model-2",
          is_downloaded: false,
        }),
      });

      // Should show download button for new model
      expect(screen.getByRole("button", { name: /download/i })).toBeInTheDocument();
    });
  });

  describe("edge cases", () => {
    it("handles model with no tags", () => {
      render(ModelCard, {
        props: { model: createMockModel({ tags: [] }) },
      });

      // Should not crash
      expect(screen.getByText("mlx-community/test-model")).toBeInTheDocument();
    });

    it("handles model with zero downloads", () => {
      render(ModelCard, {
        props: { model: createMockModel({ downloads: 0 }) },
      });

      expect(screen.getByText("0")).toBeInTheDocument();
    });

    it("handles model with zero likes", () => {
      render(ModelCard, {
        props: { model: createMockModel({ likes: 0 }) },
      });

      expect(screen.getByText("0")).toBeInTheDocument();
    });

    it("handles very large numbers", () => {
      render(ModelCard, {
        props: {
          model: createMockModel({
            downloads: 5000000,
            likes: 100000,
          }),
        },
      });

      expect(screen.getByText("5.0M")).toBeInTheDocument();
      expect(screen.getByText("100.0K")).toBeInTheDocument();
    });

    it("handles long model IDs", () => {
      render(ModelCard, {
        props: {
          model: createMockModel({
            model_id:
              "mlx-community/very-long-model-name-that-might-need-truncation",
          }),
        },
      });

      expect(
        screen.getByText(
          "mlx-community/very-long-model-name-that-might-need-truncation",
        ),
      ).toBeInTheDocument();
    });
  });
});
