import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/svelte";
import userEvent from "@testing-library/user-event";
import HuggingFaceSettings from "./HuggingFaceSettings.svelte";
import { settings } from "$lib/api/client";

// Mock API
vi.mock("$lib/api/client", () => ({
  settings: {
    getHuggingFaceStatus: vi.fn(),
    saveHuggingFaceToken: vi.fn(),
    testHuggingFaceToken: vi.fn(),
    deleteHuggingFaceToken: vi.fn(),
  },
}));

describe("HuggingFaceSettings", () => {
  beforeEach(() => {
    vi.mocked(settings.getHuggingFaceStatus).mockResolvedValue({
      configured: false,
    });
    vi.mocked(settings.saveHuggingFaceToken).mockResolvedValue({
      configured: true,
    });
    vi.mocked(settings.testHuggingFaceToken).mockResolvedValue({
      success: true,
      username: "testuser",
    });
    vi.mocked(settings.deleteHuggingFaceToken).mockResolvedValue(undefined);
  });

  describe("loading state", () => {
    it("shows loading text on mount", () => {
      // Use a promise that never resolves to keep loading state
      vi.mocked(settings.getHuggingFaceStatus).mockReturnValue(
        new Promise(() => {}),
      );

      render(HuggingFaceSettings);

      expect(
        screen.getByText("Loading HuggingFace settings..."),
      ).toBeInTheDocument();
    });

    it("hides loading text after status loads", async () => {
      render(HuggingFaceSettings);

      await waitFor(() => {
        expect(
          screen.queryByText("Loading HuggingFace settings..."),
        ).not.toBeInTheDocument();
      });
    });
  });

  describe("not configured state", () => {
    it("shows no token configured status", async () => {
      render(HuggingFaceSettings);

      await waitFor(() => {
        expect(
          screen.getByText("No token configured (using public access)"),
        ).toBeInTheDocument();
      });
    });

    it("shows default placeholder for token input", async () => {
      render(HuggingFaceSettings);

      await waitFor(() => {
        expect(
          screen.getByPlaceholderText("hf_xxxxxxxxxxxxxxxxxxxxxxxxx"),
        ).toBeInTheDocument();
      });
    });

    it("does not show Test Connection button", async () => {
      render(HuggingFaceSettings);

      await waitFor(() => {
        expect(screen.getByText("Save & Test")).toBeInTheDocument();
      });

      expect(
        screen.queryByRole("button", { name: /Test Connection/ }),
      ).not.toBeInTheDocument();
    });

    it("does not show Remove button", async () => {
      render(HuggingFaceSettings);

      await waitFor(() => {
        expect(screen.getByText("Save & Test")).toBeInTheDocument();
      });

      expect(
        screen.queryByRole("button", { name: /Remove/ }),
      ).not.toBeInTheDocument();
    });
  });

  describe("configured state", () => {
    beforeEach(() => {
      vi.mocked(settings.getHuggingFaceStatus).mockResolvedValue({
        configured: true,
      });
    });

    it("shows token configured status", async () => {
      render(HuggingFaceSettings);

      await waitFor(() => {
        expect(screen.getByText("Token configured")).toBeInTheDocument();
      });
    });

    it("shows saved placeholder for token input", async () => {
      render(HuggingFaceSettings);

      await waitFor(() => {
        expect(
          screen.getByPlaceholderText(
            "****...saved (enter new token to update)",
          ),
        ).toBeInTheDocument();
      });
    });

    it("shows Test Connection button", async () => {
      render(HuggingFaceSettings);

      await waitFor(() => {
        expect(
          screen.getByRole("button", { name: /Test Connection/ }),
        ).toBeInTheDocument();
      });
    });

    it("shows Remove button", async () => {
      render(HuggingFaceSettings);

      await waitFor(() => {
        expect(
          screen.getByRole("button", { name: /Remove/ }),
        ).toBeInTheDocument();
      });
    });
  });

  describe("save & test action", () => {
    it("disables Save button when input is empty", async () => {
      render(HuggingFaceSettings);

      await waitFor(() => {
        expect(
          screen.getByRole("button", { name: /Save & Test/ }),
        ).toBeDisabled();
      });
    });

    it("enables Save button when token entered", async () => {
      const user = userEvent.setup();
      render(HuggingFaceSettings);

      await waitFor(() => {
        expect(screen.getByLabelText("Access Token")).toBeInTheDocument();
      });

      await user.type(screen.getByLabelText("Access Token"), "hf_test123");

      expect(
        screen.getByRole("button", { name: /Save & Test/ }),
      ).toBeEnabled();
    });

    it("calls saveHuggingFaceToken then testHuggingFaceToken", async () => {
      const user = userEvent.setup();
      render(HuggingFaceSettings);

      await waitFor(() => {
        expect(screen.getByLabelText("Access Token")).toBeInTheDocument();
      });

      await user.type(screen.getByLabelText("Access Token"), "hf_test123");
      await user.click(screen.getByRole("button", { name: /Save & Test/ }));

      await waitFor(() => {
        expect(settings.saveHuggingFaceToken).toHaveBeenCalledWith("hf_test123");
      });

      await waitFor(() => {
        expect(settings.testHuggingFaceToken).toHaveBeenCalled();
      });
    });

    it("trims token before saving", async () => {
      const user = userEvent.setup();
      render(HuggingFaceSettings);

      await waitFor(() => {
        expect(screen.getByLabelText("Access Token")).toBeInTheDocument();
      });

      await user.type(screen.getByLabelText("Access Token"), "  hf_test123  ");
      await user.click(screen.getByRole("button", { name: /Save & Test/ }));

      await waitFor(() => {
        expect(settings.saveHuggingFaceToken).toHaveBeenCalledWith("hf_test123");
      });
    });

    it("shows success with username after save and test", async () => {
      const user = userEvent.setup();
      render(HuggingFaceSettings);

      await waitFor(() => {
        expect(screen.getByLabelText("Access Token")).toBeInTheDocument();
      });

      await user.type(screen.getByLabelText("Access Token"), "hf_test123");
      await user.click(screen.getByRole("button", { name: /Save & Test/ }));

      await waitFor(() => {
        expect(screen.getByText("testuser")).toBeInTheDocument();
      });
    });

    it("clears input after successful save", async () => {
      const user = userEvent.setup();
      render(HuggingFaceSettings);

      await waitFor(() => {
        expect(screen.getByLabelText("Access Token")).toBeInTheDocument();
      });

      await user.type(screen.getByLabelText("Access Token"), "hf_test123");
      await user.click(screen.getByRole("button", { name: /Save & Test/ }));

      await waitFor(() => {
        expect(
          (screen.getByLabelText("Access Token") as HTMLInputElement).value,
        ).toBe("");
      });
    });

    it("shows saving loading state", async () => {
      const user = userEvent.setup();
      let resolveSave: () => void;
      const savePromise = new Promise<void>((resolve) => {
        resolveSave = resolve;
      });
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      vi.mocked(settings.saveHuggingFaceToken).mockReturnValue(savePromise as any);

      render(HuggingFaceSettings);

      await waitFor(() => {
        expect(screen.getByLabelText("Access Token")).toBeInTheDocument();
      });

      await user.type(screen.getByLabelText("Access Token"), "hf_test123");
      await user.click(screen.getByRole("button", { name: /Save & Test/ }));

      expect(screen.getByText("Saving...")).toBeInTheDocument();

      resolveSave!();
      await savePromise;
    });

    it("sets configured to true after successful save", async () => {
      const user = userEvent.setup();
      render(HuggingFaceSettings);

      await waitFor(() => {
        expect(screen.getByLabelText("Access Token")).toBeInTheDocument();
      });

      await user.type(screen.getByLabelText("Access Token"), "hf_test123");
      await user.click(screen.getByRole("button", { name: /Save & Test/ }));

      await waitFor(() => {
        expect(screen.getByText("Token configured")).toBeInTheDocument();
      });
    });
  });

  describe("save error handling", () => {
    it("shows error message when save fails", async () => {
      const user = userEvent.setup();
      vi.mocked(settings.saveHuggingFaceToken).mockRejectedValue(
        new Error("Network error"),
      );

      render(HuggingFaceSettings);

      await waitFor(() => {
        expect(screen.getByLabelText("Access Token")).toBeInTheDocument();
      });

      await user.type(screen.getByLabelText("Access Token"), "hf_test123");
      await user.click(screen.getByRole("button", { name: /Save & Test/ }));

      await waitFor(() => {
        expect(screen.getByText("Network error")).toBeInTheDocument();
      });
    });

    it("shows generic error when save fails with non-Error", async () => {
      const user = userEvent.setup();
      vi.mocked(settings.saveHuggingFaceToken).mockRejectedValue("unknown");

      render(HuggingFaceSettings);

      await waitFor(() => {
        expect(screen.getByLabelText("Access Token")).toBeInTheDocument();
      });

      await user.type(screen.getByLabelText("Access Token"), "hf_test123");
      await user.click(screen.getByRole("button", { name: /Save & Test/ }));

      await waitFor(() => {
        expect(screen.getByText("Failed to save token")).toBeInTheDocument();
      });
    });

    it("shows saved but test failed when test fails after save", async () => {
      const user = userEvent.setup();
      vi.mocked(settings.testHuggingFaceToken).mockRejectedValue(
        new Error("Invalid token"),
      );

      render(HuggingFaceSettings);

      await waitFor(() => {
        expect(screen.getByLabelText("Access Token")).toBeInTheDocument();
      });

      await user.type(screen.getByLabelText("Access Token"), "hf_test123");
      await user.click(screen.getByRole("button", { name: /Save & Test/ }));

      await waitFor(() => {
        expect(
          screen.getByText(
            "Saved but connection test failed - check your token",
          ),
        ).toBeInTheDocument();
      });
    });
  });

  describe("test connection action", () => {
    beforeEach(() => {
      vi.mocked(settings.getHuggingFaceStatus).mockResolvedValue({
        configured: true,
      });
    });

    it("calls testHuggingFaceToken", async () => {
      const user = userEvent.setup();
      render(HuggingFaceSettings);

      await waitFor(() => {
        expect(
          screen.getByRole("button", { name: /Test Connection/ }),
        ).toBeInTheDocument();
      });

      await user.click(
        screen.getByRole("button", { name: /Test Connection/ }),
      );

      await waitFor(() => {
        expect(settings.testHuggingFaceToken).toHaveBeenCalled();
      });
    });

    it("shows success with username", async () => {
      const user = userEvent.setup();
      render(HuggingFaceSettings);

      await waitFor(() => {
        expect(
          screen.getByRole("button", { name: /Test Connection/ }),
        ).toBeInTheDocument();
      });

      await user.click(
        screen.getByRole("button", { name: /Test Connection/ }),
      );

      await waitFor(() => {
        expect(screen.getByText("testuser")).toBeInTheDocument();
      });
    });

    it("shows error on test failure", async () => {
      const user = userEvent.setup();
      vi.mocked(settings.testHuggingFaceToken).mockRejectedValue(
        new Error("Connection refused"),
      );

      render(HuggingFaceSettings);

      await waitFor(() => {
        expect(
          screen.getByRole("button", { name: /Test Connection/ }),
        ).toBeInTheDocument();
      });

      await user.click(
        screen.getByRole("button", { name: /Test Connection/ }),
      );

      await waitFor(() => {
        expect(screen.getByText("Connection refused")).toBeInTheDocument();
      });
    });

    it("shows generic error on test failure with non-Error", async () => {
      const user = userEvent.setup();
      vi.mocked(settings.testHuggingFaceToken).mockRejectedValue("unknown");

      render(HuggingFaceSettings);

      await waitFor(() => {
        expect(
          screen.getByRole("button", { name: /Test Connection/ }),
        ).toBeInTheDocument();
      });

      await user.click(
        screen.getByRole("button", { name: /Test Connection/ }),
      );

      await waitFor(() => {
        expect(screen.getByText("Connection test failed")).toBeInTheDocument();
      });
    });

    it("shows testing loading state", async () => {
      const user = userEvent.setup();
      let resolveTest: () => void;
      const testPromise = new Promise<void>((resolve) => {
        resolveTest = resolve;
      });
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      vi.mocked(settings.testHuggingFaceToken).mockReturnValue(testPromise as any);

      render(HuggingFaceSettings);

      await waitFor(() => {
        expect(
          screen.getByRole("button", { name: /Test Connection/ }),
        ).toBeInTheDocument();
      });

      await user.click(
        screen.getByRole("button", { name: /Test Connection/ }),
      );

      expect(screen.getByText("Testing...")).toBeInTheDocument();

      resolveTest!();
      await testPromise;
    });
  });

  describe("delete action", () => {
    beforeEach(() => {
      vi.mocked(settings.getHuggingFaceStatus).mockResolvedValue({
        configured: true,
      });
    });

    it("calls deleteHuggingFaceToken when Remove clicked", async () => {
      const user = userEvent.setup();
      render(HuggingFaceSettings);

      await waitFor(() => {
        expect(
          screen.getByRole("button", { name: /Remove/ }),
        ).toBeInTheDocument();
      });

      await user.click(screen.getByRole("button", { name: /Remove/ }));

      await waitFor(() => {
        expect(settings.deleteHuggingFaceToken).toHaveBeenCalled();
      });
    });

    it("sets configured to false after successful delete", async () => {
      const user = userEvent.setup();
      render(HuggingFaceSettings);

      await waitFor(() => {
        expect(screen.getByText("Token configured")).toBeInTheDocument();
      });

      await user.click(screen.getByRole("button", { name: /Remove/ }));

      await waitFor(() => {
        expect(
          screen.getByText("No token configured (using public access)"),
        ).toBeInTheDocument();
      });
    });

    it("hides Test Connection and Remove buttons after delete", async () => {
      const user = userEvent.setup();
      render(HuggingFaceSettings);

      await waitFor(() => {
        expect(
          screen.getByRole("button", { name: /Remove/ }),
        ).toBeInTheDocument();
      });

      await user.click(screen.getByRole("button", { name: /Remove/ }));

      await waitFor(() => {
        expect(
          screen.queryByRole("button", { name: /Test Connection/ }),
        ).not.toBeInTheDocument();
        expect(
          screen.queryByRole("button", { name: /Remove/ }),
        ).not.toBeInTheDocument();
      });
    });

    it("shows error when delete fails", async () => {
      const user = userEvent.setup();
      vi.mocked(settings.deleteHuggingFaceToken).mockRejectedValue(
        new Error("Delete failed"),
      );

      render(HuggingFaceSettings);

      await waitFor(() => {
        expect(
          screen.getByRole("button", { name: /Remove/ }),
        ).toBeInTheDocument();
      });

      await user.click(screen.getByRole("button", { name: /Remove/ }));

      await waitFor(() => {
        expect(screen.getByText("Delete failed")).toBeInTheDocument();
      });
    });
  });

  describe("feedback clearing", () => {
    it("clears error when typing new token", async () => {
      const user = userEvent.setup();
      vi.mocked(settings.getHuggingFaceStatus).mockResolvedValue({
        configured: true,
      });
      vi.mocked(settings.testHuggingFaceToken).mockRejectedValue(
        new Error("Bad token"),
      );

      render(HuggingFaceSettings);

      await waitFor(() => {
        expect(
          screen.getByRole("button", { name: /Test Connection/ }),
        ).toBeInTheDocument();
      });

      // Trigger an error
      await user.click(
        screen.getByRole("button", { name: /Test Connection/ }),
      );

      await waitFor(() => {
        expect(screen.getByText("Bad token")).toBeInTheDocument();
      });

      // Type in input should clear error
      await user.type(screen.getByLabelText("Access Token"), "h");

      await waitFor(() => {
        expect(screen.queryByText("Bad token")).not.toBeInTheDocument();
      });
    });

    it("clears test result when typing new token", async () => {
      const user = userEvent.setup();
      vi.mocked(settings.getHuggingFaceStatus).mockResolvedValue({
        configured: true,
      });

      render(HuggingFaceSettings);

      await waitFor(() => {
        expect(
          screen.getByRole("button", { name: /Test Connection/ }),
        ).toBeInTheDocument();
      });

      // Trigger a successful test
      await user.click(
        screen.getByRole("button", { name: /Test Connection/ }),
      );

      await waitFor(() => {
        expect(screen.getByText("testuser")).toBeInTheDocument();
      });

      // Type in input should clear test result
      await user.type(screen.getByLabelText("Access Token"), "h");

      await waitFor(() => {
        expect(screen.queryByText("testuser")).not.toBeInTheDocument();
      });
    });
  });

  describe("load status error", () => {
    it("shows error when status load fails", async () => {
      vi.mocked(settings.getHuggingFaceStatus).mockRejectedValue(
        new Error("Server error"),
      );

      render(HuggingFaceSettings);

      await waitFor(() => {
        expect(
          screen.getByText("Failed to load HuggingFace token status"),
        ).toBeInTheDocument();
      });
    });
  });
});
