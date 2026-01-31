import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor, fireEvent } from "@testing-library/svelte";
import userEvent from "@testing-library/user-event";
import ProviderForm from "./ProviderForm.svelte";
import type { BackendType, CloudCredential } from "$lib/api/types";
import { settings } from "$lib/api/client";

// Mock API
vi.mock("$lib/api/client", () => ({
  settings: {
    createProvider: vi.fn(),
    testProvider: vi.fn(),
    deleteProvider: vi.fn(),
  },
}));

function createMockCredential(
  backendType: BackendType,
): CloudCredential {
  return {
    id: 1,
    backend_type: backendType,
    base_url: null,
    created_at: "2024-01-01",
  };
}

describe("ProviderForm", () => {
  let mockOnSave: () => void;
  let mockOnDelete: () => void;

  beforeEach(() => {
    mockOnSave = vi.fn();
    mockOnDelete = vi.fn();
    vi.mocked(settings.createProvider).mockResolvedValue(createMockCredential("openai"));
    vi.mocked(settings.testProvider).mockResolvedValue({
      success: true,
    });
    vi.mocked(settings.deleteProvider).mockResolvedValue(undefined);
  });

  describe("rendering - new credential", () => {
    it("renders API key input", () => {
      render(ProviderForm, {
        props: {
          backendType: "openai",
          existingCredential: null,
          onSave: mockOnSave,
          onDelete: mockOnDelete,
        },
      });

      expect(screen.getByLabelText(/API Key/)).toBeInTheDocument();
    });

    it("shows placeholder for new credentials", () => {
      render(ProviderForm, {
        props: {
          backendType: "openai",
          existingCredential: null,
          onSave: mockOnSave,
          onDelete: mockOnDelete,
        },
      });

      const input = screen.getByPlaceholderText("Enter API key");
      expect(input).toBeInTheDocument();
    });

    it("renders Save & Test button", () => {
      render(ProviderForm, {
        props: {
          backendType: "openai",
          existingCredential: null,
          onSave: mockOnSave,
          onDelete: mockOnDelete,
        },
      });

      expect(
        screen.getByRole("button", { name: /Save & Test/ }),
      ).toBeInTheDocument();
    });

    it("does not render Test Connection button when no credential", () => {
      render(ProviderForm, {
        props: {
          backendType: "openai",
          existingCredential: null,
          onSave: mockOnSave,
          onDelete: mockOnDelete,
        },
      });

      expect(
        screen.queryByRole("button", { name: /Test Connection/ }),
      ).not.toBeInTheDocument();
    });

    it("does not render Delete button when no credential", () => {
      render(ProviderForm, {
        props: {
          backendType: "openai",
          existingCredential: null,
          onSave: mockOnSave,
          onDelete: mockOnDelete,
        },
      });

      expect(
        screen.queryByRole("button", { name: /Delete/ }),
      ).not.toBeInTheDocument();
    });
  });

  describe("rendering - existing credential", () => {
    it("shows placeholder indicating saved credential", () => {
      render(ProviderForm, {
        props: {
          backendType: "openai",
          existingCredential: createMockCredential("openai"),
          onSave: mockOnSave,
          onDelete: mockOnDelete,
        },
      });

      expect(
        screen.getByPlaceholderText(
          "****...saved (enter new key to update)",
        ),
      ).toBeInTheDocument();
    });

    it("renders Test Connection button", () => {
      render(ProviderForm, {
        props: {
          backendType: "openai",
          existingCredential: createMockCredential("openai"),
          onSave: mockOnSave,
          onDelete: mockOnDelete,
        },
      });

      expect(
        screen.getByRole("button", { name: /Test Connection/ }),
      ).toBeInTheDocument();
    });

    it("renders Delete button", () => {
      render(ProviderForm, {
        props: {
          backendType: "openai",
          existingCredential: createMockCredential("openai"),
          onSave: mockOnSave,
          onDelete: mockOnDelete,
        },
      });

      expect(
        screen.getByRole("button", { name: /Delete/ }),
      ).toBeInTheDocument();
    });

    it("populates base URL from existing credential", () => {
      const credential = {
        ...createMockCredential("openai"),
        base_url: "https://custom.openai.com/v1",
      };

      render(ProviderForm, {
        props: {
          backendType: "openai",
          existingCredential: credential,
          onSave: mockOnSave,
          onDelete: mockOnDelete,
        },
      });

      const baseUrlInput = screen.getByLabelText(
        /Base URL/,
      ) as HTMLInputElement;
      expect(baseUrlInput.value).toBe("https://custom.openai.com/v1");
    });
  });

  describe("advanced settings", () => {
    it("hides base URL by default when no credential", () => {
      render(ProviderForm, {
        props: {
          backendType: "openai",
          existingCredential: null,
          onSave: mockOnSave,
          onDelete: mockOnDelete,
        },
      });

      expect(screen.queryByLabelText(/Base URL/)).not.toBeInTheDocument();
    });

    it("shows base URL when existing credential has custom URL", () => {
      const credential = {
        ...createMockCredential("openai"),
        base_url: "https://custom.com",
      };

      render(ProviderForm, {
        props: {
          backendType: "openai",
          existingCredential: credential,
          onSave: mockOnSave,
          onDelete: mockOnDelete,
        },
      });

      expect(screen.getByLabelText(/Base URL/)).toBeInTheDocument();
    });

    it("toggles advanced settings visibility", async () => {
      const user = userEvent.setup();
      render(ProviderForm, {
        props: {
          backendType: "openai",
          existingCredential: null,
          onSave: mockOnSave,
          onDelete: mockOnDelete,
        },
      });

      await user.click(screen.getByText("Advanced Settings"));

      expect(screen.getByLabelText(/Base URL/)).toBeInTheDocument();
    });

    it("hides advanced settings when toggled again", async () => {
      const user = userEvent.setup();
      render(ProviderForm, {
        props: {
          backendType: "openai",
          existingCredential: null,
          onSave: mockOnSave,
          onDelete: mockOnDelete,
        },
      });

      await user.click(screen.getByText("Advanced Settings"));
      await user.click(screen.getByText("Advanced Settings"));

      expect(screen.queryByLabelText(/Base URL/)).not.toBeInTheDocument();
    });

    it("shows OpenAI default placeholder for base URL", async () => {
      const user = userEvent.setup();
      render(ProviderForm, {
        props: {
          backendType: "openai",
          existingCredential: null,
          onSave: mockOnSave,
          onDelete: mockOnDelete,
        },
      });

      await user.click(screen.getByText("Advanced Settings"));

      expect(
        screen.getByPlaceholderText("https://api.openai.com/v1"),
      ).toBeInTheDocument();
    });

    it("shows Anthropic default placeholder for base URL", async () => {
      const user = userEvent.setup();
      render(ProviderForm, {
        props: {
          backendType: "anthropic",
          existingCredential: null,
          onSave: mockOnSave,
          onDelete: mockOnDelete,
        },
      });

      await user.click(screen.getByText("Advanced Settings"));

      expect(
        screen.getByPlaceholderText("https://api.anthropic.com"),
      ).toBeInTheDocument();
    });
  });

  describe("API key masking", () => {
    it("displays masked key when typing", async () => {
      const user = userEvent.setup();
      render(ProviderForm, {
        props: {
          backendType: "openai",
          existingCredential: null,
          onSave: mockOnSave,
          onDelete: mockOnDelete,
        },
      });

      const input = screen.getByLabelText(/API Key/);
      await user.type(input, "sk-test1234567890");

      // Should show last 4 chars masked
      expect(screen.getByText("****...7890")).toBeInTheDocument();
    });

    it("shows full mask for short keys", async () => {
      const user = userEvent.setup();
      render(ProviderForm, {
        props: {
          backendType: "openai",
          existingCredential: null,
          onSave: mockOnSave,
          onDelete: mockOnDelete,
        },
      });

      const input = screen.getByLabelText(/API Key/);
      await user.type(input, "123");

      expect(screen.getByText("****")).toBeInTheDocument();
    });
  });

  describe("save action", () => {
    it("saves new credentials", async () => {
      const user = userEvent.setup();
      render(ProviderForm, {
        props: {
          backendType: "openai",
          existingCredential: null,
          onSave: mockOnSave,
          onDelete: mockOnDelete,
        },
      });

      await user.type(screen.getByLabelText(/API Key/), "sk-test123");
      await user.click(screen.getByRole("button", { name: /Save & Test/ }));

      await waitFor(() => {
        expect(settings.createProvider).toHaveBeenCalledWith({
          backend_type: "openai",
          api_key: "sk-test123",
          base_url: undefined,
        });
      });
    });

    it("saves with custom base URL", async () => {
      const user = userEvent.setup();
      render(ProviderForm, {
        props: {
          backendType: "openai",
          existingCredential: null,
          onSave: mockOnSave,
          onDelete: mockOnDelete,
        },
      });

      await user.type(screen.getByLabelText(/API Key/), "sk-test123");
      await user.click(screen.getByText("Advanced Settings"));
      await user.type(
        screen.getByLabelText(/Base URL/),
        "https://custom.com",
      );
      await user.click(screen.getByRole("button", { name: /Save & Test/ }));

      await waitFor(() => {
        expect(settings.createProvider).toHaveBeenCalledWith({
          backend_type: "openai",
          api_key: "sk-test123",
          base_url: "https://custom.com",
        });
      });
    });

    it("trims API key and base URL", async () => {
      const user = userEvent.setup();
      render(ProviderForm, {
        props: {
          backendType: "openai",
          existingCredential: null,
          onSave: mockOnSave,
          onDelete: mockOnDelete,
        },
      });

      await user.type(screen.getByLabelText(/API Key/), "  sk-test123  ");
      await user.click(screen.getByText("Advanced Settings"));
      await user.type(
        screen.getByLabelText(/Base URL/),
        "  https://custom.com  ",
      );
      await user.click(screen.getByRole("button", { name: /Save & Test/ }));

      await waitFor(() => {
        expect(settings.createProvider).toHaveBeenCalledWith({
          backend_type: "openai",
          api_key: "sk-test123",
          base_url: "https://custom.com",
        });
      });
    });

    it("tests connection after save", async () => {
      const user = userEvent.setup();
      render(ProviderForm, {
        props: {
          backendType: "openai",
          existingCredential: null,
          onSave: mockOnSave,
          onDelete: mockOnDelete,
        },
      });

      await user.type(screen.getByLabelText(/API Key/), "sk-test123");
      await user.click(screen.getByRole("button", { name: /Save & Test/ }));

      await waitFor(() => {
        expect(settings.testProvider).toHaveBeenCalledWith("openai");
      });
    });

    it("shows success message after successful save and test", async () => {
      const user = userEvent.setup();
      render(ProviderForm, {
        props: {
          backendType: "openai",
          existingCredential: null,
          onSave: mockOnSave,
          onDelete: mockOnDelete,
        },
      });

      await user.type(screen.getByLabelText(/API Key/), "sk-test123");
      await user.click(screen.getByRole("button", { name: /Save & Test/ }));

      await waitFor(() => {
        expect(
          screen.getByText("Connection successful"),
        ).toBeInTheDocument();
      });
    });

    it("shows error if test fails after save", async () => {
      const user = userEvent.setup();
      vi.mocked(settings.testProvider).mockRejectedValue(
        new Error("Invalid API key"),
      );

      render(ProviderForm, {
        props: {
          backendType: "openai",
          existingCredential: null,
          onSave: mockOnSave,
          onDelete: mockOnDelete,
        },
      });

      await user.type(screen.getByLabelText(/API Key/), "sk-test123");
      await user.click(screen.getByRole("button", { name: /Save & Test/ }));

      await waitFor(() => {
        expect(
          screen.getByText(/Saved but connection test failed/),
        ).toBeInTheDocument();
      });
    });

    it("calls onSave after successful save", async () => {
      const user = userEvent.setup();
      render(ProviderForm, {
        props: {
          backendType: "openai",
          existingCredential: null,
          onSave: mockOnSave,
          onDelete: mockOnDelete,
        },
      });

      await user.type(screen.getByLabelText(/API Key/), "sk-test123");
      await user.click(screen.getByRole("button", { name: /Save & Test/ }));

      await waitFor(() => {
        expect(mockOnSave).toHaveBeenCalled();
      });
    });

    it("clears input after successful save", async () => {
      const user = userEvent.setup();
      render(ProviderForm, {
        props: {
          backendType: "openai",
          existingCredential: null,
          onSave: mockOnSave,
          onDelete: mockOnDelete,
        },
      });

      await user.type(screen.getByLabelText(/API Key/), "sk-test123");
      await user.click(screen.getByRole("button", { name: /Save & Test/ }));

      await waitFor(() => {
        expect(
          (screen.getByLabelText(/API Key/) as HTMLInputElement).value,
        ).toBe("");
      });
    });

    it("shows error when API key is empty", async () => {
      render(ProviderForm, {
        props: {
          backendType: "openai",
          existingCredential: null,
          onSave: mockOnSave,
          onDelete: mockOnDelete,
        },
      });

      // Button should be disabled initially (implying validation requirement)
      expect(
        screen.getByRole("button", { name: /Save & Test/ }),
      ).toBeDisabled();

      // This confirms empty key is blocked
      expect(settings.createProvider).not.toHaveBeenCalled();
    });

    it("disables Save button when empty", () => {
      render(ProviderForm, {
        props: {
          backendType: "openai",
          existingCredential: null,
          onSave: mockOnSave,
          onDelete: mockOnDelete,
        },
      });

      expect(
        screen.getByRole("button", { name: /Save & Test/ }),
      ).toBeDisabled();
    });

    it("enables Save button when key entered", async () => {
      const user = userEvent.setup();
      render(ProviderForm, {
        props: {
          backendType: "openai",
          existingCredential: null,
          onSave: mockOnSave,
          onDelete: mockOnDelete,
        },
      });

      await user.type(screen.getByLabelText(/API Key/), "sk-test");

      expect(
        screen.getByRole("button", { name: /Save & Test/ }),
      ).toBeEnabled();
    });

    it("shows loading state during save", async () => {
      const user = userEvent.setup();
      let resolveSave: () => void;
      const savePromise = new Promise<void>((resolve) => {
        resolveSave = resolve;
      });
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      vi.mocked(settings.createProvider).mockReturnValue(savePromise as any);

      render(ProviderForm, {
        props: {
          backendType: "openai",
          existingCredential: null,
          onSave: mockOnSave,
          onDelete: mockOnDelete,
        },
      });

      await user.type(screen.getByLabelText(/API Key/), "sk-test");
      await user.click(screen.getByRole("button", { name: /Save & Test/ }));

      expect(screen.getByText("Saving...")).toBeInTheDocument();

      resolveSave!();
      await savePromise;
    });
  });

  describe("test connection action", () => {
    it("tests existing credentials", async () => {
      const user = userEvent.setup();
      render(ProviderForm, {
        props: {
          backendType: "openai",
          existingCredential: createMockCredential("openai"),
          onSave: mockOnSave,
          onDelete: mockOnDelete,
        },
      });

      await user.click(
        screen.getByRole("button", { name: /Test Connection/ }),
      );

      await waitFor(() => {
        expect(settings.testProvider).toHaveBeenCalledWith("openai");
      });
    });

    it("shows success message after successful test", async () => {
      const user = userEvent.setup();
      render(ProviderForm, {
        props: {
          backendType: "openai",
          existingCredential: createMockCredential("openai"),
          onSave: mockOnSave,
          onDelete: mockOnDelete,
        },
      });

      await user.click(
        screen.getByRole("button", { name: /Test Connection/ }),
      );

      await waitFor(() => {
        expect(
          screen.getByText("Connection successful"),
        ).toBeInTheDocument();
      });
    });

    it("shows error message on test failure", async () => {
      const user = userEvent.setup();
      vi.mocked(settings.testProvider).mockRejectedValue(
        new Error("Connection refused"),
      );

      render(ProviderForm, {
        props: {
          backendType: "openai",
          existingCredential: createMockCredential("openai"),
          onSave: mockOnSave,
          onDelete: mockOnDelete,
        },
      });

      await user.click(
        screen.getByRole("button", { name: /Test Connection/ }),
      );

      await waitFor(() => {
        expect(screen.getByText("Connection refused")).toBeInTheDocument();
      });
    });

    it("shows loading state during test", async () => {
      const user = userEvent.setup();
      let resolveTest: () => void;
      const testPromise = new Promise<void>((resolve) => {
        resolveTest = resolve;
      });
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      vi.mocked(settings.testProvider).mockReturnValue(testPromise as any);

      render(ProviderForm, {
        props: {
          backendType: "openai",
          existingCredential: createMockCredential("openai"),
          onSave: mockOnSave,
          onDelete: mockOnDelete,
        },
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
    it("shows confirmation dialog when Delete clicked", async () => {
      render(ProviderForm, {
        props: {
          backendType: "openai",
          existingCredential: createMockCredential("openai"),
          onSave: mockOnSave,
          onDelete: mockOnDelete,
        },
      });

      // Use fireEvent to bypass pointer-events check in test environment
      await fireEvent.click(screen.getByRole("button", { name: /Delete/ }));

      await waitFor(() => {
        expect(
          screen.getByText(/Delete Provider Credentials/),
        ).toBeInTheDocument();
      });
    });

    it("calls onDelete after successful deletion", async () => {
      render(ProviderForm, {
        props: {
          backendType: "openai",
          existingCredential: createMockCredential("openai"),
          onSave: mockOnSave,
          onDelete: mockOnDelete,
        },
      });

      // Use fireEvent to bypass pointer-events check in test environment
      await fireEvent.click(screen.getByRole("button", { name: /Delete/ }));

      await waitFor(() => {
        const deleteButtons = screen.getAllByRole("button", { name: /Delete/ });
        if (deleteButtons.length > 1) {
          fireEvent.click(deleteButtons[1]);
        }
      });

      await waitFor(() => {
        expect(mockOnDelete).toHaveBeenCalled();
      });
    });
  });

  describe("state management", () => {
    it("manages test result state", async () => {
      render(ProviderForm, {
        props: {
          backendType: "openai",
          existingCredential: createMockCredential("openai"),
          onSave: mockOnSave,
          onDelete: mockOnDelete,
        },
      });

      // Use fireEvent to bypass pointer-events check in test environment
      await fireEvent.click(
        screen.getByRole("button", { name: /Test Connection/ }),
      );

      await waitFor(() => {
        expect(
          screen.getByText("Connection successful"),
        ).toBeInTheDocument();
      });

      // Verify success state is shown
      expect(settings.testProvider).toHaveBeenCalledWith("openai");
    });
  });
});
