import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/svelte";
import userEvent from "@testing-library/user-event";

import ModelToggle from "./ModelToggle.svelte";

describe("ModelToggle", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe("rendering", () => {
    it("renders both toggle buttons", () => {
      render(ModelToggle, {
        props: { mode: "local" },
      });

      expect(screen.getByRole("button", { name: /my models/i })).toBeInTheDocument();
      expect(
        screen.getByRole("button", { name: /huggingface/i }),
      ).toBeInTheDocument();
    });

    it("highlights local button when mode is local", () => {
      render(ModelToggle, {
        props: { mode: "local" },
      });

      const localButton = screen.getByRole("button", { name: /my models/i });
      expect(localButton.classList.contains("bg-background")).toBe(true);
      expect(localButton.classList.contains("shadow-sm")).toBe(true);
    });

    it("highlights online button when mode is online", () => {
      render(ModelToggle, {
        props: { mode: "online" },
      });

      const onlineButton = screen.getByRole("button", { name: /huggingface/i });
      expect(onlineButton.classList.contains("bg-background")).toBe(true);
      expect(onlineButton.classList.contains("shadow-sm")).toBe(true);
    });

    it("applies muted styling to non-selected button", () => {
      render(ModelToggle, {
        props: { mode: "local" },
      });

      const onlineButton = screen.getByRole("button", { name: /huggingface/i });
      expect(onlineButton.classList.contains("text-muted-foreground")).toBe(true);
    });
  });

  describe("mode switching", () => {
    it("switches to online mode when HuggingFace button clicked", async () => {
      const user = userEvent.setup();
      const onModeChange = vi.fn();

      render(ModelToggle, {
        props: {
          mode: "local",
          onModeChange,
        },
      });

      const onlineButton = screen.getByRole("button", { name: /huggingface/i });
      await user.click(onlineButton);

      expect(onModeChange).toHaveBeenCalledWith("online");
    });

    it("switches to local mode when My Models button clicked", async () => {
      const user = userEvent.setup();
      const onModeChange = vi.fn();

      render(ModelToggle, {
        props: {
          mode: "online",
          onModeChange,
        },
      });

      const localButton = screen.getByRole("button", { name: /my models/i });
      await user.click(localButton);

      expect(onModeChange).toHaveBeenCalledWith("local");
    });

    it("works without onModeChange callback", async () => {
      const user = userEvent.setup();

      render(ModelToggle, {
        props: {
          mode: "local",
        },
      });

      const onlineButton = screen.getByRole("button", { name: /huggingface/i });
      await user.click(onlineButton);

      // Should not throw error
      expect(onlineButton.classList.contains("bg-background")).toBe(true);
    });

    it("updates styling when mode changes", async () => {
      const user = userEvent.setup();

      const { component: _component } = render(ModelToggle, {
        props: {
          mode: "local",
        },
      });

      const localButton = screen.getByRole("button", { name: /my models/i });
      const onlineButton = screen.getByRole("button", { name: /huggingface/i });

      // Initially local is selected
      expect(localButton.classList.contains("bg-background")).toBe(true);
      expect(onlineButton.classList.contains("text-muted-foreground")).toBe(true);

      // Click online
      await user.click(onlineButton);

      // Now online should be selected
      expect(onlineButton.classList.contains("bg-background")).toBe(true);
      expect(localButton.classList.contains("text-muted-foreground")).toBe(true);
    });
  });

  describe("accessibility", () => {
    it("renders buttons with type='button'", () => {
      const { container } = render(ModelToggle, {
        props: { mode: "local" },
      });

      const buttons = container.querySelectorAll("button");
      buttons.forEach((button: Element) => {
        expect(button).toHaveAttribute("type", "button");
      });
    });

    it("maintains focus state on buttons", async () => {
      const user = userEvent.setup();

      render(ModelToggle, {
        props: { mode: "local" },
      });

      const onlineButton = screen.getByRole("button", { name: /huggingface/i });
      await user.click(onlineButton);

      // Button should be focusable
      expect(onlineButton).toHaveFocus();
    });
  });

  describe("edge cases", () => {
    it("handles rapid mode switching", async () => {
      const user = userEvent.setup();
      const onModeChange = vi.fn();

      render(ModelToggle, {
        props: {
          mode: "local",
          onModeChange,
        },
      });

      const localButton = screen.getByRole("button", { name: /my models/i });
      const onlineButton = screen.getByRole("button", { name: /huggingface/i });

      // Rapidly switch modes
      await user.click(onlineButton);
      await user.click(localButton);
      await user.click(onlineButton);
      await user.click(localButton);

      expect(onModeChange).toHaveBeenCalledTimes(4);
      expect(onModeChange).toHaveBeenNthCalledWith(1, "online");
      expect(onModeChange).toHaveBeenNthCalledWith(2, "local");
      expect(onModeChange).toHaveBeenNthCalledWith(3, "online");
      expect(onModeChange).toHaveBeenNthCalledWith(4, "local");
    });

    it("clicking already selected button calls onModeChange", async () => {
      const user = userEvent.setup();
      const onModeChange = vi.fn();

      render(ModelToggle, {
        props: {
          mode: "local",
          onModeChange,
        },
      });

      const localButton = screen.getByRole("button", { name: /my models/i });
      await user.click(localButton);

      expect(onModeChange).toHaveBeenCalledWith("local");
    });
  });

  describe("styling consistency", () => {
    it("applies rounded-full to container", () => {
      const { container } = render(ModelToggle, {
        props: { mode: "local" },
      });

      const toggleContainer = container.querySelector(".rounded-full");
      expect(toggleContainer).toBeInTheDocument();
    });

    it("applies transition classes to buttons", () => {
      render(ModelToggle, {
        props: { mode: "local" },
      });

      const localButton = screen.getByRole("button", { name: /my models/i });
      const onlineButton = screen.getByRole("button", { name: /huggingface/i });

      expect(localButton.classList.contains("transition-all")).toBe(true);
      expect(onlineButton.classList.contains("transition-all")).toBe(true);
    });

    it("applies hover styles to non-selected button", () => {
      render(ModelToggle, {
        props: { mode: "local" },
      });

      const onlineButton = screen.getByRole("button", { name: /huggingface/i });
      expect(onlineButton.classList.contains("hover:text-foreground")).toBe(true);
    });
  });
});
