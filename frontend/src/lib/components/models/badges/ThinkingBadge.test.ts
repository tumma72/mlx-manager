import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/svelte";

import ThinkingBadge from "./ThinkingBadge.svelte";

describe("ThinkingBadge", () => {
  it("renders 'Thinking' text", () => {
    render(ThinkingBadge);
    expect(screen.getByText("Thinking")).toBeInTheDocument();
  });

  it("has teal color classes", () => {
    const { container } = render(ThinkingBadge);
    const badge = container.querySelector("div");
    expect(badge?.classList.contains("bg-teal-100")).toBe(true);
    expect(badge?.classList.contains("text-teal-800")).toBe(true);
    expect(badge?.classList.contains("border-teal-300")).toBe(true);
  });

  it("has rounded-full styling", () => {
    const { container } = render(ThinkingBadge);
    const badge = container.querySelector("div");
    expect(badge?.classList.contains("rounded-full")).toBe(true);
  });

  it("shows checkmark when verified", () => {
    const { container } = render(ThinkingBadge, { props: { verified: true } });
    const badge = container.querySelector("div");
    expect(badge?.getAttribute("title")).toBe("Verified by model probe");
    // CheckCircle2 icon renders as an SVG
    expect(badge?.querySelectorAll("svg").length).toBe(2);
  });

  it("does not show checkmark when not verified", () => {
    const { container } = render(ThinkingBadge);
    const badge = container.querySelector("div");
    expect(badge?.getAttribute("title")).toBe(
      "Detected from model configuration",
    );
    // Only Brain icon, no CheckCircle2
    expect(badge?.querySelectorAll("svg").length).toBe(1);
  });
});
