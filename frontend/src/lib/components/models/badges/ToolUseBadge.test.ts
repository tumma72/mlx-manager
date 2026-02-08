import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/svelte";

import ToolUseBadge from "./ToolUseBadge.svelte";

describe("ToolUseBadge", () => {
  it("renders 'Tool Use' text", () => {
    render(ToolUseBadge);
    expect(screen.getByText("Tool Use")).toBeInTheDocument();
  });

  it("has amber color classes", () => {
    const { container } = render(ToolUseBadge);
    const badge = container.querySelector("div");
    expect(badge?.classList.contains("bg-amber-100")).toBe(true);
    expect(badge?.classList.contains("text-amber-800")).toBe(true);
  });

  it("defaults to unverified title", () => {
    const { container } = render(ToolUseBadge);
    const badge = container.querySelector("[title]");
    expect(badge?.getAttribute("title")).toBe("Detected from model name/tags");
  });

  it("shows verified title when verified=true", () => {
    const { container } = render(ToolUseBadge, {
      props: { verified: true },
    });
    const badge = container.querySelector("[title]");
    expect(badge?.getAttribute("title")).toBe("Verified by model probe");
  });

  it("does not show checkmark icon by default", () => {
    const { container } = render(ToolUseBadge);
    // Only the Wrench icon should be present (1 svg)
    const svgs = container.querySelectorAll("svg");
    expect(svgs.length).toBe(1);
  });

  it("shows checkmark icon when verified=true", () => {
    const { container } = render(ToolUseBadge, {
      props: { verified: true },
    });
    // Wrench + CheckCircle2 icons (2 svgs)
    const svgs = container.querySelectorAll("svg");
    expect(svgs.length).toBe(2);
  });
});
