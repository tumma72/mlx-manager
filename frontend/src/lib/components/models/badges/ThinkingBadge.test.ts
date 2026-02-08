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
});
