import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/svelte";
import MetricGauge from "./MetricGauge.svelte";

describe("MetricGauge", () => {
  describe("rendering", () => {
    it("renders with required props", () => {
      render(MetricGauge, { props: { value: 50, label: "Memory" } });

      expect(screen.getByText("50%")).toBeInTheDocument();
      expect(screen.getByText("Memory")).toBeInTheDocument();
    });

    it("renders value with custom unit", () => {
      render(MetricGauge, { props: { value: 256, label: "RAM", unit: "MB" } });

      expect(screen.getByText("256MB")).toBeInTheDocument();
      expect(screen.getByText("RAM")).toBeInTheDocument();
    });

    it("renders value rounded to integer", () => {
      render(MetricGauge, { props: { value: 75.7, label: "CPU" } });

      expect(screen.getByText("76%")).toBeInTheDocument();
    });

    it("renders with custom max value", () => {
      render(MetricGauge, { props: { value: 50, max: 200, label: "Usage" } });

      // 50/200 = 25%, but value display shows the raw value
      expect(screen.getByText("50%")).toBeInTheDocument();
    });
  });

  describe("color thresholds", () => {
    it("applies green color for values below warning threshold", () => {
      const { container } = render(MetricGauge, {
        props: { value: 50, label: "Memory" },
      });

      // Progress circle should have green color class
      const progressCircle = container.querySelectorAll("circle")[1];
      expect(progressCircle.classList.contains("text-green-500")).toBe(true);
    });

    it("applies yellow color for values at warning threshold", () => {
      const { container } = render(MetricGauge, {
        props: { value: 75, label: "Memory" },
      });

      const progressCircle = container.querySelectorAll("circle")[1];
      expect(progressCircle.classList.contains("text-yellow-500")).toBe(true);
    });

    it("applies yellow color for values between warning and danger", () => {
      const { container } = render(MetricGauge, {
        props: { value: 85, label: "Memory" },
      });

      const progressCircle = container.querySelectorAll("circle")[1];
      expect(progressCircle.classList.contains("text-yellow-500")).toBe(true);
    });

    it("applies red color for values at danger threshold", () => {
      const { container } = render(MetricGauge, {
        props: { value: 90, label: "Memory" },
      });

      const progressCircle = container.querySelectorAll("circle")[1];
      expect(progressCircle.classList.contains("text-red-500")).toBe(true);
    });

    it("applies red color for values above danger threshold", () => {
      const { container } = render(MetricGauge, {
        props: { value: 95, label: "Memory" },
      });

      const progressCircle = container.querySelectorAll("circle")[1];
      expect(progressCircle.classList.contains("text-red-500")).toBe(true);
    });

    it("uses custom thresholds", () => {
      const { container } = render(MetricGauge, {
        props: {
          value: 50,
          label: "Custom",
          thresholds: { warning: 30, danger: 60 },
        },
      });

      // 50 >= 30 (warning) but < 60 (danger) = yellow
      const progressCircle = container.querySelectorAll("circle")[1];
      expect(progressCircle.classList.contains("text-yellow-500")).toBe(true);
    });

    it("applies red with custom danger threshold", () => {
      const { container } = render(MetricGauge, {
        props: {
          value: 65,
          label: "Custom",
          thresholds: { warning: 30, danger: 60 },
        },
      });

      // 65 >= 60 (danger) = red
      const progressCircle = container.querySelectorAll("circle")[1];
      expect(progressCircle.classList.contains("text-red-500")).toBe(true);
    });
  });

  describe("size variants", () => {
    it("renders medium size by default", () => {
      const { container } = render(MetricGauge, {
        props: { value: 50, label: "Memory" },
      });

      const wrapper = container.querySelector("div");
      expect(wrapper?.style.width).toBe("72px");
      expect(wrapper?.style.height).toBe("72px");
    });

    it("renders small size", () => {
      const { container } = render(MetricGauge, {
        props: { value: 50, label: "Memory", size: "sm" },
      });

      const wrapper = container.querySelector("div");
      expect(wrapper?.style.width).toBe("56px");
      expect(wrapper?.style.height).toBe("56px");
    });
  });

  describe("percentage clamping", () => {
    it("clamps percentage at 0 for negative values", () => {
      const { container } = render(MetricGauge, {
        props: { value: -10, label: "Memory" },
      });

      // Value should still display as-is, but the circle should show 0%
      expect(screen.getByText("-10%")).toBeInTheDocument();

      // Check that progress circle has the correct offset (should be at 0%)
      const progressCircle = container.querySelectorAll("circle")[1];
      const circumference = 2 * Math.PI * 40;
      // At 0%, offset = circumference (no progress shown)
      expect(progressCircle.getAttribute("stroke-dashoffset")).toBe(
        circumference.toString()
      );
    });

    it("clamps percentage at 100 for values above max", () => {
      const { container } = render(MetricGauge, {
        props: { value: 150, label: "Memory", max: 100 },
      });

      // Value should display as-is
      expect(screen.getByText("150%")).toBeInTheDocument();

      // Check that progress circle has the correct offset (should be at 100%)
      const progressCircle = container.querySelectorAll("circle")[1];
      // At 100%, offset = 0 (full progress shown)
      expect(progressCircle.getAttribute("stroke-dashoffset")).toBe("0");
    });
  });

  describe("SVG structure", () => {
    it("renders SVG with correct viewBox", () => {
      const { container } = render(MetricGauge, {
        props: { value: 50, label: "Memory" },
      });

      const svg = container.querySelector("svg");
      expect(svg?.getAttribute("viewBox")).toBe("0 0 100 100");
    });

    it("renders two circles (background and progress)", () => {
      const { container } = render(MetricGauge, {
        props: { value: 50, label: "Memory" },
      });

      const circles = container.querySelectorAll("circle");
      expect(circles).toHaveLength(2);
    });

    it("sets correct stroke-dasharray on progress circle", () => {
      const { container } = render(MetricGauge, {
        props: { value: 50, label: "Memory" },
      });

      const circumference = 2 * Math.PI * 40;
      const progressCircle = container.querySelectorAll("circle")[1];
      expect(progressCircle.getAttribute("stroke-dasharray")).toBe(
        circumference.toString()
      );
    });

    it("calculates correct stroke-dashoffset for 50%", () => {
      const { container } = render(MetricGauge, {
        props: { value: 50, label: "Memory" },
      });

      const circumference = 2 * Math.PI * 40;
      const expectedOffset = circumference - (50 / 100) * circumference;
      const progressCircle = container.querySelectorAll("circle")[1];
      expect(progressCircle.getAttribute("stroke-dashoffset")).toBe(
        expectedOffset.toString()
      );
    });

    it("calculates correct stroke-dashoffset for 0%", () => {
      const { container } = render(MetricGauge, {
        props: { value: 0, label: "Memory" },
      });

      const circumference = 2 * Math.PI * 40;
      // At 0%, offset = circumference
      const progressCircle = container.querySelectorAll("circle")[1];
      expect(progressCircle.getAttribute("stroke-dashoffset")).toBe(
        circumference.toString()
      );
    });

    it("calculates correct stroke-dashoffset for 100%", () => {
      const { container } = render(MetricGauge, {
        props: { value: 100, label: "Memory" },
      });

      // At 100%, offset = 0
      const progressCircle = container.querySelectorAll("circle")[1];
      expect(progressCircle.getAttribute("stroke-dashoffset")).toBe("0");
    });
  });
});
