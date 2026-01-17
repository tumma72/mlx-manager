import { describe, it, expect } from "vitest";
import {
  formatBytes,
  formatDuration,
  formatNumber,
  formatRelativeTime,
  truncate,
} from "./format";

describe("formatBytes", () => {
  it('returns "0 Bytes" for zero', () => {
    expect(formatBytes(0)).toBe("0 Bytes");
  });

  it("formats bytes correctly", () => {
    expect(formatBytes(500)).toBe("500 Bytes");
  });

  it("formats kilobytes correctly", () => {
    expect(formatBytes(1024)).toBe("1 KB");
    expect(formatBytes(1536)).toBe("1.5 KB");
  });

  it("formats megabytes correctly", () => {
    expect(formatBytes(1024 * 1024)).toBe("1 MB");
    expect(formatBytes(1024 * 1024 * 1.5)).toBe("1.5 MB");
  });

  it("formats gigabytes correctly", () => {
    expect(formatBytes(1024 * 1024 * 1024)).toBe("1 GB");
  });

  it("respects decimal precision", () => {
    expect(formatBytes(1536, 0)).toBe("2 KB");
    expect(formatBytes(1536, 1)).toBe("1.5 KB");
    expect(formatBytes(1536, 3)).toBe("1.5 KB");
  });

  it("handles negative decimals by using 0", () => {
    expect(formatBytes(1536, -1)).toBe("2 KB");
    expect(formatBytes(1536, -5)).toBe("2 KB");
  });
});

describe("formatDuration", () => {
  it("formats seconds only", () => {
    expect(formatDuration(45)).toBe("45s");
  });

  it("formats minutes", () => {
    expect(formatDuration(120)).toBe("2m");
    expect(formatDuration(90)).toBe("1m");
  });

  it("formats hours and minutes", () => {
    expect(formatDuration(3600)).toBe("1h 0m");
    expect(formatDuration(3660)).toBe("1h 1m");
    expect(formatDuration(7200)).toBe("2h 0m");
  });
});

describe("formatNumber", () => {
  it("returns number as string for small numbers", () => {
    expect(formatNumber(0)).toBe("0");
    expect(formatNumber(999)).toBe("999");
  });

  it("formats thousands with K suffix", () => {
    expect(formatNumber(1000)).toBe("1.0K");
    expect(formatNumber(1500)).toBe("1.5K");
    expect(formatNumber(15200)).toBe("15.2K");
  });

  it("formats millions with M suffix", () => {
    expect(formatNumber(1000000)).toBe("1.0M");
    expect(formatNumber(1500000)).toBe("1.5M");
  });
});

describe("formatRelativeTime", () => {
  it('returns "just now" for recent times', () => {
    const now = new Date();
    expect(formatRelativeTime(now)).toBe("just now");
  });

  it("formats minutes ago", () => {
    const fiveMinutesAgo = new Date(Date.now() - 5 * 60 * 1000);
    expect(formatRelativeTime(fiveMinutesAgo)).toBe("5m ago");
  });

  it("formats hours ago", () => {
    const twoHoursAgo = new Date(Date.now() - 2 * 60 * 60 * 1000);
    expect(formatRelativeTime(twoHoursAgo)).toBe("2h ago");
  });

  it("formats days ago", () => {
    const threeDaysAgo = new Date(Date.now() - 3 * 24 * 60 * 60 * 1000);
    expect(formatRelativeTime(threeDaysAgo)).toBe("3d ago");
  });

  it("accepts string dates", () => {
    const dateString = new Date(Date.now() - 60 * 1000).toISOString();
    expect(formatRelativeTime(dateString)).toBe("1m ago");
  });
});

describe("truncate", () => {
  it("returns original text if shorter than length", () => {
    expect(truncate("short", 10)).toBe("short");
  });

  it("returns original text if equal to length", () => {
    expect(truncate("exact", 5)).toBe("exact");
  });

  it("truncates and adds ellipsis", () => {
    expect(truncate("this is a long text", 7)).toBe("this is...");
  });

  it("handles empty string", () => {
    expect(truncate("", 5)).toBe("");
  });
});
