import "@testing-library/svelte/vitest";
import { vi, beforeEach } from "vitest";

// Mock fetch for API calls
global.fetch = vi.fn();

// Mock EventSource for SSE
class MockEventSource {
  onmessage: ((event: MessageEvent) => void) | null = null;
  onerror: ((event: Event) => void) | null = null;
  close = vi.fn();
}

global.EventSource = MockEventSource as unknown as typeof EventSource;

// Reset mocks between tests
beforeEach(() => {
  vi.clearAllMocks();
});
