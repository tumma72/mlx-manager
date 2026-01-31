import "@testing-library/svelte/vitest";
import "@testing-library/jest-dom/vitest";
import { vi, beforeEach } from "vitest";

// Mock fetch for API calls
global.fetch = vi.fn();

// Mock scrollIntoView for bits-ui components that use it
Element.prototype.scrollIntoView = vi.fn();

// Mock Element.animate for Svelte transitions
Element.prototype.animate = vi.fn(() => ({
  cancel: vi.fn(),
  finish: vi.fn(),
  play: vi.fn(),
  pause: vi.fn(),
  reverse: vi.fn(),
  updatePlaybackRate: vi.fn(),
  persist: vi.fn(),
  commitStyles: vi.fn(),
  addEventListener: vi.fn(),
  removeEventListener: vi.fn(),
  dispatchEvent: vi.fn(),
  oncancel: null,
  onfinish: null,
  onremove: null,
  finished: Promise.resolve(),
  ready: Promise.resolve(),
  currentTime: 0,
  effect: null,
  id: '',
  pending: false,
  playState: 'finished',
  playbackRate: 1,
  replaceState: 'active',
  startTime: 0,
  timeline: null,
// eslint-disable-next-line @typescript-eslint/no-explicit-any
})) as any;

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
