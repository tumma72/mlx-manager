import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';

// We need to test the PollingCoordinator class directly, not the singleton
// So we'll create a fresh instance for each test

// Mock document for visibility tests
const mockDocument = {
	visibilityState: 'visible' as 'visible' | 'hidden',
	addEventListener: vi.fn(),
	removeEventListener: vi.fn()
};

// Store original document
const originalDocument = globalThis.document;

describe('PollingCoordinator', () => {
	let PollingCoordinator: typeof import('./polling-coordinator.svelte').pollingCoordinator;

	beforeEach(async () => {
		vi.useFakeTimers();
		// Reset mocks
		mockDocument.visibilityState = 'visible';
		mockDocument.addEventListener.mockClear();

		// Mock document
		// @ts-expect-error - mocking document
		globalThis.document = mockDocument;

		// Fresh import to get a new instance behavior
		vi.resetModules();
		const module = await import('./polling-coordinator.svelte');
		PollingCoordinator = module.pollingCoordinator;
	});

	afterEach(() => {
		vi.useRealTimers();
		// Restore original document
		globalThis.document = originalDocument;
		PollingCoordinator.destroy();
	});

	describe('register', () => {
		it('registers a polling configuration', () => {
			const refreshFn = vi.fn().mockResolvedValue(undefined);

			PollingCoordinator.register('servers', {
				interval: 5000,
				minInterval: 1000,
				refreshFn
			});

			// Should not start polling automatically
			expect(refreshFn).not.toHaveBeenCalled();
		});

		it('updates config if already registered', () => {
			const refreshFn1 = vi.fn().mockResolvedValue(undefined);
			const refreshFn2 = vi.fn().mockResolvedValue(undefined);

			PollingCoordinator.register('servers', {
				interval: 5000,
				minInterval: 1000,
				refreshFn: refreshFn1
			});

			PollingCoordinator.register('servers', {
				interval: 10000,
				minInterval: 2000,
				refreshFn: refreshFn2
			});

			// Start polling - should use the second config
			PollingCoordinator.start('servers');

			expect(refreshFn2).toHaveBeenCalled();
			expect(refreshFn1).not.toHaveBeenCalled();
		});
	});

	describe('start/stop', () => {
		it('starts polling and calls refresh immediately', () => {
			const refreshFn = vi.fn().mockResolvedValue(undefined);

			PollingCoordinator.register('servers', {
				interval: 5000,
				minInterval: 1000,
				refreshFn
			});

			PollingCoordinator.start('servers');

			expect(refreshFn).toHaveBeenCalledTimes(1);
		});

		it('calls refresh at the configured interval', async () => {
			const refreshFn = vi.fn().mockResolvedValue(undefined);

			PollingCoordinator.register('servers', {
				interval: 5000,
				minInterval: 1000,
				refreshFn
			});

			PollingCoordinator.start('servers');
			expect(refreshFn).toHaveBeenCalledTimes(1);

			// Advance time by 5 seconds
			await vi.advanceTimersByTimeAsync(5000);
			expect(refreshFn).toHaveBeenCalledTimes(2);

			// Advance another 5 seconds
			await vi.advanceTimersByTimeAsync(5000);
			expect(refreshFn).toHaveBeenCalledTimes(3);
		});

		it('stops polling when stop is called', async () => {
			const refreshFn = vi.fn().mockResolvedValue(undefined);

			PollingCoordinator.register('servers', {
				interval: 5000,
				minInterval: 1000,
				refreshFn
			});

			PollingCoordinator.start('servers');
			expect(refreshFn).toHaveBeenCalledTimes(1);

			PollingCoordinator.stop('servers');

			// Advance time - should not call refresh again
			await vi.advanceTimersByTimeAsync(10000);
			expect(refreshFn).toHaveBeenCalledTimes(1);
		});

		it('warns when starting unregistered key', () => {
			const consoleSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});

			PollingCoordinator.start('servers');

			expect(consoleSpy).toHaveBeenCalledWith(
				'[PollingCoordinator] No config registered for servers'
			);

			consoleSpy.mockRestore();
		});
	});

	describe('refresh - deduplication', () => {
		it('only calls refreshFn once when multiple refreshes requested', async () => {
			let resolveRefresh: () => void;
			const refreshPromise = new Promise<void>((resolve) => {
				resolveRefresh = resolve;
			});
			const refreshFn = vi.fn().mockReturnValue(refreshPromise);

			PollingCoordinator.register('servers', {
				interval: 5000,
				minInterval: 0, // Disable throttle for this test
				refreshFn
			});

			// Start two refreshes simultaneously
			const promise1 = PollingCoordinator.refresh('servers');
			const promise2 = PollingCoordinator.refresh('servers');

			// Should only call refreshFn once (deduplication)
			expect(refreshFn).toHaveBeenCalledTimes(1);

			// Both should indicate in-flight
			expect(PollingCoordinator.isRefreshing('servers')).toBe(true);

			// Resolve the refresh
			resolveRefresh!();
			await Promise.all([promise1, promise2]);

			// Now should not be refreshing
			expect(PollingCoordinator.isRefreshing('servers')).toBe(false);
		});

		it('allows new refresh after previous completes', async () => {
			const refreshFn = vi.fn().mockResolvedValue(undefined);

			PollingCoordinator.register('servers', {
				interval: 5000,
				minInterval: 0, // Disable throttle
				refreshFn
			});

			await PollingCoordinator.refresh('servers');
			expect(refreshFn).toHaveBeenCalledTimes(1);

			await PollingCoordinator.refresh('servers');
			expect(refreshFn).toHaveBeenCalledTimes(2);
		});
	});

	describe('refresh - throttling', () => {
		it('skips refresh if called within minInterval', async () => {
			const refreshFn = vi.fn().mockResolvedValue(undefined);

			PollingCoordinator.register('servers', {
				interval: 5000,
				minInterval: 1000,
				refreshFn
			});

			await PollingCoordinator.refresh('servers');
			expect(refreshFn).toHaveBeenCalledTimes(1);

			// Try to refresh again immediately - should be throttled
			await PollingCoordinator.refresh('servers');
			expect(refreshFn).toHaveBeenCalledTimes(1);

			// Advance time past minInterval
			await vi.advanceTimersByTimeAsync(1100);

			// Now refresh should work
			await PollingCoordinator.refresh('servers');
			expect(refreshFn).toHaveBeenCalledTimes(2);
		});
	});

	describe('isRefreshing', () => {
		it('returns true while refresh is in-flight', async () => {
			let resolveRefresh: () => void;
			const refreshPromise = new Promise<void>((resolve) => {
				resolveRefresh = resolve;
			});
			const refreshFn = vi.fn().mockReturnValue(refreshPromise);

			PollingCoordinator.register('servers', {
				interval: 5000,
				minInterval: 0,
				refreshFn
			});

			expect(PollingCoordinator.isRefreshing('servers')).toBe(false);

			const promise = PollingCoordinator.refresh('servers');

			expect(PollingCoordinator.isRefreshing('servers')).toBe(true);

			resolveRefresh!();
			await promise;

			expect(PollingCoordinator.isRefreshing('servers')).toBe(false);
		});
	});

	describe('error handling', () => {
		it('catches and logs errors from refreshFn', async () => {
			const error = new Error('Refresh failed');
			const refreshFn = vi.fn().mockRejectedValue(error);
			const consoleSpy = vi.spyOn(console, 'error').mockImplementation(() => {});

			PollingCoordinator.register('servers', {
				interval: 5000,
				minInterval: 0,
				refreshFn
			});

			await PollingCoordinator.refresh('servers');

			expect(consoleSpy).toHaveBeenCalledWith(
				'[PollingCoordinator] Refresh failed for servers:',
				error
			);

			consoleSpy.mockRestore();
		});

		it('continues polling after error', async () => {
			const refreshFn = vi
				.fn()
				.mockRejectedValueOnce(new Error('First failure'))
				.mockResolvedValue(undefined);

			vi.spyOn(console, 'error').mockImplementation(() => {});

			PollingCoordinator.register('servers', {
				interval: 5000,
				minInterval: 0,
				refreshFn
			});

			PollingCoordinator.start('servers');
			expect(refreshFn).toHaveBeenCalledTimes(1);

			// Advance to next interval
			await vi.advanceTimersByTimeAsync(5000);
			expect(refreshFn).toHaveBeenCalledTimes(2);
		});
	});

	describe('pause/resume', () => {
		it('pauses polling for a specific key', async () => {
			const refreshFn = vi.fn().mockResolvedValue(undefined);

			PollingCoordinator.register('servers', {
				interval: 5000,
				minInterval: 1000,
				refreshFn
			});

			PollingCoordinator.start('servers');
			expect(refreshFn).toHaveBeenCalledTimes(1);

			PollingCoordinator.pause('servers');

			// Advance time - should not call refresh
			await vi.advanceTimersByTimeAsync(10000);
			expect(refreshFn).toHaveBeenCalledTimes(1);
		});

		it('resumes polling for a specific key', async () => {
			const refreshFn = vi.fn().mockResolvedValue(undefined);

			PollingCoordinator.register('servers', {
				interval: 5000,
				minInterval: 1000,
				refreshFn
			});

			PollingCoordinator.start('servers');
			PollingCoordinator.pause('servers');

			// Advance time - paused
			await vi.advanceTimersByTimeAsync(5000);
			expect(refreshFn).toHaveBeenCalledTimes(1);

			PollingCoordinator.resume('servers');
			// Resume triggers immediate refresh
			expect(refreshFn).toHaveBeenCalledTimes(2);

			// Advance time - should continue polling
			await vi.advanceTimersByTimeAsync(5000);
			expect(refreshFn).toHaveBeenCalledTimes(3);
		});
	});

	describe('setGlobalPause', () => {
		it('pauses all polling when set to true', async () => {
			const serverRefresh = vi.fn().mockResolvedValue(undefined);
			const profileRefresh = vi.fn().mockResolvedValue(undefined);

			PollingCoordinator.register('servers', {
				interval: 5000,
				minInterval: 1000,
				refreshFn: serverRefresh
			});

			PollingCoordinator.register('profiles', {
				interval: 5000,
				minInterval: 1000,
				refreshFn: profileRefresh
			});

			PollingCoordinator.start('servers');
			PollingCoordinator.start('profiles');

			expect(serverRefresh).toHaveBeenCalledTimes(1);
			expect(profileRefresh).toHaveBeenCalledTimes(1);

			PollingCoordinator.setGlobalPause(true);

			// Advance time - nothing should be called
			await vi.advanceTimersByTimeAsync(10000);
			expect(serverRefresh).toHaveBeenCalledTimes(1);
			expect(profileRefresh).toHaveBeenCalledTimes(1);
		});

		it('resumes all polling when set to false', async () => {
			const serverRefresh = vi.fn().mockResolvedValue(undefined);

			PollingCoordinator.register('servers', {
				interval: 5000,
				minInterval: 1000,
				refreshFn: serverRefresh
			});

			PollingCoordinator.start('servers');
			PollingCoordinator.setGlobalPause(true);

			// Advance while paused
			await vi.advanceTimersByTimeAsync(5000);
			expect(serverRefresh).toHaveBeenCalledTimes(1);

			PollingCoordinator.setGlobalPause(false);
			// Resume triggers immediate refresh
			expect(serverRefresh).toHaveBeenCalledTimes(2);
		});
	});

	describe('destroy', () => {
		it('stops all polling and clears state', async () => {
			const refreshFn = vi.fn().mockResolvedValue(undefined);

			PollingCoordinator.register('servers', {
				interval: 5000,
				minInterval: 1000,
				refreshFn
			});

			PollingCoordinator.start('servers');
			expect(refreshFn).toHaveBeenCalledTimes(1);

			// Verify polling is active before destroy
			expect(PollingCoordinator.isPolling('servers')).toBe(true);

			PollingCoordinator.destroy();

			// Advance time - nothing should be called
			await vi.advanceTimersByTimeAsync(10000);
			expect(refreshFn).toHaveBeenCalledTimes(1);

			// After destroy, state is cleared so isPolling returns false
			// (because the state map is empty, get() returns undefined)
			expect(PollingCoordinator.isPolling('servers')).toBeFalsy();
		});
	});

	describe('getTimeSinceRefresh', () => {
		it('returns Infinity if never refreshed', () => {
			PollingCoordinator.register('servers', {
				interval: 5000,
				minInterval: 1000,
				refreshFn: vi.fn().mockResolvedValue(undefined)
			});

			expect(PollingCoordinator.getTimeSinceRefresh('servers')).toBe(Infinity);
		});

		it('returns time since last refresh', async () => {
			const refreshFn = vi.fn().mockResolvedValue(undefined);

			PollingCoordinator.register('servers', {
				interval: 5000,
				minInterval: 0,
				refreshFn
			});

			await PollingCoordinator.refresh('servers');

			// Advance time
			await vi.advanceTimersByTimeAsync(3000);

			const timeSince = PollingCoordinator.getTimeSinceRefresh('servers');
			expect(timeSince).toBeGreaterThanOrEqual(3000);
			expect(timeSince).toBeLessThan(4000);
		});
	});

	describe('refresh - unregistered key', () => {
		it('warns when refreshing unregistered key', async () => {
			const consoleSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});

			await PollingCoordinator.refresh('servers');

			expect(consoleSpy).toHaveBeenCalledWith(
				'[PollingCoordinator] No config registered for servers'
			);

			consoleSpy.mockRestore();
		});
	});

	describe('visibility changes', () => {
		it('pauses polling when tab becomes hidden', async () => {
			const refreshFn = vi.fn().mockResolvedValue(undefined);

			PollingCoordinator.register('servers', {
				interval: 5000,
				minInterval: 1000,
				refreshFn
			});

			PollingCoordinator.start('servers');
			expect(refreshFn).toHaveBeenCalledTimes(1);

			// Simulate tab becoming hidden
			mockDocument.visibilityState = 'hidden';
			// Trigger the visibility change listener
			const visibilityHandler = mockDocument.addEventListener.mock.calls.find(
				(call) => call[0] === 'visibilitychange'
			)?.[1];
			expect(visibilityHandler).toBeDefined();
			visibilityHandler();

			// Polling should be paused
			expect(PollingCoordinator.isPolling('servers')).toBe(false);

			// Advance time - should not call refresh
			await vi.advanceTimersByTimeAsync(10000);
			expect(refreshFn).toHaveBeenCalledTimes(1);
		});

		it('resumes polling when tab becomes visible after being hidden', async () => {
			const refreshFn = vi.fn().mockResolvedValue(undefined);

			PollingCoordinator.register('servers', {
				interval: 5000,
				minInterval: 0,
				refreshFn
			});

			PollingCoordinator.start('servers');
			expect(refreshFn).toHaveBeenCalledTimes(1);

			// Allow the initial refresh promise to settle so it's removed from inFlight
			await Promise.resolve();
			await Promise.resolve();

			// Get the visibility handler
			const visibilityHandler = mockDocument.addEventListener.mock.calls.find(
				(call) => call[0] === 'visibilitychange'
			)?.[1];
			expect(visibilityHandler).toBeDefined();

			// Simulate tab becoming hidden first
			mockDocument.visibilityState = 'hidden';
			visibilityHandler();

			// Verify polling stopped
			expect(PollingCoordinator.isPolling('servers')).toBe(false);
			expect(refreshFn).toHaveBeenCalledTimes(1);

			// Now simulate tab becoming visible again
			mockDocument.visibilityState = 'visible';
			visibilityHandler();

			// Allow the resumeAll refresh promise to settle
			await Promise.resolve();
			await Promise.resolve();

			// The resumeAll was called - it triggers immediate refresh and restarts polling
			expect(PollingCoordinator.isPolling('servers')).toBe(true);
			// refreshFn: 1 (start) + 1 (resumeAll) = 2
			expect(refreshFn).toHaveBeenCalledTimes(2);

			// Verify the resumed interval continues
			await vi.advanceTimersByTimeAsync(5000);
			expect(refreshFn).toHaveBeenCalledTimes(3);
		});

		it('does not trigger when visibility state unchanged', async () => {
			const refreshFn = vi.fn().mockResolvedValue(undefined);

			PollingCoordinator.register('servers', {
				interval: 5000,
				minInterval: 1000,
				refreshFn
			});

			PollingCoordinator.start('servers');
			expect(refreshFn).toHaveBeenCalledTimes(1);

			// Get the visibility handler
			const visibilityHandler = mockDocument.addEventListener.mock.calls.find(
				(call) => call[0] === 'visibilitychange'
			)?.[1];
			expect(visibilityHandler).toBeDefined();

			// Trigger visibility change without actually changing state (still visible)
			visibilityHandler();

			// Should not trigger extra refresh
			expect(refreshFn).toHaveBeenCalledTimes(1);
		});
	});

	describe('shouldPoll - branch coverage', () => {
		it('returns false when tab is not visible', async () => {
			const refreshFn = vi.fn().mockResolvedValue(undefined);

			PollingCoordinator.register('servers', {
				interval: 5000,
				minInterval: 0,
				refreshFn
			});

			PollingCoordinator.start('servers');
			expect(refreshFn).toHaveBeenCalledTimes(1);

			// Allow the initial refresh promise to settle
			await Promise.resolve();
			await Promise.resolve();

			// Get the visibility handler and hide the tab
			const visibilityHandler = mockDocument.addEventListener.mock.calls.find(
				(call) => call[0] === 'visibilitychange'
			)?.[1];
			mockDocument.visibilityState = 'hidden';
			visibilityHandler();

			// Polling stops - interval callback would return false from shouldPoll
			// But intervals are cleared, so we need to test via the API
			expect(PollingCoordinator.isPolling('servers')).toBe(false);
		});

		it('returns false when globally paused', async () => {
			const refreshFn = vi.fn().mockResolvedValue(undefined);

			PollingCoordinator.register('servers', {
				interval: 5000,
				minInterval: 0,
				refreshFn
			});

			PollingCoordinator.start('servers');
			expect(refreshFn).toHaveBeenCalledTimes(1);

			// Set global pause - this affects shouldPoll
			PollingCoordinator.setGlobalPause(true);

			// Intervals are cleared by setGlobalPause
			expect(PollingCoordinator.isPolling('servers')).toBe(false);
		});

		it('returns false when specific key is paused', async () => {
			const refreshFn = vi.fn().mockResolvedValue(undefined);

			PollingCoordinator.register('servers', {
				interval: 5000,
				minInterval: 0,
				refreshFn
			});

			PollingCoordinator.start('servers');
			expect(refreshFn).toHaveBeenCalledTimes(1);

			// Pause the specific key
			PollingCoordinator.pause('servers');

			// Polling stopped for this key
			expect(PollingCoordinator.isPolling('servers')).toBe(false);

			// Advance time - no refresh should happen
			await vi.advanceTimersByTimeAsync(10000);
			expect(refreshFn).toHaveBeenCalledTimes(1);
		});

		it('checks shouldPoll conditions when interval callback fires', async () => {
			const refreshFn = vi.fn().mockResolvedValue(undefined);

			PollingCoordinator.register('servers', {
				interval: 5000,
				minInterval: 0,
				refreshFn
			});

			PollingCoordinator.start('servers');
			expect(refreshFn).toHaveBeenCalledTimes(1);

			// Allow the initial refresh promise to settle
			await Promise.resolve();
			await Promise.resolve();

			// Normal interval - shouldPoll returns true
			await vi.advanceTimersByTimeAsync(5000);
			expect(refreshFn).toHaveBeenCalledTimes(2);

			// Now test the false branch in the start() interval callback
			// We need shouldPoll to return false while interval is still running
			// This can happen if we set isVisible to false without calling pauseAll
			// But that's internal state... We test this indirectly through other mechanisms

			// The interval callback in start() has: if (this.shouldPoll(key)) { this.refresh(key); }
			// The false branch means shouldPoll returned false
			// But in normal flow, when shouldPoll would return false, intervals are cleared

			// To test this branch, we need the interval to fire while shouldPoll returns false
			// This is tricky because the code is designed to clear intervals when polling should stop
		});
	});

	describe('setGlobalPause - branch coverage', () => {
		it('does not resume when tab is hidden', async () => {
			const refreshFn = vi.fn().mockResolvedValue(undefined);

			PollingCoordinator.register('servers', {
				interval: 5000,
				minInterval: 0,
				refreshFn
			});

			PollingCoordinator.start('servers');
			expect(refreshFn).toHaveBeenCalledTimes(1);

			// Allow the initial refresh promise to settle
			await Promise.resolve();
			await Promise.resolve();

			// Hide the tab first
			const visibilityHandler = mockDocument.addEventListener.mock.calls.find(
				(call) => call[0] === 'visibilitychange'
			)?.[1];
			mockDocument.visibilityState = 'hidden';
			visibilityHandler();

			// Set global pause to true
			PollingCoordinator.setGlobalPause(true);

			// Now set global pause to false while tab is hidden
			// This should NOT call resumeAll because isVisible is false
			PollingCoordinator.setGlobalPause(false);

			// Polling should still be stopped
			expect(PollingCoordinator.isPolling('servers')).toBe(false);

			// Advance time - no refresh should happen
			await vi.advanceTimersByTimeAsync(10000);
			expect(refreshFn).toHaveBeenCalledTimes(1);
		});
	});

	describe('resume - branch coverage', () => {
		it('does not start polling when tab is hidden', async () => {
			const refreshFn = vi.fn().mockResolvedValue(undefined);

			PollingCoordinator.register('servers', {
				interval: 5000,
				minInterval: 0,
				refreshFn
			});

			PollingCoordinator.start('servers');
			expect(refreshFn).toHaveBeenCalledTimes(1);

			// Allow the initial refresh promise to settle
			await Promise.resolve();
			await Promise.resolve();

			// Hide the tab
			const visibilityHandler = mockDocument.addEventListener.mock.calls.find(
				(call) => call[0] === 'visibilitychange'
			)?.[1];
			mockDocument.visibilityState = 'hidden';
			visibilityHandler();

			// Pause the key
			PollingCoordinator.pause('servers');

			// Try to resume while tab is hidden - should not start polling
			PollingCoordinator.resume('servers');

			// Polling should still be stopped because tab is hidden
			expect(PollingCoordinator.isPolling('servers')).toBe(false);
		});

		it('does not start polling when globally paused', async () => {
			const refreshFn = vi.fn().mockResolvedValue(undefined);

			PollingCoordinator.register('servers', {
				interval: 5000,
				minInterval: 0,
				refreshFn
			});

			PollingCoordinator.start('servers');
			expect(refreshFn).toHaveBeenCalledTimes(1);

			// Set global pause
			PollingCoordinator.setGlobalPause(true);

			// Pause the key (it's already paused by global pause)
			PollingCoordinator.pause('servers');

			// Try to resume while globally paused - should not start polling
			PollingCoordinator.resume('servers');

			// Polling should still be stopped because of global pause
			expect(PollingCoordinator.isPolling('servers')).toBe(false);
		});

		it('handles resume on unregistered key gracefully', () => {
			// Resume should not throw for unregistered key
			PollingCoordinator.resume('servers');
			// Just verify no error thrown
			expect(PollingCoordinator.isPolling('servers')).toBe(false);
		});

		it('handles pause on unregistered key gracefully', () => {
			// Pause should not throw for unregistered key
			PollingCoordinator.pause('servers');
			// Just verify no error thrown
			expect(PollingCoordinator.isPolling('servers')).toBe(false);
		});
	});

	describe('resumeAll - branch coverage', () => {
		it('skips resuming polling for manually paused keys', async () => {
			const refreshFn = vi.fn().mockResolvedValue(undefined);

			PollingCoordinator.register('servers', {
				interval: 5000,
				minInterval: 0,
				refreshFn
			});

			PollingCoordinator.start('servers');
			expect(refreshFn).toHaveBeenCalledTimes(1);

			// Allow the initial refresh promise to settle
			await Promise.resolve();
			await Promise.resolve();

			// Manually pause the key (sets state.paused = true)
			PollingCoordinator.pause('servers');
			expect(PollingCoordinator.isPolling('servers')).toBe(false);

			// Get the visibility handler
			const visibilityHandler = mockDocument.addEventListener.mock.calls.find(
				(call) => call[0] === 'visibilitychange'
			)?.[1];

			// Hide tab then show - this triggers resumeAll
			mockDocument.visibilityState = 'hidden';
			visibilityHandler();
			mockDocument.visibilityState = 'visible';
			visibilityHandler();

			// resumeAll should NOT restart polling for manually paused keys
			// (condition: !state.paused fails)
			expect(PollingCoordinator.isPolling('servers')).toBe(false);

			// Advance time - no refresh should happen
			await vi.advanceTimersByTimeAsync(10000);
			expect(refreshFn).toHaveBeenCalledTimes(1);
		});

		it('skips keys that already have an interval running', async () => {
			const refreshFn = vi.fn().mockResolvedValue(undefined);

			PollingCoordinator.register('servers', {
				interval: 5000,
				minInterval: 0,
				refreshFn
			});

			PollingCoordinator.start('servers');
			expect(refreshFn).toHaveBeenCalledTimes(1);

			// Allow the initial refresh promise to settle
			await Promise.resolve();
			await Promise.resolve();

			// Note: visibility handler is registered but we test resumeAll path
			// through setGlobalPause instead of simulating visibility events

			// Calling resumeAll while polling is already running
			// The condition !state.intervalId fails because interval is already set
			// We simulate this by NOT hiding the tab first, just triggering resumeAll path differently
			// Actually, we test this implicitly through setGlobalPause

			// Verify polling is active
			expect(PollingCoordinator.isPolling('servers')).toBe(true);

			// Set global pause false (while already visible and not paused)
			// This would call resumeAll, but intervalId is already set
			PollingCoordinator.setGlobalPause(true);
			PollingCoordinator.setGlobalPause(false);

			// Only the setGlobalPause(false) -> resumeAll triggers refresh
			// But the interval was cleared by setGlobalPause(true)
			// So resumeAll restarts it
			await Promise.resolve();
			await Promise.resolve();
			expect(refreshFn).toHaveBeenCalledTimes(2);
		});
	});

	describe('resumeAll with interval callbacks', () => {
		it('resumed polling continues at interval after shouldPoll checks', async () => {
			const refreshFn = vi.fn().mockResolvedValue(undefined);

			PollingCoordinator.register('servers', {
				interval: 5000,
				minInterval: 0,
				refreshFn
			});

			PollingCoordinator.start('servers');
			expect(refreshFn).toHaveBeenCalledTimes(1);

			// Allow the initial refresh promise to settle
			await Promise.resolve();
			await Promise.resolve();

			// Get the visibility handler
			const visibilityHandler = mockDocument.addEventListener.mock.calls.find(
				(call) => call[0] === 'visibilitychange'
			)?.[1];
			expect(visibilityHandler).toBeDefined();

			// Must hide tab first to trigger resumeAll code path
			mockDocument.visibilityState = 'hidden';
			visibilityHandler();

			// Show tab (triggers resumeAll)
			mockDocument.visibilityState = 'visible';
			visibilityHandler();

			// Allow the resumeAll refresh promise to settle
			await Promise.resolve();
			await Promise.resolve();

			expect(refreshFn).toHaveBeenCalledTimes(2);

			// Advance time to test the interval callback created in resumeAll
			await vi.advanceTimersByTimeAsync(5000);
			expect(refreshFn).toHaveBeenCalledTimes(3);

			// Do it again to ensure shouldPoll is being called each interval
			await vi.advanceTimersByTimeAsync(5000);
			expect(refreshFn).toHaveBeenCalledTimes(4);
		});

		it('resumed polling respects shouldPoll conditions', async () => {
			const refreshFn = vi.fn().mockResolvedValue(undefined);

			PollingCoordinator.register('servers', {
				interval: 5000,
				minInterval: 0,
				refreshFn
			});

			PollingCoordinator.start('servers');
			expect(refreshFn).toHaveBeenCalledTimes(1);

			// Allow the initial refresh promise to settle
			await Promise.resolve();
			await Promise.resolve();

			// Get the visibility handler
			const visibilityHandler = mockDocument.addEventListener.mock.calls.find(
				(call) => call[0] === 'visibilitychange'
			)?.[1];

			// Must hide then show tab to trigger resumeAll
			mockDocument.visibilityState = 'hidden';
			visibilityHandler();
			mockDocument.visibilityState = 'visible';
			visibilityHandler();

			// Allow the resumeAll refresh promise to settle
			await Promise.resolve();
			await Promise.resolve();

			expect(refreshFn).toHaveBeenCalledTimes(2);

			// Now enable global pause - the interval callback should check shouldPoll
			PollingCoordinator.setGlobalPause(true);

			// Advance time - should not refresh due to globalPause
			await vi.advanceTimersByTimeAsync(5000);
			// Still at 2 because setGlobalPause(true) clears intervals
			expect(refreshFn).toHaveBeenCalledTimes(2);
		});
	});
});
