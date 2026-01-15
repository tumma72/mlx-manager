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
});
