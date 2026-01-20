<script lang="ts">
	import { goto } from '$app/navigation';
	import { resolve } from '$app/paths';
	import { auth } from '$lib/api/client';
	import { authStore } from '$lib/stores';
	import { Cpu, Loader2 } from 'lucide-svelte';

	type Mode = 'login' | 'register';

	let mode = $state<Mode>('login');
	let email = $state('');
	let password = $state('');
	let loading = $state(false);
	let error = $state<string | null>(null);
	let success = $state<string | null>(null);

	const isRegister = $derived(mode === 'register');
	const buttonText = $derived(loading ? '' : isRegister ? 'Create Account' : 'Sign In');
	const toggleText = $derived(
		isRegister ? 'Already have an account? Sign in' : "Don't have an account? Register"
	);

	function toggleMode() {
		mode = isRegister ? 'login' : 'register';
		error = null;
		success = null;
	}

	async function handleSubmit(e: SubmitEvent) {
		e.preventDefault();
		error = null;
		success = null;
		loading = true;

		try {
			if (isRegister) {
				const user = await auth.register({ email, password });

				if (user.status === 'approved') {
					// Auto-approved (first user) - login immediately
					const token = await auth.login(email, password);
					authStore.setAuth(token.access_token, user);
					await goto(resolve('/'));
				} else {
					// Pending approval
					success =
						'Registration submitted. Please wait for admin approval before you can sign in.';
					// Switch to login mode for convenience
					mode = 'login';
					password = '';
				}
			} else {
				const token = await auth.login(email, password);
				// Store token first so auth.me() can use it
				authStore.token = token.access_token;
				const user = await auth.me();

				// Handle non-approved users (backend returns 403, but just in case)
				if (user.status !== 'approved') {
					authStore.clearAuth();
					error = 'Your account is pending approval. Please wait for an administrator to approve your access.';
					loading = false;
					return;
				}

				authStore.setAuth(token.access_token, user);
				await goto(resolve('/'));
			}
		} catch (err) {
			if (err instanceof Error) {
				// Handle specific error messages
				if (err.message.includes('pending')) {
					error = 'Your account is pending approval. Please wait for an administrator to approve your access.';
				} else if (err.message.includes('disabled')) {
					error = 'Your account has been disabled. Please contact an administrator.';
				} else if (err.message.includes('401') || err.message.includes('Invalid')) {
					error = 'Invalid email or password.';
				} else if (err.message.includes('Email already registered')) {
					error = 'An account with this email already exists.';
				} else {
					error = err.message;
				}
			} else {
				error = 'An unexpected error occurred. Please try again.';
			}
		} finally {
			loading = false;
		}
	}
</script>

<div class="w-full max-w-md px-4">
	<div class="bg-white dark:bg-gray-800 shadow-lg rounded-xl p-8">
		<!-- Logo and Title -->
		<div class="flex flex-col items-center mb-8">
			<div class="flex items-center gap-2 mb-2">
				<Cpu class="w-10 h-10 text-mlx-500" />
				<span class="font-bold text-2xl text-gray-900 dark:text-white">MLX Manager</span>
			</div>
			<p class="text-gray-600 dark:text-gray-400 text-sm">
				{isRegister ? 'Create your account' : 'Sign in to your account'}
			</p>
		</div>

		<!-- Success Message -->
		{#if success}
			<div
				class="mb-6 p-4 bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 rounded-lg"
			>
				<p class="text-sm text-green-700 dark:text-green-400">{success}</p>
			</div>
		{/if}

		<!-- Error Message -->
		{#if error}
			<div
				class="mb-6 p-4 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg"
			>
				<p class="text-sm text-red-700 dark:text-red-400">{error}</p>
			</div>
		{/if}

		<!-- Form -->
		<form onsubmit={handleSubmit} class="space-y-4">
			<div>
				<label for="email" class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
					Email
				</label>
				<input
					type="email"
					id="email"
					bind:value={email}
					required
					autocomplete="email"
					disabled={loading}
					class="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg
						bg-white dark:bg-gray-700 text-gray-900 dark:text-white
						focus:ring-2 focus:ring-mlx-500 focus:border-mlx-500
						disabled:opacity-50 disabled:cursor-not-allowed"
					placeholder="you@example.com"
				/>
			</div>

			<div>
				<label
					for="password"
					class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1"
				>
					Password
				</label>
				<input
					type="password"
					id="password"
					bind:value={password}
					required
					minlength={8}
					autocomplete={isRegister ? 'new-password' : 'current-password'}
					disabled={loading}
					class="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg
						bg-white dark:bg-gray-700 text-gray-900 dark:text-white
						focus:ring-2 focus:ring-mlx-500 focus:border-mlx-500
						disabled:opacity-50 disabled:cursor-not-allowed"
					placeholder="Enter your password"
				/>
				{#if isRegister}
					<p class="mt-1 text-xs text-gray-500 dark:text-gray-400">
						Password must be at least 8 characters
					</p>
				{/if}
			</div>

			<button
				type="submit"
				disabled={loading}
				class="w-full py-2 px-4 bg-mlx-600 hover:bg-mlx-700 text-white font-medium rounded-lg
					transition-colors focus:outline-none focus:ring-2 focus:ring-mlx-500 focus:ring-offset-2
					disabled:opacity-50 disabled:cursor-not-allowed
					flex items-center justify-center gap-2"
			>
				{#if loading}
					<Loader2 class="w-5 h-5 animate-spin" />
				{/if}
				{buttonText}
			</button>
		</form>

		<!-- Toggle Mode -->
		<div class="mt-6 text-center">
			<button
				type="button"
				onclick={toggleMode}
				disabled={loading}
				class="text-sm text-mlx-600 hover:text-mlx-700 dark:text-mlx-400 dark:hover:text-mlx-300
					disabled:opacity-50 disabled:cursor-not-allowed"
			>
				{toggleText}
			</button>
		</div>
	</div>
</div>
