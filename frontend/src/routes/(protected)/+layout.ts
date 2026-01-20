import { redirect } from '@sveltejs/kit';
import { authStore } from '$lib/stores';
import { auth } from '$lib/api/client';

export const ssr = false;
export const prerender = false;

export async function load() {
	// Initialize auth store from localStorage if not done
	if (typeof window !== 'undefined' && !authStore.isAuthenticated) {
		authStore.initialize();
	}

	// Check authentication
	if (!authStore.isAuthenticated) {
		throw redirect(302, '/login');
	}

	// Validate token with backend
	try {
		const user = await auth.me();
		// Update stored user in case it changed (admin status, etc)
		authStore.setAuth(authStore.token!, user);
		return { user };
	} catch {
		// Token invalid - clear and redirect
		authStore.clearAuth();
		throw redirect(302, '/login');
	}
}
