import { redirect } from "@sveltejs/kit";
import { authStore } from "$lib/stores";
import { auth } from "$lib/api/client";

export const ssr = false;
export const prerender = false;

export async function load() {
  // Check for stored token directly from localStorage
  // (avoids $state reactivity issues in non-reactive load() context)
  const storedToken =
    typeof window !== "undefined"
      ? localStorage.getItem("mlx_auth_token")
      : null;

  if (!storedToken) {
    throw redirect(302, "/login");
  }

  // Ensure auth store is initialized from localStorage
  if (!authStore.isAuthenticated) {
    authStore.initialize();
  }

  // Validate token with backend
  try {
    const user = await auth.me();
    // Update stored user in case it changed (admin status, etc)
    authStore.setAuth(storedToken, user);
    return { user };
  } catch {
    // Token invalid - clear and redirect
    authStore.clearAuth();
    throw redirect(302, "/login");
  }
}
