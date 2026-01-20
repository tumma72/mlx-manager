/**
 * Auth state management using Svelte 5 runes.
 *
 * Manages JWT token and user state with localStorage persistence.
 * Token is stored for API client auth header injection.
 */

import type { User } from "$lib/api/types";

const TOKEN_KEY = "mlx_auth_token";
const USER_KEY = "mlx_auth_user";

class AuthStore {
  // Auth state
  token = $state<string | null>(null);
  user = $state<User | null>(null);
  loading = $state(true);

  // Derived state
  get isAuthenticated(): boolean {
    return !!this.token && !!this.user;
  }

  get isAdmin(): boolean {
    return this.user?.is_admin ?? false;
  }

  /**
   * Initialize auth state from localStorage.
   * Call this on app mount (client-side only).
   */
  initialize(): void {
    if (typeof window === "undefined") {
      this.loading = false;
      return;
    }

    try {
      const storedToken = localStorage.getItem(TOKEN_KEY);
      const storedUser = localStorage.getItem(USER_KEY);

      if (storedToken && storedUser) {
        this.token = storedToken;
        this.user = JSON.parse(storedUser);
      }
    } catch {
      // Invalid stored data, clear it
      this.clearAuth();
    } finally {
      this.loading = false;
    }
  }

  /**
   * Set auth state after successful login.
   */
  setAuth(token: string, user: User): void {
    this.token = token;
    this.user = user;

    if (typeof window !== "undefined") {
      localStorage.setItem(TOKEN_KEY, token);
      localStorage.setItem(USER_KEY, JSON.stringify(user));
    }
  }

  /**
   * Clear auth state on logout or 401.
   */
  clearAuth(): void {
    this.token = null;
    this.user = null;

    if (typeof window !== "undefined") {
      localStorage.removeItem(TOKEN_KEY);
      localStorage.removeItem(USER_KEY);
    }
  }

  /**
   * Update user data (e.g., after profile update).
   */
  updateUser(user: User): void {
    this.user = user;

    if (typeof window !== "undefined") {
      localStorage.setItem(USER_KEY, JSON.stringify(user));
    }
  }
}

export const authStore = new AuthStore();

// Auto-initialize on client-side
if (typeof window !== "undefined") {
  authStore.initialize();
}
