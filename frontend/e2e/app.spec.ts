import { test, expect, type Page, type Route } from "@playwright/test";

/**
 * E2E tests with mocked API responses.
 * Tests verify that pages load correctly and display data from the API.
 */

// Mock data matching our API types
const mockUser = {
  id: 1,
  email: "test@example.com",
  is_admin: true,
  status: "approved",
  created_at: "2024-01-15T10:00:00Z",
};

const mockProfiles = [
  {
    id: 1,
    name: "Test Profile",
    model_path: "/path/to/model",
    host: "127.0.0.1",
    port: 10240,
    max_memory_gb: 64,
    launchd_installed: false,
    created_at: "2024-01-15T10:00:00Z",
    updated_at: "2024-01-15T10:00:00Z",
  },
];

const mockServers = {
  servers: [],
  profiles: mockProfiles,
};

const mockSystemInfo = {
  os_version: "Darwin 24.0.0",
  chip: "Apple M2 Max",
  memory_gb: 64,
  python_version: "3.11.0",
  mlx_version: "0.1.0",
  mlx_openai_server_version: "1.5.0",
};

const mockMemory = {
  total_gb: 64,
  available_gb: 32,
  used_gb: 32,
  percent_used: 50,
  mlx_recommended_gb: 51.2,
};

const mockModelsSearch = {
  models: [
    {
      id: "mlx-community/test-model",
      name: "test-model",
      author: "mlx-community",
      description: "A test model",
      downloads: 1000,
      likes: 50,
      size_estimate_gb: 4.5,
      last_modified: "2024-01-15T10:00:00Z",
    },
  ],
  total: 1,
};

/**
 * Setup authentication by setting localStorage with mock token/user.
 */
async function setupAuth(page: Page) {
  await page.addInitScript(() => {
    // Set mock auth data in localStorage before the app loads
    // Keys must match auth.svelte.ts: TOKEN_KEY = "mlx_auth_token", USER_KEY = "mlx_auth_user"
    localStorage.setItem("mlx_auth_token", "mock-jwt-token");
    localStorage.setItem(
      "mlx_auth_user",
      JSON.stringify({
        id: 1,
        email: "test@example.com",
        is_admin: true,
        status: "approved",
        created_at: "2024-01-15T10:00:00Z",
      }),
    );
  });
}

/**
 * Setup API mocks for all routes.
 */
async function setupApiMocks(page: Page) {
  // Only intercept actual API calls to localhost/api
  await page.route(/\/api\//, async (route: Route) => {
    const url = route.request().url();

    // Extra safety: skip if URL doesn't have /api/ path
    const urlObj = new URL(url);
    if (!urlObj.pathname.startsWith("/api/")) {
      return route.continue();
    }

    // Auth endpoints - return mock user for authenticated requests
    if (url.includes("/api/auth/me")) {
      return route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify(mockUser),
      });
    }

    if (url.includes("/api/auth/users/pending/count")) {
      return route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({ count: 0 }),
      });
    }

    if (url.includes("/api/servers")) {
      return route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify(mockServers),
      });
    }

    if (url.includes("/api/profiles") && !url.includes("/api/profiles/")) {
      return route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify(mockProfiles),
      });
    }

    if (url.match(/\/api\/profiles\/\d+/)) {
      return route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify(mockProfiles[0]),
      });
    }

    if (url.includes("/api/system/info")) {
      return route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify(mockSystemInfo),
      });
    }

    if (url.includes("/api/system/memory")) {
      return route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify(mockMemory),
      });
    }

    if (url.includes("/api/models/search")) {
      return route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify(mockModelsSearch),
      });
    }

    if (url.includes("/api/models/local")) {
      return route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify([]),
      });
    }

    if (url.includes("/api/models/downloads/active")) {
      return route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify([]),
      });
    }

    if (url.includes("/api/system/parser-options")) {
      return route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          tool_call_parsers: ["hermes", "qwen"],
          reasoning_parsers: ["deepseek"],
          message_converters: ["default"],
        }),
      });
    }

    // Default: return empty success for unhandled routes
    return route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({}),
    });
  });
}

test.describe("Page Loading", () => {
  test.beforeEach(async ({ page }) => {
    await setupAuth(page);
    await setupApiMocks(page);
  });

  test("homepage loads with correct title", async ({ page }) => {
    await page.goto("/");
    await expect(page).toHaveTitle(/MLX Model Manager/);
  });

  test("servers page loads without errors", async ({ page }) => {
    await page.goto("/servers");
    await expect(page).toHaveTitle(/MLX Model Manager/);
    // Page should have loaded - check for main content area
    await expect(page.locator("main")).toBeVisible();
  });

  test("models page loads without errors", async ({ page }) => {
    await page.goto("/models");
    await expect(page).toHaveURL(/\/models/);
    await expect(page.locator("main")).toBeVisible();
  });

  test("profiles page loads without errors", async ({ page }) => {
    await page.goto("/profiles");
    await expect(page).toHaveURL(/\/profiles/);
    await expect(page.locator("main")).toBeVisible();
  });

  test("new profile page loads", async ({ page }) => {
    await page.goto("/profiles/new");
    await expect(page).toHaveURL(/\/profiles\/new/);
  });
});

test.describe("Navigation", () => {
  test.beforeEach(async ({ page }) => {
    await setupAuth(page);
    await setupApiMocks(page);
  });

  test("can navigate between all pages", async ({ page }) => {
    await page.goto("/servers");
    await expect(page).toHaveURL(/\/servers/);

    // Navigate using text links in navbar
    await page.getByRole("link", { name: "Models" }).click();
    await expect(page).toHaveURL(/\/models/);

    await page.getByRole("link", { name: "Profiles" }).click();
    await expect(page).toHaveURL(/\/profiles/);

    await page.getByRole("link", { name: "Servers" }).click();
    await expect(page).toHaveURL(/\/servers/);
  });
});

test.describe("API Integration", () => {
  test.beforeEach(async ({ page }) => {
    await setupAuth(page);
    await setupApiMocks(page);
  });

  test("servers page makes API calls", async ({ page }) => {
    // Track API calls
    const apiCalls: string[] = [];
    page.on("request", (request) => {
      if (request.url().includes("/api/")) {
        apiCalls.push(request.url());
      }
    });

    await page.goto("/servers");
    // Wait for page to fully load and make API calls
    await expect(page.locator("main")).toBeVisible();
    await page.waitForTimeout(500);

    // Verify API calls were made
    expect(apiCalls.length).toBeGreaterThan(0);
  });

  test("profiles page loads with API mocking", async ({ page }) => {
    await page.goto("/profiles");
    // Verify page loads correctly with mocked API
    await expect(page.locator("main")).toBeVisible();
    await expect(page).toHaveURL(/\/profiles/);
  });
});

test.describe("Error Handling", () => {
  test.beforeEach(async ({ page }) => {
    await setupAuth(page);
  });

  test("handles 500 errors gracefully", async ({ page }) => {
    await page.route("**/api/**", async (route) => {
      await route.fulfill({
        status: 500,
        contentType: "application/json",
        body: JSON.stringify({ detail: "Internal server error" }),
      });
    });

    await page.goto("/servers");
    // Page should still load without crashing
    await expect(page).toHaveTitle(/MLX Model Manager/);
  });

  test("handles network errors gracefully", async ({ page }) => {
    await page.route("**/api/**", async (route) => {
      await route.abort("failed");
    });

    await page.goto("/servers");
    // Page should still load without crashing
    await expect(page).toHaveTitle(/MLX Model Manager/);
  });
});
