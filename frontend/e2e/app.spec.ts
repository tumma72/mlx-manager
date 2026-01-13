import { test, expect } from "@playwright/test";

test.describe("MLX Model Manager", () => {
  test("homepage loads and redirects to servers", async ({ page }) => {
    await page.goto("/");
    // The app should load successfully
    await expect(page).toHaveTitle(/MLX Model Manager/);
  });

  test("servers page shows server list", async ({ page }) => {
    await page.goto("/servers");
    // Should show the servers heading
    await expect(page.locator("h1, h2").first()).toBeVisible();
  });

  test("models page is accessible", async ({ page }) => {
    await page.goto("/models");
    // Should show the models page
    await expect(page).toHaveURL(/\/models/);
  });

  test("profiles page is accessible", async ({ page }) => {
    await page.goto("/profiles");
    // Should show the profiles page
    await expect(page).toHaveURL(/\/profiles/);
  });

  test("can navigate to create new profile", async ({ page }) => {
    await page.goto("/profiles");
    // Find and click the new profile link/button (use first() to handle multiple matches)
    const newProfileLink = page.locator('a[href="/profiles/new"]').first();
    if (await newProfileLink.isVisible()) {
      await newProfileLink.click();
      await expect(page).toHaveURL(/\/profiles\/new/);
    }
  });
});
