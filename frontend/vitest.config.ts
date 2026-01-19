import { defineConfig } from "vitest/config";
import { sveltekit } from "@sveltejs/kit/vite";

export default defineConfig({
  plugins: [sveltekit()],
  resolve: {
    // Ensure browser conditions are used for Svelte packages
    conditions: ["browser"],
  },
  test: {
    include: ["src/**/*.{test,spec}.{js,ts}"],
    environment: "jsdom",
    globals: true,
    setupFiles: ["./src/tests/setup.ts"],
    coverage: {
      provider: "v8",
      reporter: ["text", "json", "html", "lcov", "json-summary"],
      exclude: [
        "node_modules/**",
        "src/tests/**",
        "**/*.d.ts",
        "**/*.config.*",
        "src/lib/components/ui/**", // Exclude UI primitives from coverage
        "**/index.ts", // Exclude barrel re-exports
        "src/lib/api/types.ts", // Exclude type definitions
      ],
      thresholds: {
        statements: 95,
        branches: 90, // Lower due to Svelte 5 compiled template artifacts
        functions: 95,
        lines: 95,
      },
    },
  },
});
