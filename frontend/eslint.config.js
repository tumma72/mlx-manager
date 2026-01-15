import js from "@eslint/js";
import svelte from "eslint-plugin-svelte";
import globals from "globals";
import ts from "typescript-eslint";
import svelteParser from "svelte-eslint-parser";

export default [
  js.configs.recommended,
  ...ts.configs.recommended,
  ...svelte.configs["flat/recommended"],
  {
    languageOptions: {
      globals: {
        ...globals.browser,
        ...globals.node,
      },
    },
  },
  {
    files: ["**/*.svelte"],
    languageOptions: {
      parserOptions: {
        parser: ts.parser,
      },
    },
    rules: {
      // Configure navigation rule to allow resolve() with dynamic paths
      // The rule doesn't fully support TypeScript's strict route typing
      "svelte/no-navigation-without-resolve": [
        "error",
        {
          // Links in components like Button.svelte receive resolved paths from callers
          ignoreLinks: false,
          ignoreGoto: false,
        },
      ],
    },
  },
  {
    // Handle .svelte.ts files (Svelte 5 runes mode)
    files: ["**/*.svelte.ts"],
    languageOptions: {
      parser: svelteParser,
      parserOptions: {
        parser: ts.parser,
      },
    },
  },
  {
    ignores: ["build/", ".svelte-kit/", "dist/"],
  },
];
