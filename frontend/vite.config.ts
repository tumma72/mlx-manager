import { sveltekit } from "@sveltejs/kit/vite";
import { defineConfig } from "vite";

const apiPort = process.env.VITE_API_PORT || "10242";

export default defineConfig({
  plugins: [sveltekit()],
  server: {
    proxy: {
      "/api": {
        target: `http://localhost:${apiPort}`,
        changeOrigin: true,
      },
      // MLX Server inference endpoints - embedded in MLX Manager
      "/v1": {
        target: `http://localhost:${apiPort}`,
        changeOrigin: true,
      },
    },
  },
});
