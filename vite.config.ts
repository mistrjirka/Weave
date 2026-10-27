import { defineConfig } from "vite";
export default defineConfig({
  root: "frontend",
  server: {
    host: "0.0.0.0",
    port: 5173,
    proxy: { "/api": "http://127.0.0.1:8173" },
  },
  build: {
    outDir: "../dist",
    emptyOutDir: true,
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (id.includes("/node_modules/zod/")) return "validation";
          if (
            id.includes("/node_modules/@xyflow/") ||
            id.includes("/node_modules/@dagrejs/")
          )
            return "graph-engine";
          if (
            id.includes("/node_modules/react/") ||
            id.includes("/node_modules/react-dom/")
          )
            return "react";
        },
      },
    },
  },
});
