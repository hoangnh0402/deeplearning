import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Proxy /api → FastAPI (port 8000) để tránh CORS khi phát triển
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: { "/api": "http://localhost:8000" },
  },
});
