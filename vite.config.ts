import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "path";
import fs from "fs";
import tailwindcss from "@tailwindcss/vite";

import { vitePluginForArco } from "@arco-plugins/vite-react";

export default defineConfig({
  plugins: [react(), vitePluginForArco(), tailwindcss()],
  optimizeDeps: {
    exclude: ["@/utils/loader.js", "@/cpp/build/opencv_wasm.js"],
  },
  server: {
    host: true, // 添加这行，允许通过 IP 访问
    https: {
      key: fs.readFileSync("./localhost+3-key.pem"),
      cert: fs.readFileSync("./localhost+3.pem"),
    },
    cors: {
      origin: "*",
      methods: ["GET", "POST", "OPTIONS"],
      allowedHeaders: "*",
    },
    headers: {
      "Cross-Origin-Embedder-Policy": "require-corp",
      "Cross-Origin-Opener-Policy": "same-origin",
      "Cross-Origin-Resource-Policy": "cross-origin",
    },
  },
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
});
