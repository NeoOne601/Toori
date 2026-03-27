import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { readFileSync, writeFileSync } from "fs";
import { join } from "path";

// Post-build plugin: strip `crossorigin` from dist/index.html so Electron
// can load the bundle via file:// without CORS blocking module loading.
function electronCompatBuild() {
  return {
    name: "electron-compat-build",
    closeBundle() {
      const htmlPath = join(__dirname, "dist", "index.html");
      try {
        const html = readFileSync(htmlPath, "utf8");
        const patched = html.replace(/ crossorigin(?:="[^"]*")?/g, "");
        writeFileSync(htmlPath, patched);
        console.log("[electron-compat] Stripped crossorigin from dist/index.html");
      } catch {
        // dist may not exist during dev
      }
    },
  };
}

export default defineConfig({
  base: "./",
  plugins: [react(), electronCompatBuild()],
  build: {
    outDir: "dist",
    emptyOutDir: true,
  },
  server: {
    port: 5173,
    strictPort: true,
    proxy: {
      "/v1/file": {
        target: "http://127.0.0.1:7777",
        changeOrigin: true,
      },
    },
  },
});
