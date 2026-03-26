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
  // Force react-grid-layout to be pre-bundled as CJS so named exports
  // (Responsive, WidthProvider) are available in the ESM context.
  optimizeDeps: {
    include: ["react-grid-layout"],
  },
  build: {
    outDir: "dist",
    emptyOutDir: true,
    commonjsOptions: {
      include: [/react-grid-layout/, /node_modules/],
    },
  },
  server: {
    port: 5173,
    strictPort: true,
  },
});
