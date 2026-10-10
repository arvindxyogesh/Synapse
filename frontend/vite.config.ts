import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// `vite build --mode demo` makes the static GitHub Pages demo: it reads
// .env.demo (VITE_STATIC_DEMO=true) and is served under /<repo>/.
export default defineConfig(({ mode }) => ({
  plugins: [react()],
  base: mode === "demo" ? "/Synapse/" : "/",
  server: { port: 5173 },
}));
