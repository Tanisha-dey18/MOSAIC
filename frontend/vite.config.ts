/// <reference types="vitest" />
import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

/**
 * On Netlify (NETLIFY=true) an optional default backend URL is baked into the build. Leaving it
 * unset is fine: visitors then connect at runtime (?api=<url> link or the "Connect" form). A
 * localhost or plain-http default, however, would ship a site that cannot work, so it fails the build.
 */
function checkDeployedApiUrl(mode: string) {
  if (process.env.NETLIFY !== "true") return;
  const url = (loadEnv(mode, process.cwd(), "VITE_").VITE_API_BASE_URL || process.env.VITE_API_BASE_URL || "").trim();
  const problem = !url
    ? null
    : /\/\/(localhost|127\.0\.0\.1|0\.0\.0\.0)(:|\/|$)/.test(url)
      ? `VITE_API_BASE_URL points to ${url}, which visitors' browsers cannot reach`
      : !url.startsWith("https://")
        ? `VITE_API_BASE_URL must use https:// on an https site (got ${url})`
        : null;
  if (problem) {
    throw new Error(
      `${problem}. Deploy the backend first (see docs/deployment.md) and set ` +
        "VITE_API_BASE_URL=https://<your-backend> in Netlify: Site configuration > Environment variables, then redeploy.",
    );
  }
}

export default defineConfig(({ mode }) => {
  checkDeployedApiUrl(mode);
  return {
    plugins: [react()],
    server: { port: 5173 },
    build: {
      rollupOptions: {
        output: { manualChunks: { react: ["react", "react-dom", "react-router-dom"], charts: ["recharts"] } },
      },
    },
    test: { environment: "node" },
  };
});
