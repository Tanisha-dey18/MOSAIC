/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        brand: { 50: "#eef8f7", 100: "#d5efeb", 200: "#a9ddd5", 300: "#7dc8bc", 400: "#46aa9c", 500: "#2a8c80", 600: "#1f6f67", 700: "#185a55", 800: "#134846", 900: "#0f3b3a" },
        ink: { DEFAULT: "#1f2933", soft: "#52606d", faint: "#7b8794" },
      },
      fontFamily: { sans: ["Inter", "Segoe UI", "system-ui", "sans-serif"] },
    },
  },
  plugins: [],
};
