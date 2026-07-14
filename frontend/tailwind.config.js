/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        tokyo: {
          bg: "#1a1b26",
          dark: "#16161e",
          panel: "#1f2335",
          border: "rgba(122, 162, 247, 0.15)",
          text: "#a9b1d6",
          muted: "#565f89",
          blue: "#7aa2f7",
          cyan: "#7dcfff",
          magenta: "#bb9af7",
          orange: "#ff9e64",
          yellow: "#e0af68",
          red: "#f7768e",
          green: "#9ece6a",
          teal: "#1abc9c",
        }
      },
      fontFamily: {
        sans: ["Inter", "sans-serif"],
        mono: ["JetBrains Mono", "monospace"],
      },
      backdropBlur: {
        xs: "2px",
      }
    },
  },
  plugins: [],
}
