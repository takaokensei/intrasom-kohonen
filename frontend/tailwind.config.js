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
          textDim: "#9aa5ce",
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
      // 2xs/3xs: the dashboard's dense metric/label UI relies on sub-xs text
      // (previously hardcoded as text-[10px] / text-[9.5px] ~40 times across
      // components with no scale entry). Centralizing these two sizes here
      // keeps the type scale explicit instead of ad hoc per call site.
      fontSize: {
        "2xs": ["0.625rem", { lineHeight: "0.875rem" }], // 10px
        "3xs": ["0.59375rem", { lineHeight: "0.75rem" }], // 9.5px
      },
      backdropBlur: {
        xs: "2px",
      }
    },
  },
  plugins: [],
}
