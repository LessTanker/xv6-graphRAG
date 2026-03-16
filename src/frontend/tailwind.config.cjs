/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        bg: "#f6f3ee",
        panel: "#fffaf2",
        ink: "#202020",
        accent: "#b6462a",
        border: "#d7c7b9"
      },
      boxShadow: {
        card: "0 10px 30px rgba(0, 0, 0, 0.08)"
      },
      fontFamily: {
        serifui: ["Georgia", "Times New Roman", "serif"]
      }
    }
  },
  plugins: []
};
