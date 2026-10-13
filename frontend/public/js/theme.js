// Runs before the page renders so a saved light/dark choice does not flash the other theme.
try {
  var theme = localStorage.getItem("yue2.theme");
  if (theme === "light" || theme === "dark") document.documentElement.dataset.theme = theme;
} catch (error) {
  // Storage blocked: follow the system theme.
}
