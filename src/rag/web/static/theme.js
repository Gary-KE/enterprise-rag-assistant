// Runs before the page paints so the chosen theme never flashes. Storage access is guarded.
(function () {
  var theme = null;
  try { theme = localStorage.getItem("dondoo-theme"); } catch (e) { /* ignore */ }
  if (theme !== "light" && theme !== "dark") {
    theme = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  }
  var p = new URLSearchParams(location.search).get("theme");
  if (p === "light" || p === "dark") theme = p;
  document.documentElement.setAttribute("data-theme", theme);
})();
