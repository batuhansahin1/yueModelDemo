/** Entry point: views and routing, model server status, theme, keyboard shortcut. */
import {api} from "./api.js";
import {createCover} from "./cover.js";
import {createHistory} from "./history.js";
import {processTags} from "./logic.js";
import {createTextToSong} from "./t2s.js";
import {$, $$, h, mdElement, storage, toast} from "./ui.js";

const VIEWS = ["t2s", "cover", "history", "server"];
const HEALTH_INTERVAL_MS = 5000;

const current = () => (VIEWS.includes(location.hash.slice(1)) ? location.hash.slice(1) : "t2s");

let pastJobs;
const onFinished = () => {
  if (current() === "history") pastJobs.refresh({quiet: true});
};
const t2s = createTextToSong($("#view-t2s"), {onFinished});
const cover = createCover($("#view-cover"), {onFinished});
pastJobs = createHistory($("#view-history"), {
  isVisible: () => current() === "history" && document.visibilityState === "visible",
  onRestore(values) {
    (values.mode === "t2s" ? t2s : cover).restore(values);
    location.hash = values.mode;
    toast("Girdiler ve ayarlar forma geri yüklendi.", "success");
  },
});

// --- Routing (#t2s, #cover, #history, #server) -----------------------------------
function route() {
  const name = current();
  for (const view of VIEWS) $(`#view-${view}`).hidden = view !== name;
  for (const link of $$(".tabs a")) {
    if (link.dataset.view === name) link.setAttribute("aria-current", "page");
    else link.removeAttribute("aria-current");
  }
  if (name === "history") pastJobs.refresh({quiet: true});
  if (name === "server") checkHealth();
  window.scrollTo({top: 0});
}

// --- Model server status ---------------------------------------------------------
const pill = $("#status-pill");
const banner = $("#offline-banner");

/** {state, text, detail} of the model server, as the Gradio UI's connection line. */
function describeHealth(health, error) {
  if (error) {
    return error.status === 0
      ? {state: "down", text: "Web sunucusuna ulaşılamıyor", detail: error.message}
      : {state: "down", text: "Model sunucusu bağlı değil", detail: error.message};
  }
  // "NVIDIA L4" rather than PyTorch's "cuda:0"; older servers only report the latter.
  const device = health.device_name || health.device;
  if (health.mode === "mock") return {state: "mock", text: "Test modu (mock)", detail: "Gerçek model yok, deneme tonu üretilir."};
  if (health.load_error) return {state: "down", text: "Model yüklenemedi", detail: health.load_error};
  if (health.model_loaded) return {state: "ready", text: "YuE2 hazır", detail: `Cihaz: ${device}`};
  return {state: "idle", text: "Sunucuya bağlı",
    detail: `Model boşta, bellekte değil (üretim başlayınca yüklenir)${device ? ` · cihaz: ${device}` : ""}`};
}

const basename = (path) => String(path || "").split(/[\\/]/).filter(Boolean).pop() || "—";
const yesNo = (value) => (value == null ? "—" : value ? "Evet" : "Hayır");

function renderServer(health, error, status) {
  $("#server-status").replaceChildren(h("p", {class: `server-state state-${status.state}`},
    h("span", {class: "dot", "aria-hidden": "true"}), h("strong", {}, status.text), status.detail ? ` · ${status.detail}` : ""));
  const facts = health ? [
    ["Mod", health.mode === "mock" ? "mock (modelsiz test)" : health.mode],
    ["Cihaz", health.device_name || health.device || "—"],
    ["Model", health.model ? basename(health.model) : "—"],
    ["VAE", health.vae ? basename(health.vae) : "—"],
    ["Model bellekte", yesNo(health.model_loaded)],
    ["FP8 desteği", yesNo(health.fp8)],
    ["SheetSage2", health.sheetsage ? "Kurulu" : "Kurulu değil"],
    ["Kuyruk", `çalışan: ${(health.running || []).length}, sırada: ${health.queued ?? 0}`],
  ] : [];
  $("#server-facts").replaceChildren(...facts.map(([key, value]) => h("div", {}, h("dt", {}, key), h("dd", {}, value))));
  $("#server-json").textContent = JSON.stringify(health || {status: "unreachable", error: error?.message}, null, 2);
}

async function checkHealth() {
  let health = null;
  let error = null;
  try {
    health = await api.health();
  } catch (caught) {
    error = caught;
  }
  const status = describeHealth(health, error);
  pill.dataset.state = status.state;
  const busyCount = health ? (health.running || []).length + (health.queued || 0) : 0;
  $(".status-text", pill).textContent = status.text + (busyCount ? ` · ${busyCount} iş` : "");
  pill.title = status.detail || status.text;
  banner.hidden = !error;
  if (error) {
    banner.replaceChildren(mdElement("p", error.status === 0
      ? "🔴 **Web sunucusuna ulaşılamıyor.** Sayfa açık ama sunucu kapanmış olabilir."
      : "🔴 **Model sunucusu bağlı değil.** Arayüz çalışıyor; üretim için model sunucusunu başlatın "
        + "(`./start_model_server.sh`, Windows'ta `start_model_server.bat`) ya da modelsiz denemek için "
        + "`./frontend/start.sh --mock` (Windows'ta `frontend\\start.bat --mock`) kullanın."));
  }
  for (const mode of [t2s, cover]) mode.settings.setFp8Supported(health ? Boolean(health.fp8) : null);
  cover.setSheetsageAvailable(health ? Boolean(health.sheetsage) : null);
  renderServer(health, error, status);
  return health;
}

$("#server-check").addEventListener("click", async () => {
  const health = await checkHealth();
  toast(health ? "Model sunucusuna bağlanıldı." : "Model sunucusuna ulaşılamadı.", health ? "success" : "error");
});
// Polls pause while the tab is in the background.
setInterval(() => {
  if (document.visibilityState === "visible") checkHealth();
}, HEALTH_INTERVAL_MS);
document.addEventListener("visibilitychange", () => {
  if (document.visibilityState === "visible") checkHealth();
});
window.addEventListener("hashchange", route);
route();
if (current() !== "server") checkHealth();

// --- Theme -------------------------------------------------------------------------
$("#theme-toggle").addEventListener("click", () => {
  const root = document.documentElement;
  const dark = root.dataset.theme ? root.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
  root.dataset.theme = dark ? "light" : "dark";
  storage.set("yue2.theme", root.dataset.theme);
});

// --- Keyboard: ⌘/Ctrl + Enter generates in the open mode ---------------------------------
document.addEventListener("keydown", (event) => {
  if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
    const mode = {t2s, cover}[current()];
    if (mode) {
      event.preventDefault();
      mode.panel.generate();
    }
  }
});

// --- Data and the last jobs of this browser ---------------------------------------------
fetch("data/tags.json")
  .then((response) => (response.ok ? response.json() : Promise.reject(new Error(`HTTP ${response.status}`))))
  .then((data) => cover.setSuggestions(processTags(data)))
  .catch(() => toast("Etiket listesi yüklenemedi; etiketleri elle yazabilirsiniz.", "warning"));

for (const [mode, controller] of [["t2s", t2s], ["cover", cover]]) {
  const id = storage.get(`yue2.job.${mode}`);
  if (id) controller.panel.resume(id);
}
