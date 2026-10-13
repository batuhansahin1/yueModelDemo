/**
 * 🗂 Geçmiş: the model server's past jobs (outputs/<job-id>/job.json survives restarts).
 */
import {api} from "./api.js";
import {FINAL, STATUS_LABELS, jobKind, restoreValues, shortStyle} from "./logic.js";
import {promptText, renderJob, renderPlayer} from "./output.js";
import {$, $$, copyText, formatDate, h, reportError} from "./ui.js";

const KIND_ICONS = {t2s: "✦", cover: "🎵", other: "♪"};
const KIND_NAMES = {t2s: "Text to Song", cover: "Cover", other: "Üretim"};
const STATUS_TONES = {queued: "neutral", running: "active", complete: "ok", failed: "error", cancelled: "neutral"};

export function createHistory(view, {onRestore, isVisible}) {
  const list = $("#history-list", view);
  const detail = $("#history-detail", view);
  const refreshButton = $("#history-refresh", view);
  const filterButtons = $$("[data-filter]", view);
  let jobs = [];
  let selected = null;
  let filter = "all";
  let timer = null;

  const visible = () => jobs.filter((job) => filter === "all" || jobKind(job.request) === filter);

  function renderList() {
    const items = visible();
    if (!items.length) {
      list.replaceChildren(h("li", {class: "empty"}, jobs.length ? "Bu filtrede üretim yok." : "Henüz üretim yok."));
      return;
    }
    list.replaceChildren(...items.map((job) => {
      const kind = jobKind(job.request);
      return h("li", {},
        h("button", {type: "button", class: "history-item", "aria-current": String(job.id === selected),
          onclick: () => select(job.id)},
          h("span", {class: "history-kind", title: KIND_NAMES[kind]}, KIND_ICONS[kind]),
          h("span", {class: "history-text"},
            h("span", {class: "history-style"}, shortStyle(job.request.style, 60) || "—"),
            h("span", {class: "history-meta"}, `${formatDate(job.created)} · `, h("span", {class: "mono"}, job.id))),
          h("span", {class: `badge badge-${STATUS_TONES[job.status] || "neutral"}`},
            STATUS_LABELS[job.status] || job.status)));
    }));
  }

  function renderEmptyDetail(text = "Soldan bir üretim seçin.") {
    detail.replaceChildren(h("p", {class: "empty"}, text));
  }

  async function show(id) {
    clearTimeout(timer);
    let job;
    try {
      job = await api.job(id);
    } catch (error) {
      reportError(error);
      return;
    }
    if (id !== selected) return;
    const kind = jobKind(job.request);
    const stats = FINAL.has(job.status) ? null : await api.stats();
    detail.replaceChildren(
      h("div", {class: "history-detail-head"},
        h("h2", {}, `${KIND_ICONS[kind]} ${KIND_NAMES[kind]}`),
        h("button", {type: "button", class: "btn btn-sm", onclick: () => restore(job)}, "Ayarları forma geri yükle")),
      renderJob(job, stats),
      job.status === "complete" ? renderPlayer(job) : "",
      job.request.reference?.file ? h("p", {class: "hint"}, `Referans: ${job.request.reference.file}`) : "",
      h("details", {class: "group"}, h("summary", {}, h("span", {}, "Modele giden girdi")),
        h("pre", {class: "code-block"}, promptText(job.request))),
      job.result?.abc ? h("details", {class: "group"}, h("summary", {}, h("span", {}, "ABC nota planı")),
        h("pre", {class: "code-block"}, job.result.abc),
        h("div", {class: "group-actions"},
          h("button", {type: "button", class: "btn btn-sm btn-ghost", onclick: () => copyText(job.result.abc)}, "Kopyala")))
        : "");
    // A running job is followed while it stays selected and the history is on screen.
    if (!FINAL.has(job.status)) {
      timer = setTimeout(() => {
        if (selected === id && isVisible()) show(id);
      }, 2000);
    }
  }

  function select(id) {
    selected = id;
    renderList();
    show(id);
  }

  function restore(job) {
    onRestore(restoreValues(job.request));
  }

  async function refresh({quiet = false} = {}) {
    try {
      jobs = await api.jobs(50);
    } catch (error) {
      if (!quiet) reportError(error);
      jobs = [];
      renderList();
      renderEmptyDetail("Model sunucusuna bağlanılamadı; geçmiş gösterilemiyor.");
      return;
    }
    if (!jobs.some((job) => job.id === selected)) selected = visible()[0]?.id ?? null;
    renderList();
    if (selected) show(selected);
    else renderEmptyDetail(jobs.length ? "Soldan bir üretim seçin." : "Henüz üretim yok. Text to Song ya da Cover ile ilk şarkınızı üretin.");
  }

  for (const button of filterButtons) {
    button.addEventListener("click", () => {
      filter = button.dataset.filter;
      for (const other of filterButtons) other.setAttribute("aria-pressed", String(other === button));
      if (selected && !visible().some((job) => job.id === selected)) selected = visible()[0]?.id ?? null;
      renderList();
      if (selected) show(selected);
      else renderEmptyDetail("Bu filtrede üretim yok.");
    });
  }
  refreshButton.addEventListener("click", () => refresh());
  renderEmptyDetail();

  return {refresh};
}
