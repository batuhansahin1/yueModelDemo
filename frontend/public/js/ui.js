/** Small DOM helpers shared by the views: element builder, inline markdown, toasts, busy buttons. */
import {ApiError} from "./api.js";
import {ValidationError} from "./logic.js";

const PROPERTIES = new Set(["value", "checked", "disabled", "hidden", "selected", "open"]);

/** Element builder: h("button", {class: "btn", onclick}, "Label", child...). */
export function h(tag, props, ...children) {
  const element = document.createElement(tag);
  for (const [key, value] of Object.entries(props || {})) {
    if (value == null || value === false) continue;
    if (key === "class") element.className = value;
    else if (key === "dataset") Object.assign(element.dataset, value);
    else if (key.startsWith("on") && typeof value === "function") element.addEventListener(key.slice(2), value);
    else if (PROPERTIES.has(key)) element[key] = value;
    else element.setAttribute(key, value === true ? "" : value);
  }
  for (const child of children.flat(Infinity)) {
    if (child == null || child === false) continue;
    element.append(child instanceof Node ? child : String(child));
  }
  return element;
}

export const $ = (selector, root = document) => root.querySelector(selector);
export const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];

export function escapeHtml(text) {
  return String(text ?? "").replace(/[&<>"']/g, (c) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"})[c]);
}

/** Inline markdown of the UI's own messages: **bold**, *italic*, `code`. Everything else is escaped. */
export function md(text) {
  return String(text ?? "").split(/(`[^`\n]*`)/).map((part) => {
    if (part.length >= 2 && part.startsWith("`") && part.endsWith("`")) return `<code>${escapeHtml(part.slice(1, -1))}</code>`;
    return escapeHtml(part)
      .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
      .replace(/\*([^*\s][^*]*?)\*/g, "<em>$1</em>");
  }).join("");
}

/** Element with inline markdown content. */
export function mdElement(tag, text, props = {}) {
  const element = h(tag, props);
  element.innerHTML = md(text);
  return element;
}

const TOAST_ICONS = {info: "ℹ️", success: "✅", warning: "⚠️", error: "⛔"};

export function toast(message, kind = "info", timeout) {
  const box = document.getElementById("toasts");
  if (!box) return;
  const close = h("button", {class: "toast-close", type: "button", "aria-label": "Kapat"}, "×");
  const element = h("div", {class: `toast toast-${kind}`, role: kind === "error" ? "alert" : "status"},
    h("span", {class: "toast-icon", "aria-hidden": "true"}, TOAST_ICONS[kind] || ""),
    mdElement("div", String(message).replace(/^⚠️\s*/, ""), {class: "toast-body"}),
    close);
  const dismiss = () => {
    clearTimeout(timer);
    element.classList.add("leaving");
    setTimeout(() => element.remove(), 180);
  };
  close.addEventListener("click", dismiss);
  box.append(element);
  // Errors stay longer; at most five messages at a time.
  const timer = setTimeout(dismiss, timeout ?? {error: 12000, warning: 9000}[kind] ?? 5000);
  while (box.children.length > 5) box.firstElementChild.remove();
}

/** Show an error from an action: form problems and server messages as they are, bugs with a hint. */
export function reportError(error) {
  if (error instanceof ValidationError) {
    toast(error.message, "warning");
  } else if (error instanceof ApiError) {
    toast(error.message, error.status === 409 ? "warning" : "error");
  } else {
    console.error(error);
    toast(`Beklenmeyen hata: ${error?.message || error}`, "error");
  }
}

/** Button in its working state (spinner, label, elapsed seconds); returns the function that restores it. */
export function busy(button, label) {
  const children = [...button.childNodes];
  const wasDisabled = button.disabled;
  const started = Date.now();
  const text = h("span", {}, label);
  button.disabled = true;
  button.classList.add("is-busy");
  button.replaceChildren(h("span", {class: "spinner", "aria-hidden": "true"}), text);
  const timer = setInterval(() => {
    const seconds = Math.round((Date.now() - started) / 1000);
    if (seconds >= 2) text.textContent = `${label} ${seconds} sn`;
  }, 1000);
  return () => {
    clearInterval(timer);
    button.replaceChildren(...children);
    button.classList.remove("is-busy");
    button.disabled = wasDisabled;
  };
}

export const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

/** localStorage that never throws (private windows, blocked storage). */
export const storage = {
  get(key) {
    try {
      return localStorage.getItem(key);
    } catch {
      return null;
    }
  },
  set(key, value) {
    try {
      localStorage.setItem(key, value);
    } catch {
      // Remembering is a convenience only.
    }
  },
  remove(key) {
    try {
      localStorage.removeItem(key);
    } catch {
      // Same as above.
    }
  },
};

export function downloadFile(name, content, type = "application/json") {
  const url = URL.createObjectURL(new Blob([content], {type}));
  const link = h("a", {href: url, download: name});
  document.body.append(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export async function copyText(text) {
  try {
    await navigator.clipboard.writeText(text);
  } catch {
    // The Clipboard API needs a secure context (https or localhost); fall back to a selection.
    const area = h("textarea", {class: "visually-hidden"});
    area.value = text;
    document.body.append(area);
    area.select();
    document.execCommand("copy");
    area.remove();
  }
  toast("Panoya kopyalandı.", "success", 2500);
}

export function formatBytes(bytes) {
  return bytes >= 1048576 ? `${(bytes / 1048576).toFixed(1)} MB` : `${Math.max(1, Math.round(bytes / 1024))} KB`;
}

export function formatDate(epochSeconds) {
  if (!epochSeconds) return "";
  return new Date(epochSeconds * 1000).toLocaleString("tr-TR", {dateStyle: "short", timeStyle: "short"});
}

/** Character counter under a text field ("12/3000"), warning near the limit. */
export function bindCounter(field, counter, limit) {
  const update = () => {
    const length = field.value.length;
    counter.textContent = `${length}/${limit}`;
    counter.classList.toggle("near-limit", length > limit * 0.9);
  };
  field.addEventListener("input", update);
  update();
  return update;
}

/** Tabs inside a card ("Prompt" / "Model ayarları"). */
export function bindSubtabs(root) {
  const tabs = $$("[data-subtab]", root);
  const select = (name) => {
    for (const tab of tabs) {
      const active = tab.dataset.subtab === name;
      tab.setAttribute("aria-selected", String(active));
      tab.tabIndex = active ? 0 : -1;
    }
    for (const panel of $$("[data-panel]", root)) panel.hidden = panel.dataset.panel !== name;
  };
  for (const tab of tabs) tab.addEventListener("click", () => select(tab.dataset.subtab));
  return select;
}
