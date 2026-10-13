/**
 * Multi-select tag field with suggestions and custom values (Gradio: Dropdown with
 * multiselect + allow_custom_value), used for the Cover style tags.
 */
import {h, toast} from "./ui.js";

const MAX_SUGGESTIONS = 80;
let instances = 0;

export class TagInput {
  constructor({label, max = 50, placeholder = "", onChange} = {}) {
    this.id = `tags-${++instances}`;
    this.max = max;
    this.placeholder = placeholder;
    this.onChange = onChange;
    this.tags = [];
    this.suggestions = [];
    this.matches = [];
    this.active = -1;

    this.chips = h("span", {class: "tag-chips"});
    this.input = h("input", {type: "text", class: "tag-input", placeholder, autocomplete: "off", spellcheck: "false",
      role: "combobox", "aria-label": label, "aria-autocomplete": "list", "aria-expanded": "false", "aria-controls": `${this.id}-list`});
    this.list = h("ul", {class: "tag-menu", id: `${this.id}-list`, role: "listbox", hidden: true});
    this.element = h("div", {class: "tag-field"},
      h("div", {class: "tag-box", onclick: (event) => { if (event.target === event.currentTarget) this.input.focus(); }},
        this.chips, this.input),
      this.list);

    this.input.addEventListener("input", () => this.open());
    this.input.addEventListener("focus", () => this.open());
    this.input.addEventListener("keydown", (event) => this.keydown(event));
    // Typed text is kept as a tag when the field loses focus (unless a suggestion was clicked).
    this.input.addEventListener("blur", () => setTimeout(() => {
      if (!this.element.contains(document.activeElement)) {
        this.commit(this.input.value);
        this.close();
      }
    }, 120));
    this.render();
  }

  setSuggestions(list) {
    this.suggestions = list;
  }

  get values() {
    return [...this.tags];
  }

  set values(list) {
    this.tags = [];
    for (const tag of list || []) this.add(tag, {quiet: true});
    this.render();
  }

  add(raw, {quiet = false} = {}) {
    const tag = String(raw || "").trim().replace(/\s+/g, " ");
    if (!tag || this.tags.some((existing) => existing.toLowerCase() === tag.toLowerCase())) return false;
    if (this.tags.length >= this.max) {
      if (!quiet) toast(`En fazla ${this.max} etiket seçilebilir.`, "warning");
      return false;
    }
    this.tags.push(tag);
    return true;
  }

  remove(index) {
    this.tags.splice(index, 1);
    this.render();
    this.changed();
  }

  /** Add the typed text (one or more comma-separated tags). */
  commit(text) {
    let added = false;
    for (const part of String(text || "").split(",")) added = this.add(part) || added;
    this.input.value = "";
    if (added) {
      this.render();
      this.changed();
    }
  }

  changed() {
    this.onChange?.(this.values);
  }

  render() {
    this.chips.replaceChildren(...this.tags.map((tag, i) => h("span", {class: "tag-chip"}, tag,
      h("button", {type: "button", class: "tag-remove", "aria-label": `${tag} etiketini kaldır`,
        onclick: () => this.remove(i)}, "×"))));
    this.input.placeholder = this.tags.length ? "" : this.placeholder;
  }

  open() {
    const query = this.input.value.trim().toLowerCase();
    const taken = new Set(this.tags.map((tag) => tag.toLowerCase()));
    const free = this.suggestions.filter((tag) => !taken.has(tag.toLowerCase()));
    // Prefix matches first, then the ones that contain the query.
    this.matches = query
      ? [...free.filter((tag) => tag.toLowerCase().startsWith(query)),
        ...free.filter((tag) => !tag.toLowerCase().startsWith(query) && tag.toLowerCase().includes(query))]
      : free;
    this.matches = this.matches.slice(0, MAX_SUGGESTIONS);
    // Enter adds exactly what was typed unless it is in the list; arrows pick a suggestion.
    this.active = query ? this.matches.findIndex((tag) => tag.toLowerCase() === query) : -1;
    const custom = query && this.active < 0;
    this.list.replaceChildren(
      ...this.matches.map((tag, i) => h("li", {role: "option", id: `${this.id}-opt-${i}`, class: "tag-option",
        onmousedown: (event) => {
          event.preventDefault();
          this.pick(i);
        }}, tag)),
      custom ? h("li", {class: "tag-option tag-custom", onmousedown: (event) => {
        event.preventDefault();
        this.commit(this.input.value);
        this.open();
      }}, `"${this.input.value.trim()}" ekle (Enter)`) : null,
    );
    this.list.hidden = !this.list.children.length;
    this.input.setAttribute("aria-expanded", String(!this.list.hidden));
    this.highlight();
  }

  close() {
    this.list.hidden = true;
    this.input.setAttribute("aria-expanded", "false");
    this.active = -1;
  }

  pick(index) {
    if (this.add(this.matches[index])) {
      this.render();
      this.changed();
    }
    this.input.value = "";
    this.open();
  }

  highlight() {
    [...this.list.querySelectorAll(".tag-option")].forEach((option, i) => {
      option.classList.toggle("is-active", i === this.active);
      option.setAttribute("aria-selected", String(i === this.active));
    });
    const option = this.list.querySelector(".tag-option.is-active");
    if (option) {
      this.input.setAttribute("aria-activedescendant", option.id || "");
      option.scrollIntoView({block: "nearest"});
    } else {
      this.input.removeAttribute("aria-activedescendant");
    }
  }

  keydown(event) {
    if (event.key === "ArrowDown" || event.key === "ArrowUp") {
      if (this.list.hidden) this.open();
      const count = this.matches.length;
      if (!count) return;
      event.preventDefault();
      const step = event.key === "ArrowDown" ? 1 : -1;
      this.active = this.active < 0 ? (step > 0 ? 0 : count - 1) : (this.active + step + count) % count;
      this.highlight();
    } else if (event.key === "Enter") {
      event.preventDefault();
      if (this.active >= 0 && this.active < this.matches.length && !this.list.hidden) {
        this.pick(this.active);
      } else {
        this.commit(this.input.value);
        this.open();
      }
    } else if (event.key === ",") {
      event.preventDefault();
      this.commit(this.input.value);
      this.open();
    } else if (event.key === "Backspace" && !this.input.value && this.tags.length) {
      this.remove(this.tags.length - 1);
    } else if (event.key === "Escape") {
      this.close();
    }
  }
}
