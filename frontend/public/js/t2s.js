/**
 * ✦ Text to Song: description + lyrics, voice and style; the model composes melody and chords itself.
 */
import {api, ensureServer} from "./api.js";
import {DEFAULT_ODE_STEPS, DESCRIPTION_LIMIT, FAST_ODE_STEPS, LYRICS_LIMIT, T2S_STYLES, T2S_VOICES,
  buildT2SPayload} from "./logic.js";
import {GenerationPanel} from "./output.js";
import {ModelSettings} from "./settings.js";
import {$, $$, bindCounter, bindSubtabs, h, toast} from "./ui.js";

const EXAMPLE = {
  description: "120 BPM'de, sürükleyici ritimli ve kadın vokalli siberpunk bir synth-pop şarkısı.",
  lyrics: "[verse]\nNeon ışıklar soluyor caddede\nHer gölge seni getiriyor aklıma\n\n"
    + "[chorus]\nGece kapansa da üstüme\nYine sana dönerim",
};

export function createTextToSong(view, {onFinished}) {
  const description = $("#t2s-description", view);
  const lyrics = $("#t2s-lyrics", view);
  const customStyle = $("#t2s-custom-style", view);
  const voiceButton = $("#t2s-voice-button", view);
  const styleButton = $("#t2s-style-button", view);
  const voicePanel = $("#t2s-voice-panel", view);
  const stylePanel = $("#t2s-style-panel", view);
  const updateDescriptionCount = bindCounter(description, $("#t2s-description-count", view), DESCRIPTION_LIMIT);
  const updateLyricsCount = bindCounter(lyrics, $("#t2s-lyrics-count", view), LYRICS_LIMIT);
  bindSubtabs($("#t2s-card", view));

  // Voice and style open as small panels; picking a voice or pressing Tamam closes them.
  $("#t2s-voices", view).append(...T2S_VOICES.map((voice) => h("label", {class: "pill"},
    h("input", {type: "radio", name: "t2s-voice", value: voice.value, checked: voice.value === "default"}),
    h("span", {}, voice.label))));
  $("#t2s-styles", view).append(...T2S_STYLES.map((style) => h("label", {class: "pill"},
    h("input", {type: "checkbox", name: "t2s-style", value: style.value}), h("span", {}, style.label))));
  const voiceInputs = $$("input[name=t2s-voice]", view);
  const styleInputs = $$("input[name=t2s-style]", view);
  const popovers = [[voiceButton, voicePanel], [styleButton, stylePanel]];

  function showPopover(panel) {
    for (const [button, element] of popovers) {
      element.hidden = element !== panel || !element.hidden;
      button.setAttribute("aria-expanded", String(!element.hidden));
    }
  }
  const closePopovers = () => showPopover(null);
  voiceButton.addEventListener("click", () => showPopover(voicePanel));
  styleButton.addEventListener("click", () => showPopover(stylePanel));
  $("#t2s-style-close", view).addEventListener("click", closePopovers);
  document.addEventListener("click", (event) => {
    if (!popovers.some(([button, panel]) => button.contains(event.target) || panel.contains(event.target))) closePopovers();
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") closePopovers();
  });

  const voice = () => voiceInputs.find((input) => input.checked)?.value || "default";
  const styles = () => styleInputs.filter((input) => input.checked).map((input) => input.value);
  function updateLabels() {
    voiceButton.textContent = `🎙 Vokal: ${T2S_VOICES.find((item) => item.value === voice()).label}`;
    const count = styles().length + (customStyle.value.trim() ? 1 : 0);
    styleButton.textContent = count ? `🎚 Tür (${count})` : "🎚 Tür";
  }
  for (const input of voiceInputs) {
    input.addEventListener("change", () => {
      updateLabels();
      closePopovers();
    });
  }
  for (const input of styleInputs) input.addEventListener("change", updateLabels);
  customStyle.addEventListener("input", updateLabels);
  customStyle.addEventListener("keydown", (event) => {
    if (event.key === "Enter") closePopovers();
  });

  $("#t2s-example", view).addEventListener("click", () => {
    description.value = EXAMPLE.description;
    lyrics.value = EXAMPLE.lyrics;
    updateDescriptionCount();
    updateLyricsCount();
  });

  let panel;
  // ⚡ Hızlı sentez is a preset of the ODE step slider; either one updates the other.
  const settings = new ModelSettings({onOdeChange: (steps) => panel.setFast(steps === FAST_ODE_STEPS)});
  $("[data-panel=settings]", view).append(settings.element);

  async function generate() {
    const form = {description: description.value, lyrics: lyrics.value, voice: voice(), styles: styles(),
      customStyle: customStyle.value};
    const {payload, warnings} = buildT2SPayload(form, settings.values);
    warnings.forEach((warning) => toast(warning, "warning"));
    await ensureServer();
    const stats = await api.stats();
    closePopovers();
    await panel.follow(payload, stats);
  }

  panel = new GenerationPanel({
    mode: "t2s", title: "Text to Song", generateLabel: "⚡ Şarkı üret", onGenerate: generate, onFinished,
    onFastChange: (on) => {
      settings.odeSteps = on ? FAST_ODE_STEPS : DEFAULT_ODE_STEPS;
    },
    onUseAbc: (abc) => {
      settings.setAbc(abc);
      toast("ABC, *Model ayarları → Harici ABC nota* alanına kopyalandı.", "success");
    },
  });
  $("#t2s-workspace", view).append(panel.element);

  return {
    panel,
    settings,
    /** Inputs of a past Text to Song job (history → "Ayarları forma geri yükle"). */
    restore({form, settings: values}) {
      description.value = form.description;
      lyrics.value = form.lyrics;
      for (const input of voiceInputs) input.checked = input.value === form.voice;
      for (const input of styleInputs) input.checked = form.styles.includes(input.value);
      customStyle.value = form.customStyle;
      updateDescriptionCount();
      updateLyricsCount();
      updateLabels();
      settings.set(values);
      panel.setFast(Number(values.odeSteps) === FAST_ODE_STEPS);
    },
  };
}
