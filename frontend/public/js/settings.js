/**
 * "Model ayarları" panel: every per-job YuE2 setting of the Gradio UI's Model settings tab.
 *
 * Each mode (Text to Song, Cover) has its own instance, so a cover analysis that switches the
 * CoT mode or enlarges the song token budget does not change the other mode's settings.
 */
import {h} from "./ui.js";
import {ABC_DEFAULTS, CFG_RANGE, COT_CHOICES, DEFAULT_ODE_STEPS, FAST_ODE_STEPS, ODE_RANGE, SAMPLING_CONTROLS,
  SEMANTIC_DEFAULTS, defaultSettings} from "./logic.js";

let instances = 0;

/** A slider with a number box next to it; the number box holds the value. */
function rangeControl({label, info, min, max, step, onInput}) {
  const range = h("input", {type: "range", min, max, step, "aria-label": label});
  const number = h("input", {type: "number", class: "num", min, max, step: "any", "aria-label": label});
  range.addEventListener("input", () => {
    number.value = range.value;
    onInput?.();
  });
  number.addEventListener("input", () => {
    if (number.value !== "") range.value = number.value;
    onInput?.();
  });
  const element = h("div", {class: "control"},
    h("div", {class: "control-head"}, h("span", {class: "control-label"}, label), number),
    range,
    info && h("p", {class: "hint"}, info));
  return {
    element,
    get value() {
      return number.value;
    },
    set value(value) {
      number.value = value;
      range.value = value;
    },
    set disabled(disabled) {
      range.disabled = disabled;
      number.disabled = disabled;
      element.classList.toggle("is-disabled", disabled);
    },
  };
}

function numberControl({label, info, min, onInput}) {
  const number = h("input", {type: "number", class: "num num-wide", min, step: 1, "aria-label": label});
  number.addEventListener("input", () => onInput?.());
  const element = h("div", {class: "control"},
    h("div", {class: "control-head"}, h("span", {class: "control-label"}, label), number),
    info && h("p", {class: "hint"}, info));
  return {
    element,
    get value() {
      return number.value;
    },
    set value(value) {
      number.value = value;
    },
  };
}

function toggle(label, onChange) {
  const input = h("input", {type: "checkbox", role: "switch"});
  input.addEventListener("change", () => onChange?.(input.checked));
  return {input, element: h("label", {class: "switch"}, input, h("span", {class: "switch-track", "aria-hidden": "true"}), h("span", {}, label))};
}

/** Sampling controls of one stage (ABC plan or song tokens) in a collapsible group. */
function samplingGroup(title, defaults) {
  const meta = h("span", {class: "group-meta"});
  const controls = {};
  const update = () => {
    const changed = SAMPLING_CONTROLS.filter(({key}) => Number(controls[key].value) !== defaults[key]).length;
    meta.textContent = changed ? `${changed} değişiklik` : "varsayılan";
    meta.classList.toggle("is-changed", changed > 0);
  };
  for (const control of SAMPLING_CONTROLS) {
    controls[control.key] = (control.number ? numberControl : rangeControl)({...control, onInput: update});
  }
  const element = h("details", {class: "group"},
    h("summary", {}, h("span", {}, title), meta),
    h("div", {class: "control-grid"}, SAMPLING_CONTROLS.map(({key}) => controls[key].element)));
  return {
    element,
    get values() {
      return Object.fromEntries(SAMPLING_CONTROLS.map(({key}) => [key, controls[key].value]));
    },
    set values(values) {
      for (const {key} of SAMPLING_CONTROLS) controls[key].value = values[key] ?? defaults[key];
      update();
    },
    get maxTokens() {
      return Number(controls.max_tokens.value);
    },
    set maxTokens(value) {
      controls.max_tokens.value = value;
      update();
    },
  };
}

export class ModelSettings {
  /**
   * onAbcChange(text): the external score changed (typed or set); onOdeChange(steps): the synthesis
   * steps were moved by the user; onReset(): "Varsayılanlara dön" was pressed.
   */
  constructor({onAbcChange, onOdeChange, onReset} = {}) {
    this.name = `settings-${++instances}`;
    this.element = this.render();
    this.set(defaultSettings());
    // Callbacks only after the initial values: the owner is still being set up during the constructor.
    this.onAbcChange = onAbcChange;
    this.onOdeChange = onOdeChange;
    this.onReset = onReset;
  }

  render() {
    this.cotInputs = COT_CHOICES.map((choice) => h("input", {type: "radio", name: `${this.name}-cot`, value: choice.value}));
    const cot = h("section", {class: "setting"},
      h("div", {class: "setting-head"}, h("h4", {}, "CoT modu"),
        h("p", {class: "hint"}, "Tam/Melodi önce bir ABC nota planı üretir, sonra şarkıyı ona göre üretir.")),
      h("div", {class: "choices choices-3", role: "radiogroup", "aria-label": "CoT modu"},
        COT_CHOICES.map((choice, i) => h("label", {class: "choice"}, this.cotInputs[i],
          h("span", {class: "choice-title"}, choice.label), h("span", {class: "choice-detail"}, choice.detail)))));

    this.randomSeed = toggle("Rastgele seed", () => this.syncDisabled());
    this.seed = h("input", {type: "number", class: "num num-wide", min: 0, step: 1, "aria-label": "Seed"});
    const seed = h("section", {class: "setting"},
      h("div", {class: "setting-head"}, h("h4", {}, "Seed"),
        h("p", {class: "hint"}, "Aynı seed ve aynı ayarlar aynı şarkıyı verir. Sonuçları karşılaştırmak için "
          + "rastgele seed'i kapatıp bir seed girin.")),
      h("div", {class: "setting-row"}, this.randomSeed.element, h("label", {class: "inline-field"}, "Seed", this.seed)));

    this.customCfg = toggle("CFG'yi özelleştir", () => this.syncDisabled());
    this.cfg = rangeControl({label: "CFG scale", min: CFG_RANGE[0], max: CFG_RANGE[1], step: 0.05,
      info: "1'den büyük: stil etiketlerine ve sözlere daha sıkı uyar (notaya değil). 1'den farklı her değer "
        + "şarkı aşamasını ~2 kat yavaşlatır."});
    const cfg = h("section", {class: "setting"},
      h("div", {class: "setting-head"}, h("h4", {}, "CFG"),
        h("p", {class: "hint"}, "Kapalıyken model varsayılanı kullanılır (full/melody 1.0, off 1.01).")),
      this.customCfg.element, this.cfg.element);

    this.ode = rangeControl({label: "Ses sentezi adımı (ODE)", min: ODE_RANGE[0], max: ODE_RANGE[1], step: 1,
      info: `Varsayılan ${DEFAULT_ODE_STEPS}. Az adım: hızlı, kalite düşebilir. Çok adım: yavaş. `
        + `⚡ Hızlı sentez = ${FAST_ODE_STEPS}.`,
      onInput: () => this.onOdeChange?.(Number(this.ode.value))});
    const ode = h("section", {class: "setting"}, this.ode.element);

    this.fp8 = toggle("FP8 (deneysel)");
    this.fp8Note = h("p", {class: "hint hint-warn", hidden: true},
      "Bu sunucunun cihazı FP8 desteklemiyor (compute capability 8.9+ NVIDIA gerekir).");
    const fp8 = h("section", {class: "setting"}, this.fp8.element,
      h("p", {class: "hint"}, "Token üretiminde FP8 ağırlıklar. Yalnızca compute 8.9+ NVIDIA (ör. L4). Kalite etkisi "
        + "denenmedi; yue2 FP8'de CUDA Graph kullanmadığı için daha yavaş da olabilir."),
      this.fp8Note);

    this.abcSampling = samplingGroup("ABC (nota planı) örnekleme", ABC_DEFAULTS);
    this.semanticSampling = samplingGroup("Şarkı (semantik) örnekleme", SEMANTIC_DEFAULTS);

    this.abcMeta = h("span", {class: "group-meta"});
    this.abcText = h("textarea", {class: "code-area", rows: 10, spellcheck: "false", "aria-label": "Harici ABC nota",
      placeholder: "X:1\nL:1/32\n...", oninput: () => this.abcChanged()});
    this.abcGroup = h("details", {class: "group"},
      h("summary", {}, h("span", {}, "Harici ABC nota"), this.abcMeta),
      h("p", {class: "hint"}, "Doluysa model nota planı üretmez, bu notayı kullanır. Önceki bir üretimin ABC'sini buraya "
        + "yapıştırıp sözleri/stili değiştirebilirsiniz."),
      this.abcText,
      h("div", {class: "group-actions"},
        h("button", {type: "button", class: "btn btn-sm", onclick: () => this.setAbc("")}, "Notayı temizle")));

    const reset = h("button", {type: "button", class: "btn btn-ghost btn-sm", onclick: () => this.reset()},
      "↺ Varsayılanlara dön");
    return h("form", {class: "settings", novalidate: true, onsubmit: (event) => event.preventDefault()},
      cot, seed, cfg, ode, fp8,
      h("div", {class: "groups"}, this.abcSampling.element, this.semanticSampling.element, this.abcGroup),
      h("div", {class: "settings-foot"},
        h("p", {class: "hint"}, "Bellek ve arka uç ayarları (--budget, --offload-ar, --backend) model sunucusu "
          + "başlatılırken verilir; müziği değiştirmezler."),
        reset));
  }

  syncDisabled() {
    this.seed.disabled = this.randomSeed.input.checked;
    this.cfg.disabled = !this.customCfg.input.checked;
  }

  abcChanged() {
    const lines = this.abcText.value.trim() ? this.abcText.value.trim().split("\n").length : 0;
    this.abcMeta.textContent = lines ? `${lines} satır` : "boş";
    this.abcMeta.classList.toggle("is-changed", lines > 0);
    this.onAbcChange?.(this.abcText.value);
  }

  get values() {
    return {
      cot: this.cotInputs.find((input) => input.checked)?.value || "full",
      randomSeed: this.randomSeed.input.checked,
      seed: this.seed.value,
      customCfg: this.customCfg.input.checked,
      cfgScale: this.cfg.value,
      odeSteps: this.ode.value,
      fp8: this.fp8.input.checked,
      abc: this.abcText.value,
      abcSampling: this.abcSampling.values,
      semanticSampling: this.semanticSampling.values,
    };
  }

  /** Apply (some of) the values; keys left out keep their current value. */
  set(values) {
    if (values.cot) this.setCot(values.cot);
    if (values.randomSeed != null) this.randomSeed.input.checked = values.randomSeed;
    if (values.seed != null) this.seed.value = values.seed;
    if (values.customCfg != null) this.customCfg.input.checked = values.customCfg;
    if (values.cfgScale != null) this.cfg.value = values.cfgScale;
    if (values.odeSteps != null) this.ode.value = values.odeSteps;
    if (values.fp8 != null) this.fp8.input.checked = values.fp8 && !this.fp8.input.disabled;
    if (values.abcSampling) this.abcSampling.values = values.abcSampling;
    if (values.semanticSampling) this.semanticSampling.values = values.semanticSampling;
    if (values.abc != null) this.setAbc(values.abc);
    this.syncDisabled();
  }

  setCot(cot) {
    for (const input of this.cotInputs) input.checked = input.value === cot;
  }

  get abc() {
    return this.abcText.value;
  }

  setAbc(text) {
    this.abcText.value = text || "";
    this.abcChanged();
  }

  get semanticMax() {
    return this.semanticSampling.maxTokens;
  }

  set semanticMax(value) {
    this.semanticSampling.maxTokens = value;
  }

  set odeSteps(steps) {
    this.ode.value = steps;
  }

  /** FP8 runs only on NVIDIA compute capability 8.9+; null while the server is unknown. */
  setFp8Supported(supported) {
    const unsupported = supported === false;
    if (unsupported) this.fp8.input.checked = false;
    this.fp8.input.disabled = unsupported;
    this.fp8Note.hidden = !unsupported;
  }

  /** Everything back to the model's defaults; the external score is content and stays. */
  reset() {
    const {abc, ...defaults} = defaultSettings();
    this.set(defaults);
    this.onOdeChange?.(DEFAULT_ODE_STEPS);
    this.onReset?.();
  }
}
