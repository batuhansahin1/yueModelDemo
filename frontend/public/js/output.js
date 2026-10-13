/**
 * "Üretim" panel: submits a job, follows it until it ends and shows the song, the score and the inputs.
 *
 * renderJob() is shared with the history view.
 */
import {ApiError, api} from "./api.js";
import {FAST_ODE_STEPS, FINAL, progressInfo} from "./logic.js";
import {busy, copyText, h, mdElement, reportError, sleep, storage, toast} from "./ui.js";

// The four stages of a song (Gradio shows them as text); "loading" and "saving" are short.
const STEPS = [["abc", "Nota planı"], ["semantic", "Şarkı token'ları"], ["synthesize", "Ses sentezi"], ["decode", "Ses çözme"]];
const ORDER = ["queued", "loading", "abc", "semantic", "synthesize", "decode", "saving", "done"];
const STATUS_TONES = {queued: "neutral", running: "active", complete: "ok", failed: "error", cancelled: "neutral"};

function stepStates(job) {
  const req = job.request || {};
  const at = ORDER.indexOf(job.stage);
  // With cot=off or an external score the plan stage passes without tokens.
  const skipped = (key) => key === "abc" && (req.cot === "off" || req.abc) && !(job.tokens || {}).abc;
  return STEPS.map(([key]) => {
    const index = ORDER.indexOf(key);
    if (job.status === "complete" || at > index) return skipped(key) ? "skipped" : "done";
    if (at === index) return job.status === "failed" ? "failed" : job.status === "cancelled" ? "cancelled" : "active";
    return "pending";
  });
}

function meter(token) {
  const fill = h("div", {class: "meter-fill"});
  // Width through the CSSOM: the page's Content-Security-Policy forbids inline style attributes.
  fill.style.width = `${Math.min(100, (token.value / token.max) * 100)}%`;
  return h("div", {class: "meter"},
    h("div", {class: "meter-head"}, h("span", {}, token.label), h("span", {class: "mono"}, `${token.value} / ${token.max}`)),
    h("div", {class: "meter-track"}, fill));
}

/** Status, stages, settings and token counts of a job. */
export function renderJob(job, stats) {
  const info = progressInfo(job, stats);
  const states = stepStates(job);
  const stepper = h("ol", {class: "stepper"}, STEPS.map(([, label], i) => h("li", {class: `step is-${states[i]}`},
    h("span", {class: "step-dot", "aria-hidden": "true"}), h("span", {class: "step-label"}, label),
    states[i] === "skipped" && h("span", {class: "step-note"}, "atlandı"))));
  // Synthesis and decoding report no token counts; a moving bar shows they are running.
  const indeterminate = !FINAL.has(job.status) && ["loading", "synthesize", "decode", "saving"].includes(job.stage);
  return h("div", {class: "job"},
    h("div", {class: "job-head"},
      h("span", {class: `badge badge-${STATUS_TONES[job.status] || "neutral"}`}, info.status),
      h("span", {class: "job-stage"}, info.stage)),
    stepper,
    info.tokens.map(meter),
    indeterminate && h("div", {class: "meter-track is-indeterminate", "aria-hidden": "true"}, h("div", {class: "meter-fill"})),
    h("dl", {class: "facts"},
      h("div", {class: "fact-wide"}, h("dt", {}, "İş"), h("dd", {class: "mono"}, info.id)),
      h("div", {}, h("dt", {}, "Seed"), h("dd", {class: "mono"}, String(info.seed ?? "—"))),
      h("div", {}, h("dt", {}, "Süre"), h("dd", {}, `${info.elapsed} sn`)),
      info.audioSeconds && h("div", {}, h("dt", {}, "Ses"), h("dd", {}, `${info.audioSeconds.toFixed(1)} sn`))),
    h("p", {class: "job-settings"}, h("strong", {}, "Ayarlar: "), info.settings),
    info.estimate && h("p", {class: "job-note"}, `⏱ Tahmini toplam: ${info.estimate}`),
    info.error && h("p", {class: "callout callout-error"}, `⚠️ ${info.error}`),
    info.truncated.length > 0 && h("p", {class: "callout callout-warn"},
      `⚠️ Token sınırına ulaşıldı (${info.truncated.join(", ")}); max token değerini artırmayı deneyin.`));
}

/** File name for a job's audio; old FLAC results are served as MP3 by the model server. */
export function audioFileName(job) {
  const suffix = (job.result?.audio || "audio.mp3").match(/\.[a-z0-9]+$/i)?.[0] || ".mp3";
  return `yue2-${job.id}${suffix.toLowerCase() === ".flac" ? ".mp3" : suffix}`;
}

/** Audio player with a download link for a finished job. */
export function renderPlayer(job) {
  const url = api.audioUrl(job.id);
  return h("div", {class: "player"},
    h("audio", {controls: true, preload: "metadata", src: url}),
    h("a", {class: "btn btn-sm", href: url, download: audioFileName(job)}, "⬇ İndir"));
}

export function promptText(req) {
  return `[Tags]\n${req.style || ""}\n\n[Lyrics]\n${req.lyrics || ""}`;
}

export class GenerationPanel {
  /**
   * onGenerate(): builds and runs the mode's generation (calls follow()); onUseAbc(abc): copy a generated
   * score into the mode's settings; onFastChange(on): ⚡ Hızlı sentez toggled; onFinished(job).
   */
  constructor({mode, title, generateLabel, onGenerate, onUseAbc, onFastChange, onFinished}) {
    this.mode = mode;
    this.title = title;
    this.storageKey = `yue2.job.${mode}`;
    this.onGenerate = onGenerate;
    this.onUseAbc = onUseAbc;
    this.onFinished = onFinished;
    this.jobId = null;
    this.running = false;

    this.generateButton = h("button", {type: "button", class: "btn btn-primary btn-generate", onclick: () => this.generate()},
      generateLabel);
    this.cancelButton = h("button", {type: "button", class: "btn btn-danger", disabled: true, onclick: () => this.cancel()},
      "İptal");
    this.fastInput = h("input", {type: "checkbox", role: "switch", onchange: () => onFastChange?.(this.fastInput.checked)});
    this.status = h("div", {class: "status", "aria-live": "polite"});
    this.playerSlot = h("div", {class: "player-slot"});
    this.promptView = h("pre", {class: "code-block"});
    this.promptGroup = h("details", {class: "group", hidden: true}, h("summary", {}, h("span", {}, "Modele giden girdi")),
      this.promptView);
    this.abcView = h("pre", {class: "code-block"});
    this.abcGroup = h("details", {class: "group", hidden: true},
      h("summary", {}, h("span", {}, "Üretilen ABC nota planı")),
      this.abcView,
      h("div", {class: "group-actions"},
        h("button", {type: "button", class: "btn btn-sm", onclick: () => this.onUseAbc?.(this.abcView.textContent)},
          "Bu ABC'yi harici nota olarak kullan"),
        h("button", {type: "button", class: "btn btn-sm btn-ghost", onclick: () => copyText(this.abcView.textContent)},
          "Kopyala")));
    this.element = h("aside", {class: "panel output"},
      h("div", {class: "output-head"}, h("h2", {}, "Üretim"), h("span", {class: "kbd-hint"}, "⌘/Ctrl + Enter")),
      h("div", {class: "output-actions"}, this.generateButton, this.cancelButton),
      h("label", {class: "switch"}, this.fastInput, h("span", {class: "switch-track", "aria-hidden": "true"}),
        h("span", {}, "⚡ Hızlı sentez")),
      h("p", {class: "hint"}, `Ses sentezi 32 yerine ${FAST_ODE_STEPS} adım: toplam süre kısalır, kalite biraz düşebilir.`),
      this.status, this.playerSlot, this.promptGroup, this.abcGroup);
    this.showEmpty();
  }

  get fast() {
    return this.fastInput.checked;
  }

  setFast(on) {
    this.fastInput.checked = on;
  }

  showEmpty() {
    this.status.replaceChildren(h("p", {class: "empty"}, "Henüz üretim yapılmadı."));
  }

  /** A step before the job exists, e.g. the reference analysis of a cover. */
  showMessage(text) {
    this.status.replaceChildren(h("div", {class: "job"}, h("p", {class: "job-wait"}, h("span", {class: "spinner"}),
      mdElement("span", text))));
  }

  showPrompt(req) {
    this.promptView.textContent = promptText(req);
    this.promptGroup.hidden = false;
  }

  clearResult() {
    this.playerSlot.replaceChildren();
    this.abcGroup.hidden = true;
    this.abcView.textContent = "";
  }

  async generate() {
    if (this.running) return;
    this.running = true;
    const done = busy(this.generateButton, "Çalışıyor…");
    try {
      await this.onGenerate();
    } catch (error) {
      reportError(error);
      if (!this.jobId && this.status.querySelector(".job-wait")) this.showEmpty();
    } finally {
      done();
      this.running = false;
    }
  }

  /** Submit a generation and follow it until it ends. */
  async follow(payload, stats) {
    this.clearResult();
    this.showPrompt(payload);
    const job = await api.submit(payload);
    return this.track(job, stats);
  }

  /** Poll a job every second until it is complete, failed or cancelled. */
  async track(job, stats) {
    this.jobId = job.id;
    storage.set(this.storageKey, job.id);
    this.cancelButton.disabled = false;
    this.render(job, stats);
    let failures = 0;
    try {
      while (!FINAL.has(job.status)) {
        await sleep(1000);
        try {
          job = await api.job(job.id);
          failures = 0;
        } catch (error) {
          if (!(error.unreachable || error.status === 502 || error.status === 504)) throw error;
          failures += 1;
          if (failures > 10) throw new ApiError("Model sunucusuyla bağlantı koptu.", {unreachable: true});
          continue;
        }
        this.render(job, stats);
      }
    } finally {
      this.jobId = null;
      this.cancelButton.disabled = true;
    }
    this.finish(job, true);
    this.onFinished?.(job);
    return job;
  }

  render(job, stats) {
    this.status.replaceChildren(renderJob(job, stats));
  }

  /** Result of a finished job: the song and its score. */
  finish(job, announce = false) {
    this.clearResult();
    const abc = job.result?.abc;
    if (abc) {
      this.abcView.textContent = abc;
      this.abcGroup.hidden = false;
    }
    if (job.status === "complete") this.playerSlot.replaceChildren(renderPlayer(job));
    if (!announce) return;
    if (job.status === "complete") toast(`**${this.title}:** şarkı hazır 🎵`, "success");
    else if (job.status === "failed") toast(`**${this.title}:** üretim başarısız. ${job.error || ""}`, "error");
    else toast(`**${this.title}:** üretim iptal edildi.`, "info");
  }

  /** After a page reload: show this mode's last job and keep following it while it runs. */
  async resume(id) {
    let job;
    try {
      job = await api.job(id);
    } catch (error) {
      if (error.status === 404) storage.remove(this.storageKey);
      return;
    }
    this.showPrompt(job.request);
    if (FINAL.has(job.status)) {
      this.render(job, null);
      this.finish(job);
      return;
    }
    if (this.running) return;
    this.running = true;
    const done = busy(this.generateButton, "Çalışıyor…");
    try {
      await this.track(job, await api.stats());
    } catch (error) {
      reportError(error);
    } finally {
      done();
      this.running = false;
    }
  }

  async cancel() {
    if (!this.jobId) {
      toast("Çalışan bir üretim yok.", "info");
      return;
    }
    try {
      await api.cancel(this.jobId);
      toast("İptal isteği gönderildi.", "info");
    } catch (error) {
      reportError(error);
    }
  }
}
