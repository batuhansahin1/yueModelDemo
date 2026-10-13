/**
 * 🎵 Cover: a reference song's melody, chords and structure with new style tags and lyrics.
 *
 * YuE2 cannot take audio as input but follows a given ABC score, so the model server turns the
 * reference into that score (SheetSage2 or server/cover.py) and the score is sent with the job.
 */
import {api, ensureServer} from "./api.js";
import {COVER_COT, COVER_MODES, DEFAULT_ODE_STEPS, FAST_ODE_STEPS, LANGUAGE_CHOICES, LONG_ESTIMATE_MINUTES, TAG_LIMIT,
  TRANSCRIBERS, ValidationError, analyzeParams, buildCoverPayload, clock, coverInfo, coverScore, estimateRange,
  estimateText, isAudioFile, projectData, readProject, semanticBudget, structureProblem, timeline} from "./logic.js";
import {GenerationPanel} from "./output.js";
import {ModelSettings} from "./settings.js";
import {TagInput} from "./tags.js";
import {$, $$, bindSubtabs, busy, downloadFile, formatBytes, h, mdElement, reportError, toast} from "./ui.js";

const INTRO = "MP3 yüklerseniz yeni şarkı onun melodisini, akorlarını ve yapısını takip eder; tarz, enstrümanlar ve "
  + "vokal aşağıdaki etiketlerden gelir. Orijinal sesin tınısı/aranjmanı kopyalanmaz. Sözlü bir şarkıysa "
  + "**✨ Sözleri tanı** sözleri de şarkıdaki yerleriyle birlikte çıkarır.";

function choiceCards(container, name, options, value) {
  container.append(...options.map((option) => h("label", {class: "choice"},
    h("input", {type: "radio", name, value: option.value, checked: option.value === value}),
    h("span", {class: "choice-title"}, option.label),
    h("span", {class: "choice-detail"}, option.detail))));
  return $$(`input[name=${name}]`, container);
}

/** Timeline of the lyrics over the analyzed sections (see logic.timeline). */
function renderTimeline(container, {tone, message, rows}) {
  const time = (value) => h("span", {class: "tl-time"}, value);
  const items = rows.map((row) => {
    if (row.kind === "music") return h("li", {class: "tl-row tl-music"}, time(row.time), h("span", {}, `(${row.name}, müzik)`));
    if (row.kind === "line") return h("li", {class: "tl-row tl-line"}, time(row.time), h("span", {}, row.text));
    if (row.kind === "end") return h("li", {class: "tl-row tl-end"}, time(row.time), h("span", {}, "(son)"));
    return h("li", {class: `tl-row tl-section${row.kind === "orphan" ? " tl-orphan" : ""}`}, time(row.time),
      h("span", {class: "tl-label"}, `[${row.label}]`), row.mark && h("span", {class: "tl-mark"}, `← ${row.mark}`));
  });
  container.replaceChildren(
    message ? mdElement("p", message, {class: `timeline-message tone-${tone}`}) : "",
    items.length ? h("ol", {class: "timeline-rows"}, items) : "");
}

export function createCover(view, {onFinished}) {
  const fileInput = $("#cover-file", view);
  const drop = $("#cover-drop", view);
  const fileCard = $("#cover-file-card", view);
  const fileAudio = $("#cover-file-audio", view);
  const fileInfo = $("#cover-file-info", view);
  const start = $("#cover-start", view);
  const duration = $("#cover-duration", view);
  const bpm = $("#cover-bpm", view);
  const previewButton = $("#cover-preview", view);
  const recognizeButton = $("#cover-recognize", view);
  const info = $("#cover-info", view);
  const language = $("#cover-language", view);
  const lyrics = $("#cover-lyrics", view);
  const timelineBox = $("#cover-timeline", view);
  const sheetsageNote = $("#cover-sheetsage-note", view);
  bindSubtabs($("#cover-card", view));

  // The reference file and the server's id for it; the last analysis {abc, score, lyrics, summary, reference}:
  // "score" is the analyzed full score, "abc" what is sent for the reference melody mode.
  const state = {file: null, uploadId: null, objectUrl: null, analysis: null, sheetsage: null, analyzing: false};

  const transcriberInputs = choiceCards($("#cover-transcribers", view), "cover-transcriber", TRANSCRIBERS, "sheetsage2");
  // Serbest by default: the mode YuE2 recommends when the style changes.
  const modeInputs = choiceCards($("#cover-modes", view), "cover-mode", COVER_MODES, "free");
  const transcriber = () => transcriberInputs.find((input) => input.checked)?.value || "sheetsage2";
  const mode = () => modeInputs.find((input) => input.checked)?.value || "free";
  language.append(...LANGUAGE_CHOICES.map(([value, label]) => h("option", {value}, label)));

  const tags = new TagInput({label: "Tür / stil etiketleri", max: TAG_LIMIT,
    placeholder: "Etiket yazın ya da listeden seçin (ör. pop, female vocal)"});
  $("#cover-tags", view).append(tags.element);

  const showInfo = (lines) => info.replaceChildren(...lines.map((line) => mdElement("p", line)));
  const updateTimeline = () => renderTimeline(timelineBox, timeline(lyrics.value, settings.abc, state.analysis));

  let panel;
  const settings = new ModelSettings({
    onAbcChange: updateTimeline,
    onOdeChange: (steps) => panel.setFast(steps === FAST_ODE_STEPS),
    // A score from the reference keeps the CoT mode of its reference melody mode.
    onReset: () => {
      if (state.analysis && state.analysis.abc.trim() === settings.abc.trim()) settings.setCot(COVER_COT[mode()]);
    },
  });
  $("[data-panel=settings]", view).append(settings.element);

  // --- Reference file ----------------------------------------------------
  function setFile(file) {
    if (file && !isAudioFile(file.name)) {
      toast("MP3, WAV, FLAC veya OGG yükleyin.", "warning");
      return;
    }
    if (state.objectUrl) URL.revokeObjectURL(state.objectUrl);
    state.file = file || null;
    state.uploadId = null;
    state.objectUrl = file ? URL.createObjectURL(file) : null;
    // A new reference invalidates the extracted score.
    state.analysis = null;
    settings.setAbc("");
    showInfo([INTRO]);
    fileCard.hidden = !file;
    drop.hidden = Boolean(file);
    if (file) {
      $("#cover-file-name", view).textContent = file.name;
      fileInfo.textContent = formatBytes(file.size);
      fileAudio.src = state.objectUrl;
    } else {
      fileAudio.removeAttribute("src");
      fileAudio.load();
    }
  }
  fileInput.addEventListener("change", () => {
    setFile(fileInput.files[0]);
    fileInput.value = "";
  });
  fileAudio.addEventListener("loadedmetadata", () => {
    if (state.file && Number.isFinite(fileAudio.duration)) {
      fileInfo.textContent = `${formatBytes(state.file.size)} · ${clock(fileAudio.duration)}`;
    }
  });
  for (const type of ["dragenter", "dragover"]) {
    drop.addEventListener(type, (event) => {
      event.preventDefault();
      drop.classList.add("is-over");
    });
  }
  for (const type of ["dragleave", "drop"]) drop.addEventListener(type, () => drop.classList.remove("is-over"));
  drop.addEventListener("drop", (event) => {
    event.preventDefault();
    if (event.dataTransfer.files[0]) setFile(event.dataTransfer.files[0]);
  });
  $("#cover-file-change", view).addEventListener("click", () => fileInput.click());
  $("#cover-file-remove", view).addEventListener("click", () => setFile(null));

  // New analysis settings leave the extracted score stale (the timeline says so).
  const invalidateScore = () => {
    if (state.file) settings.setAbc("");
  };
  for (const field of [start, duration, bpm]) field.addEventListener("input", invalidateScore);
  const updateSheetsageNote = () => {
    sheetsageNote.hidden = !(state.sheetsage === false && transcriber() === "sheetsage2");
  };
  for (const input of transcriberInputs) {
    input.addEventListener("change", () => {
      invalidateScore();
      updateSheetsageNote();
    });
  }

  // Switch the analyzed score between full and melody-only, unless it was edited by hand.
  for (const input of modeInputs) {
    input.addEventListener("change", () => {
      const current = state.analysis;
      if (!current || !settings.abc.trim()) return;
      if (settings.abc.trim() !== current.abc.trim()) {
        toast("Nota elle değiştirilmiş; mod değişikliği notaya uygulanmadı. Yeniden uygulamak için "
          + "**Melodiyi çıkar**'a basın.", "warning");
        return;
      }
      const score = coverScore(current.score, mode());
      state.analysis = {...current, abc: score, reference: {...current.reference, mode: mode()}};
      settings.setAbc(score);
      settings.setCot(COVER_COT[mode()]);
    });
  }

  // --- Analysis ------------------------------------------------------------
  async function upload() {
    if (!state.uploadId) state.uploadId = (await api.upload(state.file)).upload_id;
    return state.uploadId;
  }

  /** Analysis state of the reference song for the current reference settings. */
  async function analyzeReference({labels = "", recognize = false, languageCode = null} = {}) {
    const referenceMode = mode();
    const method = transcriber();
    // Out-of-range fields are reported before the upload.
    const params = analyzeParams({start: start.value, duration: duration.value, bpm: bpm.value, lyrics: labels,
      recognize, language: languageCode, transcriber: method});
    params.upload_id = await upload();
    const result = await api.analyze(params);
    // Servers that do not report the transcriber used the one that was asked for.
    const summary = {transcriber: method, ...result.summary};
    const reference = {file: state.file.name, start: params.start, duration: params.duration, ...summary,
      mode: referenceMode};
    delete reference.lines;
    return {abc: coverScore(result.abc, referenceMode), score: result.abc, lyrics: result.lyrics, summary, reference};
  }

  function applyAnalysis(result, stats) {
    state.analysis = result;
    settings.setAbc(result.abc);
    settings.setCot(COVER_COT[result.reference.mode]);
    settings.semanticMax = semanticBudget(result.summary, settings.semanticMax);
    showInfo(coverInfo(result.summary, panel.fast, stats));
  }

  function lock(on) {
    state.analyzing = on;
    previewButton.disabled = on;
    recognizeButton.disabled = on;
  }

  async function runAnalysis(button, label, action) {
    if (state.analyzing) return;
    lock(true);
    const done = busy(button, label);
    try {
      await action();
    } catch (error) {
      reportError(error);
    } finally {
      done();
      lock(false);
    }
  }

  previewButton.addEventListener("click", () => {
    if (!state.file) {
      toast("Önce bir referans şarkı yükleyin.", "warning");
      return;
    }
    runAnalysis(previewButton, "Nota çıkarılıyor…", async () => {
      toast("Referans şarkının notası çıkarılıyor...", "info");
      const result = await analyzeReference({labels: lyrics.value});
      applyAnalysis(result, await api.stats());
    });
  });

  recognizeButton.addEventListener("click", () => {
    if (!state.file) {
      toast("Önce bir referans şarkı yükleyin; sözler o şarkıdan tanınır.", "warning");
      return;
    }
    runAnalysis(recognizeButton, "Sözler tanınıyor…", async () => {
      toast("Sözler tanınıyor (vokal ayırma + Whisper); şarkının uzunluğuna göre birkaç dakika sürebilir. "
        + "Aynı şarkı ve ayarlar ikinci kez anında gelir.", "info");
      // Recognized lyrics name the sections themselves, so the box's labels are not sent.
      const result = await analyzeReference({recognize: true,
        languageCode: language.value === "auto" ? null : language.value});
      const status = result.summary.lyrics_status;
      if (status === "ok") {
        lyrics.value = result.lyrics || "";
        toast("Sözler tanındı. Yanlış kelimeleri düzeltebilirsiniz; bölüm etiketlerini silmeyin.", "success");
      } else {
        const begin = Number(start.value) || 0;
        const length = Number(duration.value) || 0;
        const span = length ? `${clock(begin)}-${clock(begin + length)}` : `${clock(begin)}'dan sona kadar`;
        toast(status === "instrumental"
          ? `Seçilen aralıkta (${span}) vokal yok. Vokalli bir bölüm seçin ya da Süre'yi 0 yapın (tüm şarkı).`
          : `Seçilen aralıkta (${span}) vokal var ama kelimeler anlaşılamadı. Sözleri elle yazın.`, "warning");
      }
      applyAnalysis(result, await api.stats());
    });
  });

  lyrics.addEventListener("input", updateTimeline);

  // --- Generation ----------------------------------------------------------
  const form = () => ({tags: tags.values, lyrics: lyrics.value});

  async function generate() {
    if (state.analyzing) throw new ValidationError("Referans analizi sürüyor; bitince tekrar deneyin.");
    buildCoverPayload(form(), settings.values);   // validate the prompt before anything slow
    await ensureServer();
    const stats = await api.stats();
    if (state.file && !settings.abc.trim()) {
      panel.showMessage("⏳ Referans şarkı analiz ediliyor (vokal ayırma + melodi çıkarma)...");
      lock(true);
      try {
        const current = await analyzeReference({labels: lyrics.value});
        applyAnalysis(current, stats);
        const seconds = current.summary.seconds;
        const span = estimateRange(seconds, panel.fast, stats);
        if (span && span[1] > LONG_ESTIMATE_MINUTES) {
          toast(`Uzun sürecek: tahmini ${estimateText(seconds, panel.fast, stats)}. İptal edip Başlangıç/Süre ile `
            + "daha kısa bir bölüm seçebilirsiniz.", "warning");
        }
      } finally {
        lock(false);
      }
    }
    // The score came from this reference (now or via Sözleri tanı / önizle): size the song to it.
    let reference = null;
    const current = state.analysis;
    if (state.file && current && current.abc.trim() === settings.abc.trim()) {
      reference = current.reference;
      settings.semanticMax = semanticBudget(current.summary, settings.semanticMax);
    }
    const problem = structureProblem(lyrics.value, settings.abc);
    if (problem) toast(problem, "warning");
    const payload = buildCoverPayload(form(), settings.values);
    payload.reference = reference;
    await panel.follow(payload, stats);
  }

  panel = new GenerationPanel({
    mode: "cover", title: "Cover", generateLabel: "🎵 Şarkı üret", onGenerate: generate, onFinished,
    onFastChange: (on) => {
      settings.odeSteps = on ? FAST_ODE_STEPS : DEFAULT_ODE_STEPS;
    },
    onUseAbc: (abc) => {
      settings.setAbc(abc);
      toast("ABC, *Model ayarları → Harici ABC nota* alanına kopyalandı.", "success");
    },
  });
  $("#cover-workspace", view).append(panel.element);

  // --- Project load / save (YuE-UI compatible genre_text / lyrics_text) --------
  function applyProject(project) {
    tags.values = project.tags;
    lyrics.value = project.lyrics;
    settings.setCot(project.cot);
    settings.setAbc(project.abc);
  }
  $("#cover-example", view).addEventListener("click", async () => {
    try {
      const response = await fetch("data/example.json");
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      applyProject(readProject(await response.json()));
    } catch (error) {
      toast(`Örnek yüklenemedi: ${error.message}`, "error");
    }
  });
  $("#cover-project-file", view).addEventListener("change", async (event) => {
    const file = event.target.files[0];
    event.target.value = "";
    if (!file) return;
    try {
      applyProject(readProject(JSON.parse(await file.text())));
      toast("Proje yüklendi.", "success");
    } catch {
      toast("Proje dosyası okunamadı; YuE2 proje dosyası (JSON) seçin.", "error");
    }
  });
  $("#cover-project-save", view).addEventListener("click", () => {
    const pad = (n) => String(n).padStart(2, "0");
    const now = new Date();
    const stamp = `${now.getFullYear()}${pad(now.getMonth() + 1)}${pad(now.getDate())}-`
      + `${pad(now.getHours())}${pad(now.getMinutes())}${pad(now.getSeconds())}`;
    downloadFile(`yue2_project_${stamp}.json`,
      JSON.stringify(projectData(tags.values, lyrics.value, settings.values.cot, settings.abc), null, 2));
  });

  showInfo([INTRO]);
  updateTimeline();

  return {
    panel,
    settings,
    setSuggestions: (list) => tags.setSuggestions(list),
    /** SheetSage2 installed on the server (null while unknown). */
    setSheetsageAvailable(available) {
      state.sheetsage = available;
      updateSheetsageNote();
    },
    /** Inputs of a past cover job (history → "Ayarları forma geri yükle"); the reference file stays as it is. */
    restore({form: values, settings: settingValues}) {
      tags.values = values.tags;
      lyrics.value = values.lyrics;
      if (values.coverMode) for (const input of modeInputs) input.checked = input.value === values.coverMode;
      if (values.transcriber) for (const input of transcriberInputs) input.checked = input.value === values.transcriber;
      updateSheetsageNote();
      settings.set(settingValues);
      panel.setFast(Number(settingValues.odeSteps) === FAST_ODE_STEPS);
      updateTimeline();
    },
  };
}
