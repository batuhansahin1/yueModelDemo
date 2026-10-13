/**
 * Pure logic of the YuE2 web UI: request building, validation, progress, cover and timeline texts.
 *
 * Ported from source/yue2_ui.py so the web UI and the Gradio UI send the model server the same
 * requests. No DOM access here; tests/logic.test.mjs runs this module under Node.
 */

// Defaults mirror yue2.protocol.GenerationConfig (models/YuE2-3B/yue2_generation_config.json).
export const ABC_DEFAULTS = Object.freeze({
  temperature: 0.7, top_p: 0.9, top_k: 30, repetition_penalty: 1.005, penalty_window: 100,
  min_tokens: 32, max_tokens: 4096,
});
export const SEMANTIC_DEFAULTS = Object.freeze({
  temperature: 1.0, top_p: 0.95, top_k: 100, repetition_penalty: 1.2, penalty_window: 50,
  min_tokens: 200, max_tokens: 9000,
});
export const SAMPLING_KEYS = ["temperature", "top_p", "top_k", "repetition_penalty", "penalty_window",
  "min_tokens", "max_tokens"];
// Controls of one sampling stage, in SAMPLING_KEYS order. Sliders unless `number` is set.
export const SAMPLING_CONTROLS = [
  {key: "temperature", label: "Temperature", min: 0, max: 5, step: 0.05,
    info: "Düşük: daha tutarlı, tekrara yatkın. Yüksek: daha yaratıcı, dağınık. 0 = hep en olası token."},
  {key: "top_p", label: "Top-p", min: 0.05, max: 1, step: 0.01,
    info: "Olasılıkları toplamı bu orana ulaşan tokenlar arasından seçer. Düşük = daha güvenli."},
  {key: "top_k", label: "Top-k", min: 1, max: 500, step: 1, integer: true,
    info: "Her adımda en olası bu kadar token arasından seçer."},
  {key: "repetition_penalty", label: "Tekrar cezası", min: 0.5, max: 3, step: 0.005,
    info: "1 = ceza yok. Büyüdükçe son tokenları tekrarlamak zorlaşır (döngüleri kırar)."},
  {key: "penalty_window", label: "Tekrar cezası penceresi", min: 1, max: 100, step: 1, integer: true,
    info: "Cezanın baktığı son token sayısı."},
  {key: "min_tokens", label: "Min token", min: 0, step: 1, integer: true, number: true,
    info: "Bu kadar token üretilmeden bitiş tokenına izin verilmez."},
  {key: "max_tokens", label: "Max token", min: 1, step: 1, integer: true, number: true,
    info: "Üst sınır. Şarkıda saniye başına 25 token; dolarsa şarkı kesilir."},
];
const FLOAT_KEYS = new Set(["temperature", "top_p", "repetition_penalty"]);

export const DEFAULT_SEED = 831001;
// yue2 accepts seeds in [0, 2**63); larger integers lose precision in JavaScript.
export const MAX_SEED = Number.MAX_SAFE_INTEGER;
export const DEFAULT_ODE_STEPS = 32;
export const FAST_ODE_STEPS = 16;
export const ODE_RANGE = [4, 64];
export const CFG_RANGE = [0, 20];

// Reference melody modes, as in YuE2's cover guide: full score (cot=full) or melody-only score (cot=melody).
export const COVER_MODES = [
  {value: "close", label: "Yakın", detail: "Melodi + akorlar + yapı. Akorlu nota, CoT 'Tam'; orijinale daha sadık."},
  {value: "free", label: "Serbest", detail: "Sadece melodi; akorlar silinir, CoT 'Melodi'. Eşlik yeni stile göre kendi akorlarını seçer (stil değiştirirken YuE2'nin önerdiği mod)."},
];
export const COVER_COT = {close: "full", free: "melody"};
// Who writes the reference song's score: SheetSage2 (YuE2's own transcriber) or server/cover.py.
export const TRANSCRIBERS = [
  {value: "sheetsage2", label: "SheetSage2", detail: "YuE2 ekibinin cover için kullandığı model; melodi, akor, bölüm ve enstrüman melodisini çıkarır."},
  {value: "basic", label: "Basit analiz", detail: "Demucs + librosa. Enstrümantal kayıtların melodisini vokale taşır ve elle BPM kabul eder."},
];
export const TRANSCRIBER_NAMES = {sheetsage2: "SheetSage2", basic: "Basit analiz"};

export const COT_CHOICES = [
  {value: "full", label: "Tam", detail: "Akorlu ABC nota planı → şarkı"},
  {value: "melody", label: "Melodi", detail: "Akorsuz ABC planı → şarkı"},
  {value: "off", label: "Kapalı", detail: "Nota planı yok, doğrudan şarkı"},
];
export const STAGE_LABELS = {
  queued: "Sırada bekliyor",
  loading: "Model yükleniyor (ilk çalıştırmada ağırlıklar indirilebilir)",
  abc: "Nota planı (ABC) üretiliyor",
  semantic: "Şarkı token'ları üretiliyor",
  synthesize: "Ses sentezleniyor",
  decode: "Ses çözülüyor (VAE)",
  saving: "Kaydediliyor",
  done: "Bitti",
  cancelled: "İptal edildi",
};
export const STATUS_LABELS = {queued: "Sırada", running: "Çalışıyor", complete: "Tamamlandı",
  failed: "Hata", cancelled: "İptal edildi"};
export const FINAL = new Set(["complete", "failed", "cancelled"]);
// Covers came out up to ~1.6x the reference length.
export const COVER_LENGTH_FACTOR = [1.0, 1.6];
// Above this estimate the UI suggests generating a shorter part first.
export const LONG_ESTIMATE_MINUTES = 20;
// YuE2 emits ~25 semantic tokens per second of audio.
export const TOKENS_PER_AUDIO_SECOND = 25;
// Whisper language codes for lyric recognition.
export const LANGUAGE_CHOICES = [["auto", "Otomatik algıla"], ["tr", "Türkçe"], ["en", "İngilizce"],
  ["de", "Almanca"], ["fr", "Fransızca"], ["es", "İspanyolca"], ["it", "İtalyanca"], ["pt", "Portekizce"],
  ["ru", "Rusça"], ["ar", "Arapça"], ["fa", "Farsça"], ["az", "Azerice"], ["ja", "Japonca"],
  ["ko", "Korece"], ["zh", "Çince"]];
// Score sections the model plays without singing (cover.lyric_labels skips the same names).
export const NON_VOCAL = new Set(["intro", "outro", "inst", "instrumental", "end", "silence", "interlude",
  "fade-out", "solo"]);
// Text to Song: style chips (tag sent to the model, label), voice choices and their tags, field limits.
export const T2S_STYLES = [
  {value: "pop", label: "Pop"}, {value: "chill", label: "Sakin"}, {value: "energetic", label: "Enerjik"},
  {value: "emotional", label: "Duygusal"}, {value: "rock", label: "Rock"},
  {value: "electronic", label: "Elektronik"}, {value: "hip-hop", label: "Hip-hop"},
  {value: "lo-fi", label: "Lo-Fi"}, {value: "soundtrack", label: "Film müziği"}, {value: "jazz", label: "Caz"},
  {value: "classical", label: "Klasik"}, {value: "ambient", label: "Ambient"},
];
export const T2S_VOICES = [{value: "default", label: "Varsayılan"}, {value: "female", label: "Kadın"},
  {value: "male", label: "Erkek"}];
export const VOICE_TAGS = {default: null, female: "female vocal", male: "male vocal"};
export const DESCRIPTION_LIMIT = 3000;
export const LYRICS_LIMIT = 5000;
export const TAG_LIMIT = 50;
export const AUDIO_EXTENSIONS = [".mp3", ".wav", ".flac", ".ogg"];
export const TIMELINE_HINT = ("Referans şarkı yükleyip **✨ Sözleri tanı**'ya basın: sözler şarkıda söylendikleri "
  + "bölümlere yerleşir ve her satırın zamanı burada görünür.");

/** A form value the user has to fix; shown as a message, not as a crash. */
export class ValidationError extends Error {
  constructor(message) {
    super(message);
    this.name = "ValidationError";
  }
}

/** Lines of a text like Python's str.splitlines(): no trailing empty line. */
export function splitLines(text) {
  const lines = String(text ?? "").split(/\r\n|[\n\r\v\f\x1c\x1d\x1e\x85]/);
  if (lines[lines.length - 1] === "") lines.pop();
  return lines;
}

/** Deduplicated, sorted tag list from YuE-UI's top_200_tags.json. */
export function processTags(data) {
  const unique = new Map();
  for (const category of ["genre", "timbre", "gender", "mood", "instrument"]) {
    for (const raw of data?.[category] || []) {
      const item = String(raw).trim();
      if (item && !unique.has(item.toLowerCase())) unique.set(item.toLowerCase(), item);
    }
  }
  return [...unique.entries()].sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0)).map(([, item]) => item);
}

export function splitTags(style) {
  return String(style || "").split(",").map((tag) => tag.trim()).filter(Boolean);
}

/** Python's round() and "{:.0f}": halves go to the even neighbour (6.5 -> 6), unlike Math.round. */
export function pyRound(value) {
  const rounded = Math.round(value);
  return Math.abs(value % 1) === 0.5 && rounded % 2 !== 0 ? rounded - 1 : rounded;
}

export function clock(seconds) {
  const s = Math.max(0, Number(seconds) || 0);
  return `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
}

/** A float setting as Python prints it (1.0, not 1), so job summaries read the same in both UIs. */
function pyNumber(key, value) {
  return FLOAT_KEYS.has(key) && Number.isInteger(value) ? value.toFixed(1) : String(value);
}

/** Sampling values over their defaults; missing or null values keep the default. */
export function mergeSampling(defaults, values) {
  const merged = {...defaults};
  for (const key of SAMPLING_KEYS) {
    if (values && values[key] != null) merged[key] = values[key];
  }
  return merged;
}

export function defaultSettings() {
  return {
    cot: "full", randomSeed: true, seed: DEFAULT_SEED, customCfg: false, cfgScale: 1.0,
    odeSteps: DEFAULT_ODE_STEPS, fp8: false, abc: "",
    abcSampling: {...ABC_DEFAULTS}, semanticSampling: {...SEMANTIC_DEFAULTS},
  };
}

/** One-line summary of a job's model settings, for comparing runs. */
export function settingsText(req) {
  const sampling = (name, values, defaults) => {
    // Temperature always; the other knobs only when changed (max_tokens is sized automatically for covers).
    const merged = mergeSampling(defaults, values);
    const changed = SAMPLING_KEYS
      .filter((key) => key !== "temperature" && key !== "max_tokens" && merged[key] !== defaults[key])
      .map((key) => `${key}=${pyNumber(key, merged[key])}`);
    return `${name} T=${pyNumber("temperature", merged.temperature)}` + (changed.length ? ` (${changed.join(", ")})` : "");
  };
  const reference = req.reference || {};
  const parts = [
    `CoT ${req.cot}`,
    `CFG ${req.cfg_scale != null ? pyNumber("temperature", req.cfg_scale) : "varsayılan"}`,
    sampling("ABC", req.abc_sampling, ABC_DEFAULTS),
    sampling("Şarkı", req.semantic_sampling, SEMANTIC_DEFAULTS),
    `ODE ${req.ode_steps || DEFAULT_ODE_STEPS}`,
  ];
  if (req.quantization === "fp8") parts.push("FP8");
  if (reference.mode) {
    const transcriber = reference.transcriber || "basic";
    parts.push(`cover ${{close: "Yakın", free: "Serbest"}[reference.mode] || reference.mode}`
      + ` (${TRANSCRIBER_NAMES[transcriber] || transcriber})`);
  }
  return parts.join(" · ");
}

/** (low, high) minutes for a cover of this length on the server's device, or null before it was measured. */
export function estimateRange(referenceSeconds, fast, stats) {
  stats = stats || {};
  // Fall back to the other synthesis mode's speed before this one has been measured.
  const rate = stats[fast ? "fast_rate" : "rate"] || stats.rate || stats.fast_rate;
  return rate ? COVER_LENGTH_FACTOR.map((factor) => referenceSeconds * factor * rate / 60) : null;
}

export function estimateText(referenceSeconds, fast, stats) {
  const span = estimateRange(referenceSeconds, fast, stats);
  if (!span) return "ilk üretimden sonra bu cihaz için hesaplanacak";
  const [low, high] = span.map((minutes) => Math.max(1, pyRound(minutes)));
  return `${low === high ? `~${high}` : `${low}-${high}`} dk (${stats.device})`;
}

export function maxTokens(req) {
  return [req?.abc_sampling?.max_tokens || ABC_DEFAULTS.max_tokens,
    req?.semantic_sampling?.max_tokens || SEMANTIC_DEFAULTS.max_tokens];
}

/** Everything the progress view shows about a job (Gradio: progress_markdown). */
export function progressInfo(job, stats) {
  const req = job.request || {};
  const [abcMax, semanticMax] = maxTokens(req);
  const reference = req.reference;
  const steps = req.ode_steps;
  const fast = steps === FAST_ODE_STEPS;
  const info = {
    status: STATUS_LABELS[job.status] || job.status,
    stage: STAGE_LABELS[job.stage] || job.stage,
    id: job.id,
    seed: req.seed,
    elapsed: pyRound(job.elapsed || 0),
    settings: settingsText(req),
    estimate: null,
    tokens: [],
    error: job.error || null,
    audioSeconds: null,
    truncated: [],
  };
  // Speeds are measured only for the default and fast synthesis settings.
  if (reference?.seconds && !FINAL.has(job.status) && [null, undefined, DEFAULT_ODE_STEPS, FAST_ODE_STEPS].includes(steps)
      && estimateRange(reference.seconds, fast, stats)) {
    info.estimate = estimateText(reference.seconds, fast, stats);
  }
  const tokens = job.tokens || {};
  if (tokens.abc) info.tokens.push({key: "abc", label: "ABC token", value: tokens.abc, max: abcMax});
  if (tokens.semantic) info.tokens.push({key: "semantic", label: "Şarkı token", value: tokens.semantic, max: semanticMax});
  const result = job.result || {};
  if (result.audio_seconds) info.audioSeconds = result.audio_seconds;
  info.truncated = Object.entries(result.truncated || {}).filter(([, value]) => value).map(([key]) => key);
  return info;
}

/** Lines (inline markdown) describing a reference analysis (Gradio: cover_markdown). */
export function coverInfo(summary, fast = false, stats = null) {
  const sections = summary.sections || [];
  const sung = sections.filter((section) => !NON_VOCAL.has(section.name)).length;
  const source = {
    vocals: "vokal",
    instrument: "**enstrüman** (kayıtta vokal bulunamadı; sözleriniz bu melodiyle söylenecek)",
    none: "⚠️ **vokal melodisi bulunamadı**; notada yalnızca enstrüman melodisi var, yeni şarkıda vokal olmayabilir. "
      + "Enstrümantal bir kayıtsa *Basit analiz*'i deneyin (enstrüman melodisini vokale taşır).",
  }[summary.melody_source || "vocals"];
  const transcriber = summary.transcriber || "basic";
  const lines = [
    `✅ **Melodi çıkarıldı** (${TRANSCRIBER_NAMES[transcriber] || transcriber}) · Tempo **${pyRound(summary.tempo)} BPM** · `
      + `Ton **${summary.key}** · ${summary.bars} ölçü (${pyRound(summary.seconds)} sn) · `
      + `melodili kısım %${pyRound(summary.vocal_ratio * 100)}`,
    `Melodi kaynağı: ${source}`,
    `Bölümler: ${sections.length} (${sung} şarkılı), zamanları sözlerin altındaki zaman çizelgesinde`,
    `⏱ Tahmini üretim süresi: **${estimateText(summary.seconds, fast, stats)}**`,
  ];
  const status = summary.lyrics_status || "off";
  if (status === "ok") {
    const model = summary.asr_model ? ` · model \`${summary.asr_model}\`` : "";
    const placement = transcriber === "sheetsage2" ? "sözler SheetSage2'nin bölümlerine yerleştirildi"
      : "bölümler sözlere göre belirlendi";
    lines.splice(1, 0, `✅ **Sözler tanındı** · ${(summary.lines || []).length} satır · dil \`${summary.language}\``
      + `${model} · ${placement}`);
  } else if (status === "none") {
    lines.splice(1, 0, "⚠️ Vokal var ama kelimeler anlaşılamadı. Sözleri elle yazıp **Melodiyi çıkar**'a basın; "
      + "bölümler sözlerinizdeki etiketlere göre adlandırılır.");
  } else if (status === "instrumental") {
    lines.splice(1, 0, "⚠️ Seçilen aralıkta vokal yok, tanınacak söz bulunamadı. Vokalli bir bölüm seçin "
      + "(Başlangıç/Süre) ya da sözleri elle yazın.");
  }
  if (summary.vocal_ratio < 0.05) lines.push("⚠️ Neredeyse hiç melodi bulunamadı; sonuç sadece akorları takip eder.");
  const span = estimateRange(summary.seconds, fast, stats);
  if (span && span[1] > LONG_ESTIMATE_MINUTES) {
    lines.push("⚠️ Üretim uzun sürecek. Önce kısa bir deneme için *Başlangıç/Süre* ile bir bölüm seçebilirsiniz.");
  }
  lines.push("Nota, *Model ayarları → Harici ABC nota* alanında; isterseniz düzenleyebilirsiniz. "
    + (transcriber === "basic" ? "Tempo yanlışsa BPM girip yeniden çıkarın."
      : "Tempo yanlışsa *Basit analiz*'e geçip BPM girebilirsiniz (SheetSage2 vuruşları kendisi bulur)."));
  return lines;
}

/** [[label, lines]] of the lyrics' `[section]` blocks; lines before any label get label "". */
export function lyricBlocks(lyrics) {
  const blocks = [];
  for (const raw of splitLines(lyrics)) {
    const line = raw.trim();
    const match = /^\[([^\]]+)\]$/.exec(line);
    if (match) {
      blocks.push([match[1].trim().toLowerCase(), []]);
    } else if (line) {
      if (!blocks.length) blocks.push(["", []]);
      blocks[blocks.length - 1][1].push(line);
    }
  }
  return blocks;
}

/** Section names of an ABC score, from its `% name` comment lines. */
export function scoreSections(abc) {
  return [...String(abc || "").matchAll(/^%(?!%)\s*(.+?)\s*$/gm)].map((match) => match[1].toLowerCase());
}

/**
 * Why the lyrics would not line up with the score's sung sections, or null.
 *
 * YuE2 sings the n-th `[section]` block of the lyrics over the n-th sung `% section`
 * of the score, so both must list the same sections in the same order.
 */
export function structureProblem(lyrics, abc) {
  const sung = scoreSections(abc).filter((name) => !NON_VOCAL.has(name));
  const blocks = lyricBlocks(lyrics).map(([label]) => label).filter((label) => !NON_VOCAL.has(label));
  if (!sung.length || !blocks.length || (sung.length === blocks.length && sung.every((name, i) => name === blocks[i]))) {
    return null;
  }
  if (blocks.includes("")) {
    return "Sözlerin başında bölüm etiketi olmayan satırlar var; ilk satırın üstüne `[verse]` gibi bir etiket ekleyin.";
  }
  const common = Math.min(sung.length, blocks.length);
  let first = 0;
  while (first < common && sung[first] === blocks[first]) first += 1;
  const detail = first < common
    ? `${first + 1}. şarkılı bölüm notada \`[${sung[first]}]\`, sözlerde \`[${blocks[first]}]\``
    : `notada ${sung.length}, sözlerde ${blocks.length} şarkılı bölüm var`;
  return `Sözlerin bölümleri notayla eşleşmiyor (${detail}). Model sözleri bu yüzden başka yerlere kaydırabilir.`;
}

/**
 * When each lyric line will be sung: the analyzed sections with times, filled with the current lyrics.
 *
 * Returns {tone: hint|info|ok|warn, message (inline markdown), rows}; rows are
 * {kind: music|section|line|orphan|end, time, ...} (Gradio: timeline_markdown).
 */
export function timeline(lyrics, abc, analysis) {
  if (!String(abc || "").trim()) {
    return analysis
      ? {tone: "warn", rows: [], message: "⚠️ Referans ayarları değişti; zaman çizelgesi için **Sözleri tanı** ya da "
        + "**Melodiyi çıkar**'a yeniden basın."}
      : {tone: "hint", rows: [], message: TIMELINE_HINT};
  }
  const problem = structureProblem(lyrics, abc);
  const head = problem ? `⚠️ ${problem}` : "";
  if (!analysis || analysis.abc.trim() !== abc.trim()) {
    // Score pasted or restored from history: no times, only the structure can be checked.
    return {tone: problem ? "warn" : "info", rows: [],
      message: head || "Nota bu referans analizinden gelmiyor; zaman çizelgesi gösterilemiyor."};
  }
  const summary = analysis.summary;
  if (!summary.sections?.length) return {tone: problem ? "warn" : "info", rows: [], message: head};
  const blocks = lyricBlocks(lyrics).filter(([label]) => !NON_VOCAL.has(label));
  const recognized = new Map();
  for (const line of summary.lines || []) {
    if (!recognized.has(line.block)) recognized.set(line.block, []);
    recognized.get(line.block).push(line);
  }
  const rows = [];
  let k = 0;
  for (const section of summary.sections) {
    const time = clock(section.start);
    if (NON_VOCAL.has(section.name)) {
      rows.push({kind: "music", time, name: section.name});
      continue;
    }
    const [label, texts] = k < blocks.length ? blocks[k] : ["", []];
    // Line times stay valid while a block keeps its recognized number of lines.
    const times = recognized.get(k) || [];
    const mark = label !== section.name ? `notada [${section.name}]`
      : times.length && texts.length && times.length !== texts.length ? "satır sayısı değişti, satır zamanları yok" : "";
    rows.push({kind: "section", time, label: label || "?", mark});
    texts.forEach((text, i) => {
      rows.push({kind: "line", time: times.length === texts.length ? clock(times[i].start) : "", text});
    });
    k += 1;
  }
  for (const [label, texts] of blocks.slice(k)) {
    rows.push({kind: "orphan", time: "-", label, mark: "notada karşılığı yok"});
    for (const text of texts) rows.push({kind: "line", time: "", text});
  }
  rows.push({kind: "end", time: clock(summary.sections[summary.sections.length - 1].end)});
  const message = head || (blocks.length
    ? `✅ Sözler notayla eşleşiyor: ${k} şarkılı bölüm. Kelimeleri düzeltebilirsiniz; bölüm etiketlerini silmeyin `
      + "ve sırasını değiştirmeyin."
    : "Notadaki bölümler aşağıda. Sözleri bu sırayla `[bölüm]` etiketleriyle yazın.");
  return {tone: problem ? "warn" : blocks.length ? "ok" : "info", rows, message};
}

/** Melody-only score: chord symbols ("Gm") removed from the music lines; headers and comments are kept. */
export function stripChords(abc) {
  return splitLines(abc).map((line) => (/^\s*(%|[A-Za-z]:)/.test(line) ? line : line.replace(/"[^"]*"/g, "")))
    .join("\n") + "\n";
}

/** The analyzed full score as sent for the chosen reference melody mode. */
export function coverScore(abc, mode) {
  return mode === "free" ? stripChords(abc) : abc;
}

/** Song token limit that fits a cover as long as the reference (the default cuts long songs). */
export function semanticBudget(summary, current) {
  return Math.min(20000, Math.max(Math.trunc(Number(current)),
    Math.trunc(summary.seconds * TOKENS_PER_AUDIO_SECOND * 1.3)));
}

/** The model's [Tags] text from the Text to Song form: styles, voice, then the free description. */
export function t2sStyle(description, styles, customStyle, voice) {
  const parts = [...(styles || []), String(customStyle || "").trim(), VOICE_TAGS[voice],
    String(description || "").trim()];
  return parts.filter(Boolean).join(", ");
}

/** Lyrics with `[Verse]`-style labels lowercased, as in YuE's own examples. */
export function normalizeLyrics(lyrics) {
  return String(lyrics || "").trim()
    .replace(/^[ \t]*\[([^\]]+)\][ \t]*$/gm, (_, label) => `[${label.trim().toLowerCase()}]`);
}

export function hasSectionLabels(lyrics) {
  return /^\s*\[[^\]]+\]\s*$/m.test(String(lyrics || ""));
}

function toNumber(value) {
  return typeof value === "string" && value.trim() === "" ? NaN : Number(value);
}

/** Sampling overrides of one stage, validated against the ranges of its controls. */
export function samplingDict(name, values) {
  const sampling = {};
  for (const control of SAMPLING_CONTROLS) {
    const value = toNumber(values?.[control.key]);
    const label = `${name} örneklemesinde ${control.label}`;
    if (!Number.isFinite(value)) throw new ValidationError(`${label} bir sayı olmalı.`);
    if (control.integer && !Number.isInteger(value)) throw new ValidationError(`${label} tam sayı olmalı.`);
    if (value < control.min || (control.max != null && value > control.max)) {
      throw new ValidationError(control.max != null ? `${label} ${control.min} ile ${control.max} arasında olmalı.`
        : `${label} en az ${control.min} olmalı.`);
    }
    sampling[control.key] = value;
  }
  if (sampling.min_tokens > sampling.max_tokens) {
    throw new ValidationError(`${name} örneklemesinde Min token, Max token'dan büyük olamaz.`);
  }
  return sampling;
}

/** Request fields that come from the Model settings panel (both modes). */
export function modelFields(settings) {
  const abc = String(settings.abc || "").trim() || null;
  if (abc && settings.cot === "off") {
    throw new ValidationError("Harici ABC nota kullanmak için CoT modu 'Tam' veya 'Melodi' olmalı.");
  }
  let seed = null;
  if (!settings.randomSeed) {
    seed = toNumber(settings.seed);
    if (!Number.isInteger(seed) || seed < 0 || seed > MAX_SEED) {
      throw new ValidationError("Seed 0 ya da daha büyük bir tam sayı olmalı (ya da Rastgele seed'i açın).");
    }
  }
  let cfgScale = null;
  if (settings.customCfg) {
    cfgScale = toNumber(settings.cfgScale);
    if (!(cfgScale >= CFG_RANGE[0] && cfgScale <= CFG_RANGE[1])) {
      throw new ValidationError(`CFG scale ${CFG_RANGE[0]} ile ${CFG_RANGE[1]} arasında olmalı.`);
    }
  }
  const steps = toNumber(settings.odeSteps);
  if (!Number.isInteger(steps) || steps < ODE_RANGE[0] || steps > ODE_RANGE[1]) {
    throw new ValidationError(`Ses sentezi adımı (ODE) ${ODE_RANGE[0]} ile ${ODE_RANGE[1]} arasında bir tam sayı olmalı.`);
  }
  return {
    cot: settings.cot,
    seed,
    cfg_scale: cfgScale,
    abc,
    abc_sampling: samplingDict("ABC", settings.abcSampling),
    semantic_sampling: samplingDict("Şarkı", settings.semanticSampling),
    ode_steps: steps === DEFAULT_ODE_STEPS ? null : steps,
    quantization: settings.fp8 ? "fp8" : null,
  };
}

/** {payload, warnings} of a Text to Song generation; throws ValidationError on missing input. */
export function buildT2SPayload(form, settings) {
  const style = t2sStyle(form.description, form.styles, form.customStyle, form.voice);
  if (!style) throw new ValidationError("Şarkı açıklaması yazın ya da en az bir tür seçin.");
  if (!String(form.lyrics || "").trim()) throw new ValidationError("Şarkı sözlerini girin.");
  const warnings = hasSectionLabels(form.lyrics) ? []
    : ["Sözlerde bölüm etiketi yok. Model en iyi `[verse]`, `[chorus]` gibi bölümlerle çalışır."];
  const payload = {
    style, lyrics: normalizeLyrics(form.lyrics), ...modelFields(settings),
    prompt: {description: String(form.description || "").trim(), styles: [...(form.styles || [])],
      custom_style: String(form.customStyle || "").trim(), voice: form.voice || "default"},
  };
  return {payload, warnings};
}

/** Payload of a Cover generation (the caller adds `reference`); throws ValidationError on missing input. */
export function buildCoverPayload(form, settings) {
  const style = (form.tags || []).map((tag) => String(tag).trim()).filter(Boolean).join(", ");
  if (!style) throw new ValidationError("En az bir tür/stil etiketi seçin.");
  if (!String(form.lyrics || "").trim()) {
    throw new ValidationError("Şarkı sözlerini girin ya da referans şarkıdan ✨ Sözleri tanı ile çıkarın.");
  }
  return {style, lyrics: normalizeLyrics(form.lyrics), ...modelFields(settings), reference: null};
}

/** Body of POST /analyze from the reference fields; throws ValidationError on out-of-range values. */
export function analyzeParams({uploadId, start, duration, bpm, lyrics, recognize, language, transcriber}) {
  const begin = toNumber(start || 0);
  const length = toNumber(duration || 0);
  const tempo = toNumber(bpm || 0);
  if (!(begin >= 0)) throw new ValidationError("Başlangıç 0 ya da daha büyük olmalı.");
  if (!(length >= 0)) throw new ValidationError("Süre 0 ya da daha büyük olmalı (0 = tüm şarkı).");
  if (!(tempo === 0 || (tempo > 30 && tempo < 300))) {
    throw new ValidationError("Tempo 30 ile 300 BPM arasında olmalı (0 = otomatik).");
  }
  return {upload_id: uploadId, start: begin, duration: length || null, bpm: tempo || null, lyrics: lyrics || "",
    recognize_lyrics: Boolean(recognize), language: language || null, transcriber};
}

export function isAudioFile(name) {
  const lower = String(name || "").toLowerCase();
  return AUDIO_EXTENSIONS.some((extension) => lower.endsWith(extension));
}

/** Which form a job came from: Text to Song stores its form, covers their reference. */
export function jobKind(req) {
  return req?.prompt ? "t2s" : req?.reference ? "cover" : "other";
}

export function shortStyle(style, limit = 50) {
  style = String(style || "");
  return style.length <= limit ? style : `${style.slice(0, limit - 3)}...`;
}

/** Form and settings values that bring a past job's inputs back (Gradio: restore). */
export function restoreValues(req) {
  const settings = {
    cot: req.cot || "full",
    randomSeed: false,
    seed: req.seed ?? DEFAULT_SEED,
    customCfg: req.cfg_scale != null,
    cfgScale: req.cfg_scale ?? 1.0,
    odeSteps: req.ode_steps || DEFAULT_ODE_STEPS,
    fp8: req.quantization === "fp8",
    abc: req.abc || "",
    abcSampling: mergeSampling(ABC_DEFAULTS, req.abc_sampling),
    semanticSampling: mergeSampling(SEMANTIC_DEFAULTS, req.semantic_sampling),
  };
  const prompt = req.prompt;
  if (prompt) {
    return {mode: "t2s", settings, form: {description: prompt.description || "", lyrics: req.lyrics || "",
      voice: prompt.voice || "default", styles: prompt.styles || [], customStyle: prompt.custom_style || ""}};
  }
  const reference = req.reference || {};
  return {mode: "cover", settings, form: {tags: splitTags(req.style), lyrics: req.lyrics || "",
    coverMode: reference.mode || null, transcriber: reference.transcriber || null}};
}

/** YuE-UI compatible project file (genre_text / lyrics_text) with the YuE2 settings. */
export function projectData(tags, lyrics, cot, abc) {
  return {genre_text: [...(tags || [])], lyrics_text: lyrics || "", yue2: {cot, abc: abc || null}};
}

export function readProject(data) {
  const settings = data?.yue2 || {};
  return {tags: Array.isArray(data?.genre_text) ? data.genre_text.map(String) : [],
    lyrics: data?.lyrics_text || "", cot: settings.cot || "full", abc: settings.abc || ""};
}
