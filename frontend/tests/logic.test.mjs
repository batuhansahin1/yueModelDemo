// Unit tests of public/js/logic.js (no browser, no server): node --test tests/logic.test.mjs
import assert from "node:assert/strict";
import {test} from "node:test";

import {
  ABC_DEFAULTS, SEMANTIC_DEFAULTS, TIMELINE_HINT, ValidationError, analyzeParams, buildCoverPayload, buildT2SPayload,
  clock, coverInfo, coverScore, defaultSettings, estimateRange, estimateText, jobKind, lyricBlocks, normalizeLyrics,
  processTags, progressInfo, pyRound, readProject, restoreValues, scoreSections, semanticBudget, settingsText,
  splitLines, stripChords, structureProblem, t2sStyle, timeline,
} from "../public/js/logic.js";

// A small analyzed reference: intro (music), verse, chorus; two recognized verse lines and one chorus line.
const ABC = ["X:1", "M:4/4", "L:1/32", "Q:1/4=120", "K:C", "% intro", "V: Vocal", "z32|", "% verse", "V: Vocal",
  '"C"E8E8G8E8|"G"D8D8z16|', "% chorus", "V: Vocal", '"F"A8A8G8F8|', ""].join("\n");
const SUMMARY = {
  tempo: 120.0, key: "C", bars: 8, seconds: 16.0, vocal_ratio: 0.6, melody_source: "vocals", transcriber: "sheetsage2",
  lyrics_status: "ok", language: "tr", asr_model: "whisper-large-v3-turkish-general",
  sections: [{name: "intro", start: 0, end: 4}, {name: "verse", start: 4, end: 10}, {name: "chorus", start: 10, end: 16}],
  lines: [{section: "verse", block: 0, text: "bir", start: 4.2, end: 6}, {section: "verse", block: 0, text: "iki", start: 7.5, end: 9},
    {section: "chorus", block: 1, text: "üç", start: 10.1, end: 12}],
};
const ANALYSIS = {abc: ABC, summary: SUMMARY};
const LYRICS = "[verse]\nbir\niki\n\n[chorus]\nüç";
const STATS = {device: "Apple MPS", rate: 6.5, fast_rate: 5.0, jobs: 3};

const t2sForm = (overrides = {}) => ({description: "", lyrics: "", voice: "default", styles: [], customStyle: "", ...overrides});
const settings = (overrides = {}) => ({...defaultSettings(), ...overrides});

test("splitLines behaves like Python's str.splitlines", () => {
  assert.deepEqual(splitLines(""), []);
  assert.deepEqual(splitLines("a"), ["a"]);
  assert.deepEqual(splitLines("a\n"), ["a"]);
  assert.deepEqual(splitLines("a\n\nb\r\nc\n"), ["a", "", "b", "c"]);
});

test("pyRound rounds halves to even like Python", () => {
  assert.deepEqual([0.5, 1.5, 2.5, 6.5, 6.4, 6.6].map(pyRound), [0, 2, 2, 6, 6, 7]);
});

test("processTags deduplicates case-insensitively and sorts", () => {
  const tags = processTags({genre: ["Pop", " rock ", "pop"], mood: ["Calm"], instrument: [], gender: ["female"], timbre: ["Airy"]});
  assert.deepEqual(tags, ["Airy", "Calm", "female", "Pop", "rock"]);
});

test("Text to Song tags: styles, custom style, voice, then the description", () => {
  assert.equal(t2sStyle(" Gece ", ["pop", "chill"], " Anadolu rock ", "female"), "pop, chill, Anadolu rock, female vocal, Gece");
  assert.equal(t2sStyle("", [], "", "default"), "");
});

test("normalizeLyrics lowercases whole-line section labels only", () => {
  assert.equal(normalizeLyrics("  [Verse 1]  \nHello [World]\n[CHORUS]\n\n"), "[verse 1]\nHello [World]\n[chorus]");
});

test("buildT2SPayload validates and builds the request", () => {
  assert.throws(() => buildT2SPayload(t2sForm(), settings()), {name: "ValidationError", message: /açıklaması yazın/});
  assert.throws(() => buildT2SPayload(t2sForm({description: "x"}), settings()), {message: "Şarkı sözlerini girin."});
  const {payload, warnings} = buildT2SPayload(
    t2sForm({description: " Sakin bir gece ", lyrics: "[Verse]\nA\n", voice: "male", styles: ["pop"], customStyle: "bağlama"}),
    settings());
  assert.deepEqual(warnings, []);
  assert.deepEqual(payload, {
    style: "pop, bağlama, male vocal, Sakin bir gece", lyrics: "[verse]\nA", cot: "full", seed: null, cfg_scale: null,
    abc: null, abc_sampling: {...ABC_DEFAULTS}, semantic_sampling: {...SEMANTIC_DEFAULTS}, ode_steps: null, quantization: null,
    prompt: {description: "Sakin bir gece", styles: ["pop"], custom_style: "bağlama", voice: "male"},
  });
});

test("buildT2SPayload warns about lyrics without section labels", () => {
  const {warnings} = buildT2SPayload(t2sForm({description: "x", lyrics: "sadece bir satır"}), settings());
  assert.equal(warnings.length, 1);
});

test("model settings: seed, CFG, fast synthesis, FP8 and external score", () => {
  const form = t2sForm({description: "x", lyrics: "[verse]\na"});
  const {payload} = buildT2SPayload(form, settings({randomSeed: false, seed: "42", customCfg: true, cfgScale: "1.5",
    odeSteps: "16", fp8: true, abc: "  X:1\n% verse\n  "}));
  assert.equal(payload.seed, 42);
  assert.equal(payload.cfg_scale, 1.5);
  assert.equal(payload.ode_steps, 16);
  assert.equal(payload.quantization, "fp8");
  assert.equal(payload.abc, "X:1\n% verse");
  assert.throws(() => buildT2SPayload(form, settings({abc: "X:1", cot: "off"})), /CoT modu 'Tam' veya 'Melodi'/);
  assert.throws(() => buildT2SPayload(form, settings({randomSeed: false, seed: ""})), ValidationError);
  assert.throws(() => buildT2SPayload(form, settings({randomSeed: false, seed: "-1"})), ValidationError);
  assert.throws(() => buildT2SPayload(form, settings({customCfg: true, cfgScale: "25"})), /CFG scale/);
  assert.throws(() => buildT2SPayload(form, settings({odeSteps: "70"})), /ODE/);
});

test("sampling values are checked like the Gradio UI and yue2", () => {
  const form = t2sForm({description: "x", lyrics: "[verse]\na"});
  const semantic = {...SEMANTIC_DEFAULTS, min_tokens: 9500};
  assert.throws(() => buildT2SPayload(form, settings({semanticSampling: semantic})),
    {message: "Şarkı örneklemesinde Min token, Max token'dan büyük olamaz."});
  assert.throws(() => buildT2SPayload(form, settings({abcSampling: {...ABC_DEFAULTS, top_k: "2.5"}})), /tam sayı/);
  assert.throws(() => buildT2SPayload(form, settings({abcSampling: {...ABC_DEFAULTS, penalty_window: "101"}})), /1 ile 100/);
  const {payload} = buildT2SPayload(form, settings({abcSampling: {...ABC_DEFAULTS, temperature: "0.85", top_k: "50"}}));
  assert.equal(payload.abc_sampling.temperature, 0.85);
  assert.equal(payload.abc_sampling.top_k, 50);
});

test("buildCoverPayload needs tags and lyrics", () => {
  assert.throws(() => buildCoverPayload({tags: [" "], lyrics: "a"}, settings()), /etiketi seçin/);
  assert.throws(() => buildCoverPayload({tags: ["pop"], lyrics: " "}, settings()), /Sözleri tanı/);
  const payload = buildCoverPayload({tags: ["pop", " Turkish "], lyrics: "[Chorus]\na"}, settings({cot: "melody"}));
  assert.equal(payload.style, "pop, Turkish");
  assert.equal(payload.lyrics, "[chorus]\na");
  assert.equal(payload.cot, "melody");
  assert.equal(payload.reference, null);
});

test("settingsText matches the Gradio UI's summary line", () => {
  assert.equal(settingsText({cot: "full", cfg_scale: null, ode_steps: null}), "CoT full · CFG varsayılan · ABC T=0.7 · Şarkı T=1.0 · ODE 32");
  assert.equal(settingsText({
    cot: "melody", cfg_scale: 2, ode_steps: 16, quantization: "fp8",
    abc_sampling: {...ABC_DEFAULTS, top_k: 50, min_tokens: null},
    semantic_sampling: {...SEMANTIC_DEFAULTS, temperature: 0.9, max_tokens: 12000, top_p: 1},
    reference: {mode: "free", transcriber: "sheetsage2"},
  }), "CoT melody · CFG 2.0 · ABC T=0.7 (top_k=50) · Şarkı T=0.9 (top_p=1.0) · ODE 16 · FP8 · cover Serbest (SheetSage2)");
  assert.match(settingsText({cot: "full", reference: {mode: "close"}}), /cover Yakın \(Basit analiz\)$/);
});

test("time estimates use the device's measured speed", () => {
  assert.equal(estimateRange(60, false, {}), null);
  assert.equal(estimateText(60, false, {}), "ilk üretimden sonra bu cihaz için hesaplanacak");
  assert.deepEqual(estimateRange(60, true, STATS), [5, 8]);
  assert.equal(estimateText(60, false, STATS), "6-10 dk (Apple MPS)");
  // Before a fast generation was measured, the normal speed stands in.
  assert.equal(estimateText(60, true, {device: "NVIDIA L4", rate: 1}), "1-2 dk (NVIDIA L4)");
});

test("semanticBudget sizes the song tokens to the reference", () => {
  assert.equal(semanticBudget({seconds: 292.9}, 9000), 9519);
  assert.equal(semanticBudget({seconds: 10}, 9000), 9000);
  assert.equal(semanticBudget({seconds: 1000}, 9000), 20000);
});

test("stripChords removes chord symbols from music lines only", () => {
  const abc = 'X:1\nT:"Başlık"\n% verse\nV: Vocal\n"C"E8E8 "G7"D8|\n';
  assert.equal(stripChords(abc), 'X:1\nT:"Başlık"\n% verse\nV: Vocal\nE8E8 D8|\n');
  assert.equal(coverScore(abc, "close"), abc);
  assert.equal(coverScore(abc, "free"), stripChords(abc));
});

test("score sections and lyric blocks", () => {
  assert.deepEqual(scoreSections(ABC), ["intro", "verse", "chorus"]);
  assert.deepEqual(scoreSections("%%MIDI program 1\n% Verse \n"), ["verse"]);
  assert.deepEqual(lyricBlocks("ön satır\n[Verse]\na\n\nb\n[chorus]\n"), [["", ["ön satır"]], ["verse", ["a", "b"]], ["chorus", []]]);
});

test("structureProblem explains lyrics that do not follow the score", () => {
  assert.equal(structureProblem(LYRICS, ABC), null);
  assert.equal(structureProblem("", ABC), null);
  assert.match(structureProblem("a\n[verse]\nb", ABC), /bölüm etiketi olmayan satırlar/);
  assert.equal(structureProblem("[chorus]\na\n[verse]\nb", ABC),
    "Sözlerin bölümleri notayla eşleşmiyor (1. şarkılı bölüm notada `[verse]`, sözlerde `[chorus]`). "
    + "Model sözleri bu yüzden başka yerlere kaydırabilir.");
  assert.match(structureProblem("[verse]\na", ABC), /notada 2, sözlerde 1 şarkılı bölüm var/);
});

test("timeline places the lyric lines on the analyzed sections", () => {
  const result = timeline(LYRICS, ABC, ANALYSIS);
  assert.equal(result.tone, "ok");
  assert.match(result.message, /^✅ Sözler notayla eşleşiyor: 2 şarkılı bölüm/);
  assert.deepEqual(result.rows, [
    {kind: "music", time: "0:00", name: "intro"},
    {kind: "section", time: "0:04", label: "verse", mark: ""},
    {kind: "line", time: "0:04", text: "bir"},
    {kind: "line", time: "0:07", text: "iki"},
    {kind: "section", time: "0:10", label: "chorus", mark: ""},
    {kind: "line", time: "0:10", text: "üç"},
    {kind: "end", time: "0:16"},
  ]);
});

test("timeline marks edits that break the alignment", () => {
  const extraLine = timeline("[verse]\nbir\niki\nüç\n[chorus]\ndört", ABC, ANALYSIS);
  assert.equal(extraLine.rows[1].mark, "satır sayısı değişti, satır zamanları yok");
  assert.equal(extraLine.rows[2].time, "");
  const swapped = timeline("[chorus]\na\n[verse]\nb\n[bridge]\nc", ABC, ANALYSIS);
  assert.equal(swapped.tone, "warn");
  assert.equal(swapped.rows[1].mark, "notada [verse]");
  assert.deepEqual(swapped.rows.filter((row) => row.kind === "orphan"),
    [{kind: "orphan", time: "-", label: "bridge", mark: "notada karşılığı yok"}]);
});

test("timeline messages without a usable analysis", () => {
  assert.deepEqual(timeline(LYRICS, "", null), {tone: "hint", rows: [], message: TIMELINE_HINT});
  assert.match(timeline(LYRICS, "", ANALYSIS).message, /Referans ayarları değişti/);
  assert.match(timeline(LYRICS, ABC + "\n% outro\n", ANALYSIS).message, /analizinden gelmiyor/);
  assert.match(timeline("", ABC, ANALYSIS).message, /Sözleri bu sırayla/);
});

test("coverInfo describes an analysis like the Gradio UI", () => {
  const lines = coverInfo(SUMMARY, false, STATS);
  assert.equal(lines[0], "✅ **Melodi çıkarıldı** (SheetSage2) · Tempo **120 BPM** · Ton **C** · 8 ölçü (16 sn) · melodili kısım %60");
  assert.equal(lines[1], "✅ **Sözler tanındı** · 3 satır · dil `tr` · model `whisper-large-v3-turkish-general` · "
    + "sözler SheetSage2'nin bölümlerine yerleştirildi");
  assert.equal(lines[4], "⏱ Tahmini üretim süresi: **2-3 dk (Apple MPS)**");
  assert.match(lines.at(-1), /SheetSage2 vuruşları kendisi bulur/);
  const basic = coverInfo({...SUMMARY, transcriber: "basic", lyrics_status: "instrumental", vocal_ratio: 0.01}, false, {});
  assert.match(basic[1], /Seçilen aralıkta vokal yok/);
  assert.ok(basic.some((line) => line.includes("Neredeyse hiç melodi bulunamadı")));
  assert.match(basic.at(-1), /BPM girip yeniden çıkarın/);
});

test("progressInfo collects what the progress view shows", () => {
  const job = {id: "20261012-120000-abc123", status: "running", stage: "semantic", elapsed: 41.5,
    tokens: {abc: 300, semantic: 450}, error: null, result: null,
    request: {cot: "full", seed: 7, ode_steps: null, semantic_sampling: {...SEMANTIC_DEFAULTS, max_tokens: 9519},
      reference: {seconds: 60, mode: "free", transcriber: "sheetsage2"}}};
  const info = progressInfo(job, STATS);
  assert.equal(info.status, "Çalışıyor");
  assert.equal(info.stage, "Şarkı token'ları üretiliyor");
  assert.equal(info.elapsed, 42);
  assert.equal(info.estimate, "6-10 dk (Apple MPS)");
  assert.deepEqual(info.tokens.map((t) => [t.key, t.value, t.max]), [["abc", 300, 4096], ["semantic", 450, 9519]]);
  const done = progressInfo({...job, status: "complete", stage: "done", result: {audio_seconds: 61.25,
    truncated: {abc: false, semantic: true}}}, STATS);
  assert.equal(done.estimate, null);
  assert.equal(done.audioSeconds, 61.25);
  assert.deepEqual(done.truncated, ["semantic"]);
});

test("restoreValues brings a job back into the form it came from", () => {
  const t2s = restoreValues({style: "pop, female vocal, Gece", lyrics: "[verse]\na", cot: "melody", seed: 5, ode_steps: 16,
    prompt: {description: "Gece", styles: ["pop"], custom_style: "", voice: "female"}});
  assert.equal(t2s.mode, "t2s");
  assert.deepEqual(t2s.form, {description: "Gece", lyrics: "[verse]\na", voice: "female", styles: ["pop"], customStyle: ""});
  assert.equal(t2s.settings.randomSeed, false);
  assert.equal(t2s.settings.seed, 5);
  assert.equal(t2s.settings.odeSteps, 16);
  // Old jobs stored null for unset sampling values: those keep the default.
  const cover = restoreValues({style: "rap, Pop ", lyrics: "x", cot: "full", seed: 9, cfg_scale: 1.5, abc: "X:1",
    abc_sampling: {temperature: 0.7, min_tokens: null}, reference: {mode: "close", transcriber: "basic"}});
  assert.equal(cover.mode, "cover");
  assert.deepEqual(cover.form, {tags: ["rap", "Pop"], lyrics: "x", coverMode: "close", transcriber: "basic"});
  assert.equal(cover.settings.customCfg, true);
  assert.equal(cover.settings.abcSampling.min_tokens, ABC_DEFAULTS.min_tokens);
  assert.equal(jobKind({style: "x"}), "other");
});

test("analyzeParams checks the reference fields", () => {
  assert.deepEqual(analyzeParams({start: "5", duration: "", bpm: "0", lyrics: "", transcriber: "basic"}),
    {upload_id: undefined, start: 5, duration: null, bpm: null, lyrics: "", recognize_lyrics: false, language: null,
      transcriber: "basic"});
  assert.throws(() => analyzeParams({bpm: "20"}), /Tempo 30 ile 300/);
  assert.throws(() => analyzeParams({start: "-1"}), /Başlangıç/);
});

test("project files keep the YuE-UI format", () => {
  assert.deepEqual(readProject({genre_text: ["pop"], lyrics_text: "[verse]\na"}), {tags: ["pop"], lyrics: "[verse]\na", cot: "full", abc: ""});
  assert.deepEqual(readProject({genre_text: [], yue2: {cot: "melody", abc: "X:1"}}), {tags: [], lyrics: "", cot: "melody", abc: "X:1"});
});

test("clock formats minutes and seconds", () => {
  assert.equal(clock(0), "0:00");
  assert.equal(clock(61.9), "1:01");
  assert.equal(clock(600), "10:00");
});
