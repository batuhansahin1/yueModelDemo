// Browser test of the site against the mock model server: headless Chrome driven over the DevTools protocol.
// No packages needed (Node 22+ has WebSocket). Skipped when Chrome or yue-ui/venv is missing.
//
//   node tests/e2e.mjs                         (or ./frontend/test.sh e2e)
//   E2E_SCREENSHOTS=/tmp/shots node tests/e2e.mjs   also saves screenshots (desktop, dark, mobile)
//   CHROME=/path/to/chrome                         another Chrome/Chromium
import {spawn} from "node:child_process";
import {existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync} from "node:fs";
import net from "node:net";
import {tmpdir} from "node:os";
import {dirname, join} from "node:path";
import {fileURLToPath} from "node:url";

const FRONTEND = join(dirname(fileURLToPath(import.meta.url)), "..");
const PROJECT = join(FRONTEND, "..");
const CHROME = [process.env.CHROME, "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
  "/Applications/Chromium.app/Contents/MacOS/Chromium", "/usr/bin/google-chrome", "/usr/bin/chromium",
  "/usr/bin/chromium-browser"].find((path) => path && existsSync(path));
const MOCK_PYTHON = process.env.YUE2_MOCK_PYTHON || join(PROJECT, "venv", "bin", "python");
const PYTHON = process.env.PYTHON || "python3";
const SHOTS = process.env.E2E_SCREENSHOTS;

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

function freePort() {
  return new Promise((resolve, reject) => {
    const server = net.createServer().listen(0, "127.0.0.1", () => {
      const {port} = server.address();
      server.close(() => resolve(port));
    }).on("error", reject);
  });
}

async function waitFor(check, what, timeout = 20000) {
  const deadline = Date.now() + timeout;
  let last;
  while (Date.now() < deadline) {
    try {
      last = await check();
      if (last) return last;
    } catch (error) {
      last = error;
    }
    await sleep(100);
  }
  throw new Error(`zaman aşımı: ${what}${last instanceof Error ? ` (${last.message})` : ""}`);
}

function start(command, args) {
  const child = spawn(command, args, {stdio: ["ignore", "pipe", "pipe"]});
  child.output = "";
  child.stdout.on("data", (data) => (child.output += data));
  child.stderr.on("data", (data) => (child.output += data));
  return child;
}

async function ok(url) {
  try {
    return (await fetch(url)).ok;
  } catch {
    return false;
  }
}

/** Minimal DevTools protocol client. */
class Cdp {
  static async connect(url) {
    const cdp = new Cdp();
    cdp.ws = new WebSocket(url);
    cdp.next = 0;
    cdp.pending = new Map();
    cdp.handlers = new Map();
    cdp.ws.addEventListener("message", ({data}) => {
      const message = JSON.parse(data);
      if (message.id && cdp.pending.has(message.id)) {
        const {resolve, reject} = cdp.pending.get(message.id);
        cdp.pending.delete(message.id);
        if (message.error) reject(new Error(message.error.message));
        else resolve(message.result);
      } else if (message.method) {
        for (const handler of cdp.handlers.get(message.method) || []) handler(message.params);
      }
    });
    await new Promise((resolve, reject) => {
      cdp.ws.addEventListener("open", resolve, {once: true});
      cdp.ws.addEventListener("error", reject, {once: true});
    });
    return cdp;
  }

  send(method, params = {}) {
    const id = ++this.next;
    this.ws.send(JSON.stringify({id, method, params}));
    return new Promise((resolve, reject) => this.pending.set(id, {resolve, reject}));
  }

  on(event, handler) {
    this.handlers.set(event, [...(this.handlers.get(event) || []), handler]);
  }
}

function toneWav(path, seconds = 2, rate = 16000) {
  const samples = seconds * rate;
  const buffer = Buffer.alloc(44 + samples * 2);
  buffer.write("RIFF", 0);
  buffer.writeUInt32LE(36 + samples * 2, 4);
  buffer.write("WAVEfmt ", 8);
  buffer.writeUInt32LE(16, 16);
  buffer.writeUInt16LE(1, 20);
  buffer.writeUInt16LE(1, 22);
  buffer.writeUInt32LE(rate, 24);
  buffer.writeUInt32LE(rate * 2, 28);
  buffer.writeUInt16LE(2, 32);
  buffer.writeUInt16LE(16, 34);
  buffer.write("data", 36);
  buffer.writeUInt32LE(samples * 2, 40);
  for (let i = 0; i < samples; i++) buffer.writeInt16LE(Math.round(8000 * Math.sin(2 * Math.PI * 440 * i / rate)), 44 + i * 2);
  writeFileSync(path, buffer);
}

// Helpers evaluated in the page with each expression (DevTools evaluation is not subject to the page's CSP).
const PAGE_HELPERS = `
  const $ = (s) => document.querySelector(s);
  const $$ = (s) => [...document.querySelectorAll(s)];
  const need = (s) => { const el = $(s); if (!el) throw new Error("öğe yok: " + s); return el; };
  const set = (s, v) => { const el = need(s); el.value = v; el.dispatchEvent(new Event("input", {bubbles: true}));
    el.dispatchEvent(new Event("change", {bubbles: true})); };
  const click = (s) => need(s).click();
  const clickText = (s, t) => { const el = $$(s).find((e) => e.textContent.includes(t));
    if (!el) throw new Error("öğe yok: " + s + " / " + t); el.click(); };
  const key = (s, k) => need(s).dispatchEvent(new KeyboardEvent("keydown", {key: k, bubbles: true}));
  const text = (s) => $(s)?.textContent ?? "";
`;

async function main() {
  if (!CHROME) {
    console.log("Chrome bulunamadı: tarayıcı testleri atlandı (CHROME=/yol/chrome ile belirtin).");
    return;
  }
  if (!existsSync(MOCK_PYTHON)) {
    console.log(`Mock model sunucusu için ${MOCK_PYTHON} yok: tarayıcı testleri atlandı.`);
    return;
  }
  const tmp = mkdtempSync(join(tmpdir(), "yue2-e2e-"));
  const processes = [];
  let cdp;
  let failures = 0;
  try {
    const [mockPort, webPort] = [await freePort(), await freePort()];
    processes.push(start(MOCK_PYTHON, [join(PROJECT, "server", "yue2_server.py"), "--mock", "--port", String(mockPort),
      "--outputs", join(tmp, "outputs")]));
    await waitFor(() => ok(`http://127.0.0.1:${mockPort}/health`), "mock model sunucusu");
    // --analyze-wait 0: every analysis takes the background path the real model's minutes-long analyses take.
    processes.push(start(PYTHON, [join(FRONTEND, "server.py"), "--port", String(webPort), "--api", `http://127.0.0.1:${mockPort}`,
      "--analyze-wait", "0"]));
    const base = `http://127.0.0.1:${webPort}`;
    await waitFor(() => ok(`${base}/`), "web sunucusu");

    const profile = join(tmp, "chrome");
    processes.push(start(CHROME, ["--headless=new", "--remote-debugging-port=0", `--user-data-dir=${profile}`,
      "--no-first-run", "--no-default-browser-check", "--disable-extensions", "--window-size=1440,1000", "about:blank"]));
    const devtoolsPort = await waitFor(() => {
      const file = join(profile, "DevToolsActivePort");
      return existsSync(file) && readFileSync(file, "utf8").split("\n")[0].trim();
    }, "Chrome DevTools portu");
    const pages = await waitFor(async () => {
      const list = await (await fetch(`http://127.0.0.1:${devtoolsPort}/json/list`)).json();
      return list.filter((target) => target.type === "page").length && list;
    }, "Chrome sekmesi");
    cdp = await Cdp.connect(pages.find((target) => target.type === "page").webSocketDebuggerUrl);

    const problems = [];
    cdp.on("Runtime.exceptionThrown", ({exceptionDetails}) =>
      problems.push(`istisna: ${exceptionDetails.exception?.description || exceptionDetails.text}`));
    cdp.on("Runtime.consoleAPICalled", ({type, args}) => {
      if (type === "error") problems.push(`console.error: ${args.map((a) => a.value ?? a.description).join(" ")}`);
    });
    cdp.on("Log.entryAdded", ({entry}) => {
      if (entry.level === "error") problems.push(`${entry.source}: ${entry.text} ${entry.url || ""}`);
    });
    await cdp.send("Runtime.enable");
    await cdp.send("Log.enable");
    await cdp.send("Page.enable");

    const evaluate = async (expression) => {
      const {result, exceptionDetails} = await cdp.send("Runtime.evaluate", {
        expression: `(async () => { ${PAGE_HELPERS} ${expression} })()`, awaitPromise: true, returnByValue: true});
      if (exceptionDetails) throw new Error(exceptionDetails.exception?.description || exceptionDetails.text);
      return result.value;
    };
    const until = (expression, what, timeout) => waitFor(() => evaluate(expression), what, timeout);
    const screenshot = async (name, fullPage = true) => {
      if (!SHOTS) return;
      mkdirSync(SHOTS, {recursive: true});
      let clip;
      if (fullPage) {
        // The sticky top bar is drawn where the page is scrolled to.
        await evaluate("window.scrollTo(0, 0)");
        const {cssContentSize} = await cdp.send("Page.getLayoutMetrics");
        clip = {x: 0, y: 0, width: cssContentSize.width, height: Math.min(cssContentSize.height, 6000), scale: 1};
      }
      const {data} = await cdp.send("Page.captureScreenshot", {format: "png", captureBeyondViewport: fullPage, clip});
      writeFileSync(join(SHOTS, `${name}.png`), Buffer.from(data, "base64"));
    };
    const step = async (name, action) => {
      try {
        await action();
        console.log(`✔ ${name}`);
      } catch (error) {
        failures += 1;
        console.log(`✘ ${name}\n    ${error.message}`);
        console.log(`    bildirimler: ${await evaluate("return $$('.toast').map((t) => t.textContent).join(' | ')").catch(() => "?")}`);
      }
    };

    await cdp.send("Page.navigate", {url: `${base}/#t2s`});

    await step("Sayfa açılır, durum çubuğu test modunu gösterir", async () => {
      await until("return $('#status-pill')?.dataset.state === 'mock'", "mock durumu");
      await until("return $$('#view-t2s .output').length === 1 && $('#view-cover').hidden", "görünümler");
    });

    await step("Text to Song: boş formda üretim bir uyarı gösterir, istek gönderilmez", async () => {
      await evaluate("click('#view-t2s .btn-generate')");
      await until("return text('.toast-warning').includes('Şarkı açıklaması yazın')", "doğrulama uyarısı");
    });

    await step("Text to Song: vokal ve tür seçilir, şarkı üretilir", async () => {
      await evaluate(`
        set('#t2s-description', 'Yağmurlu bir gecede sakin bir şarkı');
        set('#t2s-lyrics', '[Verse]\\nNeon ışıklar soluyor\\n\\n[chorus]\\nYine sana dönerim');
        click('#t2s-voice-button'); clickText('#t2s-voices .pill', 'Kadın');
        click('#t2s-style-button'); clickText('#t2s-styles .pill', 'Pop'); clickText('#t2s-styles .pill', 'Caz');
        set('#t2s-custom-style', 'bağlama'); click('#t2s-style-close');`);
      const labels = await evaluate("return [text('#t2s-voice-button'), text('#t2s-style-button')]");
      if (labels[0] !== "🎙 Vokal: Kadın" || labels[1] !== "🎚 Tür (3)") throw new Error(`etiketler: ${labels}`);
      await evaluate("click('#view-t2s .btn-generate')");
      await until("return text('#view-t2s .status').includes('Tamamlandı') && $('#view-t2s .player audio')", "üretimin bitmesi", 30000);
      const prompt = await evaluate("return text('#view-t2s .output pre')");
      if (!prompt.startsWith("[Tags]\npop, jazz, bağlama, female vocal, Yağmurlu bir gecede sakin bir şarkı\n\n[Lyrics]\n[verse]")) {
        throw new Error(`modele giden girdi: ${prompt}`);
      }
      const audio = await evaluate("const a = $('#view-t2s .player audio'); await new Promise((r) => a.readyState >= 1 ? r() : a.addEventListener('loadedmetadata', r)); return a.duration");
      if (!(audio > 3)) throw new Error(`ses süresi: ${audio}`);
      await until("return text('#view-t2s .output .group:last-child pre').startsWith('X:1')", "üretilen ABC");
    });
    await screenshot("1-text-to-song");

    await step("Model ayarları: ⚡ Hızlı sentez ODE adımını 16 yapar, ayarlar işle gider", async () => {
      await evaluate(`click('#t2s-card [data-subtab=settings]'); click('#view-t2s .output .switch input');`);
      await until(`return $('#t2s-card input.num[aria-label="Ses sentezi adımı (ODE)"]').value === '16'`, "ODE 16");
      await evaluate(`
        $$('#t2s-card .switch input')[0].click();   // Rastgele seed kapalı
        set('#t2s-card input[aria-label=Seed]', '1234');
        click('#view-t2s .btn-generate');`);
      await until("return text('#view-t2s .status').includes('Tamamlandı') && text('#view-t2s .status').includes('1234')", "seed'li üretim", 30000);
      const settings = await evaluate("return text('#view-t2s .job-settings')");
      if (!settings.includes("ODE 16")) throw new Error(`ayarlar satırı: ${settings}`);
    });
    await screenshot("2-model-ayarlari");

    await step("Cover: referans yüklenir, sözler tanınır, zaman çizelgesi dolar", async () => {
      const wav = join(tmp, "referans.wav");
      toneWav(wav);
      await evaluate("location.hash = '#cover'");
      await until("return !$('#view-cover').hidden", "Cover görünümü");
      const {root} = await cdp.send("DOM.getDocument", {depth: -1});
      const {nodeId} = await cdp.send("DOM.querySelector", {nodeId: root.nodeId, selector: "#cover-file"});
      await cdp.send("DOM.setFileInputFiles", {nodeId, files: [wav]});
      await until("return !$('#cover-file-card').hidden && text('#cover-file-name') === 'referans.wav'", "dosya kartı");
      await evaluate("set('#cover-language', 'tr'); click('#cover-recognize')");
      await until("return $('#cover-lyrics').value.startsWith('[verse]')", "tanınan sözler", 30000);
      await until("return $$('#cover-timeline .tl-row').length > 5 && text('#cover-timeline').includes('Sözler notayla eşleşiyor')", "zaman çizelgesi");
      const state = await evaluate(`return {abc: $('#cover-card textarea.code-area').value,
        cot: $('#cover-card input[name$="-cot"]:checked').value, info: text('#cover-info')}`);
      // Serbest (default): melody-only score, CoT melody.
      if (state.cot !== "melody" || state.abc.includes('"C"') || !state.abc.includes("% verse")) {
        throw new Error(`nota/CoT: ${state.cot} ${state.abc.slice(0, 80)}`);
      }
      if (!state.info.includes("Sözler tanındı")) throw new Error(`analiz özeti: ${state.info}`);
    });

    await step("Cover: Yakın moduna geçince akorlu nota ve CoT 'Tam' gelir", async () => {
      await evaluate("click('#cover-modes input[value=close]')");
      await until(`return $('#cover-card textarea.code-area').value.includes('"C"') && $('#cover-card input[name$="-cot"]:checked').value === 'full'`, "akorlu nota");
      await evaluate("click('#cover-modes input[value=free]')");
      await until(`return !$('#cover-card textarea.code-area').value.includes('"C"')`, "Serbest'e dönüş");
    });

    await step("Cover: sözlerin bölümleri bozulunca zaman çizelgesi uyarır", async () => {
      await evaluate("set('#cover-lyrics', $('#cover-lyrics').value.replace('[verse]', '[bridge]'))");
      await until("return text('#cover-timeline').includes('notayla eşleşmiyor')", "eşleşme uyarısı");
      await evaluate("set('#cover-lyrics', $('#cover-lyrics').value.replace('[bridge]', '[verse]'))");
      await until("return text('#cover-timeline').includes('Sözler notayla eşleşiyor')", "eşleşmenin dönmesi");
    });

    await step("Cover: etiketler eklenir ve referansın notasıyla şarkı üretilir", async () => {
      await evaluate(`
        set('#cover-tags input.tag-input', 'pop'); key('#cover-tags input.tag-input', 'Enter');
        set('#cover-tags input.tag-input', 'Turkish, female vocal'); key('#cover-tags input.tag-input', 'Enter');`);
      // A typed tag that is in the list takes the list's spelling ("pop" -> "Pop"); others stay as typed.
      const chips = await evaluate("return $$('#cover-tags .tag-chip').map((c) => c.firstChild.textContent)");
      if (chips.join("|") !== "Pop|Turkish|female vocal") throw new Error(`etiketler: ${chips}`);
      if (await evaluate("return text('#cover-tags .tag-menu').includes('ekle')")) throw new Error("öneri listesi yenilenmedi");
      await evaluate("click('#view-cover .btn-generate')");
      await until("return text('#view-cover .status').includes('Tamamlandı') && $('#view-cover .player audio')", "cover üretimi", 30000);
      const settings = await evaluate("return text('#view-cover .job-settings')");
      if (!settings.includes("cover Serbest (SheetSage2)") || !settings.includes("CoT melody")) throw new Error(`ayarlar: ${settings}`);
    });
    await screenshot("3-cover");

    await step("Geçmiş: üretimler listelenir, ayarlar forma geri yüklenir", async () => {
      await evaluate("location.hash = '#history'");
      await until("return $$('#history-list .history-item').length === 3", "geçmiş listesi");
      await evaluate("clickText('#history-list .history-item', 'jazz')");
      await until("return text('#history-detail').includes('Text to Song') && $('#history-detail audio')", "iş ayrıntısı");
      await screenshot("4-gecmis");
      await evaluate("set('#t2s-description', '')");
      await evaluate("clickText('#history-detail .btn', 'Ayarları forma geri yükle')");
      await until("return location.hash === '#t2s' && $('#t2s-description').value === 'Yağmurlu bir gecede sakin bir şarkı'", "geri yükleme");
    });

    await step("Sayfa yenilenince son iş paneli geri gelir", async () => {
      await cdp.send("Page.reload");
      await until("return text('#view-t2s .status').includes('Tamamlandı') && $('#view-t2s .player audio')", "son iş");
    });

    await step("Sunucu sekmesi model sunucusunun durumunu gösterir", async () => {
      await evaluate("location.hash = '#server'");
      await until("return text('#server-facts').includes('mock (modelsiz test)')", "sunucu bilgileri");
    });
    await screenshot("5-sunucu");

    if (SHOTS) {
      await evaluate("location.hash = '#cover'");
      await cdp.send("Emulation.setEmulatedMedia", {features: [{name: "prefers-color-scheme", value: "dark"}]});
      await sleep(300);
      await screenshot("6-cover-koyu");
      await cdp.send("Emulation.setEmulatedMedia", {features: []});
      await evaluate("location.hash = '#t2s'");
      await cdp.send("Emulation.setDeviceMetricsOverride", {width: 390, height: 844, deviceScaleFactor: 2, mobile: true});
      await sleep(300);
      await screenshot("7-mobil");
      await cdp.send("Emulation.clearDeviceMetricsOverride");
    }

    await step("Konsolda hata, istisna ya da CSP ihlali yok", async () => {
      if (problems.length) throw new Error(problems.join("\n    "));
    });
  } catch (error) {
    failures += 1;
    console.log(`✘ kurulum: ${error.message}`);
    for (const child of processes) if (child.output) console.log(child.output.slice(-1500));
  } finally {
    try {
      await cdp?.send("Browser.close");
    } catch {
      // Chrome is killed below anyway.
    }
    for (const child of processes) child.kill();
    await sleep(300);
    rmSync(tmp, {recursive: true, force: true});
  }
  if (failures) {
    console.log(`${failures} tarayıcı testi başarısız.`);
    process.exit(1);
  }
  console.log("Tarayıcı testleri geçti.");
}

await main();
