/**
 * Model server API (server/yue2_server.py) through the web server's /api proxy.
 *
 * URLs are relative, so the site keeps working when it is served under a sub-path.
 */
const BASE = "api";
const ANALYSIS_POLL_MS = 1500;
const wait = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

export class ApiError extends Error {
  constructor(message, {status = 0, unreachable = false} = {}) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    // The model server (or the web server itself) could not be reached at all.
    this.unreachable = unreachable;
  }
}

/** Readable text of a FastAPI error `detail` (a string, or a list of validation errors). */
function describe(detail) {
  if (Array.isArray(detail)) {
    return detail.map((item) => {
      const where = (item.loc || []).filter((part) => part !== "body").join(".");
      return where ? `${where}: ${item.msg}` : item.msg;
    }).join("; ");
  }
  if (detail && typeof detail === "object") return JSON.stringify(detail);
  return String(detail ?? "");
}

async function request(method, path, {json, body, params} = {}) {
  const url = `${BASE}/${path}${params ? `?${new URLSearchParams(params)}` : ""}`;
  const init = {method, headers: {}};
  if (json !== undefined) {
    init.body = JSON.stringify(json);
    init.headers["Content-Type"] = "application/json";
  } else if (body !== undefined) {
    init.body = body;
    init.headers["Content-Type"] = "application/octet-stream";
  }
  let response;
  try {
    response = await fetch(url, init);
  } catch {
    throw new ApiError("Web sunucusuna ulaşılamadı. Bağlantınızı ve sunucunun çalıştığını kontrol edin.",
      {unreachable: true});
  }
  const text = await response.text();
  let data = null;
  try {
    data = text ? JSON.parse(text) : null;
  } catch {
    data = null;
  }
  if (!response.ok) {
    const detail = describe(data?.detail ?? text) || response.statusText;
    const unreachable = Boolean(data?.unreachable);
    // 409: the GPU is busy with another task; 413 and 502-504 come from the proxy. Their messages say what to do.
    const plain = unreachable || [409, 413, 502, 503, 504].includes(response.status);
    throw new ApiError(plain ? detail : `Sunucu hatası (${response.status}): ${detail}`,
      {status: response.status, unreachable});
  }
  return data;
}

export const api = {
  health: () => request("GET", "health"),
  /** Measured speed of the server's device; empty when unknown or unreachable (estimates are optional). */
  async stats() {
    try {
      return (await request("GET", "stats")) || {};
    } catch {
      return {};
    }
  },
  submit: (payload) => request("POST", "jobs", {json: payload}),
  job: (id) => request("GET", `jobs/${encodeURIComponent(id)}`),
  jobs: (limit = 50) => request("GET", "jobs", {params: {limit}}),
  cancel: (id) => request("POST", `jobs/${encodeURIComponent(id)}/cancel`),
  /** The same file always gets the same id on the server, so its analyses stay cached. */
  upload: (file) => request("POST", "uploads", {body: file, params: {filename: file.name}}),
  /**
   * Reference analysis (Demucs, Whisper, SheetSage2) can take minutes. The web server then answers
   * {pending, task} at once and keeps waiting itself; the page asks for the result until it is there.
   */
  async analyze(params) {
    let data = await request("POST", "analyze", {json: params});
    let failures = 0;
    while (data?.pending) {
      await wait(ANALYSIS_POLL_MS);
      try {
        data = await request("GET", `analyze/${encodeURIComponent(data.task)}`);
        failures = 0;
      } catch (error) {
        // A dropped connection to the web server is retried; any answer from it (errors too) ends the wait.
        failures += 1;
        if (error.status !== 0 || failures > 10) throw error;
      }
    }
    return data;
  },
  audioUrl: (id) => `${BASE}/jobs/${encodeURIComponent(id)}/audio`,
};

/** Health of the model server before a generation; a clear message when it is not running. */
export async function ensureServer() {
  try {
    return await api.health();
  } catch (error) {
    if (error.unreachable || error.status >= 500) {
      throw new ApiError("Model sunucusuna bağlanılamadı. Sunucu sekmesinden durumu kontrol edin; yerelde "
        + "`./start_model_server.sh` (Windows'ta `start_model_server.bat`) ile başlatılır.",
        {status: error.status, unreachable: true});
    }
    throw error;
  }
}
