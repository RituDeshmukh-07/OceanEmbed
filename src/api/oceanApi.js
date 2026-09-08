// Single service file for all backend calls.
// Legacy backend contract (inspected from app.py, still supported):
//   GET /api/health            -> { status, model_loaded }
//   GET /api/region             -> { grid_size, depths, sst[][], ssh[][], sss[][] }
//   GET /api/profile?x=&y=      -> { x, y, depths, predicted[], true[], rmse, correlation }
// New teammate contract (tried first):
//   POST /predict  { lat, lon, date } -> { depth_m[], temperature_c[], rmse, bias, correlation, surface }

export const BACKEND_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";
export const DEMO_MODE_ENABLED = import.meta.env.VITE_DEMO_MODE === "true";

export const REGION_BOUNDS = { latMin: 5, latMax: 30, lonMin: 45, lonMax: 105 };
export const DEPTHS = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000];

// Backend grid has no real georeferencing (synthetic demo grid) - we map
// lat/lon linearly onto its x/y index space so clicks are at least stable
// and reproducible. Flagged as a known limitation (see final notes).
export function latLonToGrid(lat, lon, gridSize) {
  const { latMin, latMax, lonMin, lonMax } = REGION_BOUNDS;
  const x = Math.round(((lon - lonMin) / (lonMax - lonMin)) * (gridSize - 1));
  const y = Math.round(((lat - latMin) / (latMax - latMin)) * (gridSize - 1));
  return {
    x: Math.min(gridSize - 1, Math.max(0, x)),
    y: Math.min(gridSize - 1, Math.max(0, y)),
  };
}

export function isInRegion(lat, lon) {
  const { latMin, latMax, lonMin, lonMax } = REGION_BOUNDS;
  return lat >= latMin && lat <= latMax && lon >= lonMin && lon <= lonMax;
}

async function getJson(path, timeoutMs = 5000) {
  const controller = new AbortController();
  const t = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(`${BACKEND_URL}${path}`, { signal: controller.signal });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await res.json();
  } finally {
    clearTimeout(t);
  }
}

export async function checkHealth() {
  return getJson("/api/health");
}

export async function fetchRegion() {
  return getJson("/api/region");
}

function num(v) {
  return typeof v === "number" && !Number.isNaN(v) ? v : null;
}

// New contract: POST /predict { lat, lon, date }
async function postPredict(lat, lon, date, timeoutMs = 8000) {
  const controller = new AbortController();
  const t = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(`${BACKEND_URL}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ lat, lon, date }),
      signal: controller.signal,
    });
    if (res.status === 404) return { notImplemented: true };
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    return {
      depths: data.depth_m ?? DEPTHS,
      predicted: data.temperature_c,
      referenceProfile: null, // new contract has no reference-curve field
      rmse: num(data.rmse),
      bias: num(data.bias),
      correlation: num(data.correlation),
      surface: data.surface ?? null,
      isDemo: false,
    };
  } finally {
    clearTimeout(t);
  }
}

// Legacy contract: GET /api/profile?x=&y=&date=
async function getLegacyProfile(lat, lon, date, gridSize) {
  const { x, y } = latLonToGrid(lat, lon, gridSize);
  const data = await getJson(`/api/profile?x=${x}&y=${y}&date=${encodeURIComponent(date)}`);
  return {
    depths: data.depths,
    predicted: data.predicted,
    referenceProfile: Array.isArray(data.true) ? data.true : null,
    rmse: num(data.rmse),
    bias: null, // legacy backend does not return bias
    correlation: num(data.correlation),
    surface: null, // legacy /api/profile does not return per-point surface vars
    isDemo: false,
  };
}

// Tries the new /predict contract first; if that route doesn't exist on this
// backend (404), falls back to the existing legacy /api/profile endpoint so
// the current backend keeps working. Both paths send the selected date.
// Throws on genuine failure - caller decides whether to enter demo mode.
export async function runPrediction(lat, lon, date, gridSize) {
  try {
    const result = await postPredict(lat, lon, date);
    if (!result.notImplemented) return result;
  } catch (e) {
    // /predict exists but errored for a real reason - surface it directly,
    // do not mask a genuine backend error by silently trying the legacy path.
    throw e;
  }
  return getLegacyProfile(lat, lon, date, gridSize);
}

// Grid-index surface lookup from /api/region (only has SST, SSH, SSS).
export function surfaceAtGrid(regionData, x, y) {
  if (!regionData) return null;
  const at = (arr) => (arr && arr[y] && typeof arr[y][x] === "number" ? arr[y][x] : null);
  return {
    SST: at(regionData.sst),
    SSS: at(regionData.sss),
    SSH: at(regionData.ssh),
    U_cur: null, // not exposed by current backend
    V_cur: null,
    U_wind: null,
    V_wind: null,
  };
}

// ---- DEMO MODE (only used when backend is unreachable) ----
function mulberry32(seed) {
  return function () {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function generateDemoProfile(lat, lon) {
  const seed = Math.round((lat * 1000 + lon) * 97);
  const rng = mulberry32(seed);
  const sst = 26 + rng() * 4;
  const ssh = (rng() - 0.5) * 0.3;
  const sss = 34.5 + rng() * 1.2;
  const deepT = 4.0;
  const center = Math.min(400, Math.max(20, 100 + ssh * 300));

  const referenceProfile = DEPTHS.map((z) => {
    const transition = 1 / (1 + Math.exp((z - center) / 30));
    return deepT + (sst - deepT) * transition + 0.05 * (sss - 35);
  });
  const predicted = referenceProfile.map((t, i) => {
    const depthFactor = i / (DEPTHS.length - 1);
    return t + (rng() - 0.5) * (0.6 + depthFactor * 2.5);
  });

  const n = DEPTHS.length;
  const rmse = Math.sqrt(
    predicted.reduce((s, v, i) => s + (v - referenceProfile[i]) ** 2, 0) / n
  );
  const bias = predicted.reduce((s, v, i) => s + (v - referenceProfile[i]), 0) / n;
  const ma = predicted.reduce((s, v) => s + v, 0) / n;
  const mb = referenceProfile.reduce((s, v) => s + v, 0) / n;
  let numerator = 0, da = 0, db = 0;
  predicted.forEach((v, i) => {
    numerator += (v - ma) * (referenceProfile[i] - mb);
    da += (v - ma) ** 2;
    db += (referenceProfile[i] - mb) ** 2;
  });
  const correlation = numerator / (Math.sqrt(da * db) || 1);

  return {
    depths: DEPTHS,
    predicted,
    referenceProfile,
    rmse,
    bias,
    correlation,
    surface: { SST: sst, SSS: sss, SSH: ssh, U_cur: null, V_cur: null, U_wind: null, V_wind: null },
    isDemo: true,
  };
}
