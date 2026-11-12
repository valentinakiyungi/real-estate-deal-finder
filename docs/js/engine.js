// DealScout engine: the Python pipeline re-implemented for the browser.
// Mirrors dealfinder/parser.py (parse_rules), model.py (FeatureEngineer), agent.py (deal rules)
// and services.py (explain). tests/test_static_site.py checks it against Python on every run.

// ------------------------------------------------------------------ parser --
const WORDS = { one: 1, two: 2, three: 3, four: 4, five: 5, six: 6, seven: 7 };
const N = "(\\d+(?:\\.\\d+)?|one|two|three|four|five|six|seven)";
const RED = [
  ["foundation issues", /foundation/],
  ["cash only / unfinanceable", /cash only|not qualify for financ/],
  ["flood zone", /flood zone/],
  ["roof end of life", /roof (?:at|near) end/],
  ["tenant occupied", /tenant/],
  ["as-is / estate sale", /as-is|estate sale/],
  ["unpermitted work", /unpermitted/],
];
// same limits as the ListingFeatures schema; out-of-range values are dropped
const LIMITS = { beds: [0, 12], baths: [0, 10], sqft: [200, 20000], lot_sqft: [0, 2000000],
  year_built: [1800, 2027], garage_spaces: [0, 6] };

const num = (s) => (s.toLowerCase() in WORDS ? WORDS[s.toLowerCase()] : parseFloat(s.replace(/,/g, "")));
const replaceAll = (t, s) => t.split(s).join(" ");

export function parseListing(text) {
  const t = String(text || "").toLowerCase();
  const f = { beds: null, baths: null, sqft: null, lot_sqft: null, year_built: null, renovated: false,
    condition: "good", garage_spaces: 0, pool: false, view: false, red_flags: [] };
  let m = t.match(/lot\s*(?:of\s*)?(\d*\.?\d+)\s*acres?/) || t.match(/(\d*\.?\d+)\s*acres?/);
  let woLot = t;
  if (m) {
    f.lot_sqft = Math.trunc(parseFloat(m[1]) * 43560);
    woLot = replaceAll(t, m[0]);
  } else {
    m = t.match(/([\d,]+)\s*(?:sq\.?\s*ft\.?|sqft|sf)\s*lot/);
    if (m) { f.lot_sqft = parseInt(m[1].replace(/,/g, ""), 10); woLot = replaceAll(t, m[0]); }
  }
  m = woLot.match(/(\d+(?:\.\d)?)k\s*sq\.?\s*ft/) || woLot.match(/([\d,]{3,})\s*(?:sq\.?\s*ft\.?|sqft|sf)\b/);
  if (m) {
    const v = m[1].replace(/,/g, "");
    f.sqft = m[0].split(/\s+/)[0].includes("k") ? Math.trunc(parseFloat(v) * 1000) : parseInt(v, 10);
  }
  m = t.match(new RegExp(N + "\\s*(?:bed(?:room)?s?|br|bdrm|bd)\\b"));
  if (m) f.beds = Math.trunc(num(m[1]));
  m = t.match(/(\d+)\s*full\s*\+\s*(\d+)\s*half/);
  if (m) f.baths = parseInt(m[1], 10) + 0.5 * parseInt(m[2], 10);
  else {
    m = t.match(new RegExp(N + "\\s*(?:bath(?:room)?s?|ba)\\b"));
    if (m) f.baths = num(m[1]);
  }
  m = t.match(/(?:built|circa|yr blt|constructed in)\s*(?:in\s*)?((?:18|19|20)\d\d)/);
  if (m) f.year_built = parseInt(m[1], 10);
  m = t.match(/(\d)\s*-?\s*car\s*gar/) || t.match(/garage\s*\((\d)\)/);
  f.garage_spaces = m ? parseInt(m[1], 10) : 0;
  f.pool = t.includes("pool");
  f.view = /\bviews?\b/.test(t);
  f.renovated = /renovat|remodel|gut reno/.test(t);
  if (/needs (?:tlc|major)|handyman|fixer/.test(t)) f.condition = "poor";
  else if (/updating needed|original kitchen|cosmetic/.test(t)) f.condition = "fair";
  else if (/immaculate|pristine|turnkey|designer/.test(t)) f.condition = "excellent";
  f.red_flags = RED.filter(([, re]) => re.test(t)).map(([k]) => k);
  for (const [k, [lo, hi]] of Object.entries(LIMITS)) {
    if (f[k] !== null && (Number.isNaN(f[k]) || f[k] < lo || f[k] > hi)) f[k] = k === "garage_spaces" ? 0 : null;
  }
  return f;
}

// ------------------------------------------------------------------- model --
function features(model, r) {
  const age = model.current_year - r.year_built;
  const sqft = r.sqft;
  const lot = r.lot_sqft;
  const beds1 = Math.max(r.beds, 1);
  const cond = model.cond_ord[r.condition];
  const hood = r.neighborhood in model.hood_te ? model.hood_te[r.neighborhood] : model.hood_global;
  const v = {
    log_sqft: Math.log(sqft), log_lot: Math.log(Math.max(lot, 500)), beds: r.beds, baths: r.baths,
    sqft_per_bed: sqft / beds1, bath_bed_ratio: r.baths / beds1, age, renovated: r.renovated ? 1 : 0,
    effective_age: r.renovated ? Math.min(age, 8) : age, condition: cond, cond_x_age: cond * Math.log1p(age),
    garage: r.garage_spaces, pool: r.pool ? 1 : 0, view: r.view ? 1 : 0, dist_center_km: r.dist_center_km,
    school_rating: r.school_rating, lot_to_sqft: lot / sqft, hood_ppsf: hood, hood_x_sqft: hood + Math.log(sqft),
  };
  return model.features.map((name) => {
    if (!(name in v)) throw new Error("unknown model feature " + name);
    return v[name];
  });
}

// Some split points equal a training value exactly (e.g. log(1500)). JavaScript's Math.log can differ from
// numpy's in the last binary digit, which would send the home down the other branch. Treat values within
// one part in a trillion of the split point as equal, as Python sees them.
const goesLeft = (v, t) => v <= t || v - t <= 1e-12 * Math.max(1, Math.abs(t));

export function score(m, x) {
  let total = m.baseline;
  for (const [feat, thr, left, right] of m.trees) {
    let i = 0;
    while (feat[i] !== -1) i = goesLeft(x[feat[i]], thr[i]) ? left[i] : right[i];
    total += thr[i];
  }
  return total;
}

// Python's round() and % formatting send exact halves to the even neighbour; match that.
export function roundEven(x, digits = 0) {
  const m = 10 ** digits, v = x * m, f = Math.floor(v);
  return (v - f === 0.5 ? (f % 2 === 0 ? f : f + 1) : Math.round(v)) / m;
}
const round3 = (v) => roundEven(v / 1000) * 1000;

export function predict(model, row) {
  const x = features(model, row);
  const q = model.conformal_q || 0;
  const p = Math.exp(score(model.point, x));
  const lo = Math.min(Math.exp(score(model.lo, x) - q), p);
  const hi = Math.max(Math.exp(score(model.hi, x) + q), p);
  return { fair_value: round3(p), fair_low: round3(lo), fair_high: round3(hi), raw: { p, lo, hi } };
}

export function haircut(model, flags) {
  const hit = new Set();
  for (const f of flags) for (const k of Object.keys(model.risk_haircut)) if (f.toLowerCase().includes(k)) hit.add(k);
  if (hit.has("cash only")) hit.delete("unfinanceable");
  let s = 0;
  for (const k of hit) s += model.risk_haircut[k];
  return Math.min(0.35, s);
}

// Same rules as DealTools._analyze_one in dealfinder/agent.py
export function valueHome(model, input) {
  const f = parseListing(input.description);
  const res = { id: input.id || "CHECK", address: input.address || "Your listing", neighborhood: input.neighborhood,
    asking_price: Math.round(input.asking_price), days_on_market: 0, features: f, description: input.description,
    notes: [] };
  const missing = ["beds", "baths", "sqft", "year_built"].filter((k) => f[k] === null);
  if (missing.length) return { ...res, verdict: "insufficient data", missing };
  const row = { ...f, neighborhood: input.neighborhood, dist_center_km: input.dist_center_km,
    school_rating: input.school_rating, lot_sqft: f.lot_sqft || 5000 };
  const est = predict(model, row);
  const hc = haircut(model, f.red_flags);
  const riskAdj = est.fair_value * (1 - hc);
  const ask = res.asking_price;
  const discount = 1 - ask / est.fair_value;
  const adj = 1 - ask / riskAdj;
  let verdict;
  if (hc === 0 && (discount >= 0.15 || (ask < est.fair_low && discount >= 0.10))) verdict = "strong deal";
  else if (hc === 0 && discount >= 0.10) verdict = "deal";
  else if (hc > 0 && adj >= 0.10) verdict = "speculative";
  else if (hc > 0 && discount >= 0.08) verdict = "value trap";
  else if (discount <= -0.10) verdict = "overpriced";
  else verdict = "fair";
  let confidence = est.fair_high / est.fair_low < 1.35 ? "high" : "medium";
  if (!(input.neighborhood in model.hood_te)) {
    confidence = "low";
    res.notes.push(`'${input.neighborhood}' is not in the training data, so location is valued at the market average`);
  }
  delete est.raw;
  return { ...res, ...est, risk_haircut: roundEven(hc, 3),
    risk_adjusted_value: round3(riskAdj), discount: roundEven(discount, 3),
    discount_vs_risk_adjusted: roundEven(adj, 3), price_per_sqft: roundEven(ask / f.sqft),
    verdict, confidence };
}

// ------------------------------------------------------------ presentation --
export function moneyShort(v) {
  if (v === null || v === undefined || Number.isNaN(v)) return "–";
  return v >= 1e6 ? `$${roundEven(v / 1e6, 2).toFixed(2)}M` : `$${roundEven(v / 1e3)}k`;
}

export const money = (v) => "$" + Math.round(v).toLocaleString("en-US");

export function explain(a) {
  if (a.verdict === "insufficient data")
    return `We couldn't find ${a.missing.join(", ")} in the description, so we can't value it.`;
  const gap = a.discount;
  const pct = (x) => `${roundEven(Math.abs(x) * 100)}%`;
  const rel = gap > 0.005 ? `${pct(gap)} below` : gap < -0.005 ? `${pct(gap)} above` : "right at";
  let s = `Asking ${moneyShort(a.asking_price)} is ${rel} our estimate of ${moneyShort(a.fair_value)} ` +
    `(likely range ${moneyShort(a.fair_low)} to ${moneyShort(a.fair_high)}).`;
  const flags = a.features.red_flags;
  if (flags.length)
    s += ` The listing discloses ${flags.join(", ")}; allowing ${pct(a.risk_haircut)} for that puts the value at ` +
      `${moneyShort(a.risk_adjusted_value)}.`;
  for (const n of a.notes || []) s += ` Note: ${n}.`;
  return s;
}

const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
export { esc };

const INK = "#1B2A33", SURVEY = "#2F5D8A";
const ok = (v) => v !== null && v !== undefined && !Number.isNaN(v);

// Front elevation drawn from the home's features (port of ui/components.house_svg)
export function houseSVG(r, accent = "#1E7F5C") {
  const beds = ok(r.beds) ? Math.trunc(r.beds) : 3;
  const sqft = ok(r.sqft) ? r.sqft : 1500;
  const year = ok(r.year_built) ? Math.trunc(r.year_built) : 1990;
  const g = ok(r.garage_spaces) ? Math.trunc(r.garage_spaces) : 0;
  const lot = ok(r.lot_sqft) ? r.lot_sqft : 5000;
  const ground = 118, storyH = 34;
  const stories = sqft >= 2000 ? 2 : 1;
  const w = 80 + Math.min(beds, 6) * 14;
  const gw = g ? 22 + 16 * Math.min(g, 3) : 0;
  const total = w + gw + (r.pool ? 42 : 0);
  const x0 = Math.round((320 - total) / 2 + 12);
  const top = ground - stories * storyH;
  const st = `stroke='${INK}' stroke-width='1.4' stroke-linejoin='round'`;
  const s = [`<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 6 320 124' role='img' aria-hidden='true'>`,
    "<rect width='320' height='140' fill='#EEF2F1'/><g stroke='#DDE5E3' stroke-width='0.6'>"];
  for (let x = 0; x <= 320; x += 16) s.push(`<line x1='${x}' y1='0' x2='${x}' y2='140'/>`);
  for (let y = 0; y <= 140; y += 16) s.push(`<line x1='0' y1='${y}' x2='320' y2='${y}'/>`);
  s.push("</g>");
  if (r.view) s.push(`<path d='M0 92 Q60 58 120 84 T240 70 T320 80' fill='none' stroke='${SURVEY}' stroke-opacity='0.45' stroke-width='1.2'/>`);
  for (let k = 0; k < (lot < 6000 ? 1 : 2); k++) {
    const [tx, tr] = k === 0 ? [x0 - 26, 11] : [x0 - 12, 8];
    s.push(`<line x1='${tx}' y1='${ground}' x2='${tx}' y2='${ground - 14}' ${st}/>` +
      `<circle cx='${tx}' cy='${ground - 14 - tr}' r='${tr}' fill='#DCE8E1' ${st}/>`);
  }
  s.push(`<rect x='${x0}' y='${top}' width='${w}' height='${stories * storyH}' fill='#F9FBFA' ${st}/>`);
  if (stories === 2) s.push(`<line x1='${x0}' y1='${top + storyH}' x2='${x0 + w}' y2='${top + storyH}' ${st} stroke-opacity='0.5'/>`);
  if (year >= 2005) s.push(`<rect x='${x0 - 4}' y='${top - 8}' width='${w + 8}' height='8' fill='#DCE4E2' ${st}/>`);
  else {
    const peak = top - 30;
    if (year < 1960) s.push(`<rect x='${Math.round(x0 + w * 0.72)}' y='${peak + 6}' width='10' height='22' fill='#DCE4E2' ${st}/>`);
    s.push(`<path d='M${x0 - 8} ${top} L${Math.round(x0 + w / 2)} ${peak} L${x0 + w + 8} ${top} Z' fill='#DCE4E2' ${st}/>`);
  }
  s.push(`<rect x='${Math.round(x0 + w / 2 - 7)}' y='${ground - 24}' width='14' height='24' fill='${accent}' ${st}/>`);
  const perRow = Math.max(2, Math.min(beds + 1, 5));
  for (let row = 0; row < stories; row++) {
    const wy = ground - storyH * (row + 1) + 9;
    for (let i = 0; i < perRow; i++) {
      const wx = x0 + (i + 0.5) * w / perRow - 7;
      if (row === 0 && Math.abs(wx + 7 - (x0 + w / 2)) < 16) continue;
      s.push(`<rect x='${Math.round(wx)}' y='${wy}' width='14' height='14' fill='#CFE0EA' ${st}/>` +
        `<line x1='${Math.round(wx + 7)}' y1='${wy}' x2='${Math.round(wx + 7)}' y2='${wy + 14}' stroke='${INK}' stroke-width='0.8'/>`);
    }
  }
  let gx = x0 + w;
  if (g) {
    s.push(`<rect x='${gx}' y='${ground - 30}' width='${gw}' height='30' fill='#F9FBFA' ${st}/>` +
      `<rect x='${gx + 6}' y='${ground - 22}' width='${gw - 12}' height='22' fill='#EEF2F1' ${st}/>`);
    for (let k = 1; k < 4; k++) {
      const y = (ground - 22 + k * 5.5).toFixed(1);
      s.push(`<line x1='${gx + 6}' y1='${y}' x2='${gx + gw - 6}' y2='${y}' stroke='${INK}' stroke-width='0.7'/>`);
    }
    gx += gw;
  }
  if (r.pool) s.push(`<rect x='${gx + 8}' y='${ground - 5}' width='34' height='5' rx='1' fill='#8FB8D8' ${st}/>`);
  s.push(`<line x1='6' y1='${ground}' x2='314' y2='${ground}' stroke='${INK}' stroke-width='1.6'/></svg>`);
  return s.join("");
}

// The signature price gap bar: our likely range as a band, the asking price as a dot.
export function gapBar(asking, low, high, fair, color) {
  if (![asking, low, high, fair].every(ok)) return "";
  const lo = Math.min(asking, low) * 0.92, hi = Math.max(asking, high) * 1.04;
  const pos = (v) => (100 * (v - lo)) / (hi - lo);
  const a = pos(asking), l = pos(low), h = pos(high), f = pos(fair);
  const anchored = (p, le, re) => (p <= 50 ? `left:${le.toFixed(1)}%;transform:translateX(-6px)`
    : `right:${(100 - re).toFixed(1)}%;transform:translateX(6px)`);
  return `<div class="gap" role="img" aria-label="Asking ${moneyShort(asking)}, our likely range ${moneyShort(low)} to ${moneyShort(high)}">` +
    `<div class="gap-track"></div><div class="gap-band" style="left:${l.toFixed(1)}%;width:${(h - l).toFixed(1)}%"></div>` +
    `<div class="gap-tick" style="left:${f.toFixed(1)}%"></div>` +
    `<div class="gap-dot" style="left:${a.toFixed(1)}%;background:${color}"></div>` +
    `<div class="gap-lab ask" style="${anchored(a, a, a)}">Asking ${moneyShort(asking)}</div>` +
    `<div class="gap-lab est" style="${anchored((l + h) / 2, l, h)}">Our range ${moneyShort(low)} to ${moneyShort(high)}</div></div>`;
}
