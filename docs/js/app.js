// DealScout static site. Pages: home, deals, a home's detail page, price check, saved homes, how it works.
import * as E from "./engine.js";

// ---- settings you may want to change ----------------------------------------------------------
const CONFIG = {
  repoUrl: "https://github.com/valentinakiyungi/real-estate-deal-finder",
  liveAppUrl: "", // after deploying the Streamlit app, paste its URL here to link the AI analyst
};
const PAGE_SIZE = 12;
const DEFAULT_VERDICTS = ["strong deal", "deal", "speculative"];
const VERDICT_ORDER = ["strong deal", "deal", "speculative", "value trap", "fair", "overpriced", "insufficient data"];
const PRICE_STEPS = [0, 150000, 200000, 250000, 300000, 400000, 500000, 750000];
const WATCH_KEY = "dealscout.watch.v1";

const $app = document.getElementById("app");
const esc = E.esc;
const state = {
  market: null, model: null, modelLoading: null, lastCheck: null, shown: PAGE_SIZE, map: null,
  view: readPref("dealscout.view", "cards"), clearArmed: false,
  filters: { hoods: new Set(), maxPrice: 0, beds: 0, sort: "best", verdicts: new Set(DEFAULT_VERDICTS) },
  watch: readWatch(), compare: null,
};

// ---------------------------------------------------------------- storage (may be blocked) ---
function readPref(k, d) { try { return localStorage.getItem(k) || d; } catch { return d; } }
function writePref(k, v) { try { localStorage.setItem(k, v); } catch { /* private mode */ } }
function readWatch() { try { return JSON.parse(localStorage.getItem(WATCH_KEY)) || {}; } catch { return {}; } }
function writeWatch() { try { localStorage.setItem(WATCH_KEY, JSON.stringify(state.watch)); } catch { /* ignore */ } }

// -------------------------------------------------------------------------------- helpers ---
const verdictLabel = (v) => state.market?.verdicts[v]?.label ?? v;
const verdictColor = (v) => state.market?.verdicts[v]?.color ?? "#9AA6AC";
const pct = (x) => `${E.roundEven(Math.abs(x) * 100)}%`;
const findHome = (id) => state.market.homes.find((h) => h.id === id) || state.watch[id] ||
  (state.lastCheck?.id === id ? state.lastCheck : null);

const ICONS = {
  home: '<path d="M3 11.5 12 4l9 7.5V20h-6v-6H9v6H3z"/>',
  deals: '<path d="M3 12V4h8l10 10-8 8z"/><circle cx="7.5" cy="8.5" r="1.4"/>',
  check: '<rect x="5" y="3" width="14" height="18" rx="2"/><path d="M8.5 7.5h7M8.5 12h2M13.5 12h2M8.5 16h2M13.5 16h2"/>',
  saved: '<path d="M6 3.5h12v17l-6-4-6 4z"/>',
  how: '<circle cx="12" cy="12" r="9"/><path d="M12 11v6M12 7.6v.4"/>',
};
const icon = (n) => `<svg viewBox="0 0 24 24" aria-hidden="true">${ICONS[n]}</svg>`;

function specsHTML(h) {
  const bits = [];
  if (h.beds != null) bits.push(`${h.beds} bd`);
  if (h.baths != null) bits.push(`${h.baths} ba`);
  if (h.sqft != null) bits.push(`${Number(h.sqft).toLocaleString("en-US")} sq ft`);
  if (h.year_built != null) bits.push(`Built ${h.year_built}`);
  return `<div class="specs">${bits.map((b) => `<span>${esc(b)}</span>`).join("")}</div>`;
}

function saveButton(h, compact = true) {
  const saved = Boolean(state.watch[h.id]);
  return `<button class="btn ${compact ? "small" : ""}" data-action="save" data-id="${esc(h.id)}" aria-pressed="${saved}">` +
    `${icon("saved")}<span>${saved ? "Saved" : "Save"}</span></button>`;
}

function card(h) {
  const color = verdictColor(h.verdict);
  return `<article class="card">
    <a href="#/home/${encodeURIComponent(h.id)}" class="elev" aria-label="Details for ${esc(h.address)}">${E.houseSVG(h, color)}</a>
    <div class="card-top"><span class="badge"><i style="background:${color}"></i>${esc(verdictLabel(h.verdict))}</span>
      <span class="hood">${esc(h.neighborhood)}</span></div>
    <div class="price">${E.money(h.asking_price)}</div>
    <a class="addr" href="#/home/${encodeURIComponent(h.id)}">${esc(h.address)}</a>
    ${specsHTML(h)}
    ${E.gapBar(h.asking_price, h.fair_low, h.fair_high, h.fair_value, color)}
    ${h.red_flags ? `<div class="flags">Disclosed: ${esc(h.red_flags)}</div>` : ""}
    <div class="card-actions"><a class="btn small" href="#/home/${encodeURIComponent(h.id)}">Details</a>${saveButton(h)}</div>
  </article>`;
}

function footer() {
  const m = state.market;
  return `<footer class="foot">DealScout gives estimates, not appraisals or financial advice. Check any home with an
    inspection and recent nearby sales before you make an offer. The sample market is simulated for demonstration
    (data from ${esc(m.generated)}). <a href="${CONFIG.repoUrl}">Source code on GitHub</a>.</footer>`;
}

// ------------------------------------------------------------------------------ navigation ---
const NAV = [["#/", "home", "Home"], ["#/deals", "deals", "Deals"], ["#/check", "check", "Check a price"],
  ["#/saved", "saved", "Saved"], ["#/how", "how", "How it works"]];
const SHORT = { "Check a price": "Check", "How it works": "About" };

function renderNav(route) {
  const n = Object.keys(state.watch).length;
  const items = (short) => NAV.map(([href, key, label]) => {
    const text = (short && SHORT[label]) || label;
    const count = key === "saved" && n ? ` <span class="count">(${n})</span>` : "";
    const current = route === key || (route === "detail" && key === "deals") ? ' aria-current="page"' : "";
    return `<a href="${href}"${current}>${icon(key)}<span>${text}${count}</span></a>`;
  }).join("");
  document.querySelector(".nav-desktop").innerHTML = items(false);
  document.querySelector(".tabbar").innerHTML = items(true);
}

// ------------------------------------------------------------------------------------ home ---
function pageHome() {
  const homes = state.market.homes;
  const deals = homes.filter((h) => h.verdict === "strong deal" || h.verdict === "deal").sort((a, b) => b.discount - a.discount);
  const traps = homes.filter((h) => h.verdict === "value trap").sort((a, b) => b.discount - a.discount);
  const risky = homes.filter((h) => h.verdict === "speculative").length + traps.length;
  const ex = deals[0];
  const exColor = ex ? verdictColor(ex.verdict) : "";
  const more = deals.length > 3 ? deals.slice(1, 4) : deals.slice(0, 3);
  const steps = [
    ["Read the listing", "Messy descriptions become clean facts: bedrooms, living area, condition, and any problems the seller disclosed."],
    ["Estimate the value", `A model trained on ${state.market.training_sales.toLocaleString("en-US")} past sales prices the home and gives a likely range, not just one number.`],
    ["Account for risk", "Foundation trouble, flood zones, tenants or unpermitted work lower the value before we compare it with the asking price."],
    ["Rank the gaps", "Homes listed below the likely range, with nothing alarming disclosed, rise to the top."],
  ];
  return `<section class="hero"><div class="hero-grid">
      <div><h1>Find homes listed below what they're worth.</h1>
        <p class="lede">DealScout reads every listing, estimates what the home is worth from thousands of past sales,
        and shows you the ones priced below that, with the risks spelled out.</p>
        <div class="btn-row"><a class="btn primary" href="#/deals">Browse deals</a><a class="btn" href="#/check">Check a price</a></div></div>
      ${ex ? `<div class="card"><a class="elev" href="#/home/${encodeURIComponent(ex.id)}" aria-label="Details for ${esc(ex.address)}">${E.houseSVG(ex, exColor)}</a>
        <div class="card-top"><span class="badge"><i style="background:${exColor}"></i>${esc(verdictLabel(ex.verdict))}</span><span class="hood">${esc(ex.neighborhood)}</span></div>
        <div class="price">${E.money(ex.asking_price)}</div><a class="addr" href="#/home/${encodeURIComponent(ex.id)}">${esc(ex.address)}</a>
        ${E.gapBar(ex.asking_price, ex.fair_low, ex.fair_high, ex.fair_value, exColor)}</div>` : ""}
    </div></section>
    <p class="summary-line">Today we checked ${homes.length} homes. ${deals.length} are listed well below our estimate with
      no disclosed problems, and ${risky} look cheap but come with problems you should know about.</p>
    <h2>How it works</h2>
    <div class="steps">${steps.map(([t, d], i) => `<div class="step"><b><em>${i + 1}</em>${t}</b><span>${d}</span></div>`).join("")}</div>
    <h2>Biggest gaps right now</h2>
    ${more.length ? `<div class="grid">${more.map(card).join("")}</div>
      <p style="margin-top:1rem"><a class="btn" href="#/deals">See all ${deals.length} homes below value</a></p>` : "<p>No homes are listed below value today.</p>"}
    ${traps.length ? `<h2>Cheap for a reason</h2><div class="grid">${card(traps[0])}
      <div><p>Some listings look like bargains until you read the small print. When a seller discloses something like
      foundation trouble or a cash-only sale, we lower the value to account for it. If that explains the low price, we say
      so instead of calling it a deal.</p><p class="muted">${esc(traps[0].explanation)}</p></div></div>` : ""}
    ${footer()}`;
}

// ----------------------------------------------------------------------------------- deals ---
function filtered() {
  const f = state.filters;
  let list = state.market.homes.filter((h) =>
    (!f.hoods.size || f.hoods.has(h.neighborhood)) && (!f.maxPrice || h.asking_price <= f.maxPrice) &&
    (!f.beds || (h.beds ?? 0) >= f.beds) && f.verdicts.has(h.verdict));
  const gap = (a, b) => (b.discount ?? -9) - (a.discount ?? -9);
  const tier = (h) => VERDICT_ORDER.indexOf(h.verdict);
  const by = { best: (a, b) => tier(a) - tier(b) || gap(a, b), gap, low: (a, b) => a.asking_price - b.asking_price,
    high: (a, b) => b.asking_price - a.asking_price, new: (a, b) => a.days_on_market - b.days_on_market }[f.sort];
  return list.sort(by);
}

function activeFilterCount() {
  const f = state.filters;
  const verdictsChanged = f.verdicts.size !== DEFAULT_VERDICTS.length || DEFAULT_VERDICTS.some((v) => !f.verdicts.has(v));
  return f.hoods.size + (f.maxPrice ? 1 : 0) + (f.beds ? 1 : 0) + (verdictsChanged ? 1 : 0);
}

function pageDeals() {
  const f = state.filters;
  const hoods = [...new Set(state.market.homes.map((h) => h.neighborhood))].sort();
  const present = VERDICT_ORDER.filter((v) => state.market.homes.some((h) => h.verdict === v));
  const n = activeFilterCount();
  const open = window.matchMedia("(min-width: 900px)").matches ? " open" : "";
  return `<h1>Homes priced below what they're worth</h1>
    <p class="muted">${esc(state.market.source)}. Every home is valued by the same model you can try under Check a price.</p>
    <details class="filters"${open}><summary>Filters${n ? ` (${n} on)` : ""}</summary><div class="filters-body">
      <div><label id="lab-show">Show</label><div class="chips" role="group" aria-labelledby="lab-show">
        ${present.map((v) => `<button class="chip" data-action="verdict" data-v="${v}" aria-pressed="${f.verdicts.has(v)}"><i style="background:${verdictColor(v)}"></i>${esc(verdictLabel(v))}</button>`).join("")}</div></div>
      <div><label id="lab-hood">Neighborhood</label><div class="chips" role="group" aria-labelledby="lab-hood">
        ${hoods.map((h) => `<button class="chip" data-action="hood" data-v="${esc(h)}" aria-pressed="${f.hoods.has(h)}">${esc(h)}</button>`).join("")}</div></div>
      <div class="filter-grid">
        <div><label for="f-price">Max price</label><select id="f-price" data-filter="maxPrice">
          ${PRICE_STEPS.map((p) => `<option value="${p}"${p === f.maxPrice ? " selected" : ""}>${p ? "Up to " + E.moneyShort(p) : "Any price"}</option>`).join("")}</select></div>
        <div><label for="f-beds">Bedrooms</label><select id="f-beds" data-filter="beds">
          ${[0, 1, 2, 3, 4, 5].map((b) => `<option value="${b}"${b === f.beds ? " selected" : ""}>${b ? b + "+" : "Any"}</option>`).join("")}</select></div>
        <div><label for="f-sort">Sort by</label><select id="f-sort" data-filter="sort">
          ${[["best", "Best deals first"], ["gap", "Biggest gap"], ["low", "Lowest price"], ["high", "Highest price"], ["new", "Newest listings"]]
            .map(([v, l]) => `<option value="${v}"${v === f.sort ? " selected" : ""}>${l}</option>`).join("")}</select></div>
      </div>
      <div><button class="btn small" data-action="reset">Reset filters</button></div>
    </div></details>
    <div class="toolbar"><strong id="count"></strong>
      <div class="seg" role="group" aria-label="View">${["cards", "map", "table"].map((v) =>
        `<button data-action="view" data-v="${v}" aria-pressed="${state.view === v}">${v[0].toUpperCase() + v.slice(1)}</button>`).join("")}</div></div>
    <div id="results"></div>
    <div class="btn-row" style="margin-top:1.2rem"><button class="btn small" data-action="csv-results">Download these results (CSV)</button></div>
    ${footer()}`;
}

function renderResults() {
  const list = filtered();
  const box = document.getElementById("results");
  if (!box) return;
  document.getElementById("count").textContent = `${list.length} of ${state.market.homes.length} homes match`;
  document.querySelectorAll('[data-action="view"]').forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.v === state.view)));
  if (state.map) { state.map.remove(); state.map = null; }
  if (!list.length) {
    box.innerHTML = `<div class="empty"><p><strong>No homes match these filters.</strong></p>
      <p>Raise the price limit, pick more neighborhoods, or choose more categories under Show.</p>
      <button class="btn small" data-action="reset">Reset filters</button></div>`;
    return;
  }
  if (state.view === "map") return renderMap(box, list);
  if (state.view === "table") return renderTable(box, list);
  box.innerHTML = `<div class="grid">${list.slice(0, state.shown).map(card).join("")}</div>` +
    (list.length > state.shown ? `<div class="more"><button class="btn" data-action="more">Show more (${list.length - state.shown} left)</button></div>` : "");
}

function renderTable(box, list) {
  box.innerHTML = `<div class="scroll"><table><thead><tr><th>Address</th><th>Neighborhood</th><th>Verdict</th>
    <th class="n">Asking</th><th class="n">Our estimate</th><th class="n">Gap</th><th class="n">Beds</th><th class="n">Baths</th>
    <th class="n">Sq ft</th><th class="n">Built</th><th>Disclosed problems</th></tr></thead><tbody>
    ${list.map((h) => `<tr><td><a href="#/home/${encodeURIComponent(h.id)}">${esc(h.address)}</a></td><td>${esc(h.neighborhood)}</td>
      <td><span class="badge"><i style="background:${verdictColor(h.verdict)}"></i>${esc(verdictLabel(h.verdict))}</span></td>
      <td class="n">${E.money(h.asking_price)}</td><td class="n">${h.fair_value ? E.money(h.fair_value) : "–"}</td>
      <td class="n">${h.discount == null ? "–" : (h.discount >= 0 ? pct(h.discount) + " below" : pct(h.discount) + " above")}</td>
      <td class="n">${h.beds ?? "–"}</td><td class="n">${h.baths ?? "–"}</td><td class="n">${h.sqft ? h.sqft.toLocaleString("en-US") : "–"}</td>
      <td class="n">${h.year_built ?? "–"}</td><td>${esc(h.red_flags || "None")}</td></tr>`).join("")}
    </tbody></table></div>`;
}

let leafletPromise = null;
function loadLeaflet() {
  if (window.L) return Promise.resolve(window.L);
  if (leafletPromise) return leafletPromise;
  leafletPromise = new Promise((resolve, reject) => {
    const css = document.createElement("link");
    css.rel = "stylesheet";
    css.href = "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.css";
    document.head.appendChild(css);
    const js = document.createElement("script");
    js.src = "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.js";
    js.onload = () => resolve(window.L);
    js.onerror = () => { leafletPromise = null; reject(new Error("map library unavailable")); };
    document.head.appendChild(js);
  });
  return leafletPromise;
}

function renderMap(box, list) {
  const pins = list.filter((h) => h.lat != null && h.lon != null);
  const legend = VERDICT_ORDER.slice(0, 6).map((v) => `<span class="badge" style="font-weight:500"><i style="background:${verdictColor(v)}"></i>${esc(verdictLabel(v))}</span>`).join("");
  box.innerHTML = `<div class="legend">${legend}</div><div id="map" role="region" aria-label="Map of matching homes"></div>
    <p class="muted" style="margin-top:8px">Tap a pin for details. Pin positions in the sample market are illustrative.</p>`;
  loadLeaflet().then((L) => {
    const el = document.getElementById("map");
    if (!el || state.view !== "map") return;
    const map = L.map(el, { scrollWheelZoom: false });
    state.map = map;
    L.tileLayer("https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png", {
      subdomains: "abcd", maxZoom: 19,
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
    }).addTo(map);
    const bounds = [];
    for (const h of pins) {
      const deal = h.verdict === "strong deal" || h.verdict === "deal";
      L.circleMarker([h.lat, h.lon], { radius: deal ? 9 : 6, color: "#fff", weight: 1.5, fillColor: verdictColor(h.verdict), fillOpacity: 0.9 })
        .bindPopup(`<strong>${esc(h.address)}</strong><br>${esc(h.neighborhood)}<br>Asking ${E.money(h.asking_price)}, our estimate ${E.moneyShort(h.fair_value)}<br>${esc(verdictLabel(h.verdict))}<br><a href="#/home/${encodeURIComponent(h.id)}">View details</a>`)
        .addTo(map);
      bounds.push([h.lat, h.lon]);
    }
    if (bounds.length) map.fitBounds(bounds, { padding: [24, 24], maxZoom: 13 });
    else map.setView(state.market.map_center, 11);
  }).catch(() => {
    const el = document.getElementById("map");
    if (el) el.outerHTML = `<div class="empty"><p><strong>The map couldn't load.</strong></p><p>Check your connection, or use the cards or table view.</p></div>`;
  });
}

// ---------------------------------------------------------------------------- home detail ---
function pageDetail(id) {
  const h = findHome(id);
  if (!h) return `<a class="back" href="#/deals">Back to deals</a><div class="empty"><p><strong>We couldn't find that home.</strong></p>
    <p>It may have left the market. Browse current deals instead.</p><a class="btn primary" href="#/deals">Browse deals</a></div>`;
  const color = verdictColor(h.verdict);
  const facts = [];
  if (h.lot_sqft) facts.push(`Lot ${h.lot_sqft.toLocaleString("en-US")} sq ft`);
  if (h.condition) facts.push(`Condition: ${h.condition}`);
  if (h.renovated) facts.push("Renovated");
  facts.push(h.garage_spaces ? `${h.garage_spaces}-car garage` : "No garage");
  if (h.pool) facts.push("Pool");
  if (h.view) facts.push("View");
  if (h.price_per_sqft) facts.push(`$${h.price_per_sqft} per sq ft`);
  facts.push(`Listed ${h.days_on_market || 0} days`);
  return `<a class="back" href="${id.startsWith("CHECK") ? "#/check" : "#/deals"}">Back</a>
    <div class="detail"><div><div class="elev">${E.houseSVG(h, color)}</div></div>
    <div><div class="card-top" style="margin-top:0"><span class="badge"><i style="background:${color}"></i>${esc(verdictLabel(h.verdict))}</span>
      <span class="hood">${esc(h.neighborhood)}</span></div>
      <h1 class="price" style="font-size:2.1rem;margin:.2rem 0">${E.money(h.asking_price)}</h1>
      <div class="addr" style="font-size:1.1rem">${esc(h.address)}</div>${specsHTML(h)}
      ${E.gapBar(h.asking_price, h.fair_low, h.fair_high, h.fair_value, color)}
      <p style="margin-top:10px">${esc(h.explanation)}</p>
      <div class="metrics">
        <div class="metric"><span>Our estimate</span><b>${E.moneyShort(h.fair_value)}</b></div>
        <div class="metric"><span>Likely range</span><b>${E.moneyShort(h.fair_low)} to ${E.moneyShort(h.fair_high)}</b></div>
        <div class="metric"><span>Value after risks</span><b>${E.moneyShort(h.risk_adjusted_value)}</b></div>
        <div class="metric"><span>Confidence</span><b>${esc(h.confidence ? h.confidence[0].toUpperCase() + h.confidence.slice(1) : "n/a")}</b></div>
      </div>
      ${h.red_flags ? `<p class="flags">Disclosed problems: ${esc(h.red_flags)}</p>` : ""}
      <div class="btn-row">${saveButton(h, false)}${h.verdict === "strong deal" || h.verdict === "deal" ? "" : `<a class="btn" href="#/deals">See better deals</a>`}</div>
      <h3 style="margin-top:1.6rem">What we read from the listing</h3><ul class="facts">${facts.map((f) => `<li>${esc(f)}</li>`).join("")}</ul>
      ${h.description ? `<h3 style="margin-top:1.2rem">The listing says</h3><p class="quote">${esc(h.description)}</p>` : ""}
      <p class="muted" style="font-size:.9rem">Before you offer: order an inspection and compare three recent sales nearby.</p>
    </div></div>${footer()}`;
}

// ---------------------------------------------------------------------------- price check ---
function pageCheck() {
  const hoods = Object.keys(state.market.neighborhoods);
  const d = state.market.neighborhoods[hoods[0]];
  return `<h1>Check a price</h1>
    <p class="lede">Paste a listing and its asking price. The same model behind every deal on this site estimates what the
      home is worth, right here in your browser.</p>
    <div class="detail"><form id="check-form" novalidate>
      <div class="field"><label for="c-desc">Listing description</label><textarea id="c-desc" required>Charming 3 bed / 1 full + 1 half bath, approx 1,480 SF on a 0.21 acre lot, circa 1962. Original kitchen, solid bones. 1-car garage. Offers due Monday.</textarea></div>
      <div class="field-row"><div class="field"><label for="c-hood">Neighborhood</label><select id="c-hood">${hoods.map((h) => `<option>${esc(h)}</option>`).join("")}</select></div>
        <div class="field"><label for="c-price">Asking price ($)</label><input id="c-price" type="number" inputmode="numeric" min="10000" step="1000" value="265000" required></div></div>
      <details class="filters" style="margin:0 0 14px"><summary>Location details (optional)</summary><div class="filters-body"><div class="field-row">
        <div class="field"><label for="c-dist">Distance to centre (km)</label><input id="c-dist" type="number" inputmode="decimal" step="0.5" min="0" max="50" value="${d.dist_center_km}"></div>
        <div class="field"><label for="c-school">School rating (1 to 10)</label><input id="c-school" type="number" inputmode="decimal" step="0.5" min="1" max="10" value="${d.school_rating}"></div></div>
        <p class="muted" style="margin:0;font-size:.88rem">These start at the neighborhood's typical values.</p></div></details>
      <p id="c-error" class="flags" role="alert"></p>
      <button class="btn primary" type="submit" style="width:100%">Check this price</button>
    </form><div id="check-result">${state.lastCheck ? checkResult(state.lastCheck) :
      `<div class="empty"><p>Your result appears here: the asking price against our likely range, what we read from the
      listing, and anything that lowers the value.</p></div>`}</div></div>
    ${CONFIG.liveAppUrl ? `<div class="callout">Want the AI to read tricky listings? <a href="${CONFIG.liveAppUrl}">Open the live app with the AI analyst</a>.</div>` : ""}
    ${footer()}`;
}

function checkResult(h) {
  if (h.verdict === "insufficient data")
    return `<div class="empty"><p><strong>We can't value this one yet.</strong></p><p>${esc(h.explanation)} Add those details to the description and check again.</p></div>`;
  return `${card(h)}<p style="margin-top:12px">${esc(h.explanation)}</p>
    <div class="metrics"><div class="metric"><span>Our estimate</span><b>${E.moneyShort(h.fair_value)}</b></div>
      <div class="metric"><span>Likely range</span><b>${E.moneyShort(h.fair_low)} to ${E.moneyShort(h.fair_high)}</b></div>
      <div class="metric"><span>Value after risks</span><b>${E.moneyShort(h.risk_adjusted_value)}</b></div>
      <div class="metric"><span>Confidence</span><b>${esc(h.confidence[0].toUpperCase() + h.confidence.slice(1))}</b></div></div>`;
}

function loadModel() {
  if (state.model) return Promise.resolve(state.model);
  state.modelLoading ??= fetch("data/model.json").then((r) => {
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    return r.json();
  }).then((m) => (state.model = m)).catch((e) => { state.modelLoading = null; throw e; });
  return state.modelLoading;
}

async function runCheck(form) {
  const err = form.querySelector("#c-error");
  const btn = form.querySelector("button[type=submit]");
  const desc = form.querySelector("#c-desc").value.trim();
  const price = Number(form.querySelector("#c-price").value);
  err.textContent = "";
  if (!desc) { err.textContent = "Paste a listing description first."; return; }
  if (!(price >= 10000)) { err.textContent = "Enter an asking price of at least $10,000."; return; }
  btn.disabled = true;
  btn.textContent = state.model ? "Checking" : "Loading the price model";
  try {
    const model = await loadModel();
    const n = (state.checkCount = (state.checkCount || 0) + 1);
    const a = E.valueHome(model, { id: `CHECK-${Date.now()}`, address: `Price check ${n}`, description: desc,
      asking_price: price, neighborhood: form.querySelector("#c-hood").value,
      dist_center_km: Number(form.querySelector("#c-dist").value), school_rating: Number(form.querySelector("#c-school").value) });
    a.explanation = E.explain(a);
    const flat = { ...a, ...a.features, red_flags: a.features.red_flags.join(", "), lat: null, lon: null };
    delete flat.features;
    state.lastCheck = flat;
    const box = document.getElementById("check-result");
    box.innerHTML = checkResult(flat);
    if (window.matchMedia("(max-width: 899px)").matches) box.scrollIntoView({ behavior: "smooth", block: "start" });
  } catch {
    err.textContent = "The price model couldn't load. Check your connection and try again.";
  } finally {
    btn.disabled = false;
    btn.textContent = "Check this price";
  }
}

// ---------------------------------------------------------------------------------- saved ---
function pageSaved() {
  const homes = Object.values(state.watch);
  if (!homes.length) return `<h1>Saved homes</h1><div class="empty"><p><strong>You haven't saved any homes yet.</strong></p>
    <p>Tap Save on any home, or on a price check, to keep it here and compare homes side by side.</p>
    <a class="btn primary" href="#/deals">Browse deals</a></div>${footer()}`;
  const ids = homes.map((h) => h.id);
  if (!state.compare) state.compare = new Set(ids.slice(0, 3));
  for (const id of [...state.compare]) if (!state.watch[id]) state.compare.delete(id);
  return `<h1>Saved homes</h1><p class="muted">Saved on this device. Download the list to keep a copy elsewhere.</p>
    <div class="grid">${homes.map(card).join("")}</div>
    <h2>Compare side by side</h2>
    <div class="chips" role="group" aria-label="Homes to compare">${homes.map((h) =>
      `<button class="chip" data-action="compare" data-id="${esc(h.id)}" aria-pressed="${state.compare.has(h.id)}">${esc(h.address)}</button>`).join("")}</div>
    <div id="compare" style="margin-top:14px">${compareTable()}</div>
    <div class="btn-row" style="margin-top:1.2rem"><button class="btn small" data-action="csv-saved">Download saved homes (CSV)</button>
      <button class="btn small" data-action="clear">${state.clearArmed ? "Tap again to remove all" : "Remove all"}</button></div>
    ${footer()}`;
}

const COMPARE = [
  ["Verdict", (h) => verdictLabel(h.verdict)], ["Asking price", (h) => E.money(h.asking_price)],
  ["Our estimate", (h) => (h.fair_value ? E.money(h.fair_value) : "–")],
  ["Likely range", (h) => `${E.moneyShort(h.fair_low)} to ${E.moneyShort(h.fair_high)}`],
  ["Asking vs estimate", (h) => (h.discount == null ? "–" : h.discount >= 0 ? `${pct(h.discount)} below` : `${pct(h.discount)} above`)],
  ["Value after risks", (h) => E.moneyShort(h.risk_adjusted_value)], ["Neighborhood", (h) => h.neighborhood],
  ["Bedrooms", (h) => h.beds ?? "–"], ["Bathrooms", (h) => h.baths ?? "–"],
  ["Living area", (h) => (h.sqft ? `${h.sqft.toLocaleString("en-US")} sq ft` : "–")],
  ["Price per sq ft", (h) => (h.price_per_sqft ? `$${h.price_per_sqft}` : "–")], ["Year built", (h) => h.year_built ?? "–"],
  ["Condition", (h) => h.condition ?? "–"], ["Garage", (h) => (h.garage_spaces ? `${h.garage_spaces} car` : "None")],
  ["Pool", (h) => (h.pool ? "Yes" : "No")], ["Disclosed problems", (h) => h.red_flags || "None"],
  ["Days listed", (h) => h.days_on_market ?? 0], ["Confidence", (h) => h.confidence ?? "–"],
];

function compareTable() {
  const homes = [...state.compare].map((id) => state.watch[id]).filter(Boolean);
  if (homes.length < 2) return `<p class="muted">Pick at least two homes to compare.</p>`;
  const best = homes.reduce((a, b) => ((b.discount ?? -9) > (a.discount ?? -9) ? b : a));
  return `<div class="scroll"><table class="compare"><thead><tr><th>Home</th>${homes.map((h) => `<th>${esc(h.address)}</th>`).join("")}</tr></thead>
    <tbody><tr><td>Drawing</td>${homes.map((h) => `<td><div class="elev" style="max-width:220px">${E.houseSVG(h, verdictColor(h.verdict))}</div></td>`).join("")}</tr>
    ${COMPARE.map(([label, fn]) => `<tr><td>${label}</td>${homes.map((h) => `<td>${esc(fn(h))}</td>`).join("")}</tr>`).join("")}</tbody></table></div>
    <p style="margin-top:12px">Biggest gap: <strong>${esc(best.address)}</strong>${best.discount != null ? `, listed ${pct(best.discount)} ${best.discount >= 0 ? "below" : "above"} our estimate` : ""}${best.red_flags ? `, but it discloses ${esc(best.red_flags)}.` : " with no disclosed problems."}</p>`;
}

// ---------------------------------------------------------------------------- how it works ---
function pageHow() {
  const m = state.market.metrics;
  const imp = state.market.importances;
  const max = Math.max(...imp.map((i) => i.value));
  const meanings = [
    ["strong deal", "Asking price is at least 15% below our estimate, or below the whole likely range, and the listing discloses nothing alarming."],
    ["deal", "Asking price is at least 10% below our estimate, with nothing alarming disclosed."],
    ["speculative", "The listing discloses problems, but it's still at least 10% below the value after we allow for them. Worth a look with careful checks."],
    ["value trap", "Looks cheap, but the disclosed problems explain the price."],
    ["fair", "Within 10% of our estimate either way."], ["overpriced", "Asking price is more than 10% above our estimate."],
  ];
  return `<h1>How the estimates work</h1>
    <p class="lede">DealScout combines three pieces: a parser that reads listing text, a price model trained on past sales,
      and rules that compare the two and rank the results.</p>
    <div class="metrics">
      <div class="metric"><span>Typical error</span><b>${(m.MAPE * 100).toFixed(1)}%</b></div>
      <div class="metric"><span>Simple price-per-foot error</span><b>${(m.baseline_ppsf_MAPE * 100).toFixed(1)}%</b></div>
      <div class="metric"><span>Within 10% of the sale price</span><b>${Math.round(m.within_10pct * 100)}%</b></div>
      <div class="metric"><span>Sales inside our range</span><b>${Math.round(m.interval_80_coverage * 100)}%</b></div></div>
    <p class="muted">Measured on ${m.test_rows.toLocaleString("en-US")} sales the model never saw. Our likely range is built to
      contain the real price 80% of the time.</p>
    <h2>What the labels mean</h2><div class="labels">${meanings.map(([v, t]) =>
      `<p><span class="badge"><i style="background:${verdictColor(v)}"></i>${esc(verdictLabel(v))}</span><br><span class="muted">${t}</span></p>`).join("")}</div>
    <h2>How we allow for risk</h2>
    <p>When a listing discloses a problem, we lower its value before comparing: foundation issues 12%, cash-only or
      unfinanceable 8%, flood zone 7%, tenant in place 5%, sold as-is 4%, unpermitted work 4%, roof at end of life 3%,
      up to 35% in total.</p>
    <h2>What drives price in the model</h2>
    <div class="bars">${imp.map((i) => `<div class="bar"><span>${esc(i.factor)}</span><div style="width:${(100 * i.value / max).toFixed(1)}%"
      role="img" aria-label="${esc(i.factor)}: ${i.value.toFixed(3)}"></div></div>`).join("")}</div>
    <p class="muted" style="margin-top:8px">Longer bars: the estimate gets worse when the model can't use that factor.</p>
    <h2>Under the hood</h2>
    <p>A gradient-boosted tree model on log price, with engineered features such as effective age, condition for age and a
      neighborhood price level. Two more models give the likely range, calibrated so it holds 80% of real prices. On this
      site the trees run in your browser, so price checks never leave your device. The full project, including an AI
      analyst that reads listings with Gemini, is <a href="${CONFIG.repoUrl}">on GitHub</a>${CONFIG.liveAppUrl ? `, and you can try it in the <a href="${CONFIG.liveAppUrl}">live app</a>` : ""}.</p>
    <h2>Limits</h2>
    <p>Estimates come from ${state.market.training_sales.toLocaleString("en-US")} sales in a simulated market. Listings in
      neighborhoods the model hasn't seen get a low confidence rating. Nothing here replaces an inspection, an appraisal or local advice.</p>
    ${footer()}`;
}

// ----------------------------------------------------------------------------- CSV export ---
function downloadCSV(rows, name) {
  const cols = ["id", "address", "neighborhood", "verdict", "asking_price", "fair_value", "fair_low", "fair_high", "discount",
    "risk_adjusted_value", "beds", "baths", "sqft", "year_built", "condition", "red_flags", "days_on_market", "confidence"];
  const cell = (v) => { const s = String(v ?? ""); return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s; };
  const csv = [cols.join(","), ...rows.map((r) => cols.map((c) => cell(r[c])).join(","))].join("\n");
  const a = Object.assign(document.createElement("a"), { href: URL.createObjectURL(new Blob([csv], { type: "text/csv" })), download: name });
  document.body.appendChild(a);
  a.click();
  setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 0);
}

// --------------------------------------------------------------------------------- router ---
function route() {
  const hash = location.hash.replace(/^#\/?/, "");
  const [page, arg] = hash.split("/");
  return { page: page || "home", arg: arg ? decodeURIComponent(arg) : null };
}

const TITLES = { home: "DealScout: homes listed below what they're worth", deals: "Deals | DealScout",
  check: "Check a price | DealScout", saved: "Saved homes | DealScout", how: "How it works | DealScout", detail: "DealScout" };

function render({ keepScroll = false } = {}) {
  const { page, arg } = route();
  if (state.map) { state.map.remove(); state.map = null; }
  const key = page === "home" && arg ? "detail" : page;
  const views = { home: pageHome, deals: pageDeals, check: pageCheck, saved: pageSaved, how: pageHow };
  let html;
  if (key === "detail") html = pageDetail(arg);
  else html = (views[key] || pageHome)();
  const y = window.scrollY;
  $app.innerHTML = html;
  renderNav(views[key] || key === "detail" ? key : "home");
  document.title = key === "detail" && findHome(arg) ? `${findHome(arg).address} | DealScout` : TITLES[key] || TITLES.home;
  if (key === "deals") renderResults();
  if (keepScroll) window.scrollTo(0, y);
  else { window.scrollTo(0, 0); $app.focus({ preventScroll: true }); }
}

// ------------------------------------------------------------------------------- events ---
function updateSaveButtons(id) {
  const saved = Boolean(state.watch[id]);
  document.querySelectorAll(`[data-action="save"][data-id="${CSS.escape(id)}"]`).forEach((b) => {
    b.setAttribute("aria-pressed", String(saved));
    b.querySelector("span").textContent = saved ? "Saved" : "Save";
  });
  renderNav(route().page === "home" && route().arg ? "detail" : route().page);
}

document.addEventListener("click", (e) => {
  const el = e.target.closest("[data-action]");
  if (!el) return;
  const { action, v, id } = el.dataset;
  const f = state.filters;
  if (action === "save") {
    const h = findHome(id);
    if (!h) return;
    if (state.watch[id]) delete state.watch[id];
    else state.watch[id] = h;
    writeWatch();
    if (route().page === "saved") render({ keepScroll: true });
    else updateSaveButtons(id);
  } else if (action === "verdict" || action === "hood") {
    const set = action === "verdict" ? f.verdicts : f.hoods;
    set.has(v) ? set.delete(v) : set.add(v);
    el.setAttribute("aria-pressed", String(set.has(v)));
    state.shown = PAGE_SIZE;
    updateFilterSummary();
    renderResults();
  } else if (action === "view") {
    state.view = v;
    writePref("dealscout.view", v);
    renderResults();
  } else if (action === "more") {
    state.shown += PAGE_SIZE;
    renderResults();
  } else if (action === "reset") {
    state.filters = { hoods: new Set(), maxPrice: 0, beds: 0, sort: "best", verdicts: new Set(DEFAULT_VERDICTS) };
    state.shown = PAGE_SIZE;
    render({ keepScroll: true });
  } else if (action === "compare") {
    if (state.compare.has(id)) state.compare.delete(id);
    else if (state.compare.size < 4) state.compare.add(id);
    document.querySelectorAll('[data-action="compare"]').forEach((b) => b.setAttribute("aria-pressed", String(state.compare.has(b.dataset.id))));
    document.getElementById("compare").innerHTML = compareTable();
  } else if (action === "csv-results") {
    downloadCSV(filtered(), "dealscout_results.csv");
  } else if (action === "csv-saved") {
    downloadCSV(Object.values(state.watch), "dealscout_saved.csv");
  } else if (action === "clear") {
    if (state.clearArmed) { state.watch = {}; writeWatch(); state.compare = null; state.clearArmed = false; }
    else { state.clearArmed = true; setTimeout(() => { state.clearArmed = false; }, 4000); }
    render({ keepScroll: true });
  }
});

function updateFilterSummary() {
  const s = document.querySelector(".filters summary");
  const n = activeFilterCount();
  if (s) s.textContent = `Filters${n ? ` (${n} on)` : ""}`;
}

document.addEventListener("change", (e) => {
  const key = e.target.dataset?.filter;
  if (!key) return;
  state.filters[key] = key === "sort" ? e.target.value : Number(e.target.value);
  state.shown = PAGE_SIZE;
  updateFilterSummary();
  renderResults();
});

document.addEventListener("change", (e) => {
  if (e.target.id !== "c-hood") return;
  const d = state.market.neighborhoods[e.target.value];
  if (!d) return;
  document.getElementById("c-dist").value = d.dist_center_km;
  document.getElementById("c-school").value = d.school_rating;
});

document.addEventListener("submit", (e) => {
  if (e.target.id !== "check-form") return;
  e.preventDefault();
  runCheck(e.target);
});

window.addEventListener("hashchange", () => render());

// ---------------------------------------------------------------------------------- start ---
fetch("data/market.json")
  .then((r) => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json(); })
  .then((m) => { state.market = m; render({ keepScroll: true }); })
  .catch(() => {
    $app.innerHTML = `<div class="empty" style="margin-top:2rem"><p><strong>Today's listings couldn't load.</strong></p>
      <p>Check your connection and reload the page.</p><button class="btn" onclick="location.reload()">Reload</button></div>`;
  });
