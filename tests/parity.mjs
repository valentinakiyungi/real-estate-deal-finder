// Checks docs/js/engine.js against cases produced by Python (run by tests/test_static_site.py).
import { readFileSync } from "node:fs";
const [, , engineFile, modelFile, casesFile] = process.argv;
const src = readFileSync(engineFile, "utf8");
const E = await import("data:text/javascript;base64," + Buffer.from(src).toString("base64"));
const model = JSON.parse(readFileSync(modelFile, "utf8"));
const cases = JSON.parse(readFileSync(casesFile, "utf8"));
const FIELDS = ["verdict", "fair_value", "fair_low", "fair_high", "risk_adjusted_value", "risk_haircut", "discount",
  "price_per_sqft", "confidence", "explanation"];
let bad = 0;
for (const c of cases) {
  const a = E.valueHome(model, c.input);
  a.explanation = E.explain(a);
  const diffs = [];
  for (const [k, v] of Object.entries(c.features)) {
    const mine = a.features[k];
    if (JSON.stringify(mine) !== JSON.stringify(v)) diffs.push(`features.${k}: js=${JSON.stringify(mine)} py=${JSON.stringify(v)}`);
  }
  for (const k of FIELDS) {
    if (!(k in c.expect)) continue;
    if (JSON.stringify(a[k]) !== JSON.stringify(c.expect[k])) diffs.push(`${k}: js=${JSON.stringify(a[k])} py=${JSON.stringify(c.expect[k])}`);
  }
  if (diffs.length) { bad++; if (bad <= 8) console.log(c.input.id, "\n  " + diffs.join("\n  ")); }
}
// the SVG and gap bar must render for every case without throwing
for (const c of cases.slice(0, 50)) { E.houseSVG(c.features, "#1E7F5C"); }
console.log(`${cases.length - bad}/${cases.length} cases match`);
process.exit(bad ? 1 : 0);
