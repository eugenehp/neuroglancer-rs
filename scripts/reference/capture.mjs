// Dumps neuroglancer's DOM under a selector (element boxes, fonts, colours,
// text runs), after ACTIONS as in compare.mjs: the measurements the UI port
// is built from.
//
//   [ACTIONS=…] node capture.mjs states/x.json 'css selector' [depth]
import { chromium } from "playwright-core";
import { readFileSync } from "node:fs";
const exe = process.env.CHROME ?? process.env.HOME + "/Library/Caches/ms-playwright/chromium-1243/chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing";
const [stateFile, selector, depthArg] = process.argv.slice(2);
const state = JSON.parse(readFileSync(stateFile, "utf8"));
const browser = await chromium.launch({ executablePath: exe, headless: true });
const page = await browser.newPage({ viewport: { width: 1200, height: 800 }, deviceScaleFactor: 1 });
await page.goto(`http://127.0.0.1:${process.env.PORT ?? 8792}/${process.env.PAGE ?? "ng"}/index.html${process.env.PAGE === "rs" ? "?credits=0" : ""}#!` + encodeURIComponent(JSON.stringify(state)));
await page.waitForTimeout(4000);
for (const step of (process.env.ACTIONS ?? "").split(";").filter(Boolean)) {
  const i = step.indexOf(":");
  const [kind, arg] = [step.slice(0, i), step.slice(i + 1)];
  const [x, y, d] = arg.split(",").map(Number);
  if (kind === "move") await page.mouse.move(x, y);
  else if (kind === "click") await page.mouse.click(x, y);
  else if (kind === "dblclick") await page.mouse.dblclick(x, y);
  else if (kind === "rclick") await page.mouse.click(x, y, { button: "right" });
  else if (kind === "ctrlrclick") { await page.keyboard.down("Control"); await page.mouse.click(x, y, { button: "right" }); await page.keyboard.up("Control"); }
  else if (kind === "ctrlclick") { await page.keyboard.down("Control"); await page.mouse.click(x, y); await page.keyboard.up("Control"); }
  else if (kind === "key") await page.keyboard.press(arg);
  else if (kind === "type") await page.keyboard.type(arg);
  else if (kind === "keydown") await page.keyboard.down(arg);
  else if (kind === "keyup") await page.keyboard.up(arg);
  else if (kind === "wheel") { await page.mouse.move(x, y); await page.mouse.wheel(0, d); }
  await page.waitForTimeout(500);
}
const out = await page.evaluate(([selector, maxDepth, full]) => {
  const res = [];
  // FULL=1: unrounded.
  const r2 = (v) => (full ? v : Math.round(v * 100) / 100);
  function walk(e, d) {
    const r = e.getBoundingClientRect();
    const cs = getComputedStyle(e);
    if (cs.display === "none" || (r.width === 0 && r.height === 0 && cs.display !== "contents")) return;
    const runs = [];
    for (const n of e.childNodes) if (n.nodeType === 3 && n.textContent.trim()) {
      const rg = document.createRange(); rg.selectNodeContents(n); const b = rg.getBoundingClientRect();
      runs.push([n.textContent, r2(b.x), r2(b.y), r2(b.width), r2(b.height)]);
    }
    const o = { tag: e.tagName.toLowerCase(), cls: typeof e.className === "string" ? e.className : "", rect: [r.x, r.y, r.width, r.height].map(r2) };
    const pick = { font: cs.font, color: cs.color, bg: cs.backgroundColor, border: cs.borderWidth !== "0px" ? cs.border : undefined, padding: cs.padding !== "0px" ? cs.padding : undefined, margin: cs.margin !== "0px" ? cs.margin : undefined, display: cs.display, whiteSpace: cs.whiteSpace !== "normal" ? cs.whiteSpace : undefined, cursor: cs.cursor !== "auto" ? cs.cursor : undefined };
    Object.assign(o, Object.fromEntries(Object.entries(pick).filter(([, v]) => v !== undefined)));
    if (runs.length) o.runs = runs;
    if (e.tagName === "INPUT") { o.value = e.value; o.type = e.type; o.checked = e.checked; o.placeholder = e.placeholder; }
    res.push("  ".repeat(d) + JSON.stringify(o));
    if (e.tagName.toLowerCase() === "svg" || d >= maxDepth) return;
    for (const c of e.children) walk(c, d + 1);
  }
  for (const e of document.querySelectorAll(selector)) walk(e, 0);
  return res;
}, [selector, Number(depthArg ?? 30), !!process.env.FULL]);
console.log(out.join("\n"));
await browser.close();
