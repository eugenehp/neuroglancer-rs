// Renders a viewer state in real neuroglancer (built by setup.sh; UI controls
// hidden, panel borders kept) and in neuroglancer-rs, both in headless Chrome
// at 800×600, and writes the screenshots `<out>-ng.png`, `<out>-rs.png`.
//
//   [ACTIONS="move:x,y;click:x,y;key:Enter;wheel:x,y,dy"] node compare.mjs states/x.json /tmp/x [webgl][,ui] [width×height]
//
// `ui` compares the whole page, both UIs showing; otherwise neuroglancer's
// UI is hidden and neuroglancer-rs opened with `?ui=0`, so the canvases are
// compared.
//
// Needs `playwright-core` (npm i in this directory) and Playwright's
// Chromium; CHROME overrides the executable.
import { chromium } from "playwright-core";
import { readFileSync, writeFileSync } from "node:fs";
const exe = process.env.CHROME ?? process.env.HOME + "/Library/Caches/ms-playwright/chromium-1243/chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing";
const [stateFile, out, modeArg, sizeArg] = process.argv.slice(2);
const modes = (modeArg ?? "").split(",");
const [width, height] = (sizeArg ?? "800x600").split("x").map(Number);
const state = JSON.parse(readFileSync(stateFile, "utf8"));
const base = `http://127.0.0.1:${process.env.PORT ?? 8792}`;
const hash = "#!" + encodeURIComponent(JSON.stringify(state));
const browser = await chromium.launch({ executablePath: exe, headless: true, args: ["--enable-unsafe-webgpu"] });
async function shot(url, file, prepare) {
  const page = await browser.newPage({ viewport: { width, height }, deviceScaleFactor: 1 });
  const logs = [];
  if (modes.includes("webgl")) await page.addInitScript(() => Object.defineProperty(Navigator.prototype, "gpu", { get: () => undefined }));
  page.on("console", (m) => logs.push(`[${m.type()}] ${m.text()}`));
  page.on("pageerror", (e) => logs.push(`[pageerror] ${e.message}`));
  await page.goto(url, { waitUntil: "load" });
  if (prepare) await prepare(page);
  await page.waitForTimeout(5000);
  // READY=1 (remote data): then until the viewer has its data in
  // (`viewer.isReady()`, ng-web's `isReady()`), up to 90 s, and a frame or two.
  if (process.env.READY) {
    await page.waitForFunction(() => (window.viewer ? window.viewer.isReady() : window.isReady?.()), null, { timeout: 90000, polling: 250 })
      .catch(() => logs.push("[harness] not ready after 90 s"));
    await page.waitForTimeout(1000);
  }
  // ACTIONS="move:x,y;click:x,y;…": pointer input before the screenshot.
  // `key:Enter` presses a key (`key:5`, `key:Shift+Tab`); `keydown:Shift`,
  // `keyup:KeyA` hold and release one; `type:text` types text (no `;` in
  // it); `wheel:x,y,dy`; `mousedown:x,y`, `mouseup:x,y`.
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
    else if (kind === "mousedown") { await page.mouse.move(x, y); await page.mouse.down(); }
    else if (kind === "mouseup") { await page.mouse.move(x, y); await page.mouse.up(); }
    else if (kind === "wheel") { await page.mouse.move(x, y); await page.mouse.wheel(0, d); }
    await page.waitForTimeout(700);
  }
  await page.screenshot({ path: file });
  await page.close();
  return logs.filter((l) => !l.includes("favicon") && !l.includes("404"));
}
const ngLogs = await shot(`${base}/ng/index.html${hash}`, `${out}-ng.png`, async (page) => {
  await page.waitForFunction(() => window.viewer !== undefined);
  if (modes.includes("ui")) return;
  await page.evaluate(() => {
    const c = window.viewer.uiConfiguration;
    c.showUIControls.value = false;
    c.showTopBar.value = false;

    c.showLayerPanel.value = false;
    // HTML over the panels (not canvas pixels).
    const style = document.createElement("style");
    style.textContent = ".neuroglancer-display-dimensions-widget, .neuroglancer-data-panel-layout-controls, .perspective-panel-show-slice-views { display: none !important; }";
    document.head.appendChild(style);
  });
});
// `?credits=0`: without the port's credits, which neuroglancer's page has no counterpart of.
const rsLogs = await shot(`${base}/rs/index.html${modes.includes("ui") ? "?credits=0" : "?ui=0"}${hash}`, `${out}-rs.png`);
await browser.close();
writeFileSync(`${out}-logs.txt`, "neuroglancer:\n" + ngLogs.join("\n") + "\nneuroglancer-rs:\n" + rsLogs.join("\n") + "\n");
