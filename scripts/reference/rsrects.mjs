// Dumps neuroglancer-rs's UI text runs (`uiTextRects`) after ACTIONS, as
// capture.mjs does for neuroglancer's DOM.
//
//   [ACTIONS=…] node rsrects.mjs states/x.json [filter]
import { chromium } from "playwright-core";
import { readFileSync } from "node:fs";
const exe = process.env.CHROME ?? process.env.HOME + "/Library/Caches/ms-playwright/chromium-1243/chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing";
const [stateFile, filter] = process.argv.slice(2);
const state = JSON.parse(readFileSync(stateFile, "utf8"));
const browser = await chromium.launch({ executablePath: exe, headless: true, args: ["--enable-unsafe-webgpu"] });
const page = await browser.newPage({ viewport: { width: 1200, height: 800 }, deviceScaleFactor: 1 });
await page.goto(`http://127.0.0.1:${process.env.PORT ?? 8792}/rs/index.html?credits=0#!` + encodeURIComponent(JSON.stringify(state)));
await page.waitForTimeout(5000);
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
  await page.waitForTimeout(500);
}
const rows = JSON.parse(await page.evaluate(() => window.uiTextRects()));
for (const r of rows) if (!filter || r[0].includes(filter)) console.log(JSON.stringify(r.map((v) => (typeof v === "number" && !process.env.FULL ? Math.round(v * 100) / 100 : v))));
await browser.close();
