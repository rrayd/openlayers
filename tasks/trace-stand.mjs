#!/usr/bin/env node
import fs from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';

function parseArgs(argv) {
  const args = new Map();
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (!a.startsWith('--')) continue;
    const key = a.slice(2);
    const next = argv[i + 1];у
    if (!next || next.startsWith('--')) {
      args.set(key, true);
      continue;
    }
    args.set(key, next);
    i++;
  }
  return args;
}

function toInt(value, fallback) {
  const n = Number.parseInt(String(value), 10);
  return Number.isFinite(n) ? n : fallback;
}

function tsLabel(d = new Date()) {
  const pad2 = (n) => String(n).padStart(2, '0');
  return (
    d.getFullYear() +
    pad2(d.getMonth() + 1) +
    pad2(d.getDate()) +
    '-' +
    pad2(d.getHours()) +
    pad2(d.getMinutes()) +
    pad2(d.getSeconds())
  );
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const url = String(args.get('url') || 'http://172.26.83.229:9001/');
  const durationMs = toInt(args.get('duration'), 15000);
  const outRoot = String(args.get('out') || path.resolve('artifacts/stand-trace'));
  const cdpBase = args.get('cdp') ? String(args.get('cdp')) : '';
  const wsEndpoint = args.get('ws') ? String(args.get('ws')) : '';
  const outDir = path.join(outRoot, tsLabel());

  await fs.mkdir(outDir, {recursive: true});

  const puppeteerMod = await import('puppeteer');
  const puppeteer = puppeteerMod.default ?? puppeteerMod;

  let browser;
  if (wsEndpoint || cdpBase) {
    let endpoint = wsEndpoint;
    if (!endpoint) {
      const base = cdpBase.endsWith('/') ? cdpBase.slice(0, -1) : cdpBase;
      const versionUrl = `${base}/json/version`;
      const res = await fetch(versionUrl);
      if (!res.ok) {
        throw new Error(`CDP version fetch failed: ${res.status} ${res.statusText}`);
      }
      const json = await res.json();
      endpoint = json.webSocketDebuggerUrl;
      if (!endpoint) {
        throw new Error(`CDP response missing webSocketDebuggerUrl: ${versionUrl}`);
      }
    }
    browser = await puppeteer.connect({browserWSEndpoint: endpoint});
  } else {
    const executablePath =
      typeof puppeteer.executablePath === 'function'
        ? puppeteer.executablePath()
        : undefined;
    browser = await puppeteer.launch({
      headless: 'new',
      executablePath,
      args: ['--no-sandbox', '--disable-setuid-sandbox'],
      defaultViewport: {width: 1280, height: 720},
    });
  }

  try {
    const page = await browser.newPage();

    /** @type {Array<{type: string, text: string, ts: number}>} */
    const consoleLogs = [];
    page.on('console', (msg) => {
      consoleLogs.push({
        type: msg.type(),
        text: msg.text(),
        ts: Date.now(),
      });
    });
    page.on('pageerror', (err) => {
      consoleLogs.push({
        type: 'pageerror',
        text: String(err?.stack || err),
        ts: Date.now(),
      });
    });

    // Be explicit: we want a stable initial load, then measure steady-state animation.
    await page.goto(url, {waitUntil: 'networkidle2', timeout: 60000});
    await page.waitForTimeout(1500);

    const tracePath = path.join(outDir, 'trace.json');
    await page.tracing.start({
      path: tracePath,
      screenshots: true,
    });

    const rafStats = await page.evaluate(async ({durationMs}) => {
      // Collect frame delta stats without relying on devtools overlays.
      // This will reflect main-thread rAF cadence (jank and long frames).
      const deltas = [];
      const start = performance.now();
      let last = start;
      let maxDelta = 0;
      let over25 = 0;
      let over33 = 0;
      let over50 = 0;
      let frames = 0;

      await new Promise((resolve) => {
        function step(t) {
          const d = t - last;
          last = t;
          frames++;
          if (d > maxDelta) maxDelta = d;
          if (d > 25) over25++;
          if (d > 33.4) over33++;
          if (d > 50) over50++;
          if (deltas.length < 2000) deltas.push(d);
          if (t - start >= durationMs) {
            resolve();
            return;
          }
          requestAnimationFrame(step);
        }
        requestAnimationFrame(step);
      });

      const elapsed = performance.now() - start;
      const fps = frames > 0 ? (frames * 1000) / elapsed : 0;
      return {
        elapsedMs: elapsed,
        frames,
        fps,
        maxDeltaMs: maxDelta,
        over25,
        over33,
        over50,
        deltasMs: deltas,
      };
    }, {durationMs});

    const perfMetrics = await page.metrics();

    await page.tracing.stop();
    await page.screenshot({path: path.join(outDir, 'final.png')});

    await fs.writeFile(
      path.join(outDir, 'raf.json'),
      JSON.stringify(rafStats, null, 2),
      'utf8',
    );
    await fs.writeFile(
      path.join(outDir, 'metrics.json'),
      JSON.stringify(perfMetrics, null, 2),
      'utf8',
    );
    await fs.writeFile(
      path.join(outDir, 'console.json'),
      JSON.stringify(consoleLogs, null, 2),
      'utf8',
    );

    // Small human-readable summary.
    const summary = [
      `url: ${url}`,
      `durationMs: ${durationMs}`,
      `raf.fps: ${rafStats.fps.toFixed(1)}`,
      `raf.frames: ${rafStats.frames}`,
      `raf.maxDeltaMs: ${rafStats.maxDeltaMs.toFixed(1)}`,
      `raf.over25: ${rafStats.over25}`,
      `raf.over33: ${rafStats.over33}`,
      `raf.over50: ${rafStats.over50}`,
      `outDir: ${outDir}`,
      `trace: ${tracePath}`,
    ].join('\n');
    await fs.writeFile(path.join(outDir, 'summary.txt'), summary + '\n', 'utf8');

    // eslint-disable-next-line no-console
    console.log(summary);
  } finally {
    await browser.close();
  }
}

main().catch((err) => {
  // eslint-disable-next-line no-console
  console.error(err);
  process.exitCode = 1;
});
