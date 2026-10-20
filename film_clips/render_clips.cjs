/* render_clips.cjs — redraws the film's four clips and two stills with the website's look (web/js/look.js),
   from the recorded poses in web/data (python export_web_data.py replays), with the original clips' camera.
   Each 100-frame segment is piped straight into ffmpeg (nothing big lands on disk) and finished segments
   are skipped, so an interrupted run resumes where it stopped.

   Usage, from the repo root:
     python -m http.server 8766 --bind 127.0.0.1          # serve the repo
     PUPPETEER=/path/to/puppeteer-core CHROME=/path/to/chrome node film_clips/render_clips.cjs http://127.0.0.1:8766/film_clips OUTDIR
   Then copy OUTDIR's *_clean.mp4 into brag-output/composition/assets/video/ and its two PNGs into assets/stills/. */
const puppeteer = require(process.env.PUPPETEER || "puppeteer-core");
const CHROME = process.env.CHROME;
const fs = require("fs"), path = require("path"), { spawn, execFileSync } = require("child_process");
const [base, out, ...only] = process.argv.slice(2);
const NAMES = { untrained: "untrained_normal_clean", sac229: "sac_229k_normal_clean", sac524: "sac_524k_normal_clean", pair: "a_freezes_b_gets_past_clean" };
const STILLS = { untrained: [163, "untrained_last.png"], pair: [345, "pair_6p90.png"] };
const SEG = 100;
fs.mkdirSync(out, { recursive: true });

async function page() {
  const b = await puppeteer.launch({ executablePath: CHROME, args: ["--hide-scrollbars"] });
  const p = await b.newPage();
  await p.setViewport({ width: 3840, height: 2160, deviceScaleFactor: 1 });
  p.on("pageerror", e => console.log("pageerror", e.message));
  await p.goto(`${base}/clipgen.html`, { waitUntil: "load" });
  await p.waitForFunction(() => document.title === "ready" && window.setup);
  return { b, p };
}

(async () => {
  const { b, p } = await page();
  for (const clip of only.length ? only : Object.keys(NAMES)) {
    const total = await p.evaluate(n => window.setup(n), clip);
    const segdir = path.join(out, `${clip}_segments`); fs.mkdirSync(segdir, { recursive: true });
    const t0 = Date.now();
    for (let s0 = 0; s0 < total; s0 += SEG) {
      const seg = path.join(segdir, `${String(s0 / SEG).padStart(3, "0")}.mp4`);
      if (fs.existsSync(seg)) continue;
      const tmp = seg + ".part.mp4";
      const ff = spawn("ffmpeg", ["-loglevel", "error", "-y", "-f", "image2pipe", "-framerate", "50", "-c:v", "png", "-i", "-",
        "-c:v", "libx264", "-preset", "medium", "-crf", "12", "-pix_fmt", "yuv420p", "-r", "50", tmp], { stdio: ["pipe", "inherit", "inherit"] });
      for (let f = s0; f < Math.min(total, s0 + SEG); f++) {
        await p.evaluate(n => window.draw(n), f);
        const png = await p.screenshot({ type: "png", optimizeForSpeed: true });
        if (!ff.stdin.write(png)) await new Promise(r => ff.stdin.once("drain", r));
      }
      ff.stdin.end();
      await new Promise((res, rej) => ff.on("close", c => (c === 0 ? res() : rej(new Error(`ffmpeg ${c}`)))));
      fs.renameSync(tmp, seg);
      console.log(`${clip}: frames ${s0}-${Math.min(total, s0 + SEG) - 1} of ${total} (${((Date.now() - t0) / 1000).toFixed(0)} s)`);
    }
    const list = fs.readdirSync(segdir).filter(f => /^\d{3}\.mp4$/.test(f)).sort().map(f => `file '${path.join(segdir, f)}'`).join("\n");
    fs.writeFileSync(path.join(segdir, "list.txt"), list);
    const dest = path.join(out, `${NAMES[clip]}.mp4`);
    execFileSync("ffmpeg", ["-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", path.join(segdir, "list.txt"), "-c", "copy", "-movflags", "+faststart", dest]);
    const n = execFileSync("ffprobe", ["-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", dest]).toString().trim();
    console.log(`${clip}: ${dest} has ${n} frames (expected ${total})`);
    if (STILLS[clip]) {
      const [f, name] = STILLS[clip];
      await p.evaluate(n => window.draw(n), f);
      await p.screenshot({ path: path.join(out, name), type: "png" });
      console.log(`still ${name} = frame ${f}`);
    }
  }
  await b.close();
})().catch(e => { console.error(e); process.exit(1); });
