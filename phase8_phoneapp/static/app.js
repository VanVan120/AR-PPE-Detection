/* AR Safety Monitor — phone client.
 *
 * The phone captures a frame, posts it to the laptop, and gets back the boxes as
 * NORMALISED coordinates; it draws them itself over its own live preview. The picture is
 * therefore never sent back and never stutters — only the boxes carry the round-trip lag,
 * and they are drawn faded once they are visibly out of date so a slow link looks slow
 * rather than looking wrong.
 *
 * The capture loop is single-flight: the next frame is only sent once the previous
 * response has arrived. That is the whole flow-control mechanism. Sending on a timer
 * instead would queue frames behind a busy laptop and the lag would grow without bound.
 */
"use strict";

// ---------------------------------------------------------------- the key ---
// The URL wins, then whatever was saved last, then the value baked in at serve time.
// A freshly-copied link therefore always repairs an app whose saved key has gone stale
// (the laptop's key was rotated), and the installed icon still works without one.
const BAKED = "__TOKEN__";
const urlTok = new URLSearchParams(location.search).get("t");
let T = urlTok || localStorage.getItem("ar_token") || BAKED;
try { if (T) localStorage.setItem("ar_token", T); } catch (e) { /* private mode */ }

// A stable per-device id, so the laptop can tell "the same phone reconnecting" from
// "a second phone trying to also be the camera".
let CID = null;
try { CID = localStorage.getItem("ar_cid"); } catch (e) { /* ignore */ }
if (!CID) {
  CID = Math.random().toString(36).slice(2) + Date.now().toString(36);
  try { localStorage.setItem("ar_cid", CID); } catch (e) { /* ignore */ }
}

const q = (p) => p + (p.includes("?") ? "&" : "?") + "t=" + encodeURIComponent(T);
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s).replace(/[&<>"']/g,
  (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

/* Swap which of the two media elements is showing. Both are given a `display` in the
 * stylesheet, so clearing the inline value (`style.display = ""`) falls back to the
 * stylesheet — which for `#shot` is `none`. Setting it to "" to *reveal* something
 * therefore hides it, and the glasses views would simply never appear on the phone. Both
 * values are stated explicitly here for that reason. */
function show(visibleId, hiddenId) {
  $(visibleId).style.display = "block";
  $(hiddenId).style.display = "none";
}

const SEV = { high: "#e23030", medium: "#ff9100", low: "#e8be40" };

// ------------------------------------------------------------------ state ---
const S = {
  running: false, stream: null, facing: "environment", view: "live",
  people: [], peopleAt: 0, minInterval: 90, lag: 0, rate: 0, frames: 0, tLast: 0,
  wakeLock: null, fatal: "",
};

// ------------------------------------------------------------------- tabs ---
document.querySelectorAll("[data-tab]").forEach((b) => {
  b.addEventListener("click", () => {
    document.querySelectorAll("[data-tab]").forEach((o) =>
      o.setAttribute("aria-pressed", String(o === b)));
    document.querySelectorAll(".panel").forEach((p) => p.classList.remove("on"));
    $("tab-" + b.dataset.tab).classList.add("on");
    if (b.dataset.tab === "rep") loadReport();
  });
});

// ----------------------------------------------------------------- camera ---
function camMessage(html) {
  const el = $("camwarn");
  el.style.display = html ? "block" : "none";
  el.innerHTML = html ? '<div class="warn">' + html + "</div>" : "";
}

async function startCamera() {
  if (!window.isSecureContext) {
    // Worth spelling out: this is the single most likely reason the app "does not work",
    // and the cause is the address it was opened on, not the phone or the permission.
    camMessage("<b>This page was opened over plain http.</b> Browsers only give the " +
      "camera to an https page, so the live view cannot start. Open the https link " +
      "from the laptop terminal, or use <b>Record</b> instead — that works either way.");
    return;
  }
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    camMessage("<b>This browser will not share a camera.</b> Use <b>Record</b> instead.");
    return;
  }
  camMessage("");
  $("go").disabled = true;
  try {
    S.stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: { ideal: S.facing },
               width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false,
    });
  } catch (e) {
    $("go").disabled = false;
    const why = (e && e.name) || "";
    camMessage(why === "NotAllowedError"
      ? "<b>Camera permission was refused.</b> Allow it in the browser's site settings " +
        "(the padlock/ⓘ next to the address), or use <b>Record</b> instead."
      : why === "NotFoundError"
        ? "<b>No camera found on this device.</b>"
        : why === "NotReadableError"
          ? "<b>The camera is in use by another app.</b> Close it and try again."
          : "<b>Could not start the camera.</b> " + esc(String(e && e.message || e)));
    return;
  }
  const v = $("video");
  v.srcObject = S.stream;
  try { await v.play(); } catch (e) { /* autoplay policies; the stream still renders */ }
  $("placeholder").style.display = "none";
  $("go").textContent = "Stop camera";
  $("go").disabled = false;
  S.running = true;
  keepAwake();
  loop();
}

function stopCamera() {
  S.running = false;
  if (S.stream) { S.stream.getTracks().forEach((t) => t.stop()); S.stream = null; }
  $("video").srcObject = null;
  $("go").textContent = "Start camera";
  $("placeholder").style.display = "";
  S.people = [];
  drawHud();
  releaseWake();
}

$("go").addEventListener("click", () => (S.running ? stopCamera() : startCamera()));

$("flip").addEventListener("click", async () => {
  S.facing = S.facing === "environment" ? "user" : "environment";
  if (S.running) { stopCamera(); await sleep(150); startCamera(); }
});

$("reset").addEventListener("click", async () => {
  $("reset").disabled = true;
  try {
    await fetch(q("/api/reset"), { method: "POST", cache: "no-store" });
    S.people = []; drawHud();
    renderAlerts([]); renderWorkers([]);
    $("rep").innerHTML = '<div class="empty">Nothing recorded yet.</div>';
  } catch (e) { /* shown by the next status poll */ }
  $("reset").disabled = false;
});

// Phones suspend a backgrounded tab and drop the camera track with it.
document.addEventListener("visibilitychange", () => {
  if (!document.hidden && S.running && S.stream) {
    const live = S.stream.getVideoTracks().some((t) => t.readyState === "live");
    if (!live) { stopCamera(); startCamera(); }
  }
});

async function keepAwake() {
  try {
    if ("wakeLock" in navigator) S.wakeLock = await navigator.wakeLock.request("screen");
  } catch (e) { /* not fatal — the screen just sleeps as usual */ }
}
function releaseWake() {
  if (S.wakeLock) { try { S.wakeLock.release(); } catch (e) { /* ignore */ } S.wakeLock = null; }
}
document.addEventListener("visibilitychange", () => {
  if (!document.hidden && S.running && !S.wakeLock) keepAwake();
});

// ------------------------------------------------------------ capture loop ---
const cap = document.createElement("canvas");
const capctx = cap.getContext("2d", { alpha: false });

function grab() {
  const v = $("video");
  const vw = v.videoWidth, vh = v.videoHeight;
  if (!vw || !vh) return null;
  // 640 on the long edge: the detector runs at 640 anyway, so sending more would cost
  // upload time and be thrown away at the first resize.
  const s = Math.min(1, 640 / Math.max(vw, vh));
  const w = Math.round(vw * s), h = Math.round(vh * s);
  if (cap.width !== w || cap.height !== h) { cap.width = w; cap.height = h; }
  capctx.drawImage(v, 0, 0, w, h);
  return new Promise((res) => cap.toBlob(res, "image/jpeg", 0.6));
}

async function loop() {
  while (S.running) {
    if (document.hidden) { await sleep(300); continue; }
    const t0 = performance.now();
    let blob = null;
    try { blob = await grab(); } catch (e) { /* frame not ready yet */ }
    if (!blob) { await sleep(120); continue; }

    let data = null;
    try {
      const r = await fetch(q("/api/frame") + "&cid=" + encodeURIComponent(CID) +
                            "&v=" + encodeURIComponent(S.view),
                            { method: "POST", body: blob, cache: "no-store",
                              headers: { "Content-Type": "image/jpeg" } });
      if (r.status === 403) { onForbidden(); return; }
      data = await r.json();
    } catch (e) {
      setStatus(false, "no link to laptop");
      await sleep(700);
      continue;
    }
    handleFrame(data);

    const spent = performance.now() - t0;
    // Pace to the rate the laptop told the tracker to expect. Sending much faster does
    // not make the tracker better — it makes its occlusion memory cover less real time.
    if (spent < S.minInterval) await sleep(S.minInterval - spent);
  }
}

function handleFrame(d) {
  if (!d || d.ok !== true) {
    if (d && d.starting) setStatus(false, "loading model…");
    else if (d && d.camera_busy) setStatus(false, "another phone is the camera");
    else if (d && d.busy) { /* one dropped frame; not worth showing */ }
    else if (d && d.error) { setStatus(false, "error"); S.fatal = d.error; camMessage("<b>" + esc(d.error) + "</b>"); }
    return;
  }
  S.fatal = "";
  S.people = d.people || [];
  S.peopleAt = performance.now();
  S.lag = d.ms || 0;
  S.frames++;
  const now = performance.now();
  if (S.tLast) {
    const inst = 1000 / Math.max(1, now - S.tLast);
    S.rate = S.rate ? S.rate * 0.8 + inst * 0.2 : inst;
  }
  S.tLast = now;

  if (d.view === "live") {
    show("video", "shot");
    drawHud();
  } else if (d.jpeg) {
    // The glasses views are rendered on the laptop, so here the picture itself is the
    // result; the local overlay must be cleared or the boxes would be drawn twice.
    $("shot").src = "data:image/jpeg;base64," + d.jpeg;
    show("shot", "video");
    S.people = [];
    drawHud();
  }

  $("rate").textContent = S.rate ? S.rate.toFixed(1) : "--";
  $("lag").textContent = Math.round(S.lag);
  $("nPersons").textContent = d.persons || 0;
  $("nAlerts").textContent = (d.alerts || []).length;
  $("nWorkers").textContent = (d.workers || []).length;
  renderAlerts(d.alerts || []);
  renderWorkers(d.workers || []);
  setStatus(true, (d.alerts || []).length ? "violation" : "live");
}

function onForbidden() {
  stopCamera();
  setStatus(false, "key rejected");
  try { localStorage.removeItem("ar_token"); } catch (e) { /* ignore */ }
  camMessage("<b>The laptop rejected this app's access key.</b> It was rotated or the " +
    "server was set up again. Open the fresh link printed in the laptop terminal — this " +
    "app will pick the new key up and keep working.");
}

function setStatus(ok, text) {
  $("dot").style.background = ok
    ? (text === "violation" ? "var(--high)" : "var(--ok)")
    : "var(--faint)";
  $("src").textContent = text;
}

// ---------------------------------------------------------------- overlay ---
const hud = $("hud");
const hctx = hud.getContext("2d");

function drawHud() {
  const W = hud.clientWidth, H = hud.clientHeight;
  if (!W || !H) return;
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  if (hud.width !== Math.round(W * dpr) || hud.height !== Math.round(H * dpr)) {
    hud.width = Math.round(W * dpr);
    hud.height = Math.round(H * dpr);
  }
  hctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  hctx.clearRect(0, 0, W, H);
  if (!S.people.length) return;

  // Fade the boxes as they age. They describe where people were one round trip ago; on a
  // struggling link that can be most of a second, and a crisp box in the wrong place is
  // far more misleading than a faint one.
  const age = performance.now() - S.peopleAt;
  hctx.globalAlpha = age < 400 ? 1 : Math.max(0.25, 1 - (age - 400) / 1600);

  hctx.lineWidth = 3;
  hctx.font = "600 13px -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif";
  hctx.textBaseline = "middle";

  // Draw the people at the back first, so a label that has to move ends up over someone
  // further away rather than over the person in front.
  const order = S.people.slice().sort((a, b) => a.box[3] - b.box[3]);
  const placed = [];

  for (const p of order) {
    const x = p.box[0] * W, y = p.box[1] * H;
    const w = (p.box[2] - p.box[0]) * W, h = (p.box[3] - p.box[1]) * H;
    const col = p.severity ? (SEV[p.severity] || SEV.high) : "#60c46c";
    hctx.strokeStyle = col;
    roundRect(x, y, w, h, 8);
    hctx.stroke();

    const lines = [p.label + (p.badge ? "  ▣" : "")].concat(p.violations || []);
    const wide = Math.max.apply(null, lines.map((t) => hctx.measureText(t).width));
    const bh = lines.length * 19;
    let bx = Math.max(2, Math.min(x, W - wide - 16));
    // Keep the whole stack on screen: a person at the top edge would otherwise have
    // their name drawn off the picture entirely.
    let by = y - 8 - bh;
    if (by < 4) by = Math.min(y + h + 6, H - bh - 4);
    // Nudge down past anything already drawn. Workers standing shoulder to shoulder is
    // the normal case on a site, and overlapping names render as one unreadable smear —
    // which looks like the tracker has lost its mind rather than like a layout problem.
    for (let guard = 0; guard < 12; guard++) {
      const clash = placed.find((r) => bx < r.x + r.w && bx + wide + 12 > r.x &&
                                       by < r.y + r.h && by + bh > r.y);
      if (!clash) break;
      by = clash.y + clash.h + 3;
      if (by + bh > H - 2) { by = Math.max(2, H - bh - 2); break; }
    }
    placed.push({ x: bx, y: by, w: wide + 12, h: bh });

    lines.forEach((text, i) => {
      const isName = i === 0;
      const tw = hctx.measureText(text).width;
      const ty = by + i * 19 + 10;
      hctx.fillStyle = isName ? "rgba(18,16,14,.82)" : col;
      roundRect(bx, ty - 10, tw + 12, 19, 6);
      hctx.fill();
      hctx.fillStyle = isName ? "#fff" : "#14120f";
      hctx.fillText(text, bx + 6, ty);
    });
  }
  hctx.globalAlpha = 1;
}

function roundRect(x, y, w, h, r) {
  const rr = Math.min(r, Math.abs(w) / 2, Math.abs(h) / 2);
  hctx.beginPath();
  hctx.moveTo(x + rr, y);
  hctx.arcTo(x + w, y, x + w, y + h, rr);
  hctx.arcTo(x + w, y + h, x, y + h, rr);
  hctx.arcTo(x, y + h, x, y, rr);
  hctx.arcTo(x, y, x + w, y, rr);
  hctx.closePath();
}

window.addEventListener("resize", drawHud);
window.addEventListener("orientationchange", () => setTimeout(drawHud, 250));
// Redraw between frames so the fade is visible when the link stalls, rather than the
// last boxes sitting there at full strength looking current.
setInterval(() => { if (S.running && S.view === "live") drawHud(); }, 200);

// ------------------------------------------------------------------ lists ---
function renderAlerts(list) {
  const el = $("alerts");
  if (!list.length) { el.innerHTML = '<div class="empty">All clear.</div>'; return; }
  el.innerHTML = list.map((a) =>
    '<div class="row"><span class="sev ' + esc(a.severity) + '"></span>' +
    '<span class="grow">' + esc(a.worker) + "</span>" +
    '<span class="tag">' + esc(a.label) + "</span></div>").join("");
}

function renderWorkers(list) {
  const el = $("workers");
  if (!list.length) { el.innerHTML = '<div class="empty">Nobody identified yet.</div>'; return; }
  el.innerHTML = list.map((w) => {
    const col = w.violating ? "var(--high)" : (w.present ? "var(--ok)" : "var(--faint)");
    return '<div class="row"><span class="dot" style="background:' + col + '"></span>' +
      '<span class="grow"' + (w.present ? "" : ' style="color:var(--muted)"') + ">" +
      esc(w.label) + "</span>" +
      '<span class="tag' + (w.badge ? " badge" : "") + '">' +
      (w.badge ? "badge" : "seen") + "</span></div>";
  }).join("");
}

// ------------------------------------------------------------------- view ---
document.querySelectorAll("[data-view]").forEach((b) => {
  b.addEventListener("click", () => {
    S.view = b.dataset.view;
    document.querySelectorAll("[data-view]").forEach((o) =>
      o.setAttribute("aria-pressed", String(o === b)));
    if (S.view === "live") show("video", "shot");
  });
});

// ----------------------------------------------------------------- report ---
async function loadReport() {
  try {
    const r = await fetch(q("/api/report"), { cache: "no-store" });
    if (r.status === 403) return onForbidden();
    render_report(await r.json());
  } catch (e) {
    $("rep").innerHTML = '<div class="empty">Could not reach the laptop.</div>';
  }
}

function render_report(rep) {
  const w = (rep && rep.workers) || {};
  const rows = w.per_worker || [];
  const el = $("rep");
  if (!rows.length) {
    el.innerHTML = '<div class="empty">Nothing recorded yet — start the camera.</div>';
    return;
  }
  const head = '<div class="stat" style="padding-bottom:8px">' +
    "<span><b>" + (w.workers_seen || 0) + "</b>workers</span>" +
    "<span><b>" + (w.total_violation_episodes || 0) + "</b>episodes</span>" +
    "<span><b>" + (w.total_violation_s || 0) + "</b>sec unsafe</span></div>";
  el.innerHTML = head + rows.map((r) => {
    const kinds = Object.keys(r.by_type || {}).map((k) =>
      esc(k) + " ×" + r.by_type[k]).join(", ");
    return '<div class="row"><span class="grow">' + esc(r.label) +
      (kinds ? ' <span style="color:var(--muted)">' + kinds + "</span>" : "") +
      '</span><span class="tag">' + (r.violation_s || 0) + "s</span></div>";
  }).join("");
}

$("snap").href = q("/api/snapshot");
$("raw").href = q("/api/report");

// ------------------------------------------------------- record and upload ---
$("clip").addEventListener("change", (ev) => {
  const f = ev.target.files && ev.target.files[0];
  ev.target.value = "";                     // so the same file can be picked twice
  if (f) upload(f);
});

function upWarn(html) {
  const el = $("upwarn");
  el.style.display = html ? "block" : "none";
  el.innerHTML = html ? '<div class="warn">' + html + "</div>" : "";
}

function upload(file) {
  upWarn("");
  $("reccard").style.display = "none";
  $("upstate").style.display = "block";
  $("upmsg").textContent = "Uploading…";
  $("upbar").value = 0;

  // XHR rather than fetch: only XHR reports upload progress, and a clip off a phone can
  // be a hundred megabytes over WiFi — a bar that never moves reads as a hang.
  const xhr = new XMLHttpRequest();
  xhr.open("POST", q("/api/upload") + "&name=" + encodeURIComponent(file.name || "clip"));
  xhr.upload.onprogress = (e) => {
    if (e.lengthComputable) $("upbar").value = Math.round(90 * e.loaded / e.total);
  };
  xhr.onload = () => {
    let d = {};
    try { d = JSON.parse(xhr.responseText); } catch (e) { /* handled below */ }
    if (xhr.status === 403) return onForbidden();
    if (xhr.status !== 200 || !d.job) {
      $("upstate").style.display = "none";
      upWarn("<b>Could not analyse that clip.</b> " + esc(d.error || ("HTTP " + xhr.status)));
      return;
    }
    $("upmsg").textContent = "Analysing…";
    pollJob(d.job);
  };
  xhr.onerror = () => {
    $("upstate").style.display = "none";
    upWarn("<b>Upload failed.</b> Check the phone is still on the same WiFi.");
  };
  xhr.send(file);
}

async function pollJob(jid) {
  for (;;) {
    await sleep(900);
    let d;
    try {
      const r = await fetch(q("/api/job") + "&id=" + encodeURIComponent(jid),
                            { cache: "no-store" });
      if (r.status === 403) return onForbidden();
      d = await r.json();
    } catch (e) { continue; }
    if (d.state === "running") {
      $("upbar").value = Math.max(90, 90 + Math.round(d.pct / 10));
      $("upmsg").textContent = "Analysing… " + (d.pct || 0) + "%";
      continue;
    }
    $("upstate").style.display = "none";
    if (d.state !== "done") {
      upWarn("<b>Analysis failed.</b> " + esc(d.error || d.state));
      return;
    }
    const base = q("/api/result") + "&id=" + encodeURIComponent(jid);
    $("recvid").src = base + "&f=video";
    $("recdl").href = base + "&f=video";
    $("recjson").href = base + "&f=report";
    $("recsum").textContent = d.summary || "";
    $("reccard").style.display = "";
    $("reccard").scrollIntoView({ behavior: "smooth", block: "start" });
    return;
  }
}

// ------------------------------------------------------------- status poll ---
// Runs whether or not this phone is the camera, so a second phone can watch, and so the
// pacing follows the rate the laptop told the tracker to expect.
async function pollStatus() {
  try {
    const r = await fetch(q("/api/status") + "&cid=" + encodeURIComponent(CID),
                          { cache: "no-store" });
    if (r.status === 403) return onForbidden();
    const s = await r.json();
    if (s.target_fps) S.minInterval = Math.max(60, 1000 / s.target_fps);
    // Surfaced rather than silently corrected: the fix is a flag on the laptop, and the
    // symptom (workers multiplying in the roster) otherwise gets blamed on the model.
    $("tune").style.display = s.fps_hint ? "" : "none";
    $("tune").textContent = s.fps_hint || "";
    if (s.error) { setStatus(false, "error"); camMessage("<b>" + esc(s.error) + "</b>"); return; }
    if (!s.ready) { setStatus(false, "loading model…"); return; }
    if (!S.running) {
      // Not the camera: mirror whichever phone is.
      $("nPersons").textContent = s.persons || 0;
      $("nAlerts").textContent = (s.alerts || []).length;
      $("nWorkers").textContent = (s.workers || []).length;
      renderAlerts(s.alerts || []);
      renderWorkers(s.workers || []);
      setStatus(false, s.camera_busy ? "another phone is the camera" : "camera off");
    }
    if (!s.tls && !window.isSecureContext) {
      camMessage("<b>Opened over plain http — the camera cannot start.</b> Use " +
        "<b>Record</b>, or reopen the https link from the laptop.");
    }
  } catch (e) {
    setStatus(false, "no link to laptop");
  }
}
setInterval(pollStatus, 2000);
pollStatus();

// ---------------------------------------------------------- install as app ---
let deferredInstall = null;
window.addEventListener("beforeinstallprompt", (e) => {
  e.preventDefault();
  deferredInstall = e;
  $("install").disabled = false;
  $("installhow").textContent = "Tap Install to add it to the home screen.";
});
$("install").addEventListener("click", async () => {
  if (!deferredInstall) return;
  deferredInstall.prompt();
  await deferredInstall.userChoice;
  deferredInstall = null;
  $("install").disabled = true;
});
if (window.matchMedia("(display-mode: standalone)").matches || navigator.standalone) {
  $("install").disabled = true;
  $("installhow").textContent = "Already installed — you are running the app.";
}

if ("serviceWorker" in navigator && window.isSecureContext) {
  // The token rides along so the worker's own fetches are authorised too.
  navigator.serviceWorker.register("/sw.js?t=" + encodeURIComponent(T), { scope: "/" })
    .catch(() => { /* the app works fine uninstalled */ });
}
