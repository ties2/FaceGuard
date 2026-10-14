// FaceGuard reference capture client.
// In production the FaceTec SDK owns the camera; this client shows what
// FaceGuard needs *in addition*: the nonce challenge, frame timing and
// device telemetry. Nothing here is trusted by the server.
(() => {
  const API = "";
  const $ = (id) => document.getElementById(id);
  const status = (t) => ($("status").textContent = t);
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

  async function sha256(text) {
    const buf = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(text));
    return [...new Uint8Array(buf)].map((b) => b.toString(16).padStart(2, "0")).join("").slice(0, 32);
  }

  async function deviceId() {
    const fp = [navigator.userAgent, screen.width, screen.height, navigator.hardwareConcurrency, Intl.DateTimeFormat().resolvedOptions().timeZone].join("|");
    return sha256(fp);
  }

  function telemetry(track, video, nDevices) {
    const s = track.getSettings();
    return {
      platform: "web",
      camera_label: track.label,
      track_settings: { width: s.width, height: s.height, frameRate: Math.round(s.frameRate || 0) },
      frame_size: [video.videoWidth, video.videoHeight],
      media_devices_count: nDevices,
      navigator_webdriver: !!navigator.webdriver,
      user_agent: navigator.userAgent,
    };
  }

  function grab(video, canvas) {
    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL("image/jpeg", 0.9).split(",")[1];
  }

  function renderResult(r, tel) {
    const d = $("decision");
    d.textContent = r.decision.replace("_", " ");
    d.className = "decision " + r.decision;
    const colors = { pass: "var(--ok)", review: "var(--warn)", reject: "var(--bad)", "n/a": "var(--line)" };
    $("layers").innerHTML = r.layers
      .map((l) => {
        const v = l.score == null ? (l.band === "pass" ? 0 : 1) : l.score;
        const chips = (l.reasons || []).map((c) => `<span class="chip${c.includes("=") ? " info" : ""}">${c}</span>`).join("");
        return `<div class="layer"><span>${l.name}</span><div class="bar"><i style="width:${Math.max(2, v * 100)}%;background:${colors[l.band]}"></i></div><span class="muted">${l.score == null ? l.band : l.score.toFixed(3)}</span></div>${chips ? `<div class="chips">${chips}</div>` : ""}`;
      })
      .join("");
    $("reasons").innerHTML = r.reasons.length ? r.reasons.map((x) => `<li>${x}</li>`).join("") : '<li class="muted">no findings</li>';
    $("telemetry").innerHTML = Object.entries(tel)
      .filter(([k]) => k !== "user_agent")
      .map(([k, v]) => `<tr><td>${k}</td><td>${typeof v === "object" ? JSON.stringify(v) : v}</td></tr>`)
      .join("");
  }

  async function run() {
    $("start").disabled = true;
    try {
      status("requesting camera…");
      const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720, frameRate: 30, facingMode: "user" }, audio: false });
      const video = $("preview");
      video.srcObject = stream;
      $("flashVideo").srcObject = stream;
      await new Promise((r) => (video.readyState >= 2 ? r() : (video.onloadeddata = r)));
      const track = stream.getVideoTracks()[0];
      const nDevices = (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === "videoinput").length;

      status("opening session…");
      const user = localStorage.getItem("fg_user") || "demo-" + Math.random().toString(36).slice(2, 10);
      try { localStorage.setItem("fg_user", user); } catch (e) { /* private mode */ }
      const sres = await fetch(API + "/v1/liveness/sessions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ user_ref: user, device_id: await deviceId() }),
      });
      if (!sres.ok) throw new Error("session: " + (await sres.text()));
      const s = await sres.json();
      const ch = s.challenge;
      const perSlot = Math.round(ch.n_frames / ch.colors.length);
      const interval = 1000 / ch.fps;

      // countdown on neutral grey, then the challenge
      const flash = $("flash");
      flash.style.display = "flex";
      flash.style.background = "#808080";
      for (let i = 3; i > 0; i--) { status(`challenge in ${i}…`); await sleep(400); }

      const canvas = document.createElement("canvas");
      canvas.width = 640; canvas.height = Math.round(640 * video.videoHeight / video.videoWidth);
      const frames = [], ts = [];
      const t0 = performance.now();
      for (let f = 0; f < ch.n_frames; f++) {
        flash.style.background = ch.colors[Math.floor(f / perSlot)];
        const due = t0 + f * interval;
        const wait = due - performance.now();
        if (wait > 0) await sleep(wait);
        frames.push(grab(video, canvas));
        ts.push(performance.now() - t0);
      }
      flash.style.display = "none";

      status("verifying…");
      const tel = telemetry(track, video, nDevices);
      const body = {
        session_token: s.session_token,
        frames,
        timestamps_ms: ts,
        telemetry: tel,
        vendor: { face_scan: "demo-facescan-" + s.session_id, audit_trail_image: frames[Math.floor(frames.length / 2)] },
      };
      const vres = await fetch(API + `/v1/liveness/sessions/${s.session_id}/verify`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const r = await vres.json();
      if (!vres.ok) throw new Error(JSON.stringify(r.detail || r));
      renderResult(r, tel);
      $("audit").src = "data:image/jpeg;base64," + body.vendor.audit_trail_image;
      $("audit").style.display = "block";
      video.style.display = "none";
      status(`done in ${Math.round(r.latency_ms)} ms server time`);
      stream.getTracks().forEach((t) => t.stop());
    } catch (e) {
      status("error: " + e.message);
      $("flash").style.display = "none";
    } finally {
      $("start").disabled = false;
    }
  }

  $("start").addEventListener("click", run);
})();
