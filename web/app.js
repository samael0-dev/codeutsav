/**
 * BilletVision Operator Dashboard
 * Vanilla JS — no build step, no framework.
 *
 * API used:
 *   WS  /events                        — inspection_result | kpi_update | alert
 *   GET /api/health
 *   GET /api/stats
 *   GET /api/log?n=&status=
 *   GET /api/log/count
 *   GET /api/tolerances
 *   PUT /api/tolerances/{profile}       body: {updates:{}}
 *   PUT /api/tolerances/{profile}/activate
 *   GET /api/review/pending
 *   POST /api/review/{seq}              body: {action, notes}
 *   GET /api/alerts
 *   GET /api/alerts/active
 *   POST /api/alerts/clear
 *   GET  /api/source                    — what is being analysed + progress
 *   POST /api/source/{video|image}?filename=   raw file body
 *   POST /api/source/demo | /api/source/reset
 */

(function () {
  "use strict";

  // Re-attach the MJPEG stream if it drops (pipeline restart / network blip)
  const _liveImg = document.getElementById("live-stream");
  /** Open a fresh MJPEG connection. A cut stream often fires no error event, so this is also
   *  called whenever the server restarts, the WebSocket reconnects or the source changes. */
  function _reattachStream() {
    if (_liveImg) _liveImg.src = `/video?t=${Date.now()}`;
  }
  if (_liveImg) {
    _liveImg.onerror = () => setTimeout(_reattachStream, 1000);
  }

  /* ── icons (Lucide sprite is inlined in index.html) ─────────────────── */
  const _icon = (name, cls = "") =>
    `<svg class="icon ${cls}" aria-hidden="true"><use href="#i-${name}"/></svg>`;
  const _STATUS_ICON = {
    PASS: "circle-check", FAIL: "circle-x", REWORK: "wrench", REVIEW: "eye", IDLE: "clock",
  };
  function _setStatusIcon(status) {
    const el = document.getElementById("status-icon");
    if (el) el.innerHTML = _icon(_STATUS_ICON[(status || "IDLE").toUpperCase()] || "alert", "icon-xl");
  }

  /* ── theme toggle (light / dark, persisted per browser) ─────────────── */
  function _initTheme() {
    document.getElementById("theme-toggle")?.addEventListener("click", () => {
      const next = document.documentElement.getAttribute("data-theme") === "dark" ? "light" : "dark";
      document.documentElement.setAttribute("data-theme", next);
      try { localStorage.setItem("bv-theme", next); } catch (_) {}
    });
  }

  /* Colour the stream pill's dot from the status text */
  function _initStreamPill() {
    const label = document.getElementById("stream-status");
    const pill = document.getElementById("stream-pill");
    if (!label || !pill) return;
    const sync = () => {
      const t = label.textContent.toLowerCase();
      pill.dataset.state = t === "streaming" ? "ok" : (t.includes("lost") || t === "offline") ? "bad" : "";
    };
    new MutationObserver(sync).observe(label, { childList: true, characterData: true, subtree: true });
    sync();
  }

  /* ── custom dropdowns ───────────────────────────────────────────────────
   * Each native <select> stays in the DOM (hidden) as the source of truth, so
   * existing code that reads .value / listens for "change" keeps working. The
   * menu is position:fixed so it is never clipped by a card's overflow. */
  let _openDd = null;

  function _closeDropdown() {
    if (!_openDd) return;
    _openDd.menu.classList.add("hidden");
    _openDd.trigger.setAttribute("aria-expanded", "false");
    _openDd = null;
  }

  function _enhanceSelect(sel) {
    const wrap = document.createElement("div");
    wrap.className = "dd" + (sel.classList.contains("field-xs") ? " dd-xs" : "");
    const trigger = document.createElement("button");
    trigger.type = "button";
    trigger.className = "dd-trigger";
    trigger.setAttribute("aria-haspopup", "listbox");
    trigger.setAttribute("aria-expanded", "false");
    trigger.setAttribute("aria-label", sel.getAttribute("aria-label") || "");
    trigger.innerHTML = `<span class="dd-label"></span>${_icon("chevron-down", "dd-chevron")}`;
    const menu = document.createElement("ul");
    menu.className = "dd-menu hidden";
    menu.setAttribute("role", "listbox");
    sel.parentNode.insertBefore(wrap, sel);
    wrap.append(sel, trigger);
    document.body.appendChild(menu);
    sel.classList.add("dd-native");
    sel.tabIndex = -1;
    sel.setAttribute("aria-hidden", "true");

    let active = -1;
    const opts = () => Array.from(menu.children);
    const setActive = (i) => {
      const list = opts();
      if (!list.length) return;
      active = (i + list.length) % list.length;
      list.forEach((li, k) => li.classList.toggle("active", k === active));
      list[active].scrollIntoView({ block: "nearest" });
    };

    const sync = () => {
      const cur = sel.options[sel.selectedIndex];
      trigger.querySelector(".dd-label").textContent = cur ? cur.textContent : "—";
      menu.innerHTML = Array.from(sel.options).map((o, i) =>
        `<li class="dd-option${i === sel.selectedIndex ? " selected" : ""}" role="option" data-i="${i}"
             aria-selected="${i === sel.selectedIndex}"><span>${_esc(o.textContent)}</span>${_icon("check", "dd-check")}</li>`).join("");
    };

    const place = () => {
      const r = trigger.getBoundingClientRect();
      menu.style.minWidth = `${Math.max(r.width, 140)}px`;
      const h = menu.offsetHeight;
      const below = window.innerHeight - r.bottom;
      menu.style.top = `${below < h + 8 && r.top > h + 8 ? r.top - h - 4 : r.bottom + 4}px`;
      if (sel.dataset.align === "right") { menu.style.left = "auto"; menu.style.right = `${window.innerWidth - r.right}px`; }
      else { menu.style.right = "auto"; menu.style.left = `${r.left}px`; }
    };

    const open = () => {
      _closeDropdown();
      sync();
      menu.classList.remove("hidden");
      trigger.setAttribute("aria-expanded", "true");
      place();
      setActive(Math.max(0, sel.selectedIndex));
      _openDd = { menu, trigger };
    };

    const choose = (i) => {
      if (i >= 0 && i !== sel.selectedIndex) {
        sel.selectedIndex = i;
        sel.dispatchEvent(new Event("change", { bubbles: true }));
      }
      sync();
      _closeDropdown();
      trigger.focus();
    };

    trigger.addEventListener("click", () => (_openDd && _openDd.menu === menu ? _closeDropdown() : open()));
    trigger.addEventListener("keydown", (e) => {
      const isOpen = _openDd && _openDd.menu === menu;
      if (["ArrowDown", "ArrowUp", "Enter", " "].includes(e.key) && !isOpen) { e.preventDefault(); open(); return; }
      if (!isOpen) return;
      if (e.key === "ArrowDown") { e.preventDefault(); setActive(active + 1); }
      else if (e.key === "ArrowUp") { e.preventDefault(); setActive(active - 1); }
      else if (e.key === "Home") { e.preventDefault(); setActive(0); }
      else if (e.key === "End") { e.preventDefault(); setActive(-1); }
      else if (e.key === "Enter" || e.key === " ") { e.preventDefault(); choose(active); }
      else if (e.key === "Escape" || e.key === "Tab") { e.stopPropagation(); _closeDropdown(); }
    });
    menu.addEventListener("mousedown", (e) => e.preventDefault());   // keep focus on the trigger
    menu.addEventListener("click", (e) => {
      const li = e.target.closest(".dd-option");
      if (li) choose(+li.dataset.i);
    });
    menu.addEventListener("mousemove", (e) => {
      const li = e.target.closest(".dd-option");
      if (li && +li.dataset.i !== active) setActive(+li.dataset.i);
    });

    // Options filled or value set from code (e.g. the tolerance profile list)
    new MutationObserver(sync).observe(sel, { childList: true, subtree: true, characterData: true });
    sel.addEventListener("change", sync);
    sync();
  }

  function _initDropdowns() {
    document.querySelectorAll("select.field").forEach(_enhanceSelect);
    document.addEventListener("mousedown", (e) => {
      if (_openDd && !_openDd.menu.contains(e.target) && !_openDd.trigger.contains(e.target)) _closeDropdown();
    });
    document.addEventListener("keydown", (e) => { if (e.key === "Escape") _closeDropdown(); });
    window.addEventListener("resize", _closeDropdown);
    window.addEventListener("scroll", (e) => {
      if (_openDd && !_openDd.menu.contains(e.target)) _closeDropdown();
    }, true);
  }

  /* ── state ──────────────────────────────────────────────────────────── */
  let _lastRecord = null;         // most-recent inspection result
  let _lastSeenSeq = -1;          // sequence deduplication guard
  let _logRows = [];              // current visible log rows (filtered)
  let _allRows = [];              // full fetched list (pre-filter)
  let _tolerances = {};           // {profile: {key:val}}
  let _activeProfile = null;

  /* ── audio (Web Audio API — no file needed) ─────────────────────────── */
  let _audioCtx = null;
  function _beep(frequency = 880, duration = 0.25, type = "square") {
    try {
      if (!_audioCtx) _audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      const osc  = _audioCtx.createOscillator();
      const gain = _audioCtx.createGain();
      osc.connect(gain);
      gain.connect(_audioCtx.destination);
      osc.type = type;
      osc.frequency.value = frequency;
      gain.gain.setValueAtTime(0.25, _audioCtx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.001, _audioCtx.currentTime + duration);
      osc.start(_audioCtx.currentTime);
      osc.stop(_audioCtx.currentTime + duration);
    } catch (_) { /* audio blocked — no crash */ }
  }
  function _alarmBeep()  { _beep(440, 0.4, "sawtooth"); }
  function _passBeep()   { _beep(1046, 0.12, "sine"); }

  /* ── voice alerts (browser speechSynthesis — no dependency) ─────────── */
  let _voiceOn = true;
  try { _voiceOn = localStorage.getItem("bv-voice") !== "off"; } catch (_) {}

  function _initVoice() {
    const btn = document.getElementById("voice-toggle");
    if (!btn) return;
    if (!("speechSynthesis" in window)) { btn.classList.add("hidden"); return; }
    const sync = () => {
      btn.setAttribute("aria-pressed", String(_voiceOn));
      btn.title = `Voice alerts: ${_voiceOn ? "on" : "off"}`;
    };
    sync();
    btn.addEventListener("click", () => {
      _voiceOn = !_voiceOn;
      if (!_voiceOn) window.speechSynthesis.cancel();
      try { localStorage.setItem("bv-voice", _voiceOn ? "on" : "off"); } catch (_) {}
      sync();
    });
  }

  function _speak(text) {
    if (!_voiceOn || !text) return;
    try {
      const synth = window.speechSynthesis;
      synth.cancel();   // latest billet wins; never let a backlog build up
      const u = new SpeechSynthesisUtterance(text);
      u.rate = 1.05;
      synth.speak(u);
    } catch (_) { /* speech unavailable — no crash */ }
  }

  /** Short spoken phrase for one reason, e.g. "width over tolerance". */
  function _spokenReason(reason) {
    const s = String(reason);
    const label = (s.match(/^[^\d(<>]*/)[0] || "").replace(/_/g, " ").trim();
    const range = s.match(/^[^\d]*?(-?[\d.]+)\s*\w*\s+out of range \((-?[\d.]+)/);
    if (range) return `${label} ${+range[1] > +range[2] ? "over" : "under"} tolerance`;
    if (/>\s*limit/.test(s)) return `${label} over limit`;
    return label || s;
  }

  function _speakVerdict(msg) {
    const who = msg.billet_seq != null ? `Billet ${msg.billet_seq}` : "Billet";
    const first = _reasons(msg)[0];
    const why = msg.status === "REVIEW" ? "needs review" : (first ? _spokenReason(first) : "");
    _speak(`${who} ${msg.status}${why ? ", " + why : ""}.`);
  }

  /* ── WebSocket ──────────────────────────────────────────────────────── */
  const _wsUrl = (() => {
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    return `${proto}//${location.host}/events`;
  })();

  let _ws = null;
  let _wsWasClosed = false;
  let _wsRetry = 1000;   // ms, doubles on failure up to 16 s

  function _connectWs() {
    _ws = new WebSocket(_wsUrl);

    _ws.onopen = () => {
      _wsRetry = 1000;
      _setWsStatus(true);
      if (_wsWasClosed) { _wsWasClosed = false; _reattachStream(); _fetchAlerts(); _fetchLog(); }
    };

    _ws.onmessage = (ev) => {
      try { _handleWsMsg(JSON.parse(ev.data)); }
      catch (_) { /* malformed payload — ignore */ }
    };

    _ws.onclose = () => {
      _wsWasClosed = true;
      _setWsStatus(false);
      setTimeout(_connectWs, _wsRetry);
      _wsRetry = Math.min(_wsRetry * 2, 16000);
    };

    _ws.onerror = () => _ws.close();
  }

  function _setWsStatus(connected) {
    const dot   = document.getElementById("ws-dot");
    const label = document.getElementById("ws-label");
    if (!dot) return;
    dot.className   = "ws-dot " + (connected ? "connected" : "disconnected");
    label.textContent = connected ? "Live" : "Reconnecting…";
  }

  function _handleWsMsg(msg) {
    if (msg.type === "kpi_update")        _onKpi(msg);
    else if (msg.type === "inspection_result") _onInspection(msg);
    else if (msg.type === "alert")        _onAlertMsg(msg);
    else if (msg.type === "camera_status") _onCameraStatus(msg);
    else if (msg.type === "source_changed") _onSourceChanged();
  }

  /** Reasons as an array, whether the row came from the WS (array) or the DB (";"-joined string). */
  function _reasons(r) {
    const v = r.reasons || r.fail_reasons || [];
    if (Array.isArray(v)) return v;
    return String(v).split(";").map(s => s.trim()).filter(Boolean);
  }

  /** Header indicator + stream status when the camera drops or recovers. */
  function _onCameraStatus(msg) {
    const lost = msg.status === "lost";
    const el = document.getElementById("stream-status");
    if (el) el.textContent = lost ? "CAMERA LOST" : "Streaming";
    document.body.classList.toggle("camera-lost", lost);
  }

  /* ── KPI update ─────────────────────────────────────────────────────── */
  function _onKpi(msg) {
    _setText("kpi-fps",       _fmt(msg.fps, 1));
    _setText("kpi-latency",   msg.latency_ms != null ? msg.latency_ms.toFixed(0) : "—");
    _setText("kpi-total",     msg.total ?? "—");
    _setText("kpi-pass-rate", msg.pass_rate != null ? `${(msg.pass_rate * 100).toFixed(1)}%` : "—");
    _setText("kpi-ocr-rate",  msg.ocr_rate  != null ? `${(msg.ocr_rate  * 100).toFixed(1)}%` : "—");
  }

  /* ── Inspection result ───────────────────────────────────────────────── */
  function _onInspection(msg) {
    if (msg.billet_seq && msg.billet_seq === _lastSeenSeq) return;
    if (msg.billet_seq) _lastSeenSeq = msg.billet_seq;

    _lastRecord = msg;
    _updateStatusTile(msg);
    _updateMeasDetails(msg);
    _prependLogRow(msg, true);

    if (msg.status === "FAIL" || msg.status === "REWORK") { _alarmBeep(); _showAlertBanner(msg); _speakVerdict(msg); }
    else if (msg.status === "REVIEW") { _beep(660, 0.2, "triangle"); _showAlertBanner(msg); _speakVerdict(msg); }
    else if (msg.status === "PASS") {
      _passBeep();
      // a PASS must not hide a still-relevant system alert (e.g. camera lost)
      if (document.getElementById("alert-banner")?.dataset.system !== "1") _hideAlertBanner();
    }

    // update review badge count
    if (msg.status === "REVIEW") {
      const badge = document.getElementById("review-badge");
      if (badge) {
        const cur = parseInt(badge.textContent) || 0;
        badge.textContent = cur + 1;
        badge.classList.remove("hidden");
      }
    }
  }

  function _updateStatusTile(msg) {
    const tile  = document.getElementById("status-tile");
    const text  = document.getElementById("status-text");
    const subid = document.getElementById("status-billet-id");
    if (!tile) return;
    tile.className = `status-tile ${(msg.status || "idle").toLowerCase()}`;
    _setStatusIcon(msg.status);
    text.textContent = msg.status || "?";
    subid.textContent = msg.billet_id || "—";
    const time = msg.timestamp ? String(msg.timestamp).replace("T", " ").slice(11, 19) : "";
    _setText("status-meta", [msg.billet_seq != null ? `#${msg.billet_seq}` : "", time].filter(Boolean).join(" · "));
  }

  function _updateMeasDetails(msg) {
    const fmt = (v, d=1, u="mm") => v != null ? `${(+v).toFixed(d)} ${u}` : "—";
    _setText("m-length", fmt(msg.length_mm));
    _setText("m-width",  fmt(msg.width_mm));
    _setText("m-height", fmt(msg.height_mm));
    _setText("m-diam",   fmt(msg.diameter_mm));
    _setText("m-oval",   msg.ovality != null ? `${(+msg.ovality).toFixed(2)}%` : "—");
    _setText("m-diag",   fmt(msg.diag_diff_mm));
    _setText("m-ocr",    msg.ocr_confidence != null ? `${(msg.ocr_confidence * 100).toFixed(0)}%` : "—");
    _setText("m-ms",     msg.processing_ms  != null ? `${(+msg.processing_ms).toFixed(0)} ms` : "—");

    // Show only the fields that apply to this billet's shape (idle: show the rectangular set)
    const idle = msg.length_mm == null && msg.width_mm == null && msg.diameter_mm == null;
    document.querySelectorAll("#meas-dl [data-opt]").forEach(el => {
      const val = el.querySelector(".mt-val, dd");
      el.classList.toggle("hidden", idle ? el.dataset.opt === "round" : (val && val.textContent === "—"));
    });

    const reasons = _reasons(msg);
    const box  = document.getElementById("fail-reasons-box");
    const list = document.getElementById("fail-reasons-list");
    if (!box) return;
    if (reasons.length > 0) {
      list.innerHTML = reasons.map(r => `<li>${_esc(r)}</li>`).join("");
      box.classList.remove("hidden");
    } else {
      box.classList.add("hidden");
    }
  }

  /* ── Alert banner ───────────────────────────────────────────────────── */
  function _showAlertBanner(msg) {
    const banner = document.getElementById("alert-banner");
    if (!banner) return;
    const title  = document.getElementById("alert-title");
    const detail = document.getElementById("alert-detail");
    const system = msg.kind && msg.kind !== "billet";
    banner.dataset.system = system ? "1" : "";
    banner.className = `alert-banner ${(system ? "fail" : (msg.status || "fail")).toLowerCase()}`;
    title.textContent = system ? `${msg.status}` : `${msg.status} — Billet ${msg.billet_id || "?"}`;
    detail.textContent = _reasons(msg).slice(0, 2).join(" | ");
    banner.classList.remove("hidden");

    // push to alert history list
    _addAlertHistoryItem(msg);
  }

  function _hideAlertBanner() {
    const banner = document.getElementById("alert-banner");
    if (banner) banner.classList.add("hidden");
  }

  function _addAlertHistoryItem(msg) {
    const list = document.getElementById("alert-history-list");
    if (!list) return;
    // Remove "no alerts" placeholder
    const empty = list.querySelector(".alert-empty");
    if (empty) empty.remove();

    const reasons = _reasons(msg).slice(0, 2).join("; ");
    const time = msg.timestamp ? msg.timestamp.slice(11, 19) : new Date().toLocaleTimeString();
    const li = document.createElement("li");
    li.className = `alert-item ${(msg.status || "").toLowerCase()}`;
    li.innerHTML = `
      ${_icon(_STATUS_ICON[(msg.status || "").toUpperCase()] || "alert")}
      <span class="alert-item-time">${_esc(time)}</span>
      <span class="alert-item-id">${_esc(msg.billet_id || "?")}</span>
      <span class="alert-item-body">${_esc(reasons) || msg.status}</span>`;
    list.insertBefore(li, list.firstChild);
    // Keep history list tidy
    while (list.children.length > 20) list.removeChild(list.lastChild);
  }

  function _onAlertMsg(msg) {
    // Billet verdict alerts were already shown (banner + history) by the matching
    // inspection_result message; system alerts (camera, duplicate ID) only arrive here.
    if (!msg.kind || msg.kind === "billet") return;
    _showAlertBanner(msg);
    if (msg.sound !== false) { _alarmBeep(); _speak(String(msg.status || "System alert").replace(/_/g, " ")); }
  }

  /* ── Log table ──────────────────────────────────────────────────────── */
  const LOG_MAX = 100;

  function _prependLogRow(msg, isNew = false) {
    if (isNew) msg._isNew = true;
    _allRows.unshift(msg);
    if (_allRows.length > LOG_MAX) _allRows.pop();
    _renderLogTable();
    if (isNew) {
      setTimeout(() => { msg._isNew = false; }, 2000);
    }
  }

  function _renderLogTable() {
    const idFilter  = (document.getElementById("log-filter-id")?.value || "").trim().toUpperCase();
    const stFilter  = (document.getElementById("log-filter-status")?.value || "");
    _logRows = _allRows.filter(r => {
      if (stFilter && r.status !== stFilter) return false;
      if (idFilter && !(r.billet_id || "").toUpperCase().includes(idFilter)) return false;
      return true;
    });

    const tbody = document.getElementById("log-tbody");
    if (!tbody) return;

    if (_logRows.length === 0) {
      tbody.innerHTML = `<tr><td colspan="10" class="table-empty">No records match the current filter.</td></tr>`;
      _setText("log-count-label", "0 records");
      return;
    }

    tbody.innerHTML = _logRows.map(r => `
      <tr class="${r._isNew ? 'new-row' : ''}">
        <td class="muted">${_esc(_fmtTime(r.timestamp))}</td>
        <td class="num">${r.billet_seq ?? "—"}</td>
        <td class="id">${_esc(r.billet_id || "UNKNOWN")}</td>
        <td><span class="badge-status badge-${(r.status||"").toLowerCase()}">${_esc(r.status||"—")}</span></td>
        <td class="num">${_fmtDims(r)}</td>
        <td class="num">${_fmtOval(r)}</td>
        <td class="num">${_fmtDiag(r)}</td>
        <td class="num">${r.ocr_confidence != null ? (r.ocr_confidence*100).toFixed(0)+"%" : "—"}</td>
        <td class="reasons">${_esc(_reasons(r).join("; ")||"—")}</td>
        <td><button class="icon-btn" data-drill="${r.billet_seq}" title="Details" aria-label="Details">${_icon("expand")}</button></td>
      </tr>`).join("");

    // Attach drill-down handlers (the modal loads the full record from the API)
    tbody.querySelectorAll("[data-drill]").forEach(btn => {
      btn.addEventListener("click", () => {
        const row = _logRows.find(x => String(x.billet_seq) === btn.dataset.drill);
        if (row) _openModal(row);
      });
    });

    _setText("log-count-label", `${_logRows.length} record${_logRows.length !== 1 ? "s" : ""}`);
  }

  function _fmtTime(ts) {
    if (!ts) return "—";
    return ts.replace("T", " ").slice(0, 19);
  }
  function _fmtDims(r) {
    const v = (x) => x != null ? (+x).toFixed(1) : "—";
    return `${v(r.length_mm)} × ${v(r.width_mm)} × ${v(r.height_mm)}`;
  }
  function _fmtOval(r) {
    const d = r.diameter_mm != null ? `Ø${(+r.diameter_mm).toFixed(1)}` : "";
    const o = r.ovality != null ? ` ov=${(+r.ovality).toFixed(2)}%` : "";
    return d + o || "—";
  }
  function _fmtDiag(r) {
    return r.diag_diff_mm != null ? `${(+r.diag_diff_mm).toFixed(2)} mm` : "—";
  }

  async function _fetchLog() {
    const stFilter = (document.getElementById("log-filter-status")?.value || "");
    const idFilter = (document.getElementById("log-filter-id")?.value || "").trim();
    let url = `/api/log?n=${LOG_MAX}`;
    if (stFilter) url += `&status=${stFilter}`;
    if (idFilter) url += `&q=${encodeURIComponent(idFilter)}`;
    try {
      const res = await fetch(url);
      if (!res.ok) return;
      const rows = await res.json();
      _allRows = rows;
      _renderLogTable();
      if (!_lastRecord && rows.length > 0) {
        _lastRecord = rows[0];
        _updateStatusTile(rows[0]);
        _updateMeasDetails(rows[0]);
      }
    } catch (_) {}
  }

  /* ── Review queue ───────────────────────────────────────────────────── */
  async function _fetchReviewQueue() {
    try {
      const res = await fetch("/api/review/pending?n=50");
      if (!res.ok) return;
      const items = await res.json();
      _renderReviewCards(items);

      const badge = document.getElementById("review-badge");
      if (badge) {
        badge.textContent = items.length;
        items.length > 0 ? badge.classList.remove("hidden") : badge.classList.add("hidden");
      }
    } catch (_) {}
  }

  function _renderReviewCards(items) {
    const container = document.getElementById("review-cards");
    if (!container) return;
    if (items.length === 0) {
      container.innerHTML = `<p class="table-empty">No pending reviews.</p>`;
      return;
    }
    container.innerHTML = items.map(r => _reviewCardHtml(r)).join("");
    container.querySelectorAll(".review-approve-btn").forEach(btn => {
      btn.addEventListener("click", () => {
        const inp = container.querySelector(`input[data-seq="${btn.dataset.seq}"]`);
        _resolveReview(btn.dataset.seq, "approve", "", inp ? inp.value.trim().toUpperCase() : "");
      });
    });
    container.querySelectorAll(".review-reject-btn").forEach(btn => {
      btn.addEventListener("click", () => {
        const inp = container.querySelector(`input[data-seq="${btn.dataset.seq}"]`);
        _resolveReview(btn.dataset.seq, "reject", inp ? inp.value : "");
      });
    });
  }

  function _reviewCardHtml(r) {
    const dims = _fmtDims(r);
    // Prefer the tight ID crop; fall back to the annotated billet card.
    const src = r.id_crop_url || r.image_url;
    const imgHtml = src
      ? `<img src="${_esc(src)}" alt="ID crop">`
      : `<span>No snapshot</span>`;
    const guess = (r.billet_id || "").startsWith("UNREAD") ? "" : (r.billet_id || "");

    return `<div class="review-card">
      <div class="review-card-header">
        <span class="review-card-id">${_esc(r.billet_id || "UNKNOWN")}</span>
        <span class="review-card-seq">#${r.billet_seq}</span>
      </div>
      <div class="review-card-crop">${imgHtml}</div>
      <div class="review-card-meas">${_esc(dims)} · OCR ${r.ocr_confidence != null ? (r.ocr_confidence*100).toFixed(0)+"%" : "—"}</div>
      <div class="review-card-meas">${_esc(_reasons(r).join("; "))}</div>
      <div class="review-card-actions">
        <input class="review-id-input" type="text" placeholder="Type the correct ID" data-seq="${r.billet_seq}" value="${_esc(guess)}">
        <button class="btn btn-pass btn-sm review-approve-btn" data-seq="${r.billet_seq}">${_icon("check")}Approve</button>
        <button class="btn btn-fail btn-sm review-reject-btn"  data-seq="${r.billet_seq}">${_icon("x")}Reject</button>
      </div>
      <div class="review-error hidden" data-err="${r.billet_seq}"></div>
    </div>`;
  }

  async function _resolveReview(seq, action, notes, correctedId) {
    const errEl = document.querySelector(`[data-err="${seq}"]`);
    try {
      const body = { action, notes };
      if (action === "approve" && correctedId) body.corrected_id = correctedId;
      const res = await fetch(`/api/review/${seq}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (res.ok) {
        await _fetchReviewQueue();
        _fetchLog();
      } else if (errEl) {
        const err = await res.json().catch(() => ({}));
        errEl.textContent = err.detail || `Error ${res.status}`;
        errEl.classList.remove("hidden");
      }
    } catch (e) {
      if (errEl) { errEl.textContent = `Network error: ${e.message}`; errEl.classList.remove("hidden"); }
    }
  }

  /* ── Tolerances ─────────────────────────────────────────────────────── */
  async function _fetchTolerances() {
    try {
      const res = await fetch("/api/tolerances");
      if (!res.ok) return;
      _tolerances = await res.json();
      _populateProfileSelect();
    } catch (_) {}
  }

  function _populateProfileSelect() {
    const sel = document.getElementById("profile-select");
    if (!sel) return;
    const prev = sel.value;
    sel.innerHTML = Object.keys(_tolerances).map(p =>
      `<option value="${_esc(p)}">${_esc(p)}</option>`).join("");
    if (prev && _tolerances[prev]) sel.value = prev;
    _loadProfileFields(sel.value);
  }

  function _loadProfileFields(profile) {
    const tol = _tolerances[profile];
    if (!tol) return;
    _setInputVal("tol-width-nom",   tol.width_nominal_mm);
    _setInputVal("tol-width-tol",   tol.width_tol_mm);
    _setInputVal("tol-height-nom",  tol.height_nominal_mm);
    _setInputVal("tol-height-tol",  tol.height_tol_mm);
    _setInputVal("tol-length-nom",  tol.length_nominal_mm);
    _setInputVal("tol-length-tol",  tol.length_tol_mm);
    _setInputVal("tol-diam-nom",    tol.diameter_nominal_mm);
    _setInputVal("tol-diam-tol",    tol.diameter_tol_mm);
    _setInputVal("tol-ovality",     tol.max_ovality_pct);
    _setInputVal("tol-diag-diff",   tol.max_diag_diff_mm);
    _setInputVal("tol-camber",      tol.max_camber_mm);
    _setInputVal("tol-cs-var",      tol.max_cross_section_var_mm);
    _setInputVal("tol-surface",     tol.max_surface_anomaly_score);
    _setInputVal("tol-edge",        tol.max_edge_irregularity_mm);
  }

  async function _saveTolerances() {
    const profile = document.getElementById("profile-select")?.value;
    if (!profile) return;
    const updates = {};
    const add = (id, key) => {
      const el = document.getElementById(id);
      if (el && el.value !== "") updates[key] = parseFloat(el.value);
    };
    add("tol-width-nom",  "width_nominal_mm");
    add("tol-width-tol",  "width_tol_mm");
    add("tol-height-nom", "height_nominal_mm");
    add("tol-height-tol", "height_tol_mm");
    add("tol-length-nom", "length_nominal_mm");
    add("tol-length-tol", "length_tol_mm");
    add("tol-diam-nom",   "diameter_nominal_mm");
    add("tol-diam-tol",   "diameter_tol_mm");
    add("tol-ovality",    "max_ovality_pct");
    add("tol-diag-diff",  "max_diag_diff_mm");
    add("tol-camber",     "max_camber_mm");
    add("tol-cs-var",     "max_cross_section_var_mm");
    add("tol-surface",    "max_surface_anomaly_score");
    add("tol-edge",       "max_edge_irregularity_mm");

    _showTolStatus("Saving…", "");
    try {
      const res = await fetch(`/api/tolerances/${profile}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ updates }),
      });
      if (res.ok) {
        const updated = await res.json();
        _tolerances[profile] = updated;
        _showTolStatus("Saved successfully.", "ok");
      } else {
        const err = await res.json().catch(() => ({}));
        _showTolStatus(`Error: ${err.detail || res.status}`, "err");
      }
    } catch (e) {
      _showTolStatus(`Network error: ${e.message}`, "err");
    }
  }

  async function _activateProfile() {
    const profile = document.getElementById("profile-select")?.value;
    if (!profile) return;
    try {
      const res = await fetch(`/api/tolerances/${profile}/activate`, { method: "PUT" });
      if (res.ok) {
        _activeProfile = profile;
        const badge = document.getElementById("active-profile-badge");
        const name  = document.getElementById("active-profile-name");
        if (badge) { badge.classList.remove("hidden"); }
        if (name)  { name.textContent = profile; }
        _showTolStatus(`Profile "${profile}" activated.`, "ok");
      }
    } catch (_) {}
  }

  function _showTolStatus(msg, cls) {
    const el = document.getElementById("tol-save-status");
    if (!el) return;
    el.textContent = msg;
    el.className = `tol-status ${cls}`;
    el.classList.remove("hidden");
    setTimeout(() => el.classList.add("hidden"), 4000);
  }

  /* ── Calibration (FR-3) ─────────────────────────────────────────────── */
  async function _fetchCalibration() {
    try {
      const res = await fetch("/api/calibration");
      if (!res.ok) return;
      const c = await res.json();
      _setText("cal-scale", c.mm_per_px != null ? `1 px = ${(+c.mm_per_px).toFixed(4)} mm (${(+c.px_per_mm).toFixed(2)} px/mm)` : "—");
      _setText("cal-marker", c.marker_type ? `${c.marker_type}, ${c.marker_size_mm} mm` : "—");
      _setText("cal-when", c.calibrated_at ? _fmtTime(c.calibrated_at) : "never");
      _setText("cal-reproj", c.reprojection_error != null ? `${(+c.reprojection_error).toFixed(3)} px` : "—");
    } catch (_) {}
  }

  async function _recalibrate() {
    const el = document.getElementById("cal-status");
    const show = (msg, cls) => {
      if (!el) return;
      el.textContent = msg;
      el.className = `tol-status ${cls}`;
      setTimeout(() => el.classList.add("hidden"), 6000);
    };
    const size = parseFloat(document.getElementById("cal-marker-mm")?.value || "50");
    show("Calibrating…", "");
    try {
      const res = await fetch("/api/calibration/recalibrate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ marker_size_mm: size }),
      });
      const body = await res.json().catch(() => ({}));
      if (res.ok) { show(`Calibrated: 1 px = ${(+body.mm_per_px).toFixed(4)} mm`, "ok"); _fetchCalibration(); }
      else show(body.detail || `Error ${res.status}`, "err");
    } catch (e) { show(`Network error: ${e.message}`, "err"); }
  }

  /* ── Alert REST ──────────────────────────────────────────────────────── */
  async function _fetchAlerts() {
    try {
      const res = await fetch("/api/alerts?n=20");
      if (!res.ok) return;
      const items = await res.json();
      const list  = document.getElementById("alert-history-list");
      if (!list || items.length === 0) return;
      list.innerHTML = items.map(a => {
        const reasons = (a.reasons || []).slice(0, 2).join("; ");
        return `<li class="alert-item ${(a.status||"").toLowerCase()}">
          ${_icon(_STATUS_ICON[(a.status || "").toUpperCase()] || "alert")}
          <span class="alert-item-time">${_esc(_fmtTime(a.timestamp).slice(11) || "—")}</span>
          <span class="alert-item-id">${_esc(a.billet_id||"?")}</span>
          <span class="alert-item-body">${_esc(reasons) || a.status}</span>
        </li>`;
      }).join("");
    } catch (_) {}
  }

  async function _fetchActiveAlert() {
    try {
      const res = await fetch("/api/alerts/active");
      if (!res.ok) return;
      const a = await res.json();
      if (a) _showAlertBanner(a);
      else   _hideAlertBanner();
    } catch (_) {}
  }

  async function _clearActiveAlert() {
    try {
      await fetch("/api/alerts/clear", { method: "POST" });
      _hideAlertBanner();
    } catch (_) {}
  }

  async function _clearAlertHistory() {
    try {
      await fetch("/api/alerts/clear?history=true", { method: "POST" });
      _hideAlertBanner();
      const list = document.getElementById("alert-history-list");
      if (list) list.innerHTML = `<li class="alert-empty">No alerts</li>`;
    } catch (_) {}
  }

  /* ── Drill-down modal ─────────────────────────────────────────────────── */
  function _openModal(record) {
    const backdrop = document.getElementById("modal-backdrop");
    const title    = document.getElementById("modal-title");
    const body     = document.getElementById("modal-body");
    if (!backdrop) return;

    title.textContent = `Billet ${record.billet_id || "UNKNOWN"} — ${record.status || "—"}`;

    const m = (label, val) => val != null
      ? `<div class="modal-meas-item"><div class="label">${_esc(label)}</div><div class="value">${_esc(String(val))}</div></div>`
      : "";

    const reasons = _reasons(record);
    const reasonsHtml = reasons.length
      ? `<ul class="modal-reasons">${reasons.map(r => `<li>${_esc(r)}</li>`).join("")}</ul>`
      : `<p style="color:var(--text-3);font-size:13px">All parameters within tolerance.</p>`;

    const imgUrl = record.image_url
      || (record.image_path ? `/snapshots/${record.image_path.split(/[\\/]/).pop()}` : null);
    const imgHtml = imgUrl
      ? `<img src="${_esc(imgUrl)}" alt="snapshot">`
      : `<p class="no-img">No snapshot saved.</p>`;

    body.innerHTML = `
      <div class="modal-section">
        <h3>Identity</h3>
        <div class="modal-meas-grid">
          ${m("Billet ID", record.billet_id)}
          ${m("Seq #",     record.billet_seq)}
          ${m("Batch",     record.batch_id)}
          ${m("Timestamp", _fmtTime(record.timestamp))}
          ${m("Status",    record.status)}
          ${m("Proc. time", record.processing_ms != null ? record.processing_ms.toFixed(0)+" ms" : null)}
        </div>
      </div>

      <div class="modal-section">
        <h3>Dimensions</h3>
        <div class="modal-meas-grid">
          ${m("Length (mm)",   record.length_mm  != null ? (+record.length_mm).toFixed(2)  : null)}
          ${m("Width (mm)",    record.width_mm   != null ? (+record.width_mm).toFixed(2)   : null)}
          ${m("Height (mm)",   record.height_mm  != null ? (+record.height_mm).toFixed(2)  : null)}
          ${m("Diameter (mm)", record.diameter_mm!= null ? (+record.diameter_mm).toFixed(2): null)}
          ${m("Ovality (%)",   record.ovality    != null ? (+record.ovality).toFixed(3)    : null)}
          ${m("Diag Diff (mm)",record.diag_diff_mm!=null ? (+record.diag_diff_mm).toFixed(3): null)}
        </div>
      </div>

      <div class="modal-section">
        <h3>OCR / ID</h3>
        <div class="modal-meas-grid">
          ${m("OCR Confidence", record.ocr_confidence != null ? (record.ocr_confidence*100).toFixed(1)+"%" : null)}
          ${m("Defects",        record.defects || "none")}
        </div>
      </div>

      <div class="modal-section">
        <h3>Decision Reasoning</h3>
        ${reasonsHtml}
      </div>

      <div class="modal-section modal-snapshot">
        <h3>Snapshot</h3>
        ${imgHtml}
      </div>
      <div id="modal-detail"><p style="color:var(--text-3);font-size:13px">Loading frame-level detail…</p></div>`;

    backdrop.classList.remove("hidden");
    if (record.billet_seq != null) _loadModalDetail(record.billet_seq);
  }

  /** Fetch /api/billet/{seq} and append the per-frame data, tolerances, OCR and frame images. */
  async function _loadModalDetail(seq) {
    const host = document.getElementById("modal-detail");
    if (!host) return;
    try {
      const res = await fetch(`/api/billet/${seq}`);
      const data = res.ok ? await res.json() : null;
      const d = data && data.detail;
      if (!d) { host.innerHTML = ""; return; }
      const n = (v, dp = 2) => v != null ? (+v).toFixed(dp) : "—";
      const tol = d.tolerances || {};
      const meas = d.measurement || {};
      const rows = [
        ["Length", meas.length_mm, tol.length_nominal_mm, tol.length_tol_mm],
        ["Width", meas.width_mm, tol.width_nominal_mm, tol.width_tol_mm],
        ["Diameter", meas.diameter_mm, tol.diameter_nominal_mm, tol.diameter_tol_mm],
      ].filter(r => r[1] != null && r[2] != null).map(r =>
        `<tr><td>${r[0]}</td><td>${n(r[1])} mm</td><td>${n(r[2], 1)} ± ${n(r[3], 1)} mm</td></tr>`);
      const limits = [
        ["Camber", meas.camber_mm, tol.max_camber_mm, "mm"],
        ["Cross-section var.", meas.cross_section_var_mm, tol.max_cross_section_var_mm, "mm"],
        ["Edge irregularity", meas.edge_irregularity_mm, tol.max_edge_irregularity_mm, "mm"],
        ["Surface anomaly", meas.surface_anomaly_score, tol.max_surface_anomaly_score, ""],
        ["Diagonal diff.", meas.diag_diff_mm, tol.max_diag_diff_mm, "mm"],
        ["Ovality", meas.ovality, tol.max_ovality_pct, "%"],
      ].filter(r => r[1] != null && r[2] != null).map(r =>
        `<tr><td>${r[0]}</td><td>${n(r[1], 3)} ${r[3]}</td><td>≤ ${n(r[2], 2)} ${r[3]}</td></tr>`);
      const frames = (d.frames || []).map(f =>
        `<tr><td>#${f.rank}</td><td>${n(f.sharpness, 0)}</td><td>${n(f.length_mm, 1)}</td><td>${n(f.width_mm, 2)}</td><td>${n(f.camber_mm, 2)}</td></tr>`);
      const ocr = d.ocr || {};
      const img = d.images || {};
      const thumbs = [img.id_crop, ...(img.frames || [])].filter(Boolean)
        .map(u => `<img src="${_esc(u)}" alt="frame" style="max-height:110px;margin:4px;border-radius:4px;border:1px solid var(--border)">`).join("");
      host.innerHTML = `
        <div class="modal-section"><h3>Measured vs tolerance (profile ${_esc(d.profile || "?")}, ${n(d.mm_per_px, 4)} mm/px, length: ${_esc(d.length_mode || "direct")})</h3>
          <table class="data-table"><thead><tr><th>Parameter</th><th>Measured</th><th>Limit</th></tr></thead>
          <tbody>${rows.concat(limits).join("")}</tbody></table></div>
        <div class="modal-section"><h3>OCR / ID decision</h3>
          <div class="modal-meas-grid">
            <div class="modal-meas-item"><div class="label">Read</div><div class="value">${_esc(ocr.text || "—")}</div></div>
            <div class="modal-meas-item"><div class="label">Confidence</div><div class="value">${n((ocr.confidence || 0) * 100, 0)}%</div></div>
            <div class="modal-meas-item"><div class="label">Format valid</div><div class="value">${ocr.format_valid ? "yes" : "no"}</div></div>
            <div class="modal-meas-item"><div class="label">Candidates</div><div class="value">${_esc((ocr.candidates || []).join(", ") || "—")}</div></div>
          </div></div>
        <div class="modal-section"><h3>Best frames used for the median (${d.frames_observed ?? "?"} observed)</h3>
          <table class="data-table"><thead><tr><th>Rank</th><th>Sharpness</th><th>Length mm</th><th>Width mm</th><th>Camber mm</th></tr></thead>
          <tbody>${frames.join("")}</tbody></table>
          <div style="margin-top:8px">${thumbs}</div></div>`;
    } catch (_) { host.innerHTML = ""; }
  }

  function _closeModal() {
    const backdrop = document.getElementById("modal-backdrop");
    if (backdrop) backdrop.classList.add("hidden");
  }

  /* ── Input source: upload video / image, simulated demo ─────────────── */
  const _SRC_BTNS = ["src-camera-btn", "src-demo-btn", "src-video-btn", "src-image-btn"];
  const _KIND_NAME = { video: "Uploaded video", image: "Uploaded image", demo: "Simulated demo", camera: "Live camera", configured: "Configured source" };

  function _srcBusy(busy) {
    _SRC_BTNS.forEach(id => { const b = document.getElementById(id); if (b) b.disabled = busy; });
  }

  function _srcMessage(text, ok) {
    const el = document.getElementById("src-message");
    if (!el) return;
    el.textContent = text || "";
    el.className = "tol-status " + (text ? (ok ? "ok" : "err") : "hidden");
  }

  function _srcProgress(frac) {
    const wrap = document.getElementById("src-progress");
    const bar = document.getElementById("src-progress-bar");
    if (!wrap || !bar) return;
    wrap.classList.toggle("hidden", frac == null);
    if (frac != null) bar.style.width = `${Math.round(Math.min(1, Math.max(0, frac)) * 100)}%`;
  }

  /** A new source started: clear the previous run's tile, details and banner. */
  function _onSourceChanged() {
    _lastRecord = null;
    _lastSeenSeq = -1;           // sequence numbers continue, but never skip the new run's first result
    const tile = document.getElementById("status-tile");
    if (tile) tile.className = "status-tile idle";
    _setStatusIcon("IDLE");
    _setText("status-text", "WAITING");
    _setText("status-billet-id", "—");
    _setText("status-meta", "");
    _updateMeasDetails({});
    _hideAlertBanner();
    _reattachStream();           // always start the new source on a fresh video connection
    _fetchAlerts();              // the server cleared the previous source's alerts
    _fetchSource();
  }

  function _renderSource(src) {
    document.querySelectorAll(".source-select .seg").forEach(b => b.classList.toggle("active", b.dataset.kind === src.kind));
    const kind = _KIND_NAME[src.kind] || src.kind;
    const kindWord = kind.split(" ").pop().toLowerCase();
    _setText("src-label", !src.label ? kind
      : String(src.label).toLowerCase().includes(kindWord) ? src.label : `${kind} · ${src.label}`);
    const bs = src.by_status || {};
    const counts = `${src.results} billet${src.results === 1 ? "" : "s"} (` +
      `${bs.PASS || 0} pass, ${bs.FAIL || 0} fail, ${bs.REWORK || 0} rework, ${bs.REVIEW || 0} review)`;
    const scale = `scale ${(+src.mm_per_px).toFixed(3)} mm/px (${src.scale_source})`;
    let detail;
    if (src.ended) {
      detail = src.results > 0
        ? `Analysis complete — ${counts} · ${scale}`
        : "Analysis complete — no complete billet found. Keep the whole billet inside the frame.";
    } else if (src.kind === "configured") {
      detail = `${counts} · ${scale}`;
    } else {
      detail = `Analysing… ${counts} · ${scale}`;
    }
    if (src.kind === "camera" && src.camera_ok === false) {
      detail = "Camera lost — reconnecting. Click Camera to force a retry.";
    }
    if (src.scale_source !== "marker" && (src.kind === "video" || src.kind === "image" || src.kind === "camera")) {
      detail += src.kind === "camera"
        ? " — no calibration marker in view: mm values use the stored scale and are NOT valid for this camera. " +
          "Place the marker in view, then Tolerances → Recalibrate."
        : " — no calibration marker found, so mm values use the stored scale";
    }
    _setText("src-detail", detail);

    const finite = src.kind === "video" || src.kind === "image";
    _srcProgress(finite && src.frames_total ? (src.ended ? 1 : src.frames_done / src.frames_total) : null);
  }

  let _bootId = null;      // server start id; a change means the server restarted under this page
  let _serverDown = false;

  async function _fetchSource() {
    try {
      const r = await fetch("/api/source");
      if (!r.ok) return;
      const src = await r.json();
      const restarted = (_bootId !== null && src.boot_id !== _bootId) || _serverDown;
      _bootId = src.boot_id;
      _serverDown = false;
      if (restarted) {          // recover without a manual F5
        _reattachStream(); _fetchAlerts(); _fetchLog(); _fetchTolerances();
      }
      _renderSource(src);
    } catch (_) {
      _serverDown = true;       // server restarting — reconnect as soon as it answers again
    }
  }

  async function _srcCall(url, label) {
    _srcBusy(true);
    _srcMessage(label, true);
    try {
      const res = await fetch(url, { method: "POST" });
      const body = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(body.detail || `HTTP ${res.status}`);
      _srcMessage("", true);
      _onSourceChanged();
    } catch (err) {
      _srcMessage(String(err.message || err), false);
    } finally {
      _srcBusy(false);
    }
  }

  /** Upload ``file`` as the raw request body (with progress); resolves to the parsed JSON reply. */
  function _uploadFile(kind, file) {
    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open("POST", `/api/source/${kind}?filename=${encodeURIComponent(file.name)}`);
      xhr.upload.onprogress = e => {
        if (e.lengthComputable) {
          _srcProgress(e.loaded / e.total);
          _srcMessage(`Uploading ${file.name}… ${Math.round(e.loaded / e.total * 100)}%`, true);
        }
      };
      xhr.upload.onload = () => _srcMessage(`Preparing ${file.name}…`, true);
      xhr.onerror = () => reject(new Error("Upload failed — is the server running?"));
      xhr.onload = () => {
        let body = {};
        try { body = JSON.parse(xhr.responseText); } catch (_) {}
        if (xhr.status >= 200 && xhr.status < 300) resolve(body);
        else reject(new Error(body.detail || `HTTP ${xhr.status}`));
      };
      xhr.send(file);
    });
  }

  async function _onFilePicked(kind, input) {
    const file = input.files && input.files[0];
    input.value = "";                       // allow re-picking the same file
    if (!file) return;
    _srcBusy(true);
    try {
      await _uploadFile(kind, file);
      _srcMessage("", true);
      _onSourceChanged();
    } catch (err) {
      _srcMessage(String(err.message || err), false);
      _srcProgress(null);
    } finally {
      _srcBusy(false);
    }
  }

  function _initSource() {
    const on = (id, ev, fn) => document.getElementById(id)?.addEventListener(ev, fn);
    on("src-demo-btn", "click",
      () => _srcCall("/api/source/demo", "Starting simulated conveyor (the first run renders the belt video, ~2 min)…"));
    on("src-camera-btn", "click", () => {
      const idx = document.getElementById("src-camera-index")?.value || "0";
      _srcCall(`/api/source/camera?index=${encodeURIComponent(idx)}`, "Opening the camera…");
    });
    on("src-video-btn", "click", () => document.getElementById("src-video-input")?.click());
    on("src-image-btn", "click", () => document.getElementById("src-image-input")?.click());
    on("src-video-input", "change", e => _onFilePicked("video", e.target));
    on("src-image-input", "change", e => _onFilePicked("image", e.target));
    _fetchSource();
    setInterval(_fetchSource, 1500);
  }

  /* ── Tab switching ───────────────────────────────────────────────────── */
  function _initTabs() {
    document.querySelectorAll(".tab-btn").forEach(btn => {
      btn.addEventListener("click", () => {
        document.querySelectorAll(".tab-btn").forEach(b => {
          b.classList.remove("active");
          b.setAttribute("aria-selected", "false");
        });
        document.querySelectorAll(".tab-panel").forEach(p => p.classList.add("hidden"));

        btn.classList.add("active");
        btn.setAttribute("aria-selected", "true");
        const panel = document.getElementById(`tab-${btn.dataset.tab}`);
        if (panel) panel.classList.remove("hidden");

        // Lazy-load tab data on first switch
        if (btn.dataset.tab === "log")        _fetchLog();
        if (btn.dataset.tab === "review")     _fetchReviewQueue();
        if (btn.dataset.tab === "tolerances") { _fetchTolerances(); _fetchCalibration(); }
      });
    });
  }

  /* ── Periodic refresh ────────────────────────────────────────────────── */
  function _startPolling() {
    // Pull stats every 2 s even without WS (fallback)
    setInterval(async () => {
      try {
        const r = await fetch("/api/stats");
        if (r.ok) _onKpi(await r.json());
      } catch (_) {}
    }, 2000);

    // Refresh log every 10 s if on log tab
    setInterval(() => {
      const logPanel = document.getElementById("tab-log");
      if (logPanel && !logPanel.classList.contains("hidden")) _fetchLog();
    }, 10000);

    // Refresh review queue every 15 s if on review tab
    setInterval(() => {
      const rvPanel = document.getElementById("tab-review");
      if (rvPanel && !rvPanel.classList.contains("hidden")) _fetchReviewQueue();
    }, 15000);
  }

  /* ── Utility ─────────────────────────────────────────────────────────── */
  function _setText(id, val) {
    const el = document.getElementById(id);
    if (el) el.textContent = val;
  }
  function _setInputVal(id, val) {
    const el = document.getElementById(id);
    if (el && val != null) el.value = val;
  }
  function _fmt(val, decimals = 1) {
    return val != null ? (+val).toFixed(decimals) : "—";
  }
  function _esc(str) {
    return String(str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  /* ── Init ────────────────────────────────────────────────────────────── */
  function init() {
    _initTheme();
    _initDropdowns();
    _initVoice();
    _initStreamPill();
    _initTabs();
    _initSource();

    // Tolerances tab buttons
    document.getElementById("save-tol-btn")?.addEventListener("click", _saveTolerances);
    document.getElementById("activate-profile-btn")?.addEventListener("click", _activateProfile);
    document.getElementById("recalibrate-btn")?.addEventListener("click", _recalibrate);
    document.getElementById("profile-select")?.addEventListener("change", e => {
      _loadProfileFields(e.target.value);
    });

    // Alert banner dismiss
    document.getElementById("alert-dismiss")?.addEventListener("click", _clearActiveAlert);
    document.getElementById("clear-alerts-btn")?.addEventListener("click", _clearAlertHistory);

    // Log filters
    let _idDebounce = null;
    document.getElementById("log-filter-id")?.addEventListener("input", () => {
      _renderLogTable();                       // instant client-side filter on loaded rows
      clearTimeout(_idDebounce);               // then look up older matches on the server
      _idDebounce = setTimeout(_fetchLog, 350);
    });
    document.getElementById("log-filter-status")?.addEventListener("change", () => {
      _allRows = []; // clear so fresh fetch respects status filter
      _fetchLog();
    });
    document.getElementById("log-refresh-btn")?.addEventListener("click", _fetchLog);
    document.getElementById("review-refresh-btn")?.addEventListener("click", _fetchReviewQueue);

    // Drill-down modal on "View Full" button
    document.getElementById("drill-last-btn")?.addEventListener("click", () => {
      if (_lastRecord) _openModal(_lastRecord);
    });

    // Modal close
    document.getElementById("modal-close")?.addEventListener("click", _closeModal);
    document.getElementById("modal-backdrop")?.addEventListener("click", e => {
      if (e.target === e.currentTarget) _closeModal();
    });
    document.addEventListener("keydown", e => {
      if (e.key === "Escape") _closeModal();
    });

    // Start
    _connectWs();
    _startPolling();
    _fetchAlerts();
    _fetchActiveAlert();
    _fetchTolerances();   // preload even on Live tab so save works

    // Initial log fetch silently (populates if already on log tab)
    _fetchLog();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
