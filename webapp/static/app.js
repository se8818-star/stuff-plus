// Stuff+ leaderboards and grader. Plain JS, no build step.
"use strict";

const PT = {
  FF: ["Four-seam", "#d22d49"], SI: ["Sinker", "#f08c00"], FC: ["Cutter", "#933f2c"],
  SL: ["Slider", "#c9b500"], ST: ["Sweeper", "#b8912a"], SV: ["Slurve", "#7f9cc9"],
  CU: ["Curveball", "#00a9c4"], KC: ["Knuckle curve", "#6236cd"], CH: ["Changeup", "#1d9e3a"],
  FS: ["Splitter", "#3a9a9a"], FO: ["Forkball", "#55a0a0"], SC: ["Screwball", "#60db33"],
};
const ptColor = t => (PT[t] || [t, "#888"])[1];
const ptName = t => (PT[t] || [t])[0];

// starting points for a new pitch in the grader (RHP-ish averages)
const TEMPLATES = {
  FF: { usage: 45, velo: 94.5, spin: 2300, ivb: 16, hb: 8 },
  SI: { usage: 30, velo: 93.5, spin: 2150, ivb: 8, hb: 15 },
  FC: { usage: 20, velo: 89, spin: 2400, ivb: 9, hb: -2 },
  SL: { usage: 25, velo: 85.5, spin: 2450, ivb: 2, hb: -5 },
  ST: { usage: 25, velo: 82, spin: 2600, ivb: 1, hb: -14 },
  SV: { usage: 20, velo: 80, spin: 2600, ivb: -6, hb: -13 },
  CU: { usage: 15, velo: 79, spin: 2550, ivb: -9, hb: -8 },
  KC: { usage: 15, velo: 82, spin: 2400, ivb: -8, hb: -6 },
  CH: { usage: 15, velo: 86, spin: 1800, ivb: 6, hb: 14 },
  FS: { usage: 15, velo: 86, spin: 1300, ivb: 3, hb: 10 },
};
const ARM = { ext: 6.4, rel_height: 5.8, rel_side: 1.9 };

const $ = s => document.querySelector(s);
const el = (tag, attrs = {}, html = "") => {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  e.innerHTML = html;
  return e;
};
const f1 = v => v == null ? "–" : v.toFixed(1);
const f0 = v => v == null ? "–" : Math.round(v).toLocaleString();
const pct = v => { if (v == null) return "–"; const n = Math.min(99, Math.max(1, Math.round(v))); return `${n}${ordinal(n)}`; };
function ordinal(n) { const s = ["th", "st", "nd", "rd"], v = n % 100; return s[(v - 20) % 10] || s[v] || s[0]; }
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const api = async (url, opts) => {
  const r = await fetch(url, opts);
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  return r.json();
};

// ------------------------------------------------------------- colour scale
function hexRgb(h) { h = h.trim().replace("#", ""); return [0, 2, 4].map(i => parseInt(h.slice(i, i + 2), 16)); }
const css = getComputedStyle(document.documentElement);
const GOOD = hexRgb(css.getPropertyValue("--good")), BAD = hexRgb(css.getPropertyValue("--bad"));
const MIDC = hexRgb(css.getPropertyValue("--surface-2"));
function stuffBg(v) {
  if (v == null) return "transparent";
  const t = Math.max(-1, Math.min(1, (v - 100) / 25));
  const c = t > 0 ? GOOD : BAD;
  return `rgba(${c.join(",")},${(Math.abs(t) * 0.5).toFixed(3)})`;
}
function stuffFill(v) { // solid colour for dots
  const t = Math.max(-1, Math.min(1, (v - 100) / 30)), c = t > 0 ? GOOD : BAD, a = Math.abs(t);
  return `rgb(${MIDC.map((m, i) => Math.round(m + (c[i] - m) * (0.25 + 0.75 * a))).join(",")})`;
}
const stuffCell = v => v == null ? '<span class="muted">–</span>'
  : `<span class="stuff" style="background:${stuffBg(v)}">${Math.round(v)}</span>`;

// ------------------------------------------------------------- state
const S = { meta: null, season: null, data: null, byPitcher: {}, sort: {} };

// ------------------------------------------------------------- generic table
function renderTable(table, cols, rows, key, onClick) {
  const st = S.sort[key] || (S.sort[key] = { col: cols.find(c => c.defaultSort)?.key, asc: false });
  const col = cols.find(c => c.key === st.col);
  if (col) {
    const val = col.sortVal || (r => r[col.key]);
    rows = [...rows].sort((a, b) => {
      const x = val(a), y = val(b);
      if (x == null) return 1; if (y == null) return -1;
      return (x < y ? -1 : x > y ? 1 : 0) * (st.asc ? 1 : -1);
    });
  }
  const head = cols.map(c => `<th data-k="${c.key}" class="${c.l ? "l " : ""}${c.key === st.col ? "sorted" + (st.asc ? " asc" : "") : ""}"
    ${c.title ? `title="${esc(c.title)}"` : ""}>${c.label}</th>`).join("");
  const body = rows.map((r, i) => `<tr data-i="${i}">${cols.map(c =>
    `<td class="${c.l ? "l " : ""}${c.cls || ""}">${c.render ? c.render(r, i) : esc(r[c.key])}</td>`).join("")}</tr>`).join("");
  table.innerHTML = `<thead><tr>${head}</tr></thead><tbody>${body}</tbody>`;
  table.querySelectorAll("th").forEach(th => th.onclick = () => {
    const k = th.dataset.k;
    if (k === "rank") return;
    st.asc = st.col === k ? !st.asc : false; st.col = k;
    renderTable(table, cols, rows, key, onClick);
  });
  if (onClick) table.querySelectorAll("tbody tr").forEach(tr => tr.onclick = () => onClick(rows[+tr.dataset.i]));
  return rows;
}

function toCSV(cols, rows) {
  const q = v => v == null ? "" : /[",\n]/.test(String(v)) ? `"${String(v).replace(/"/g, '""')}"` : v;
  return [cols.map(c => q(c.label)).join(","),
    ...rows.map((r, i) => cols.map(c => q(c.csv ? c.csv(r, i) : r[c.key])).join(","))].join("\n");
}
function download(name, text) {
  const a = el("a", { href: URL.createObjectURL(new Blob([text], { type: "text/csv" })), download: name });
  document.body.append(a); a.click(); a.remove();
}

function segValue(id) { return $(`#${id} .on`).dataset.v; }
function wireSeg(id, cb) {
  document.querySelectorAll(`#${id} button`).forEach(b => b.onclick = () => {
    document.querySelectorAll(`#${id} button`).forEach(x => x.classList.toggle("on", x === b)); cb();
  });
}

// ------------------------------------------------------------- pitchers tab
const pitcherState = { rows: [], cols: [] };
function renderPitchers() {
  const q = $("#p-search").value.trim().toLowerCase(), hand = segValue("p-hand"),
    role = S.meta.has_roles ? segValue("p-role") : "", min = +$("#p-min").value || 0;
  const rows = S.data.pitchers.filter(r => r.n >= min && (!hand || r.hand === hand)
    && (!role || r.role === role) && (!q || (r.name || "").toLowerCase().includes(q)));
  const types = S.meta.pitch_types.filter(t => S.data.pitch_types.some(p => p.pitch_type === t)).slice(0, 9);
  const ptOf = (r, t) => S.byPitcher[r.pitcher]?.[t];
  const cols = [
    { key: "rank", label: "#", cls: "rank", render: (r, i) => i + 1, csv: (r, i) => i + 1 },
    { key: "name", label: "Pitcher", l: 1, cls: "name" },
    { key: "hand", label: "Hand" },
    ...(S.meta.has_roles ? [{ key: "role", label: "Role" }] : []),
    { key: "n", label: "Pitches", render: r => f0(r.n) },
    { key: "fb_velo", label: "FB velo", render: r => f1(r.fb_velo), title: "Primary fastball velocity" },
    { key: "stuff_plus", label: "Stuff+", render: r => stuffCell(r.stuff_plus), defaultSort: 1,
      csv: r => r.stuff_plus?.toFixed(1) },
    { key: "grade", label: "Grade", cls: "grade", title: "20-80 scale vs. qualified pitchers" },
    ...types.map(t => ({
      key: "pt_" + t, label: `<span class="dot" style="background:${ptColor(t)}"></span>${t}`,
      title: `${ptName(t)} Stuff+ (usage)`,
      sortVal: r => ptOf(r, t)?.stuff_plus,
      render: r => { const p = ptOf(r, t); return p ? `<span title="${f1(p.usage)}% usage, ${f1(p.velo)} mph">${stuffCell(p.stuff_plus)}</span>` : ""; },
      csv: r => ptOf(r, t)?.stuff_plus?.toFixed(1),
    })),
  ];
  pitcherState.cols = cols;
  pitcherState.rows = renderTable($("#p-table"), cols, rows, "pitchers", r => openPitcher(r.pitcher));
  $("#p-count").textContent = `${rows.length} pitchers`;
}

// ------------------------------------------------------------- pitches tab
const typeFilter = { t: new Set(), x: new Set() };
function renderTypeChips(id, set, cb) {
  const types = S.meta.pitch_types.filter(t => S.data.pitch_types.some(p => p.pitch_type === t));
  const box = $(`#${id}`); box.innerHTML = "";
  for (const t of types) {
    const b = el("button", { class: "chip" + (set.has(t) ? " on" : ""), title: ptName(t) },
      `<span class="dot" style="background:${ptColor(t)}"></span>${t}`);
    b.onclick = () => { set.has(t) ? set.delete(t) : set.add(t); b.classList.toggle("on"); cb(); };
    box.append(b);
  }
}
function filteredPitches(prefix, set) {
  const q = $(`#${prefix}-search`)?.value.trim().toLowerCase() || "", hand = segValue(`${prefix}-hand`),
    min = +$(`#${prefix}-min`).value || 0;
  return S.data.pitch_types.filter(r => r.n >= min && (!hand || r.hand === hand)
    && (!set.size || set.has(r.pitch_type)) && (!q || (r.name || "").toLowerCase().includes(q)));
}
const pitchState = { rows: [], cols: [] };
function renderPitches() {
  const rows = filteredPitches("t", typeFilter.t);
  const cols = [
    { key: "rank", label: "#", cls: "rank", render: (r, i) => i + 1, csv: (r, i) => i + 1 },
    { key: "name", label: "Pitcher", l: 1, cls: "name" },
    { key: "hand", label: "Hand" },
    { key: "pitch_type", label: "Pitch", l: 1, render: r => `<span class="dot" style="background:${ptColor(r.pitch_type)}"></span>${ptName(r.pitch_type)}` },
    { key: "n", label: "Pitches", render: r => f0(r.n) },
    { key: "usage", label: "Usage %", render: r => f1(r.usage), csv: r => r.usage?.toFixed(1) },
    { key: "velo", label: "Velo", render: r => f1(r.velo), csv: r => r.velo?.toFixed(1) },
    { key: "spin", label: "Spin", render: r => f0(r.spin), csv: r => r.spin?.toFixed(0) },
    { key: "ivb", label: "IVB", render: r => f1(r.ivb), title: "Induced vertical break (in)", csv: r => r.ivb?.toFixed(1) },
    { key: "hb", label: "HB", render: r => f1(r.hb), title: "Horizontal break, + = arm side (in)", csv: r => r.hb?.toFixed(1) },
    { key: "ext", label: "Ext", render: r => f1(r.ext), csv: r => r.ext?.toFixed(2) },
    { key: "rel_height", label: "Rel Z", render: r => f1(r.rel_height), csv: r => r.rel_height?.toFixed(2) },
    { key: "stuff_plus", label: "Stuff+", render: r => stuffCell(r.stuff_plus), defaultSort: 1, csv: r => r.stuff_plus?.toFixed(1) },
    { key: "grade", label: "Grade", cls: "grade", title: "20-80 scale vs. same pitch type" },
  ];
  pitchState.cols = cols;
  pitchState.rows = renderTable($("#t-table"), cols, rows, "pitches", r => openPitcher(r.pitcher));
  $("#t-count").textContent = `${rows.length} pitches`;
}

// ------------------------------------------------------------- explore tab
const AXES = { velo: "Velocity (mph)", spin: "Spin (rpm)", ivb: "IVB (in)", hb: "HB, arm side (in)",
  ext: "Extension (ft)", rel_height: "Release height (ft)", rel_side: "Release side (ft)",
  usage: "Usage (%)", stuff_plus: "Stuff+" };
function niceTicks(lo, hi, n = 6) {
  const span = hi - lo || 1, step0 = span / n, mag = 10 ** Math.floor(Math.log10(step0));
  const step = [1, 2, 2.5, 5, 10].map(m => m * mag).find(s => s >= step0);
  const out = []; for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(+v.toFixed(6));
  return out;
}
function corr(xs, ys) {
  const n = xs.length; if (n < 3) return null;
  const mx = xs.reduce((a, b) => a + b) / n, my = ys.reduce((a, b) => a + b) / n;
  let sxy = 0, sx = 0, sy = 0;
  for (let i = 0; i < n; i++) { const dx = xs[i] - mx, dy = ys[i] - my; sxy += dx * dy; sx += dx * dx; sy += dy * dy; }
  return sxy / Math.sqrt(sx * sy);
}
function renderExplore() {
  const xk = $("#x-axis").value, yk = $("#y-axis").value;
  const rows = filteredPitches("x", typeFilter.x).filter(r => r[xk] != null && r[yk] != null);
  const box = $("#scatter"), W = Math.max(box.clientWidth - 28, 320), H = Math.min(560, Math.max(360, W * 0.62));
  const m = { l: 52, r: 14, t: 12, b: 42 };
  const xs = rows.map(r => r[xk]), ys = rows.map(r => r[yk]);
  const pad = (a, b) => { const d = (b - a) * 0.05 || 1; return [a - d, b + d]; };
  const [x0, x1] = pad(Math.min(...xs), Math.max(...xs)), [y0, y1] = pad(Math.min(...ys), Math.max(...ys));
  const sx = v => m.l + (v - x0) / (x1 - x0) * (W - m.l - m.r), sy = v => H - m.b - (v - y0) / (y1 - y0) * (H - m.t - m.b);
  let svg = `<svg width="${W}" height="${H}" viewBox="0 0 ${W} ${H}">`;
  if (rows.length) {
    for (const t of niceTicks(x0, x1)) svg += `<line class="axis" x1="${sx(t)}" x2="${sx(t)}" y1="${m.t}" y2="${H - m.b}"/><text x="${sx(t)}" y="${H - m.b + 16}" text-anchor="middle">${t}</text>`;
    for (const t of niceTicks(y0, y1)) svg += `<line class="axis" x1="${m.l}" x2="${W - m.r}" y1="${sy(t)}" y2="${sy(t)}"/><text x="${m.l - 6}" y="${sy(t) + 4}" text-anchor="end">${t}</text>`;
    // draw best last so they sit on top
    const order = rows.map((r, i) => i).sort((a, b) => Math.abs(rows[a].stuff_plus - 100) - Math.abs(rows[b].stuff_plus - 100));
    for (const i of order) { const r = rows[i];
      svg += `<circle data-i="${i}" cx="${sx(r[xk]).toFixed(1)}" cy="${sy(r[yk]).toFixed(1)}" r="4.5" fill="${stuffFill(r.stuff_plus)}" stroke="var(--surface)" stroke-width=".8"/>`; }
  } else svg += `<text x="${W / 2}" y="${H / 2}" text-anchor="middle">No pitches match these filters</text>`;
  svg += `<text x="${(m.l + W - m.r) / 2}" y="${H - 6}" text-anchor="middle">${AXES[xk]}</text>`;
  svg += `<text transform="translate(13 ${(m.t + H - m.b) / 2}) rotate(-90)" text-anchor="middle">${AXES[yk]}</text></svg>`;
  box.innerHTML = svg;
  const tip = $("#tooltip");
  box.querySelectorAll("circle").forEach(c => {
    const r = rows[+c.dataset.i];
    c.onmousemove = e => {
      tip.hidden = false; tip.style.left = e.clientX + 14 + "px"; tip.style.top = e.clientY + 10 + "px";
      tip.innerHTML = `<b>${esc(r.name)}</b> · ${ptName(r.pitch_type)}<br>Stuff+ ${f1(r.stuff_plus)} · ${f1(r.velo)} mph · ${f1(r.ivb)} IVB · ${f1(r.hb)} HB<br><span class="muted">${f0(r.n)} pitches</span>`;
    };
    c.onmouseleave = () => tip.hidden = true;
    c.onclick = () => { tip.hidden = true; openPitcher(r.pitcher); };
    c.style.cursor = "pointer";
  });
  const r = corr(xs, ys);
  $("#x-corr").textContent = r == null ? "" : `${rows.length} pitches · r = ${r.toFixed(2)}`;
}

// ------------------------------------------------------------- movement plot
function movementSVG(pitches, size = 300) {
  const lim = 25, m = 28, w = size, s = v => m + (v + lim) / (2 * lim) * (w - 2 * m);
  let g = `<svg width="${w}" height="${w}" viewBox="0 0 ${w} ${w}" role="img" aria-label="Pitch movement plot">`;
  for (const t of [-20, -10, 0, 10, 20]) {
    g += `<line class="${t ? "axis" : "zero"}" x1="${s(t)}" x2="${s(t)}" y1="${m}" y2="${w - m}"/>`;
    g += `<line class="${t ? "axis" : "zero"}" x1="${m}" x2="${w - m}" y1="${s(-t)}" y2="${s(-t)}"/>`;
    g += `<text x="${s(t)}" y="${w - m + 14}" text-anchor="middle">${t}</text><text x="${m - 5}" y="${s(-t) + 4}" text-anchor="end">${t}</text>`;
  }
  g += `<text x="${m}" y="${m - 8}">IVB (in)</text><text x="${w - m}" y="${m - 8}" text-anchor="end">arm side →</text>`;
  for (const p of pitches) {
    if (p.ivb == null || p.hb == null) continue;
    const r = 6 + Math.sqrt(Math.max(p.usage || 10, 1)) * 1.4, x = s(Math.max(-lim, Math.min(lim, p.hb))), y = s(-Math.max(-lim, Math.min(lim, p.ivb)));
    g += `<circle cx="${x}" cy="${y}" r="${r}" fill="${ptColor(p.pitch_type)}" fill-opacity=".85" stroke="var(--surface)" stroke-width="1.5"><title>${ptName(p.pitch_type)}: ${f1(p.ivb)} IVB, ${f1(p.hb)} HB</title></circle>`;
    g += `<text class="ptlabel" x="${x + r + 3}" y="${y + 4}">${p.pitch_type}</text>`;
  }
  return g + "</svg>";
}

function trendSVG(seasons, w = 300, h = 140) {
  const pts = seasons.filter(s => s.stuff_plus != null);
  if (pts.length < 2) return `<p class="muted">Only one season of data.</p>`;
  const m = { l: 34, r: 12, t: 12, b: 24 };
  const ys = pts.map(p => p.stuff_plus), lo = Math.min(90, ...ys) - 2, hi = Math.max(110, ...ys) + 2;
  const yr = pts.map(p => p.year), x0 = Math.min(...yr), x1 = Math.max(...yr);
  const sx = v => m.l + (v - x0) / ((x1 - x0) || 1) * (w - m.l - m.r), sy = v => h - m.b - (v - lo) / (hi - lo) * (h - m.t - m.b);
  let g = `<svg width="${w}" height="${h}" viewBox="0 0 ${w} ${h}">`;
  for (const t of niceTicks(lo, hi, 4)) g += `<line class="${t === 100 ? "zero" : "axis"}" x1="${m.l}" x2="${w - m.r}" y1="${sy(t)}" y2="${sy(t)}"/><text x="${m.l - 5}" y="${sy(t) + 4}" text-anchor="end">${t}</text>`;
  g += `<polyline fill="none" stroke="var(--text)" stroke-width="2" points="${pts.map(p => `${sx(p.year)},${sy(p.stuff_plus)}`).join(" ")}"/>`;
  for (const p of pts) g += `<circle cx="${sx(p.year)}" cy="${sy(p.stuff_plus)}" r="4.5" fill="${stuffFill(p.stuff_plus)}" stroke="var(--text)"><title>${p.year}: ${f1(p.stuff_plus)}</title></circle><text x="${sx(p.year)}" y="${h - 6}" text-anchor="middle">${p.year}</text>`;
  return g + "</svg>";
}

// ------------------------------------------------------------- pitcher drawer
async function openPitcher(id) {
  const d = await api(`/api/pitcher/${id}`);
  const cur = d.seasons.find(s => s.year === S.season) || d.seasons[d.seasons.length - 1];
  const arsenal = d.pitch_types.filter(p => p.year === cur.year);
  const body = $("#drawer-body");
  body.innerHTML = `
    <h2>${esc(cur.name)}</h2>
    <div class="muted">${cur.hand}HP${cur.role ? " · " + cur.role : ""} · ${cur.year} · ${f0(cur.n)} pitches</div>
    <div class="stats">
      <div><div class="label">Stuff+</div><div class="big">${f0(cur.stuff_plus)}</div></div>
      <div><div class="label">Grade</div><div class="big">${cur.grade ?? "–"}</div></div>
      ${cur.fb_velo ? `<div><div class="label">FB velo</div><div class="big">${f1(cur.fb_velo)}</div></div>` : ""}
    </div>
    <div class="card"><div class="table-wrap"><table id="d-table"></table></div></div>
    <div class="row">
      <div class="card"><h3>Movement</h3>${movementSVG(arsenal, 300)}</div>
      <div class="card"><h3>Stuff+ by season</h3>${trendSVG(d.seasons)}</div>
    </div>
    <button id="d-grade">Open this arsenal in the grader</button>`;
  S.sort["drawer:" + id] ||= { col: "usage", asc: false };
  renderTable($("#d-table"), [
    { key: "pitch_type", label: "Pitch", l: 1, render: r => `<span class="dot" style="background:${ptColor(r.pitch_type)}"></span>${ptName(r.pitch_type)}` },
    { key: "usage", label: "Usage", render: r => f1(r.usage) + "%" },
    { key: "velo", label: "Velo", render: r => f1(r.velo) },
    { key: "spin", label: "Spin", render: r => f0(r.spin) },
    { key: "ivb", label: "IVB", render: r => f1(r.ivb) },
    { key: "hb", label: "HB", render: r => f1(r.hb) },
    { key: "stuff_plus", label: "Stuff+", render: r => stuffCell(r.stuff_plus) },
    { key: "grade", label: "Grade", cls: "grade" },
  ], arsenal, "drawer:" + id);
  $("#d-grade").onclick = () => { loadArsenal(cur.hand, arsenal); closeDrawer(); switchTab("grader"); };
  $("#drawer").hidden = false;
}
function closeDrawer() { $("#drawer").hidden = true; }

// ------------------------------------------------------------- grader
const G_FIELDS = [
  ["usage", "Usage %", 1, false, "Share of pitches thrown"], ["velo", "Velo", 0.1, false, "Release speed (mph)"],
  ["spin", "Spin", 10, false, "Spin rate (rpm)"], ["ivb", "IVB", 0.1, false, "Induced vertical break (in)"],
  ["hb", "HB", 0.1, false, "Horizontal break (in), + = arm side"], ["ext", "Ext", 0.1, false, "Extension (ft)"],
  ["rel_height", "Rel Z", 0.1, false, "Release height (ft)"], ["rel_side", "Rel X", 0.1, false, "Release side (ft), + = arm side"],
  ["spin_axis", "Axis", 1, true, "Spin axis (deg, 180 = pure backspin). Optional"],
  ["arm_angle", "Arm°", 1, true, "Arm angle (deg). Optional"],
];
const G = { hand: "R", pitches: [] };
function newPitch(t) { return { pitch_type: t, ...TEMPLATES[t], ...ARM, spin_axis: null, arm_angle: null }; }
function defaultArsenal() { return [newPitch("FF"), newPitch("SL"), newPitch("CH")].map((p, i) => ({ ...p, usage: [50, 30, 20][i] })); }
function saveG() { try { localStorage.setItem("stuffplus.grader", JSON.stringify(G)); } catch (e) { /* storage unavailable */ } }
function restoreG() {
  try { const s = JSON.parse(localStorage.getItem("stuffplus.grader")); if (s?.pitches?.length) Object.assign(G, s); } catch (e) { /* ignore */ }
  if (!G.pitches.length) G.pitches = defaultArsenal();
}
function loadArsenal(hand, rows) {
  const r1 = v => v == null ? null : Math.round(v * 10) / 10;
  G.hand = hand;
  G.pitches = [...rows].sort((a, b) => b.usage - a.usage).map(p => ({
    pitch_type: p.pitch_type, usage: Math.round(p.usage), velo: r1(p.velo), spin: Math.round(p.spin / 10) * 10,
    ivb: r1(p.ivb), hb: r1(p.hb), ext: r1(p.ext), rel_height: r1(p.rel_height), rel_side: r1(p.rel_side),
    spin_axis: p.spin_axis == null ? null : Math.round(p.spin_axis), arm_angle: p.arm_angle == null ? null : Math.round(p.arm_angle),
  }));
  $("#g-note").dataset.src = `Loaded from ${rows[0]?.name} (${rows[0]?.year}).`;
  renderGraderInputs(); gradeNow();
}
function renderGraderInputs() {
  document.querySelectorAll("#g-hand button").forEach(b => b.classList.toggle("on", b.dataset.v === G.hand));
  const types = Object.keys(TEMPLATES);
  const head = `<thead><tr><th class="l">Pitch</th>${G_FIELDS.map(([, l, , , tip]) =>
    `<th title="${tip}">${l}</th>`).join("")}<th></th></tr></thead>`;
  const body = G.pitches.map((p, i) => `<tr data-i="${i}"><td class="l"><select data-k="pitch_type">${
    types.map(t => `<option value="${t}" title="${ptName(t)}" ${t === p.pitch_type ? "selected" : ""}>${t}</option>`).join("")}</select></td>${
    G_FIELDS.map(([k, , step, opt]) => `<td><input type="number" data-k="${k}" step="${step}" value="${p[k] ?? ""}" ${opt ? 'class="opt" placeholder="auto"' : ""}></td>`).join("")
    }<td><button class="icon" data-del="${i}" title="Remove" aria-label="Remove pitch">✕</button></td></tr>`).join("");
  const t = $("#g-table"); t.innerHTML = head + `<tbody>${body}</tbody>`;
  t.querySelectorAll("input, select").forEach(inp => inp.oninput = () => {
    const i = +inp.closest("tr").dataset.i, k = inp.dataset.k;
    G.pitches[i][k] = k === "pitch_type" ? inp.value : (inp.value === "" ? null : +inp.value);
    saveG(); scheduleGrade();
  });
  t.querySelectorAll("[data-del]").forEach(b => b.onclick = () => {
    if (G.pitches.length === 1) return;
    G.pitches.splice(+b.dataset.del, 1); saveG(); renderGraderInputs(); gradeNow();
  });
}
let gradeTimer = null, gradeSeq = 0;
function scheduleGrade() { clearTimeout(gradeTimer); gradeTimer = setTimeout(gradeNow, 120); }
async function gradeNow() {
  const seq = ++gradeSeq;
  const req = ["velo", "spin", "ivb", "hb", "ext", "rel_height", "rel_side"];
  const bad = G.pitches.findIndex(p => req.some(k => p[k] == null || Number.isNaN(p[k])));
  if (bad >= 0) { $("#g-note").textContent = `Fill in every required field for pitch ${bad + 1}.`; return; }
  try {
    const res = await api("/api/grade", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ hand: G.hand, season: S.season, pitches: G.pitches.map(p => ({ ...p, usage: p.usage ?? 0 })) }) });
    if (seq === gradeSeq) renderGrade(res);
  } catch (e) {
    if (seq !== gradeSeq) return;
    let msg = e.message;
    try { msg = JSON.parse(msg.slice(msg.indexOf("{"))).detail.map(d => `${d.loc.slice(-1)[0]}: ${d.msg}`).join("; "); } catch (_) { /* raw */ }
    $("#g-note").textContent = "Couldn't grade: " + msg;
  }
}
function renderGrade(res) {
  const o = res.overall;
  $("#g-overall").textContent = f0(o.stuff_plus); $("#g-overall").style.color = o.stuff_plus >= 100 ? "var(--good)" : "var(--bad)";
  $("#g-grade").textContent = o.grade ?? "–"; $("#g-pct").textContent = pct(o.percentile);
  $("#g-bars").innerHTML = res.pitches.map((p, i) => {
    const v = p.stuff_plus, lo = 50, hi = 150, x = Math.max(lo, Math.min(hi, v)), mid = 50;
    const pos = (x - lo) / (hi - lo) * 100, left = Math.min(pos, mid), width = Math.abs(pos - mid);
    return `<div class="barrow" title="${ptName(p.pitch_type)}${p.is_primary_fb ? " (primary fastball)" : ""}: ${p.percentile == null ? "" : pct(p.percentile) + " percentile among " + p.pitch_type + "s"}">
      <b><span class="dot" style="background:${ptColor(p.pitch_type)}"></span>${p.pitch_type}</b>
      <div class="bartrack"><div class="mid"></div><div class="fill" style="left:${left}%;width:${width}%;background:${stuffFill(v)}"></div></div>
      <span class="barval">${f0(v)} <span class="grade">· ${p.grade ?? "–"}${p.percentile == null ? "" : " · " + pct(p.percentile)}</span></span></div>`;
  }).join("");
  $("#g-move").innerHTML = movementSVG(G.pitches, 300);
  const fb = res.pitches.find(p => p.is_primary_fb);
  $("#g-note").textContent = [$("#g-note").dataset.src || "",
    `Scaled to ${res.season_scale} MLB pitches. Overall is usage-weighted.`,
    fb ? `Primary fastball for velo/movement differences: ${fb.pitch_type}.` : "",
    "Grades are 20-80 vs. the same pitch type; percentile is among MLB pitchers' versions of that pitch."].filter(Boolean).join(" ");
}
function wireGrader() {
  wireSeg("g-hand", () => { G.hand = segValue("g-hand"); saveG(); gradeNow(); });
  $("#g-add-type").innerHTML = Object.keys(TEMPLATES).map(t => `<option value="${t}">${t} · ${ptName(t)}</option>`).join("");
  $("#g-add").onclick = () => { if (G.pitches.length >= 12) return; G.pitches.push(newPitch($("#g-add-type").value)); saveG(); renderGraderInputs(); gradeNow(); };
  $("#g-reset").onclick = () => { G.hand = "R"; G.pitches = defaultArsenal(); delete $("#g-note").dataset.src; saveG(); renderGraderInputs(); gradeNow(); };
  const inp = $("#g-load"), list = $("#g-load-list");
  inp.oninput = () => {
    const q = inp.value.trim().toLowerCase();
    if (!q) { list.hidden = true; return; }
    const hits = S.data.pitchers.filter(r => (r.name || "").toLowerCase().includes(q)).sort((a, b) => b.n - a.n).slice(0, 12);
    list.innerHTML = hits.map(h => `<div data-id="${h.pitcher}">${esc(h.name)} <span class="muted">${h.hand}HP · ${S.season} · Stuff+ ${f0(h.stuff_plus)}</span></div>`).join("")
      || `<div class="muted">No match in ${S.season}</div>`;
    list.hidden = false;
    list.querySelectorAll("[data-id]").forEach(d => d.onclick = () => {
      const p = S.data.pitchers.find(r => r.pitcher === +d.dataset.id);
      loadArsenal(p.hand, Object.values(S.byPitcher[p.pitcher]).map(r => ({ ...r })));
      inp.value = ""; list.hidden = true;
    });
  };
  document.addEventListener("click", e => { if (!e.target.closest(".loader")) list.hidden = true; });
}

// ------------------------------------------------------------- tabs & boot
function switchTab(name) {
  document.querySelectorAll("#tabs button").forEach(b => b.classList.toggle("active", b.dataset.tab === name));
  document.querySelectorAll(".tab").forEach(t => t.classList.toggle("active", t.id === "tab-" + name));
  if (name === "explore") renderExplore();
  try { localStorage.setItem("stuffplus.tab", name); } catch (e) { /* ignore */ }
}
async function loadSeason(season) {
  S.season = season;
  $("#season-note").textContent = S.meta.season_notes[season] || "";
  S.data = await api(`/api/leaderboard/${season}`);
  S.byPitcher = {};
  for (const p of S.data.pitch_types) (S.byPitcher[p.pitcher] ||= {})[p.pitch_type] = p;
  renderTypeChips("t-types", typeFilter.t, renderPitches);
  renderTypeChips("x-types", typeFilter.x, renderExplore);
  renderPitchers(); renderPitches();
  if ($("#tab-explore").classList.contains("active")) renderExplore();
  gradeNow();
}
async function boot() {
  S.meta = await api("/api/meta");
  const m = S.meta, tr = m.train_years;
  $("#model-info").textContent = m.label;
  $("#model-info").title = `Features: ${m.features.join(", ")}\nBuilt ${m.built_at}`;
  $("#demo-banner").hidden = !m.synthetic;
  if (m.has_roles) $("#p-role").hidden = false;
  if (m.synthetic) { $("#p-min").value = 100; }
  $("#season").innerHTML = [...m.seasons].reverse().map(s => `<option>${s}</option>`).join("");
  $("#season").onchange = e => loadSeason(+e.target.value);
  document.querySelectorAll("#tabs button").forEach(b => b.onclick = () => switchTab(b.dataset.tab));
  ["p-search", "p-min"].forEach(id => $("#" + id).oninput = renderPitchers);
  ["t-search", "t-min"].forEach(id => $("#" + id).oninput = renderPitches);
  $("#x-min").oninput = renderExplore;
  wireSeg("p-hand", renderPitchers); wireSeg("p-role", renderPitchers);
  wireSeg("t-hand", renderPitches); wireSeg("x-hand", renderExplore);
  $("#p-csv").onclick = () => download(`stuffplus_pitchers_${S.season}.csv`, toCSV(pitcherState.cols.map(c => ({ ...c, label: c.label.replace(/<[^>]+>/g, "") })), pitcherState.rows));
  $("#t-csv").onclick = () => download(`stuffplus_pitches_${S.season}.csv`, toCSV(pitchState.cols, pitchState.rows));
  const opts = Object.entries(AXES).map(([k, v]) => `<option value="${k}">${v}</option>`).join("");
  $("#x-axis").innerHTML = opts; $("#y-axis").innerHTML = opts;
  $("#x-axis").value = "hb"; $("#y-axis").value = "ivb";
  $("#x-axis").onchange = $("#y-axis").onchange = renderExplore;
  window.addEventListener("resize", () => { if ($("#tab-explore").classList.contains("active")) renderExplore(); });
  $("#drawer-close").onclick = closeDrawer;
  $("#drawer").onclick = e => { if (e.target.id === "drawer") closeDrawer(); };
  document.addEventListener("keydown", e => { if (e.key === "Escape") closeDrawer(); });
  restoreG(); wireGrader(); renderGraderInputs();
  await loadSeason(m.latest_season);
  let tab = "pitchers"; try { tab = localStorage.getItem("stuffplus.tab") || tab; } catch (e) { /* ignore */ }
  switchTab(tab);
}
boot().catch(e => { document.querySelector("main").innerHTML = `<div class="card">Failed to load: ${esc(e.message)}</div>`; });
