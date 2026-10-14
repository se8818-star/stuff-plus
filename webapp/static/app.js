// Stuff+ leaderboards and grader. Plain JS, no build step.
"use strict";

// Pitch-type colours avoid red and blue, which mean better / worse than average.
// Each is at least 3:1 against the light (#fff, #f6f7f9) and dark (#171b22, #0f1217) surfaces.
const PT = {
  FF: ["Four-seam", "#b07d12"], SI: ["Sinker", "#8c6239"], FC: ["Cutter", "#7a7a1e"],
  SL: ["Slider", "#2f8a4e"], ST: ["Sweeper", "#11867c"], SV: ["Slurve", "#5c8a1e"],
  CU: ["Curveball", "#8456c9"], KC: ["Knuckle curve", "#9a5b9e"], CH: ["Changeup", "#b24f8e"],
  FS: ["Splitter", "#687585"], FO: ["Forkball", "#7a7a7a"], SC: ["Screwball", "#a08356"],
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
// cols are in CSV order; `front` lists keys shown first on screen (the CSV keeps its order)
const displayOrder = (cols, front) => [...front.map(k => cols.find(c => c.key === k)).filter(Boolean),
  ...cols.filter(c => !front.includes(c.key))];
function renderTable(table, cols, rows, key, onClick, focusKey) {
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
  const sortable = c => c.key !== "rank" && c.label !== "";
  const head = cols.map(c => {
    const sorted = c.key === st.col, aria = sorted ? ` aria-sort="${st.asc ? "ascending" : "descending"}"` : "";
    return `<th data-k="${c.key}" class="${c.l ? "l " : ""}${c.cls || ""} ${c.nx ? "nx " : ""}${sorted ? "sorted" + (st.asc ? " asc" : "") : ""}"${aria}
    ${c.title ? `title="${esc(c.title)}"` : ""}>${sortable(c) ? `<button type="button" class="th-btn">${c.label}</button>` : c.label}</th>`;
  }).join("");
  const body = rows.map((r, i) => `<tr data-i="${i}">${cols.map(c =>
    `<td class="${c.l ? "l " : ""}${c.cls || ""}${c.nx ? " nx" : ""}">${c.render ? c.render(r, i) : esc(r[c.key])}</td>`).join("")}</tr>`).join("");
  table.innerHTML = `<thead><tr>${head}</tr></thead><tbody>${body}</tbody>`;
  table.querySelectorAll("th .th-btn").forEach(b => b.onclick = () => {
    const k = b.closest("th").dataset.k;
    st.asc = st.col === k ? !st.asc : false; st.col = k;
    renderTable(table, cols, rows, key, onClick, k);
  });
  if (focusKey) table.querySelector(`th[data-k="${focusKey}"] .th-btn`)?.focus();
  // whole-row click for the mouse; the name button inside is the keyboard route
  if (onClick) table.querySelectorAll("tbody tr").forEach(tr => tr.onclick = () => onClick(rows[+tr.dataset.i], tr));
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

// name cell: a button (keyboard route to the drawer) plus a subline shown on phones
const nameCell = (r, sub) => `<button type="button" class="name-btn" data-pid="${r.pitcher}">${esc(r.name)}</button>`
  + `<span class="sub">${esc(sub.filter(Boolean).join(" · "))}</span>`;

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
    { key: "name", label: "Pitcher", l: 1, cls: "name",
      render: r => nameCell(r, [`${r.hand}HP`, r.role, `${f0(r.n)} pitches`]) },
    { key: "hand", label: "Hand", nx: 1 },
    ...(S.meta.has_roles ? [{ key: "role", label: "Role", nx: 1 }] : []),
    { key: "n", label: "Pitches", render: r => f0(r.n), nx: 1 },
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
  pitcherState.rows = renderTable($("#p-table"), displayOrder(cols, ["rank", "name", "stuff_plus", "grade"]), rows,
    "pitchers", (r, tr) => openPitcher(r.pitcher, tr));
  // "–" only appears when a pitch has a row but no score (the published season's 50-pitch minimum)
  const dash = S.data.pitch_types.some(p => p.stuff_plus == null);
  $("#p-key").textContent = dash
    ? "In the pitch columns, – means he threw it but too few times to get a score (under 50 pitches); blank means he didn't throw it."
    : "In the pitch columns, a blank cell means he didn't throw that pitch.";
  $("#p-count").textContent = `${rows.length} pitchers`;
}

// ------------------------------------------------------------- pitches tab
const typeFilter = { t: new Set(), x: new Set(), z: new Set() };
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
    { key: "name", label: "Pitcher", l: 1, cls: "name",
      render: r => nameCell(r, [ptName(r.pitch_type), `${r.hand}HP`, `${f0(r.n)} pitches`]) },
    { key: "hand", label: "Hand", nx: 1 },
    { key: "pitch_type", label: "Pitch", l: 1, render: r => `<span class="dot" style="background:${ptColor(r.pitch_type)}"></span>${ptName(r.pitch_type)}` },
    { key: "n", label: "Pitches", render: r => f0(r.n), nx: 1 },
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
  pitchState.rows = renderTable($("#t-table"), displayOrder(cols, ["rank", "name", "stuff_plus", "grade"]), rows,
    "pitches", (r, tr) => openPitcher(r.pitcher, tr));
  $("#t-key").textContent = S.data.pitch_types.some(p => p.stuff_plus == null)
    ? "– in Stuff+ means too few pitches to get a score (under 50)." : "";
  $("#t-count").textContent = `${rows.length} pitches`;
}

// ------------------------------------------------------------- explore tab
const AXES = { velo: "Velocity (mph)", spin: "Spin (rpm)", ivb: "IVB (in)", hb: "HB, arm side (in)",
  ext: "Extension (ft)", rel_height: "Release height (ft)", rel_side: "Release side (ft)",
  usage: "Usage (%)", stuff_plus: "Stuff+" };
const INCH_AXES = new Set(["ivb", "hb"]), INCH_LIM = 25;
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
  // square plot area; axes in inches use the same fixed -25..25 as the movement plot
  const box = $("#scatter"), m = { l: 52, r: 14, t: 12, b: 42 };
  const side = Math.min(Math.max(box.clientWidth - 28 - m.l - m.r, 220), 560);
  const W = side + m.l + m.r, H = side + m.t + m.b;
  const xs = rows.map(r => r[xk]), ys = rows.map(r => r[yk]);
  const pad = (a, b) => { const d = (b - a) * 0.05 || 1; return [a - d, b + d]; };
  const range = (k, vs) => INCH_AXES.has(k) ? [-INCH_LIM, INCH_LIM] : pad(Math.min(...vs), Math.max(...vs));
  const [x0, x1] = range(xk, xs), [y0, y1] = range(yk, ys);
  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
  const sx = v => m.l + (clamp(v, x0, x1) - x0) / (x1 - x0) * (W - m.l - m.r),
    sy = v => H - m.b - (clamp(v, y0, y1) - y0) / (y1 - y0) * (H - m.t - m.b);
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

// ------------------------------------------------------------- 3D explore
// A canvas scatter of HB x IVB x velocity in a fixed box (HB, IVB -25..25 in;
// velocity 65..100 mph). Drag rotates, wheel/pinch zooms, Shift-drag or two
// fingers pans; arrow keys, +/- and 0 do the same from the keyboard.
const VELO_LIM = [65, 100];
const V3 = { yaw: -0.65, pitch: 0.38, zoom: 1, panX: 0, panY: 0 };
const V3_HOME = { ...V3 };
const Z = { rows: [], proj: [], color: "stuff", raf: 0 };
function render3D() {
  Z.color = segValue("z-color");
  const rows = filteredPitches("z", typeFilter.z).filter(r => r.hb != null && r.ivb != null && r.velo != null);
  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
  // normalised to the box's -1..1 on each axis: x = HB, y = IVB (up), z = velocity
  Z.rows = rows.map(r => ({ r, x: clamp(r.hb, -INCH_LIM, INCH_LIM) / INCH_LIM, y: clamp(r.ivb, -INCH_LIM, INCH_LIM) / INCH_LIM,
    z: (clamp(r.velo, ...VELO_LIM) - VELO_LIM[0]) / (VELO_LIM[1] - VELO_LIM[0]) * 2 - 1,
    fill: Z.color === "type" ? ptColor(r.pitch_type) : stuffFill(r.stuff_plus) }));
  $("#z-count").textContent = `${rows.length} pitches`;
  const types = [...new Set(rows.map(r => r.pitch_type))];
  $("#z-legend").innerHTML = Z.color === "type"
    ? `<div class="type-key">${types.map(t => `<span><span class="dot" style="background:${ptColor(t)}"></span>${esc(ptName(t))}</span>`).join("")}</div>`
    : `<p>Colour is Stuff+: red is better than average, blue is worse.</p><div class="scale"><span>70</span><div class="bar"></div><span>130</span></div>`;
  $("#cube").setAttribute("aria-label", `3D scatter of ${rows.length} pitches: horizontal break, induced vertical break and velocity`);
  draw3D();
}
function draw3D() {
  cancelAnimationFrame(Z.raf);
  Z.raf = requestAnimationFrame(() => {
    const cv = $("#cube"), dpr = window.devicePixelRatio || 1, W = cv.clientWidth, H = cv.clientHeight;
    if (!W) return;
    if (cv.width !== Math.round(W * dpr)) { cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr); }
    const g = cv.getContext("2d"); g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, W, H);
    const cs = getComputedStyle(document.documentElement), col = k => cs.getPropertyValue(k).trim();
    const cy = Math.cos(V3.yaw), sy = Math.sin(V3.yaw), cp = Math.cos(V3.pitch), sp = Math.sin(V3.pitch);
    const scale = Math.min(W, H) * 0.25 * V3.zoom, D = 7, ox = W / 2 + V3.panX, oy = H / 2 + V3.panY;
    const P = (x, y, z) => {  // rotate (yaw about the vertical, then pitch), then perspective
      const x1 = x * cy - z * sy, z1 = x * sy + z * cy;
      const y2 = y * cp - z1 * sp, z2 = y * sp + z1 * cp;  // z2 > 0 is farther away
      const f = D / (D + z2);
      return [ox + x1 * scale * f, oy - y2 * scale * f, z2, f];
    };
    const line = (a, b, color, dash = []) => {
      const p = P(...a), q = P(...b);
      g.beginPath(); g.setLineDash(dash); g.strokeStyle = color; g.moveTo(p[0], p[1]); g.lineTo(q[0], q[1]); g.stroke();
    };
    const text = (s, at, align = "center") => { const p = P(...at); g.textAlign = align; g.fillText(s, p[0], p[1]); };
    g.lineWidth = 1; g.font = "11px -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif";
    // the box: 12 edges, plus light grid lines on the floor and back walls
    const border = col("--border"), muted = col("--muted");
    for (const t of [-0.8, -0.4, 0.4, 0.8]) {
      line([t, -1, -1], [t, -1, 1], border); line([-1, -1, t], [1, -1, t], border);
      line([t, -1, 1], [t, 1, 1], border); line([-1, t, 1], [1, t, 1], border);
    }
    for (const a of [-1, 1]) for (const b of [-1, 1]) {
      line([-1, a, b], [1, a, b], muted); line([a, -1, b], [a, 1, b], muted); line([a, b, -1], [a, b, 1], muted);
    }
    // 0,0 reference: HB = 0 and IVB = 0 on the walls, and the line through both
    line([0, -1, -1], [0, -1, 1], muted, [3, 3]); line([-1, 0, 1], [1, 0, 1], muted, [3, 3]);
    line([0, 0, -1], [0, 0, 1], muted, [3, 3]);
    g.setLineDash([]); g.fillStyle = muted;
    for (const t of [-20, -10, 0, 10, 20]) {
      text(String(t), [t / INCH_LIM, -1.08, -1.08]);
      text(String(t), [-1.1, t / INCH_LIM, -1], "right");
    }
    for (const v of [70, 80, 90, 100]) text(String(v), [1.12, -1.08, (v - VELO_LIM[0]) / (VELO_LIM[1] - VELO_LIM[0]) * 2 - 1], "left");
    g.fillStyle = col("--text"); g.font = "600 11px -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif";
    text("HB, arm side (in)", [0, -1.25, -1.25]); text("IVB (in)", [-1.25, 1.12, -1]); text("Velo (mph)", [1.25, -1.25, 0], "left");
    // points, far to near
    Z.proj = Z.rows.map(d => { const [sx, sy2, depth, f] = P(d.x, d.y, d.z); return { d, sx, sy: sy2, depth, rad: 3.6 * f * Math.sqrt(V3.zoom) }; })
      .sort((a, b) => b.depth - a.depth);
    const surf = col("--surface");
    for (const p of Z.proj) {
      g.beginPath(); g.arc(p.sx, p.sy, Math.max(1.5, p.rad), 0, 2 * Math.PI);
      g.fillStyle = p.d.fill; g.fill(); g.lineWidth = 0.6; g.strokeStyle = surf; g.stroke();
    }
    if (!Z.rows.length) { g.fillStyle = muted; g.textAlign = "center"; g.fillText("No pitches match these filters", W / 2, H / 2); }
  });
}
function hit3D(e) {
  const r = $("#cube").getBoundingClientRect(), x = e.clientX - r.left, y = e.clientY - r.top;
  let best = null, bd = 64;  // within 8px; the nearest-to-viewer dot wins ties
  for (let i = Z.proj.length - 1; i >= 0; i--) {
    const p = Z.proj[i], d = (p.sx - x) ** 2 + (p.sy - y) ** 2;
    if (d < bd) { bd = d; best = p; }
  }
  return best;
}
function wire3D() {
  const cv = $("#cube"), tip = $("#tooltip"), ptrs = new Map();
  let moved = 0, pinch = null;
  const reset = () => { Object.assign(V3, V3_HOME); draw3D(); };
  const zoomBy = k => { V3.zoom = Math.max(0.4, Math.min(8, V3.zoom * k)); draw3D(); };
  cv.addEventListener("pointerdown", e => { cv.setPointerCapture(e.pointerId); ptrs.set(e.pointerId, [e.clientX, e.clientY]); moved = 0; tip.hidden = true; });
  cv.addEventListener("pointermove", e => {
    if (!ptrs.has(e.pointerId)) {  // hover
      const p = hit3D(e);
      if (!p) { tip.hidden = true; cv.style.cursor = ""; return; }
      const r = p.d.r; cv.style.cursor = "pointer";
      tip.hidden = false; tip.style.left = e.clientX + 14 + "px"; tip.style.top = e.clientY + 10 + "px";
      tip.innerHTML = `<b>${esc(r.name)}</b> · ${ptName(r.pitch_type)}<br>Stuff+ ${f1(r.stuff_plus)} · ${f1(r.velo)} mph · ${f1(r.ivb)} IVB · ${f1(r.hb)} HB<br><span class="muted">${f0(r.n)} pitches</span>`;
      return;
    }
    const [px, py] = ptrs.get(e.pointerId), dx = e.clientX - px, dy = e.clientY - py;
    ptrs.set(e.pointerId, [e.clientX, e.clientY]); moved += Math.abs(dx) + Math.abs(dy);
    if (ptrs.size === 2) {  // two fingers: pinch to zoom, move together to pan
      const [a, b] = [...ptrs.values()], dist = Math.hypot(a[0] - b[0], a[1] - b[1]);
      if (pinch) V3.zoom = Math.max(0.4, Math.min(8, V3.zoom * dist / pinch));
      pinch = dist; V3.panX += dx / 2; V3.panY += dy / 2;
    } else if (e.shiftKey || e.buttons === 2) { V3.panX += dx; V3.panY += dy; }
    else { V3.yaw += dx * 0.01; V3.pitch = Math.max(-1.5, Math.min(1.5, V3.pitch + dy * 0.01)); }
    draw3D();
  });
  const up = e => {
    ptrs.delete(e.pointerId); if (ptrs.size < 2) pinch = null;
    if (e.type === "pointerup" && moved < 4) { const p = hit3D(e); if (p) { tip.hidden = true; openPitcher(p.d.r.pitcher); } }
  };
  cv.addEventListener("pointerup", up); cv.addEventListener("pointercancel", up);
  cv.addEventListener("pointerleave", () => { if (!ptrs.size) tip.hidden = true; });
  cv.addEventListener("contextmenu", e => e.preventDefault());
  cv.addEventListener("wheel", e => { e.preventDefault(); zoomBy(Math.exp(-e.deltaY * 0.0015)); }, { passive: false });
  cv.addEventListener("dblclick", reset);
  cv.addEventListener("keydown", e => {
    const step = 0.08, pan = 20, k = e.key;
    if (k === "ArrowLeft") e.shiftKey ? V3.panX -= pan : V3.yaw -= step;
    else if (k === "ArrowRight") e.shiftKey ? V3.panX += pan : V3.yaw += step;
    else if (k === "ArrowUp") e.shiftKey ? V3.panY -= pan : V3.pitch = Math.max(-1.5, V3.pitch - step);
    else if (k === "ArrowDown") e.shiftKey ? V3.panY += pan : V3.pitch = Math.min(1.5, V3.pitch + step);
    else if (k === "+" || k === "=") V3.zoom = Math.min(8, V3.zoom * 1.2);
    else if (k === "-") V3.zoom = Math.max(0.4, V3.zoom / 1.2);
    else if (k === "0") Object.assign(V3, V3_HOME);
    else return;
    e.preventDefault(); draw3D();
  });
  $("#z-zoom-in").onclick = () => zoomBy(1.25);
  $("#z-zoom-out").onclick = () => zoomBy(0.8);
  $("#z-reset").onclick = reset;
  $("#z-min").oninput = render3D;
  wireSeg("z-hand", render3D); wireSeg("z-color", render3D);
  // redraw when the theme flips so box and labels pick up the new colours
  matchMedia("(prefers-color-scheme: dark)").addEventListener?.("change", () => { if (S.tab === "explore3d") render3D(); });
}

// ------------------------------------------------------------- movement plot
function movementSVG(pitches, size = 300) {
  const lim = INCH_LIM, m = 28, w = size, s = v => m + (v + lim) / (2 * lim) * (w - 2 * m);
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
  const m = { l: 34, r: 22, t: 12, b: 24 };  // r leaves room for the last year's label
  const ys = pts.map(p => p.stuff_plus), lo = Math.min(90, ...ys) - 2, hi = Math.max(110, ...ys) + 2;
  const yr = pts.map(p => p.year), x0 = Math.min(...yr), x1 = Math.max(...yr);
  const sx = v => m.l + (v - x0) / ((x1 - x0) || 1) * (w - m.l - m.r), sy = v => h - m.b - (v - lo) / (hi - lo) * (h - m.t - m.b);
  let g = `<svg width="${w}" height="${h}" viewBox="0 0 ${w} ${h}" role="img" aria-label="Stuff+ by season, ${x0} to ${x1}">`;
  for (const t of niceTicks(lo, hi, 4)) g += `<line class="${t === 100 ? "zero" : "axis"}" x1="${m.l}" x2="${w - m.r}" y1="${sy(t)}" y2="${sy(t)}"/><text x="${m.l - 5}" y="${sy(t) + 4}" text-anchor="end">${t}</text>`;
  g += `<polyline fill="none" stroke="var(--text)" stroke-width="2" points="${pts.map(p => `${sx(p.year)},${sy(p.stuff_plus)}`).join(" ")}"/>`;
  for (const p of pts) g += `<circle cx="${sx(p.year)}" cy="${sy(p.stuff_plus)}" r="4.5" fill="${stuffFill(p.stuff_plus)}" stroke="var(--text)"><title>${p.year}: ${f1(p.stuff_plus)}</title></circle><text x="${sx(p.year)}" y="${h - 6}" text-anchor="middle">${p.year}</text>`;
  return g + "</svg>";
}

// ------------------------------------------------------------- pitcher drawer
// Opening a pitcher is a navigation (#/pitchers/<slug>), so links and Back work.
// The drawer is a modal dialog: the page behind it is inert while it is open.
const D = { id: null, opener: null, pushed: false, seq: 0 };
function openPitcher(id, from) {
  D.opener = from?.querySelector?.(".name-btn") || from || document.activeElement;
  const slug = S.slugOf[id];
  if (!slug) return showPitcher(id);
  D.pushed = true;
  go({ slug });
}
function setInert(on) {
  for (const e of [document.querySelector("header"), $("#demo-banner"), document.querySelector("main"),
    document.querySelector("footer")]) e?.toggleAttribute("inert", on);
}
async function showPitcher(id) {
  const seq = ++D.seq;
  const d = await api(`/api/pitcher/${id}`);
  if (seq !== D.seq) return;
  const cur = d.seasons.find(s => s.year === S.season) || d.seasons[d.seasons.length - 1];
  const arsenal = d.pitch_types.filter(p => p.year === cur.year);
  const body = $("#drawer-body");
  body.innerHTML = `
    <h2 id="drawer-title" tabindex="-1">${esc(cur.name)}</h2>
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
    <button type="button" id="d-grade">Open this arsenal in the grader</button>`;
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
  $("#d-grade").onclick = () => {
    loadArsenal(cur.hand, arsenal);
    D.pushed = false; D.opener = null; hideDrawer();
    go({ tab: "grader" }, true);  // replaces the drawer's history entry
  };
  const wasOpen = !$("#drawer").hidden;
  D.id = id;
  $("#drawer").hidden = false;
  if (!wasOpen) setInert(true);
  $("#drawer-title").focus();
}
// hide without touching history (the route already says "no drawer")
function hideDrawer() {
  D.seq++;  // drop a pitcher still loading
  if ($("#drawer").hidden) return;
  const id = D.id;
  $("#drawer").hidden = true; D.id = null; D.pushed = false; setInert(false);
  const back = D.opener?.isConnected && D.opener !== document.body ? D.opener
    : document.querySelector(`.tab.active [data-pid="${id}"]`);
  back?.focus(); D.opener = null;
}
// the user closes it: undo the navigation that opened it
function closeDrawer() {
  if ($("#drawer").hidden) return;
  if (D.pushed) history.back();  // hashchange hides it
  else go({ tab: S.tab }, true);
}

// ------------------------------------------------------------- grader
// spoken names for the grader's columns ("Four-seam velocity (mph)")
const G_SPOKEN = { usage: "usage (%)", velo: "velocity (mph)", spin: "spin rate (rpm)", ivb: "induced vertical break (in)",
  hb: "horizontal break (in)", ext: "extension (ft)", rel_height: "release height (ft)", rel_side: "release side (ft)",
  spin_axis: "spin axis (degrees)", arm_angle: "arm angle (degrees)" };
function gLabels(i) {
  const t = G.pitches[i].pitch_type, dup = G.pitches.filter(p => p.pitch_type === t).length > 1;
  const name = ptName(t) + (dup ? ` (row ${i + 1})` : "");
  return { type: `Pitch type, row ${i + 1}`, del: `Remove ${name}`, field: k => `${name} ${G_SPOKEN[k]}` };
}
function labelGraderRows() {
  $("#g-table").querySelectorAll("tbody tr").forEach(tr => {
    const L = gLabels(+tr.dataset.i);
    tr.querySelectorAll("[data-k]").forEach(e => e.setAttribute("aria-label",
      e.dataset.k === "pitch_type" ? L.type : L.field(e.dataset.k)));
    tr.querySelector("[data-del]")?.setAttribute("aria-label", L.del);
  });
}
const G_FIELDS = [
  ["usage", "Usage %", 1, false, "Share of pitches thrown"], ["velo", "Velo", 0.1, false, "Release speed (mph)"],
  ["spin", "Spin", 10, false, "Spin rate (rpm)"], ["ivb", "IVB", 0.1, false, "Induced vertical break (in)"],
  ["hb", "HB", 0.1, false, "Horizontal break (in), + = arm side"], ["ext", "Ext", 0.1, false, "Extension (ft)"],
  ["rel_height", "Rel Z", 0.1, false, "Release height (ft)"], ["rel_side", "Rel X", 0.1, false, "Release side (ft), + = arm side"],
  ["spin_axis", "Axis", 1, true, "Spin axis (deg, 180 = pure backspin). Optional"],
  ["arm_angle", "Arm°", 1, true, "Arm angle (deg). Optional"],
];
// basedOn: the MLB pitcher an arsenal was loaded from. The server then grades
// it with the model that never saw him, so it matches his leaderboard number.
const G = { hand: "R", pitches: [], customId: null, name: "", basedOn: null };
// Pitch types offered for a grader row: the templates, plus the row's own type
// when it has no template (FO, SC, KN, ... from a loaded arsenal), so it is
// shown as itself and isn't silently turned into the first template.
function pitchTypeOptions(types, current) {
  return !current || types.includes(current) ? types : [current, ...types];
}
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
  G.customId = null; G.name = ""; G.savedName = null;
  G.basedOn = rows[0]?.pitcher != null ? { pitcher: rows[0].pitcher, name: rows[0].name || "" } : null;
  saveG(); renderGraderInputs(); gradeNow();
}
// A saved pitcher is open in the grader when G.customId is set. Typing a
// different name means "this is another pitcher", so the main button then
// saves a new one instead of overwriting the pitcher that was opened.
const renamed = () => !!G.customId && (G.name || "").trim() !== (G.savedName ?? G.name ?? "").trim();
function updateSaveButtons() {
  $("#g-save").textContent = !G.customId ? "Save pitcher" : renamed() ? "Save as new pitcher" : "Save changes";
  $("#g-save-new").hidden = !G.customId;
  $("#g-save-new").textContent = renamed() ? `Rename “${G.savedName}”` : "Save as new";
}
function renderGraderInputs() {
  document.querySelectorAll("#g-hand button").forEach(b => b.classList.toggle("on", b.dataset.v === G.hand));
  $("#g-name").value = G.name || "";
  updateSaveButtons();
  const types = Object.keys(TEMPLATES);
  const head = `<thead><tr><th class="l">Pitch</th>${G_FIELDS.map(([, l, , , tip]) =>
    `<th title="${tip}">${l}</th>`).join("")}<th></th></tr></thead>`;
  const body = G.pitches.map((p, i) => `<tr data-i="${i}"><td class="l"><select data-k="pitch_type">${
    pitchTypeOptions(types, p.pitch_type).map(t => `<option value="${esc(t)}" title="${esc(ptName(t))}" ${t === p.pitch_type ? "selected" : ""}>${esc(t)}</option>`).join("")}</select></td>${
    G_FIELDS.map(([k, , step, opt]) => `<td><input type="number" data-k="${k}" step="${step}" value="${p[k] ?? ""}" ${opt ? 'class="opt" placeholder="auto"' : ""}></td>`).join("")
    }<td><button class="icon" data-del="${i}" title="Remove" aria-label="Remove pitch">✕</button></td></tr>`).join("");
  const t = $("#g-table"); t.innerHTML = head + `<tbody>${body}</tbody>`;
  labelGraderRows();
  t.querySelectorAll("input, select").forEach(inp => inp.oninput = () => {
    const i = +inp.closest("tr").dataset.i, k = inp.dataset.k;
    G.pitches[i][k] = k === "pitch_type" ? inp.value : (inp.value === "" ? null : +inp.value);
    if (k === "pitch_type") labelGraderRows();
    if (G.customId) $("#g-saved").textContent = "Unsaved changes";
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
      body: JSON.stringify({ hand: G.hand, season: S.season, pitcher: G.basedOn?.pitcher ?? null,
        pitches: G.pitches.map(p => ({ ...p, usage: p.usage ?? 0 })) }) });
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
  const fb = res.pitches.find(p => p.is_primary_fb), m = res.model || {};
  $("#g-note").textContent = [$("#g-note").dataset.src || "",
    m.fold != null ? `Scored like the leaderboard, by the model that never saw ${G.basedOn?.name || "this pitcher"}'s pitches.`
      : m.out_of_fold ? "Scored by the average of the leaderboard's five out-of-fold models." : "",
    `Scaled to ${res.season_scale} MLB pitches. Overall is usage-weighted.`,
    fb ? `Primary fastball for velo/movement differences: ${fb.pitch_type}.` : "",
    "Grades are 20-80 vs. the same pitch type; percentile is among MLB pitchers' versions of that pitch."].filter(Boolean).join(" ");
}
function wireGrader() {
  wireSeg("g-hand", () => { G.hand = segValue("g-hand"); saveG(); gradeNow(); });
  $("#g-add-type").innerHTML = Object.keys(TEMPLATES).map(t => `<option value="${t}">${t} · ${ptName(t)}</option>`).join("");
  $("#g-add").onclick = () => { if (G.pitches.length >= 12) return; G.pitches.push(newPitch($("#g-add-type").value)); saveG(); renderGraderInputs(); gradeNow(); };
  $("#g-reset").onclick = () => { G.hand = "R"; G.pitches = defaultArsenal(); G.customId = null; G.name = ""; G.savedName = null; G.basedOn = null; $("#g-saved").textContent = ""; delete $("#g-note").dataset.src; saveG(); renderGraderInputs(); gradeNow(); };
  $("#g-name").oninput = e => { G.name = e.target.value; saveG(); updateSaveButtons(); };
  $("#g-name").onkeydown = e => { if (e.key === "Enter") $("#g-save").click(); };
  // main button: new pitcher when unsaved or renamed; second button is the other choice
  $("#g-save").onclick = () => saveCustom(renamed());
  $("#g-save-new").onclick = () => saveCustom(!renamed());
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

// ------------------------------------------------------------- my pitchers
const C = { rows: [] };
async function saveCustom(asNew) {
  const name = (G.name || "").trim();
  if (!name) { $("#g-name").focus(); $("#g-saved").textContent = "Give the pitcher a name first."; return; }
  const body = JSON.stringify({ name, hand: G.hand, based_on: G.basedOn || null,
    pitches: G.pitches.map(p => ({ ...p, usage: p.usage ?? 0 })) });
  const opts = { headers: { "Content-Type": "application/json" }, body };
  try {
    let row = null;
    if (G.customId && !asNew) {
      try { row = await api(`/api/custom/${G.customId}`, { ...opts, method: "PUT" }); }
      catch (e) { if (!e.message.startsWith("404")) throw e; }  // deleted elsewhere: save it again as new
    }
    row ||= await api("/api/custom", { ...opts, method: "POST" });
    G.customId = row.id; G.savedName = row.name; saveG(); renderGraderInputs();
    await loadCustom();
    $("#g-saved").textContent = `Saved “${row.name}”. My pitchers now has ${C.rows.length}.`;
  } catch (e) {
    let msg = e.message;
    try { msg = JSON.parse(msg.slice(msg.indexOf("{"))).detail.map(d => `${d.loc.slice(-1)[0]}: ${d.msg}`).join("; "); } catch (_) { /* raw */ }
    $("#g-saved").textContent = "Couldn't save: " + msg;
  }
}
async function loadCustom() {
  try { C.rows = await api(`/api/custom?season=${S.season}`); C.error = null; }
  catch (e) { C.error = e.message; }  // keep showing the last good list
  renderCustom();
}
function renderCustom() {
  $("#c-error").hidden = !C.error;
  $("#c-error").textContent = C.error ? `Couldn't load saved pitchers (${C.error}). Is the server still running?` : "";
  $("#c-count").textContent = C.rows.length ? `${C.rows.length} saved` : "";
  $("#c-empty").hidden = C.rows.length > 0 || !!C.error;
  $("#c-wrap").hidden = C.rows.length === 0;
  if (!C.rows.length) return;
  const rows = C.rows.map(r => ({ ...r, stuff_plus: r.result.overall.stuff_plus, grade: r.result.overall.grade,
    percentile: r.result.overall.percentile, fb_velo: r.pitches.find((p, i) => r.result.pitches[i]?.is_primary_fb)?.velo }));
  renderTable($("#c-table"), [
    { key: "name", label: "Pitcher", l: 1, cls: "name" },
    { key: "hand", label: "Hand" },
    { key: "fb_velo", label: "FB velo", render: r => f1(r.fb_velo) },
    { key: "stuff_plus", label: "Stuff+", render: r => stuffCell(r.stuff_plus), defaultSort: 1 },
    { key: "grade", label: "Grade", cls: "grade" },
    { key: "percentile", label: "Pctl", render: r => pct(r.percentile), title: "Percentile among qualified MLB pitchers this season" },
    { key: "arsenal", label: "Arsenal", sortVal: r => r.pitches.length, render: r => `<div class="arsenal-chips">${
      r.result.pitches.map((p, i) => `<span title="${ptName(p.pitch_type)}: ${f0(r.pitches[i].usage)}% usage, ${f1(r.pitches[i].velo)} mph, grade ${p.grade ?? "–"}"><span class="dot" style="background:${ptColor(p.pitch_type)}"></span>${p.pitch_type} ${stuffCell(p.stuff_plus)}</span>`).join("")}</div>` },
    { key: "updated", label: "Saved", render: r => esc((r.updated || "").slice(0, 10)) },
    { key: "actions", label: "", sortVal: () => 0, render: r =>
      `<button class="icon" data-open="${r.id}" title="Open in the grader" aria-label="Open in the grader">✎</button><button class="icon danger" data-delc="${r.id}" title="Delete" aria-label="Delete">✕</button>` },
  ], rows, "custom", r => openCustom(r.id));
}
// One listener on the table (capture phase), so delete keeps working after the
// table is re-sorted and the row's own click (open in grader) does not also fire.
function wireCustom() {
  $("#c-table").addEventListener("click", async e => {
    const b = e.target.closest("[data-delc]"); if (!b) return;
    e.stopPropagation();
    const r = C.rows.find(x => x.id === b.dataset.delc);
    if (!r || !confirm(`Delete “${r.name}”?`)) return;
    try { await api(`/api/custom/${r.id}`, { method: "DELETE" }); } catch (err) { /* already gone */ }
    if (G.customId === r.id) { G.customId = null; G.savedName = null; saveG(); renderGraderInputs(); }
    loadCustom();
  }, true);
  // saved pitchers live on the server, so another window or browser may have
  // changed them: refresh whenever this window is looked at again
  const refresh = () => { if (S.season && document.visibilityState === "visible") loadCustom(); };
  window.addEventListener("focus", refresh);
  document.addEventListener("visibilitychange", refresh);
  window.addEventListener("pageshow", refresh);
}
function openCustom(id) {
  const r = C.rows.find(x => x.id === id);
  if (!r) return;
  G.hand = r.hand; G.name = r.name; G.customId = r.id; G.savedName = r.name; G.basedOn = r.based_on || null;
  G.pitches = r.pitches.map(p => ({ ...p }));
  $("#g-note").dataset.src = `Editing your saved pitcher “${r.name}”.`;
  $("#g-saved").textContent = "";
  saveG(); renderGraderInputs(); gradeNow(); go({ tab: "grader" });
}

// ------------------------------------------------------------- routes, tabs & boot
// #/pitchers  #/pitches  #/explore  #/grade  #/my-pitchers  #/pitchers/<slug>  (+ ?season=YYYY)
const ROUTES = { pitchers: "pitchers", pitches: "pitches", explore: "explore", explore3d: "explore-3d", grader: "grade",
  custom: "my-pitchers" };
const TAB_OF = Object.fromEntries(Object.entries(ROUTES).map(([k, v]) => [v, k]));
function slugify(name) {
  const n = String(name || "").includes(",") ? name.split(",").map(x => x.trim()).reverse().join(" ") : String(name || "");
  return n.normalize("NFD").replace(/[\u0300-\u036f]/g, "").toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
}
function buildSlugs() {
  const names = {};
  for (const r of [...S.data.pitchers, ...S.data.pitch_types]) names[r.pitcher] ??= r.name;
  const count = {};
  for (const n of Object.values(names)) count[slugify(n)] = (count[slugify(n)] || 0) + 1;
  S.slugOf = {}; S.slugIds = {};
  for (const [id, n] of Object.entries(names)) {
    const base = slugify(n) || "pitcher", slug = count[base] > 1 ? `${base}-${id}` : base;
    S.slugOf[id] = slug; S.slugIds[slug] = +id;
  }
}
function routeHash({ tab, slug, season = S.season }) {
  const q = season && season !== S.meta.latest_season ? `?season=${season}` : "";
  return "#/" + (slug ? `pitchers/${slug}` : ROUTES[tab] || "pitchers") + q;
}
function parseHash() {
  const m = location.hash.match(/^#\/([\w-]+)(?:\/([\w-]+))?\/?(?:\?(.*))?$/);
  if (!m) return {};
  return { tab: TAB_OF[m[1]], slug: m[1] === "pitchers" ? m[2] : null,
    season: +new URLSearchParams(m[3] || "").get("season") || null };
}
// navigate inside the app; Back/Forward and edited URLs arrive through hashchange
function go(route, replace) {
  const h = routeHash(route);
  if (h !== location.hash) history[replace ? "replaceState" : "pushState"](null, "", h);
  return applyRoute();
}
let routing = Promise.resolve();
const applyRoute = () => (routing = routing.then(applyRouteNow, applyRouteNow));
async function applyRouteNow() {
  const r = parseHash();
  const season = S.meta.seasons.includes(r.season) ? r.season : S.meta.latest_season;
  if (season !== S.season) { $("#season").value = season; await loadSeason(season); }
  if (r.slug) {
    if (!S.tab) showTab("pitchers");
    const id = S.slugIds[r.slug];
    if (id == null) { hideDrawer(); history.replaceState(null, "", routeHash({ tab: S.tab })); return; }
    if (D.id !== id) await showPitcher(id);
    return;
  }
  hideDrawer();
  const tab = r.tab || S.tab || "pitchers";
  if (!r.tab) history.replaceState(null, "", routeHash({ tab }));
  showTab(tab);
}
function switchTab(name) { go({ tab: name }); }
function showTab(name) {
  if (name === S.tab) return;
  S.tab = name;
  document.querySelectorAll("#tabs button").forEach(b => {
    const on = b.dataset.tab === name;
    b.classList.toggle("active", on); b.setAttribute("aria-selected", on); b.tabIndex = on ? 0 : -1;
  });
  document.querySelectorAll(".tab").forEach(t => t.classList.toggle("active", t.id === "tab-" + name));
  if (name === "explore") renderExplore();
  if (name === "explore3d") render3D();
  if (name === "custom") loadCustom();
  try { localStorage.setItem("stuffplus.tab", name); } catch (e) { /* ignore */ }
}
function wireTabs() {
  const btns = [...document.querySelectorAll("#tabs button")];
  btns.forEach(b => b.onclick = () => switchTab(b.dataset.tab));
  $("#tabs").addEventListener("keydown", e => {
    const i = btns.indexOf(document.activeElement); if (i < 0) return;
    const j = { ArrowRight: (i + 1) % btns.length, ArrowLeft: (i - 1 + btns.length) % btns.length,
      Home: 0, End: btns.length - 1 }[e.key];
    if (j == null) return;
    e.preventDefault(); btns[j].focus(); switchTab(btns[j].dataset.tab);
  });
}
// plain-language season line up top; the technical notes live under Methodology
function seasonLine(season) {
  return `${season} season${S.meta.frozen ? " · scores adjusted for small samples" : ""}`;
}
function renderMethodology() {
  const m = S.meta;
  const notes = [...m.seasons].reverse().map(y => m.season_notes[y] ? `<li>${y}: ${esc(m.season_notes[y])}</li>` : "").join("");
  $("#method-body").innerHTML = `
    <p>Stuff+ grades each pitch from its physical traits (velocity, spin, movement, release and extension), not
      from results, then averages a pitcher's pitches. 100 is league average${m.spread ? " and 10 points is one standard deviation" : ""}.
      Grades put the same number on the 20–80 scouting scale.${m.frozen ? " Scores are shrunk toward average for pitchers with few pitches." : ""}</p>
    <p>Model: ${esc(m.label)}${m.synthetic ? " · synthetic demo data" : ""}. Built ${esc(m.built_at)}.</p>
    <p>Features: ${esc(m.features.join(", "))}.</p>
    ${notes ? `<ul>${notes}</ul>` : ""}`;
}
async function loadSeason(season) {
  S.season = season;
  $("#season-note").textContent = seasonLine(season);
  S.data = await api(`/api/leaderboard/${season}`);
  S.byPitcher = {};
  for (const p of S.data.pitch_types) (S.byPitcher[p.pitcher] ||= {})[p.pitch_type] = p;
  buildSlugs();
  renderTypeChips("t-types", typeFilter.t, renderPitches);
  renderTypeChips("x-types", typeFilter.x, renderExplore);
  renderTypeChips("z-types", typeFilter.z, render3D);
  renderPitchers(); renderPitches();
  if ($("#tab-explore").classList.contains("active")) renderExplore();
  if ($("#tab-explore3d").classList.contains("active")) render3D();
  if ($("#tab-custom").classList.contains("active")) loadCustom();
  gradeNow();
}
async function boot() {
  S.meta = await api("/api/meta");
  const m = S.meta, tr = m.train_years;
  renderMethodology();
  $("#demo-banner").hidden = !m.synthetic;
  if (m.has_roles) $("#p-role").hidden = false;
  if (m.synthetic) { $("#p-min").value = 100; }
  $("#season").innerHTML = [...m.seasons].reverse().map(s => `<option>${s}</option>`).join("");
  $("#season").onchange = e => go({ tab: S.tab, season: +e.target.value });
  wireTabs();
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
  window.addEventListener("resize", () => { if ($("#tab-explore").classList.contains("active")) renderExplore();
    if ($("#tab-explore3d").classList.contains("active")) draw3D(); });
  wire3D();
  $("#drawer-close").onclick = closeDrawer;
  $("#drawer").onclick = e => { if (e.target.id === "drawer") closeDrawer(); };
  document.addEventListener("keydown", e => { if (e.key === "Escape" && !$("#drawer").hidden) closeDrawer(); });
  $("#c-new").onclick = () => { $("#g-reset").click(); go({ tab: "grader" }).then(() => $("#g-name").focus()); };
  restoreG(); wireGrader(); wireCustom(); renderGraderInputs();
  window.addEventListener("hashchange", applyRoute);
  if (!parseHash().tab) {  // no link: reopen the last tab
    let tab = "pitchers"; try { tab = localStorage.getItem("stuffplus.tab") || tab; } catch (e) { /* ignore */ }
    history.replaceState(null, "", "#/" + (ROUTES[tab] || "pitchers"));
  }
  await applyRoute();
}
boot().catch(e => { document.querySelector("main").innerHTML = `<div class="card">Failed to load: ${esc(e.message)}</div>`; });
