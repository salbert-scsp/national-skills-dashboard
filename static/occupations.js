// Occupations page. Reads the embedded payload; all text goes through textContent.
(function () {
  "use strict";
  const node = document.getElementById("occupations-data");
  if (!node) return;
  const data = JSON.parse(node.textContent || "{}");
  const rows = data.rows || [];
  if (!rows.length) return;

  const $ = (id) => document.getElementById(id);
  const body = $("occ-body"), search = $("occ-search"), group = $("occ-group");
  const minSkills = $("occ-min"), measure = $("occ-measure"), detail = $("occ-detail");
  const count = $("occ-count");
  const fmtInt = new Intl.NumberFormat("en-US");
  const fmtUsd = new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 0 });
  const pct = (v) => (v === null || v === undefined ? "—" : (v * 100).toFixed(0) + "%");
  const signed = (v) => (v === null || v === undefined ? "—" : (v > 0 ? "+" : "") + v.toFixed(1) + "%");

  let sortKey = "employment", sortDir = -1, selected = null;

  // Group filter options, in SOC order.
  const groups = new Map();
  rows.forEach((r) => groups.set(r.major_group, r.major_group_name));
  [...groups.entries()].sort().forEach(([code, name]) => {
    const o = document.createElement("option");
    o.value = code; o.textContent = name;
    group.appendChild(o);
  });

  function el(tag, cls, text) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text !== undefined) e.textContent = text;
    return e;
  }

  function value(r, key) {
    return key === "measure" ? r[measure.value] : r[key];
  }

  function filtered() {
    const q = search.value.trim().toLowerCase();
    const g = group.value, min = Number(minSkills.value);
    return rows.filter((r) =>
      (!g || r.major_group === g) &&
      (r.n_skills || 0) >= min &&
      (!q || r.title.toLowerCase().includes(q) || r.soc_code.includes(q)));
  }

  function render() {
    const list = filtered().sort((a, b) => {
      const x = value(a, sortKey), y = value(b, sortKey);
      if (x === null || x === undefined) return 1;
      if (y === null || y === undefined) return -1;
      return (typeof x === "string" ? x.localeCompare(y) : x - y) * sortDir;
    });
    body.textContent = "";
    list.forEach((r) => {
      const tr = el("tr", "occ__row");
      tr.tabIndex = 0;
      if (selected === r.soc_code) tr.setAttribute("aria-current", "true");
      const name = el("td");
      name.appendChild(el("div", "occ__title", r.title));
      name.appendChild(el("div", "occ__sub", r.soc_code + " · " + r.major_group_name));
      tr.appendChild(name);
      const emp = el("td", "num", r.employment === null ? "—" : fmtInt.format(r.employment));
      if (r.oews_match === "broad (shared)") emp.appendChild(el("span", "occ__flag", " shared"));
      tr.appendChild(emp);
      tr.appendChild(el("td", "num", r.median_annual_wage === null
        ? (r.wage_top_coded ? "≥ top code" : "—") : fmtUsd.format(r.median_annual_wage)));
      const ch = el("td", "num", signed(r.proj_change_pct));
      if (r.proj_change_pct !== null) ch.classList.add(r.proj_change_pct < 0 ? "occ__neg" : "occ__pos");
      tr.appendChild(ch);
      tr.appendChild(el("td", "num", fmtInt.format(r.n_skills || 0)));
      const share = value(r, "measure");
      const excl = measure.value === "share_ai_or_enabling_excl_office";
      const other = excl ? r.share_ai_or_enabling : r.share_ai_or_enabling_excl_office;
      const cell = el("td", "occ__sharecell");
      const top = el("div", "occ__share");
      const meter = el("div", "occ__meter");
      const fill = el("div", "occ__fill");
      fill.style.width = share === null ? "0" : Math.round(share * 100) + "%";
      meter.appendChild(fill);
      top.appendChild(meter);
      top.appendChild(el("span", "occ__pct", pct(share)));
      cell.appendChild(top);
      cell.appendChild(el("div", "occ__other",
        (excl ? "With Excel & Office: " : "Without Excel & Office: ") + pct(other)));
      tr.appendChild(cell);
      const open = () => { selected = r.soc_code; showDetail(r); render(); };
      tr.addEventListener("click", open);
      tr.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); open(); } });
      body.appendChild(tr);
    });
    count.textContent = fmtInt.format(list.length) + " of " + fmtInt.format(rows.length) + " occupations";
    document.querySelectorAll(".th-sort").forEach((b) => {
      const th = b.closest("th");
      th.setAttribute("aria-sort", b.dataset.key === sortKey ? (sortDir > 0 ? "ascending" : "descending") : "none");
    });
  }

  function kv(dl, label, text) {
    dl.appendChild(el("dt", null, label));
    dl.appendChild(el("dd", null, text));
  }

  function chipList(title, items, empty) {
    const box = el("div", "occ__list");
    box.appendChild(el("h3", "panel__subhead", title));
    if (!items.length) { box.appendChild(el("p", "panel__note", empty)); return box; }
    const wrap = el("div", "tagline");
    items.forEach((s) => wrap.appendChild(el("span", "tag", s)));
    box.appendChild(wrap);
    return box;
  }

  function showDetail(r) {
    detail.textContent = "";
    detail.appendChild(el("h2", null, r.title));
    detail.appendChild(el("div", "occ__sub", r.soc_code + " · " + r.major_group_name));

    // Composition bar of the listed skills.
    const total = r.n_skills || 0;
    const stack = el("div", "stack occ__stack");
    stack.setAttribute("role", "img");
    stack.setAttribute("aria-label", `${r.n_ai_skill} AI Skills, ${r.n_ai_enabling} AI Enabling, ${r.n_not_ai} Not AI of ${total} listed skills`);
    [["ai", r.n_ai_skill], ["enabling", r.n_ai_enabling], ["nonai", r.n_not_ai]].forEach(([k, n]) => {
      const seg = el("div", "stack__seg stack__seg--" + k);
      seg.style.width = total ? (n / total) * 100 + "%" : "0";
      stack.appendChild(seg);
    });
    detail.appendChild(stack);
    const legend = el("div", "legend");
    [["ai", "AI Skill", r.n_ai_skill], ["enabling", "AI Enabling", r.n_ai_enabling], ["nonai", "Not AI", r.n_not_ai]].forEach(([k, label, n]) => {
      const item = el("span", "legend__item");
      item.appendChild(el("span", "legend__swatch legend__swatch--" + k));
      item.appendChild(document.createTextNode(label + " "));
      item.appendChild(el("span", "legend__count", String(n)));
      legend.appendChild(item);
    });
    detail.appendChild(legend);

    const dl = el("dl", "occ__facts");
    kv(dl, "AI / AI Enabling share (all skills)", pct(r.share_ai_or_enabling));
    kv(dl, "Excluding Excel & Office", pct(r.share_ai_or_enabling_excl_office));
    kv(dl, "Employment", (r.employment === null ? "Not published" : fmtInt.format(r.employment)) +
       (r.oews_match === "broad (shared)" ? " (combined BLS group)" : ""));
    kv(dl, "Median annual wage", r.median_annual_wage === null ? "—" : fmtUsd.format(r.median_annual_wage));
    kv(dl, `Projected change ${data.proj_years.base}–${data.proj_years.end}`,
       signed(r.proj_change_pct) + (r.proj_match === "broad (shared)" ? " (combined group)" : ""));
    detail.appendChild(dl);

    detail.appendChild(chipList("AI Skills", r.ai_skills, "None listed for this occupation."));
    detail.appendChild(chipList("Most widely used AI Enabling skills", r.top_ai_enabling_skills, "None listed."));
  }

  document.querySelectorAll(".th-sort").forEach((b) => b.addEventListener("click", () => {
    const key = b.dataset.key;
    if (key === sortKey) sortDir = -sortDir; else { sortKey = key; sortDir = key === "title" ? 1 : -1; }
    render();
  }));
  [search, group, minSkills, measure].forEach((c) => c.addEventListener("input", render));

  render();
})();
