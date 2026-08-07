/* Dashboard interactivity. No libraries, no network calls.
 *
 * The server embeds every scored skill as JSON in the page, so filtering, sorting and
 * drill-down are local and instant. Two rules hold throughout:
 *
 *   1. Bar length is min-max normalized WITHIN the currently filtered slice, so the
 *      strongest visible skill reads 1.00. The raw cosine score is never normalized
 *      and is always reachable in the tooltip, the drawer, and the table view.
 *   2. AI category is bucketed from the RAW score only. Bucketing a normalized value
 *      would make a skill's class change as the filters change.
 *
 * All text is written with textContent, never innerHTML: skill names and summaries are
 * scraped from the open web and are not trusted markup.
 */
(function () {
  "use strict";

  var node = document.getElementById("dashboard-data");
  var DATA;
  try {
    DATA = JSON.parse(node.textContent);
  } catch (err) {
    console.error("dashboard: could not parse embedded data", err);
    return;
  }

  var SKILLS = DATA.skills || [];
  var TRENDS = DATA.trends || {};
  var CATEGORIES = DATA.categories || [];

  // Maps a category name to its CSS modifier. Kept in one place so the palette and
  // the markup cannot drift apart.
  var CAT_SUFFIX = {
    "AI Skill": "ai",
    "AI Enabling Skill": "enabling",
    "Not AI Skill": "nonai"
  };

  function suffixFor(category) {
    return CAT_SUFFIX[category] || "nonai";
  }

  var activeCategories = new Set(CATEGORIES);

  // ---------------------------------------------------------------- utilities

  function el(tag, className, text) {
    var n = document.createElement(tag);
    if (className) n.className = className;
    if (text !== undefined && text !== null) n.textContent = String(text);
    return n;
  }

  function fmt(value, places) {
    if (value === null || value === undefined) return "n/a";
    return Number(value).toFixed(places === undefined ? 4 : places);
  }

  /* Min-max across the slice. When every value is equal (including a single-item
   * view) the range is zero: return 1 rather than dividing by it, because each item
   * is simultaneously the maximum of what is visible. */
  function normalizer(values) {
    var usable = values.filter(function (v) { return v !== null && v !== undefined; });
    if (!usable.length) return function () { return 0; };
    var min = Math.min.apply(null, usable);
    var max = Math.max.apply(null, usable);
    var range = max - min;
    if (range <= 0) return function () { return 1; };
    return function (v) {
      if (v === null || v === undefined) return 0;
      return (v - min) / range;
    };
  }

  // ------------------------------------------------------------------ tooltip

  var tip = document.getElementById("tip");

  function showTip(event, skill, normalized) {
    while (tip.firstChild) tip.removeChild(tip.firstChild);
    tip.appendChild(el("strong", null, skill.skill_name));

    var dl = document.createElement("dl");
    [
      ["Raw AI score", fmt(skill.ai_score)],
      ["Normalized here", fmt(normalized, 2)],
      ["Class", skill.category_bucket],
      ["Tech / ML / Embedded",
       fmt(skill.tech_base_sim, 2) + " / " + fmt(skill.ml_pipeline_sim, 2) +
       " / " + fmt(skill.embedded_ai_sim, 2)],
      ["Occupations", String(skill.occupations.length)]
    ].forEach(function (pair) {
      dl.appendChild(el("dt", null, pair[0]));
      dl.appendChild(el("dd", null, pair[1]));
    });
    tip.appendChild(dl);

    tip.setAttribute("data-open", "true");
    moveTip(event);
  }

  function moveTip(event) {
    var pad = 14;
    var rect = tip.getBoundingClientRect();
    var x = event.clientX + pad;
    var y = event.clientY + pad;
    if (x + rect.width > window.innerWidth) x = event.clientX - rect.width - pad;
    if (y + rect.height > window.innerHeight) y = event.clientY - rect.height - pad;
    tip.style.left = Math.max(4, x) + "px";
    tip.style.top = Math.max(4, y) + "px";
  }

  function hideTip() {
    tip.setAttribute("data-open", "false");
  }

  // ------------------------------------------------------------------- drawer

  var drawer = document.getElementById("drawer");
  var scrim = document.getElementById("scrim");
  var drawerTitle = document.getElementById("drawer-title");
  var drawerContent = document.getElementById("drawer-content");
  var lastFocused = null;

  function openDrawer(skill) {
    lastFocused = document.activeElement;
    drawerTitle.textContent = skill.skill_name;

    while (drawerContent.firstChild) drawerContent.removeChild(drawerContent.firstChild);

    var dl = document.createElement("dl");
    function pair(label, value) {
      dl.appendChild(el("dt", null, label));
      dl.appendChild(el("dd", null, value));
    }
    pair("AI class", skill.category_bucket);
    pair("Raw AI score", fmt(skill.ai_score));
    pair("Tech base", fmt(skill.tech_base_sim));
    pair("ML pipeline", fmt(skill.ml_pipeline_sim));
    pair("Embedded AI", fmt(skill.embedded_ai_sim));
    pair("O*NET category", skill.category || "n/a");
    pair("Source", skill.best_source_name || "n/a");
    pair("Snapshot", skill.snapshot_date || "n/a");
    drawerContent.appendChild(dl);

    if (skill.wikipedia_summary) {
      drawerContent.appendChild(el("div", "drawer__section", "Definition"));
      drawerContent.appendChild(el("p", "drawer__quote", skill.wikipedia_summary));
    }

    if (skill.resolved_title) {
      var link = el("a", null, skill.resolved_title);
      link.href = "https://en.wikipedia.org/wiki/" +
        encodeURIComponent(String(skill.resolved_title).replace(/ /g, "_"));
      link.target = "_blank";
      link.rel = "noopener noreferrer";
      var p = el("p", "drawer__body");
      p.appendChild(document.createTextNode("Reference page: "));
      p.appendChild(link);
      drawerContent.appendChild(p);
    }

    drawerContent.appendChild(el("div", "drawer__section", "Occupations"));
    if (!skill.occupations.length) {
      drawerContent.appendChild(el("p", "muted", "Not mapped to any occupation."));
    } else {
      var list = el("ul", "sources");
      skill.occupations.forEach(function (occ) {
        var li = document.createElement("li");
        li.appendChild(el("span", "mono", occ.onet_code));
        li.appendChild(document.createTextNode(" " + (occ.onet_title || "")+ " "));
        li.appendChild(el("span", occ.is_hot_tech ? "badge badge--edit" : "tag",
                          occ.is_hot_tech ? "Hot tech" : "Standard"));
        list.appendChild(li);
      });
      drawerContent.appendChild(list);
    }

    drawerContent.appendChild(buildReportForm(skill));

    drawer.setAttribute("data-open", "true");
    drawer.setAttribute("aria-hidden", "false");
    scrim.setAttribute("data-open", "true");
    document.getElementById("drawer-close").focus();
  }

  /* Report control. Built per open rather than once, because it carries the skill_name of
   * whatever is currently open and must reset its state between skills -- a stale success
   * message on the next skill would be a lie.
   *
   * Submits with fetch so the drawer stays open and the reader keeps their filters and
   * scroll position. The endpoint always answers 200 with an ok flag, so a non-ok result
   * is a message to render rather than an exception to swallow. */
  var REPORT_REASONS = [
    ["wrong_page", "The reference page is not about this skill"],
    ["wrong_definition", "The definition text is wrong or misleading"],
    ["outdated", "The page or definition is out of date"],
    ["other", "Something else"]
  ];

  function buildReportForm(skill) {
    var wrap = el("div", "drawer__report");
    wrap.appendChild(el("div", "drawer__section", "Report a problem"));
    wrap.appendChild(el("p", "muted",
      "If the reference page or the definition above is wrong, flag it for review. " +
      "The score stays as it is until someone reviews the report."));

    var toggle = el("button", "ghost", "Report this skill");
    toggle.type = "button";
    toggle.setAttribute("aria-expanded", "false");

    var form = el("div", "drawer__reportform hidden");

    var reasonId = "report-reason";
    var reasonLabel = el("label", null, "What is wrong");
    reasonLabel.setAttribute("for", reasonId);
    var select = el("select");
    select.id = reasonId;
    REPORT_REASONS.forEach(function (pair) {
      var option = el("option", null, pair[1]);
      option.value = pair[0];
      select.appendChild(option);
    });

    var noteId = "report-note";
    var noteLabel = el("label", null, "Detail (optional)");
    noteLabel.setAttribute("for", noteId);
    var note = el("textarea");
    note.id = noteId;
    note.rows = 3;
    note.placeholder = "What should it say instead, or which page is the right one?";

    var send = el("button", "primary", "Submit report");
    send.type = "button";

    var status = el("p", "drawer__reportstatus");
    status.setAttribute("role", "status");
    status.setAttribute("aria-live", "polite");

    [reasonLabel, select, noteLabel, note, send, status].forEach(function (n) {
      form.appendChild(n);
    });

    toggle.addEventListener("click", function () {
      var open = form.classList.toggle("hidden") === false;
      toggle.setAttribute("aria-expanded", open ? "true" : "false");
      if (open) select.focus();
    });

    send.addEventListener("click", function () {
      send.disabled = true;
      status.textContent = "Sending...";
      status.className = "drawer__reportstatus";

      var body = new URLSearchParams();
      body.append("skill_name", skill.skill_name);
      body.append("reason", select.value);
      body.append("note", note.value);

      fetch("/report", {
        method: "POST",
        headers: { "Content-Type": "application/x-www-form-urlencoded" },
        body: body.toString()
      }).then(function (response) {
        return response.json();
      }).then(function (result) {
        status.textContent = result.message;
        status.className = "drawer__reportstatus " +
          (result.ok ? "drawer__reportstatus--ok" : "drawer__reportstatus--bad");
        // Left disabled after a successful report so one click cannot become five.
        if (!result.ok) send.disabled = false;
      }).catch(function (err) {
        console.error("report failed", err);
        status.textContent = "The report could not be sent. Check your connection.";
        status.className = "drawer__reportstatus drawer__reportstatus--bad";
        send.disabled = false;
      });
    });

    wrap.appendChild(toggle);
    wrap.appendChild(form);
    return wrap;
  }

  function closeDrawer() {
    drawer.setAttribute("data-open", "false");
    drawer.setAttribute("aria-hidden", "true");
    scrim.setAttribute("data-open", "false");
    if (lastFocused && lastFocused.focus) lastFocused.focus();
  }

  document.getElementById("drawer-close").addEventListener("click", closeDrawer);
  scrim.addEventListener("click", closeDrawer);
  document.addEventListener("keydown", function (event) {
    if (event.key === "Escape") { closeDrawer(); hideTip(); }
  });

  // --------------------------------------------------------------- bar charts

  /* Renders one bar per skill. `scoreOf` lets the role panel rank on the same raw
   * score while normalizing within its own slice. */
  function renderBars(container, skills) {
    while (container.firstChild) container.removeChild(container.firstChild);

    var norm = normalizer(skills.map(function (s) { return s.ai_score; }));

    skills.forEach(function (skill) {
      var value = norm(skill.ai_score);

      var row = el("li");
      var button = el("button", "bar");
      button.type = "button";

      var label = el("div", "bar__label");
      label.appendChild(el("span", "bar__swatch bar__swatch--" + suffixFor(skill.category_bucket)));
      label.appendChild(el("span", "bar__name", skill.skill_name));

      var track = el("div", "bar__track");
      var fill = el("span", "bar__fill");
      fill.style.width = (value * 100).toFixed(2) + "%";
      track.appendChild(fill);

      button.appendChild(label);
      button.appendChild(track);
      button.appendChild(el("div", "bar__value", fmt(value, 2)));

      button.setAttribute(
        "aria-label",
        skill.skill_name + ", " + skill.category_bucket +
        ", normalized " + fmt(value, 2) + ", raw " + fmt(skill.ai_score)
      );

      button.addEventListener("mouseenter", function (e) { showTip(e, skill, value); });
      button.addEventListener("mousemove", moveTip);
      button.addEventListener("mouseleave", hideTip);
      button.addEventListener("focus", function (e) { showTip(e, skill, value); });
      button.addEventListener("blur", hideTip);
      button.addEventListener("click", function () { hideTip(); openDrawer(skill); });

      row.appendChild(button);
      container.appendChild(row);
    });
  }

  function renderTable(container, skills) {
    while (container.firstChild) container.removeChild(container.firstChild);

    var norm = normalizer(skills.map(function (s) { return s.ai_score; }));
    var table = el("table", "dtable");
    table.appendChild(el("caption", null,
      "Raw scores are the measured values; normalized is relative to this filtered view."));

    var head = document.createElement("thead");
    var hrow = document.createElement("tr");
    ["Skill", "Class", "Normalized", "Raw AI", "Tech base", "ML pipeline", "Embedded AI", "Occupations"]
      .forEach(function (name) { hrow.appendChild(el("th", null, name)); });
    head.appendChild(hrow);
    table.appendChild(head);

    var body = document.createElement("tbody");
    skills.forEach(function (skill) {
      var tr = document.createElement("tr");
      tr.appendChild(el("td", null, skill.skill_name));
      tr.appendChild(el("td", null, skill.category_bucket));
      tr.appendChild(el("td", "num", fmt(norm(skill.ai_score), 2)));
      tr.appendChild(el("td", "num", fmt(skill.ai_score)));
      tr.appendChild(el("td", "num", fmt(skill.tech_base_sim)));
      tr.appendChild(el("td", "num", fmt(skill.ml_pipeline_sim)));
      tr.appendChild(el("td", "num", fmt(skill.embedded_ai_sim)));
      tr.appendChild(el("td", "num", String(skill.occupations.length)));
      body.appendChild(tr);
    });
    table.appendChild(body);
    container.appendChild(table);
  }

  function renderLegend(container, skills) {
    while (container.firstChild) container.removeChild(container.firstChild);
    var counts = {};
    CATEGORIES.forEach(function (c) { counts[c] = 0; });
    skills.forEach(function (s) { counts[s.category_bucket] = (counts[s.category_bucket] || 0) + 1; });

    CATEGORIES.forEach(function (category) {
      var item = el("span", "legend__item");
      item.appendChild(el("span", "legend__swatch legend__swatch--" + suffixFor(category)));
      item.appendChild(el("span", null, category));
      item.appendChild(el("span", "legend__count", counts[category]));
      container.appendChild(item);
    });
  }

  // ------------------------------------------------------------ ranked panel

  var rankedBars = document.getElementById("ranked-bars");
  var rankedTable = document.getElementById("ranked-table");
  var rankedEmpty = document.getElementById("ranked-empty");
  var rankedLegend = document.getElementById("ranked-legend");
  var searchInput = document.getElementById("skill-search");
  var sortSelect = document.getElementById("ranked-sort");
  var rankedCount = document.getElementById("ranked-count");

  /* Sorts a copy, never SKILLS itself: the role panel and the trend selector read the
   * same array, and reordering it in place would silently reorder them too.
   *
   * Nulls sort last in both directions. An unscored skill is not "the lowest" -- it is
   * absent -- so parking it at the end is the honest position either way. (In practice
   * dashboard_data.py's INNER JOIN excludes them, so this is belt and braces.)
   *
   * Ties break on name so the order is total and does not shuffle between redraws. */
  function sortSkills(skills, direction) {
    var sign = direction === "asc" ? 1 : -1;
    return skills.slice().sort(function (a, b) {
      var av = a.ai_score, bv = b.ai_score;
      var aNull = av === null || av === undefined;
      var bNull = bv === null || bv === undefined;
      if (aNull && bNull) return a.skill_name.localeCompare(b.skill_name);
      if (aNull) return 1;
      if (bNull) return -1;
      if (av === bv) return a.skill_name.localeCompare(b.skill_name);
      return (av - bv) * sign;
    });
  }

  function rankedSelection() {
    var term = searchInput.value.trim().toLowerCase();
    var matched = SKILLS.filter(function (skill) {
      if (!skill.is_hot_tech_anywhere) return false;
      if (!activeCategories.has(skill.category_bucket)) return false;
      if (term && skill.skill_name.toLowerCase().indexOf(term) === -1) return false;
      return true;
    });
    return sortSkills(matched, sortSelect.value);
  }

  function drawRanked() {
    var selection = rankedSelection();
    renderBars(rankedBars, selection);
    renderTable(rankedTable, selection);
    renderLegend(rankedLegend, selection);
    rankedEmpty.classList.toggle("hidden", selection.length > 0);

    /* Stating the count matters now that the list is a bounded scroll box: without it,
     * ten visible rows read as ten results. Sorting low-to-high changes which end is at
     * the top but never how many there are, so the count is a filter readout only. */
    var total = SKILLS.filter(function (s) { return s.is_hot_tech_anywhere; }).length;
    if (!selection.length) {
      rankedCount.textContent = "";
    } else if (selection.length === total) {
      rankedCount.textContent = selection.length + " skills. Scroll within the list to see them all.";
    } else {
      rankedCount.textContent = selection.length + " of " + total +
        " skills match. Scroll within the list to see them all.";
    }
  }

  // Category chips. Selection uses aria-pressed, which the stylesheet renders with a
  // check glyph as well as a fill, so it is never colour-alone.
  var catFilter = document.getElementById("cat-filter");
  CATEGORIES.forEach(function (category) {
    var chip = el("button", "chip");
    chip.type = "button";
    chip.setAttribute("aria-pressed", "true");
    chip.appendChild(el("span", "legend__swatch legend__swatch--" + suffixFor(category)));
    chip.appendChild(el("span", null, category));
    chip.addEventListener("click", function () {
      if (activeCategories.has(category)) {
        activeCategories.delete(category);
        chip.setAttribute("aria-pressed", "false");
      } else {
        activeCategories.add(category);
        chip.setAttribute("aria-pressed", "true");
      }
      drawRanked();
    });
    catFilter.appendChild(chip);
  });

  searchInput.addEventListener("input", drawRanked);
  sortSelect.addEventListener("change", drawRanked);

  // ------------------------------------------------------------ bucket panel

  function drawBuckets() {
    var stack = document.getElementById("bucket-stack");
    var legend = document.getElementById("bucket-legend");
    var table = document.getElementById("buckets-table");
    while (stack.firstChild) stack.removeChild(stack.firstChild);
    while (legend.firstChild) legend.removeChild(legend.firstChild);

    var buckets = DATA.buckets || [];
    var total = buckets.reduce(function (sum, b) { return sum + b.count; }, 0);

    buckets.forEach(function (bucket) {
      var share = total ? bucket.count / total : 0;
      var seg = el("div", "stack__seg stack__seg--" + suffixFor(bucket.category));
      seg.style.width = (share * 100).toFixed(2) + "%";
      seg.title = bucket.category + ": " + bucket.count;
      stack.appendChild(seg);

      var item = el("span", "legend__item");
      item.appendChild(el("span", "legend__swatch legend__swatch--" + suffixFor(bucket.category)));
      item.appendChild(el("span", null, bucket.category));
      item.appendChild(el("span", "legend__count",
        bucket.count + (total ? " (" + (share * 100).toFixed(0) + "%)" : "")));
      legend.appendChild(item);
    });

    stack.setAttribute("aria-label", buckets.map(function (b) {
      return b.category + " " + b.count;
    }).join(", "));

    while (table.firstChild) table.removeChild(table.firstChild);
    var dt = el("table", "dtable");
    var thead = document.createElement("thead");
    var hr = document.createElement("tr");
    ["AI class", "Skills", "Share"].forEach(function (n) { hr.appendChild(el("th", null, n)); });
    thead.appendChild(hr);
    dt.appendChild(thead);
    var tb = document.createElement("tbody");
    buckets.forEach(function (bucket) {
      var tr = document.createElement("tr");
      tr.appendChild(el("td", null, bucket.category));
      tr.appendChild(el("td", "num", String(bucket.count)));
      tr.appendChild(el("td", "num", total ? ((bucket.count / total) * 100).toFixed(1) + "%" : "n/a"));
      tb.appendChild(tr);
    });
    dt.appendChild(tb);
    table.appendChild(dt);
  }

  // -------------------------------------------------------------- role panel

  var occSelect = document.getElementById("occupation-select");
  var roleBars = document.getElementById("role-bars");
  var roleTable = document.getElementById("role-table");
  var roleEmpty = document.getElementById("role-empty");
  var roleLegend = document.getElementById("role-legend");

  (DATA.occupations || []).forEach(function (occ) {
    var option = el("option", null, (occ.onet_title || occ.onet_code) + " (" + occ.onet_code + ")");
    option.value = occ.onet_code;
    occSelect.appendChild(option);
  });

  function drawRole() {
    var code = occSelect.value;
    // Hot status is read from the matching occupation entry, not from the skill-level
    // flag. That is the whole point: a skill can be hot here and not hot elsewhere.
    var selection = SKILLS.filter(function (skill) {
      return skill.occupations.some(function (occ) {
        return occ.onet_code === code && occ.is_hot_tech;
      });
    });
    renderBars(roleBars, selection);
    renderTable(roleTable, selection);
    renderLegend(roleLegend, selection);
    roleEmpty.classList.toggle("hidden", selection.length > 0);
  }

  occSelect.addEventListener("change", drawRole);

  // ------------------------------------------------------------- trend panel

  var trendSelect = document.getElementById("trend-select");
  var trendChart = document.getElementById("trend-chart");
  var trendNote = document.getElementById("trend-note");
  var trendTable = document.getElementById("trend-table");

  Object.keys(TRENDS).sort().forEach(function (name) {
    var option = el("option", null, name + " (" + TRENDS[name].length + ")");
    option.value = name;
    trendSelect.appendChild(option);
  });

  function svg(tag, attrs) {
    var n = document.createElementNS("http://www.w3.org/2000/svg", tag);
    Object.keys(attrs || {}).forEach(function (k) { n.setAttribute(k, attrs[k]); });
    return n;
  }

  function drawTrend() {
    var name = trendSelect.value;
    var series = TRENDS[name] || [];
    while (trendChart.firstChild) trendChart.removeChild(trendChart.firstChild);
    while (trendTable.firstChild) trendTable.removeChild(trendTable.firstChild);

    if (!series.length) {
      trendNote.textContent = "No snapshots recorded.";
      return;
    }

    var W = 900, H = 220, padL = 46, padR = 12, padT = 12, padB = 30;
    trendChart.setAttribute("viewBox", "0 0 " + W + " " + H);
    trendChart.setAttribute("preserveAspectRatio", "none");

    var scores = series.map(function (p) { return p.score === null ? 0 : p.score; });
    var lo = Math.min.apply(null, scores);
    var hi = Math.max.apply(null, scores);
    // Pad the domain so a flat or near-flat series is not drawn hugging an edge.
    var span = hi - lo;
    if (span <= 0) { lo -= 0.05; hi += 0.05; } else { lo -= span * 0.15; hi += span * 0.15; }

    function px(i) {
      if (series.length === 1) return (padL + W - padR) / 2;
      return padL + (i / (series.length - 1)) * (W - padL - padR);
    }
    function py(v) {
      return padT + (1 - (v - lo) / (hi - lo)) * (H - padT - padB);
    }

    [0, 0.5, 1].forEach(function (frac) {
      var value = lo + frac * (hi - lo);
      var y = py(value);
      trendChart.appendChild(svg("line", {
        x1: padL, y1: y, x2: W - padR, y2: y, class: "trend__grid"
      }));
      var label = svg("text", { x: 4, y: y + 4, class: "trend__tick" });
      label.textContent = value.toFixed(3);
      trendChart.appendChild(label);
    });

    trendChart.appendChild(svg("line", {
      x1: padL, y1: H - padB, x2: W - padR, y2: H - padB, class: "trend__axis"
    }));

    if (series.length > 1) {
      trendChart.appendChild(svg("polyline", {
        class: "trend__line",
        points: series.map(function (p, i) {
          return px(i) + "," + py(p.score === null ? 0 : p.score);
        }).join(" ")
      }));
    }

    series.forEach(function (point, i) {
      trendChart.appendChild(svg("circle", {
        cx: px(i), cy: py(point.score === null ? 0 : point.score), r: 4, class: "trend__dot"
      }));
      // Label first and last only; a number on every point is noise.
      if (i === 0 || i === series.length - 1) {
        var t = svg("text", { x: px(i), y: H - padB + 16, class: "trend__tick", "text-anchor": "middle" });
        t.textContent = point.date;
        trendChart.appendChild(t);
      }
    });

    trendChart.setAttribute("aria-label",
      "AI score for " + name + " across " + series.length + " snapshots");

    trendNote.textContent = series.length === 1
      ? "One snapshot so far, so there is no movement to show yet. A second point appears " +
        "after the next 90-day re-summarization."
      : series.length + " snapshots from " + series[0].date + " to " + series[series.length - 1].date + ".";

    var table = el("table", "dtable");
    var thead = document.createElement("thead");
    var hr = document.createElement("tr");
    ["Snapshot", "Raw AI score"].forEach(function (n) { hr.appendChild(el("th", null, n)); });
    thead.appendChild(hr);
    table.appendChild(thead);
    var tb = document.createElement("tbody");
    series.forEach(function (point) {
      var tr = document.createElement("tr");
      tr.appendChild(el("td", null, point.date));
      tr.appendChild(el("td", "num", fmt(point.score)));
      tb.appendChild(tr);
    });
    table.appendChild(tb);
    trendTable.appendChild(table);
  }

  trendSelect.addEventListener("change", drawTrend);

  // --------------------------------------------------------- table view toggles

  document.querySelectorAll("[data-table-toggle]").forEach(function (button) {
    var key = button.getAttribute("data-table-toggle");
    var tableId = key + "-table";
    button.addEventListener("click", function () {
      var showing = button.getAttribute("aria-pressed") === "true";
      button.setAttribute("aria-pressed", showing ? "false" : "true");
      document.getElementById(tableId).classList.toggle("hidden", showing);

      // The chart and its table are alternates, not companions, except for the trend
      // panel where the table is always visible.
      var chart = key === "ranked" ? rankedBars
        : key === "role" ? roleBars
        : document.getElementById("bucket-stack");
      if (chart) chart.classList.toggle("hidden", !showing);
      var legend = key === "buckets" ? document.getElementById("bucket-legend") : null;
      if (legend) legend.classList.toggle("hidden", !showing);
    });
  });

  // ------------------------------------------------------------------- tabs

  var tabs = Array.prototype.slice.call(document.querySelectorAll(".tab"));
  tabs.forEach(function (tab) {
    tab.addEventListener("click", function () {
      tabs.forEach(function (other) {
        var selected = other === tab;
        other.setAttribute("aria-selected", selected ? "true" : "false");
        document.getElementById(other.getAttribute("aria-controls"))
          .classList.toggle("hidden", !selected);
      });
    });
  });

  // ------------------------------------------------------------------- start

  drawRanked();
  drawBuckets();
  if (occSelect.options.length) drawRole();
  if (trendSelect.options.length) drawTrend();
})();
