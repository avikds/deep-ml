
(function () {
  "use strict";

  var CFG = JSON.parse(document.getElementById("cfg").textContent);
  var DIFF = ["easy", "medium", "hard", "unrated"];
  var DCLASS = { easy: "e", medium: "m", hard: "h", unrated: "u" };
  var $ = function (id) { return document.getElementById(id); };

  function esc(s) {
    return String(s).replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }
  function n(v) { return Number(v || 0).toLocaleString("en-US"); }
  function day(iso) { return String(iso || "").slice(0, 10); }

  /* ---------- theme ---------- */
  var root = document.documentElement;
  try {
    var saved = localStorage.getItem("dml-theme");
    if (saved) root.setAttribute("data-theme", saved);
  } catch (e) {}
  $("theme").addEventListener("click", function () {
    var isDark = root.getAttribute("data-theme") === "dark" ||
      (!root.getAttribute("data-theme") && matchMedia("(prefers-color-scheme: dark)").matches);
    var next = isDark ? "light" : "dark";
    root.setAttribute("data-theme", next);
    try { localStorage.setItem("dml-theme", next); } catch (e) {}
  });

  /* ---------- data ----------
     progress.json comes off raw.githubusercontent so the page is never stale
     relative to the repo, even if only the data file was committed. The
     relative copy is the fallback that keeps a local preview working. */
  function getJSON(urls) {
    var i = 0;
    function attempt() {
      if (i >= urls.length) return Promise.reject(new Error("unreachable"));
      var url = urls[i++];
      return fetch(url, { cache: "no-cache" }).then(function (r) {
        if (!r.ok) throw new Error(String(r.status));
        return r.json();
      }).catch(attempt);
    }
    return attempt();
  }

  var RAW = "https://raw.githubusercontent.com/" + CFG.login + "/" + CFG.repo + "/" + CFG.branch + "/";

  Promise.all([
    getJSON(["./catalog.json"]),
    getJSON([RAW + "progress.json", "../progress.json", "./progress.json"])
  ]).then(function (res) { boot(res[0], res[1]); }).catch(function () {
    $("app").innerHTML = '<div class="err">Could not read this repository&rsquo;s progress data. ' +
      'If the repo was just created, GitHub may still be publishing it.</div>';
    $("app").classList.remove("gone");
    $("boot").classList.add("gone");
  });

  /* ---------- model ---------- */
  function boot(catalog, progress) {
    var cats = catalog.categories;
    // Catalog problems arrive as [id, title, categoryIndex, difficultyIndex].
    var problems = catalog.problems.map(function (p) {
      return { id: String(p[0]), title: p[1], cat: cats[p[2]], diff: DIFF[p[3]] || "unrated" };
    });

    var items = [];
    var byKey = {};
    Object.keys(progress.items || {}).forEach(function (k) {
      var r = progress.items[k];
      if (!r || !r.path) return;
      r.key = k;
      r.dayStr = day(r.first_solved_at);
      items.push(r);
      byKey[k] = r;
    });
    items.sort(function (a, b) { return a.dayStr < b.dayStr ? -1 : a.dayStr > b.dayStr ? 1 : 0; });

    var state = {
      catalog: problems, items: items, byKey: byKey,
      days: uniqueDays(items),
      cut: null,
      repoTree: "https://github.com/" + CFG.login + "/" + CFG.repo + "/tree/" + CFG.branch
    };
    state.cut = state.days.length ? state.days[state.days.length - 1] : null;

    $("boot").classList.add("gone");
    $("app").classList.remove("gone");

    renderHero(state);
    renderRail(state);
    buildScrub(state);
    renderMap(state);
    renderMomentum(state);
    renderMix(state);
    renderNext(state);
    renderLists(state);
    renderFooter(progress);
    wireTooltip();
    wireShare(state);
    wireExport(state);
  }

  function uniqueDays(items) {
    var seen = {}, out = [];
    items.forEach(function (r) {
      if (r.dayStr && !seen[r.dayStr]) { seen[r.dayStr] = 1; out.push(r.dayStr); }
    });
    return out.sort();
  }

  /** Items solved on or before the scrubber's cutoff. */
  function upTo(state) {
    if (!state.cut) return state.items;
    return state.items.filter(function (r) { return r.dayStr <= state.cut; });
  }

  function solvedIndex(list) {
    var m = {};
    list.forEach(function (r) { m[r.key] = r; });
    return m;
  }

  /* ---------- hero ---------- */
  function renderHero(state) {
    var total = state.catalog.length + CFG.totals.labs + CFG.totals.math;
    var pct = total ? (state.items.length / total) * 100 : 0;
    $("who").textContent = CFG.login;
    $("bignum").textContent = pct >= 10 ? pct.toFixed(0) : pct.toFixed(1);
    $("bigl1").textContent = "percent of Deep-ML solved";
    $("bigl2").textContent = n(state.items.length) + " of " + n(total) + " problems, labs and math";

    var first = state.days[0];
    $("lede").textContent = first
      ? "Machine learning practice, worked by hand since " + first +
        ". Every solution below is committed in this repository."
      : "Machine learning practice from Deep-ML.";
  }

  /* ---------- stat rail ---------- */
  function renderRail(state) {
    var items = state.items;
    var byKind = function (k) { return items.filter(function (r) { return r.kind === k; }).length; };
    var hard = items.filter(function (r) { return (r.difficulty || "").toLowerCase() === "hard"; }).length;
    var lines = items.reduce(function (a, r) { return a + (r.lines || 0); }, 0);
    var streak = longestStreak(state.days);

    var rail = [
      [n(byKind("problem")), "problems"],
      [n(byKind("lab")), "labs"],
      [n(hard), "hard solved"],
      [n(streak) + (streak === 1 ? " day" : " days"), "longest streak"],
      [lines ? n(lines) : String(state.days.length), lines ? "lines written" : "active days"]
    ];
    $("rail").innerHTML = rail.map(function (s) {
      return '<div class="cellstat"><div class="v tnum">' + esc(s[0]) + '</div><div class="k">' + esc(s[1]) + "</div></div>";
    }).join("");
  }

  /** Longest run of consecutive calendar days with at least one solve. */
  function longestStreak(days) {
    var best = 0, run = 0, prev = null;
    days.forEach(function (d) {
      var t = Date.parse(d + "T00:00:00Z");
      run = prev !== null && t - prev === 86400000 ? run + 1 : 1;
      prev = t;
      if (run > best) best = run;
    });
    return best;
  }

  /* ---------- scrubber: the time machine ---------- */
  function buildScrub(state) {
    var slider = $("slider");
    if (state.days.length < 2) { $("scrub").classList.add("gone"); return; }

    slider.min = "0";
    slider.max = String(state.days.length - 1);
    slider.value = String(state.days.length - 1);
    stamp(state);

    slider.addEventListener("input", function () {
      state.cut = state.days[Number(slider.value)];
      stamp(state);
      paint(state);
      renderRailLive(state);
    });

    var timer = null;
    $("play").addEventListener("click", function () {
      if (timer) { stop(); return; }
      // Restart from the beginning when the scrubber is parked at the end.
      if (Number(slider.value) >= state.days.length - 1) slider.value = "0";
      $("play").innerHTML = ICON_PAUSE;
      var stepMs = Math.max(16, Math.min(60, 4200 / state.days.length));
      timer = setInterval(function () {
        var v = Number(slider.value) + 1;
        if (v > state.days.length - 1) { stop(); return; }
        slider.value = String(v);
        slider.dispatchEvent(new Event("input"));
      }, stepMs);
    });
    function stop() { clearInterval(timer); timer = null; $("play").innerHTML = ICON_PLAY; }
  }

  var ICON_PLAY = '<svg viewBox="0 0 10 12"><path d="M0 0l10 6-10 6z"/></svg>';
  var ICON_PAUSE = '<svg viewBox="0 0 10 12"><path d="M0 0h3.2v12H0zM6.8 0H10v12H6.8z"/></svg>';

  function stamp(state) {
    var shown = upTo(state).length;
    $("stamp").innerHTML = "<b>" + n(shown) + "</b> by " + esc(state.cut || "");
  }

  /** The rail follows the scrubber, so dragging back in time is legible. */
  function renderRailLive(state) {
    var list = upTo(state);
    var hard = list.filter(function (r) { return (r.difficulty || "").toLowerCase() === "hard"; }).length;
    var cells = $("rail").children;
    if (cells.length >= 3) {
      cells[0].children[0].textContent = n(list.filter(function (r) { return r.kind === "problem"; }).length);
      cells[1].children[0].textContent = n(list.filter(function (r) { return r.kind === "lab"; }).length);
      cells[2].children[0].textContent = n(hard);
    }
  }

  /* ---------- coverage map ---------- */
  function renderMap(state) {
    var groups = {};
    state.catalog.forEach(function (p) {
      (groups[p.cat] = groups[p.cat] || []).push(p);
    });

    var MIN = 10;
    var rows = [], tail = [];
    Object.keys(groups).forEach(function (cat) {
      if (groups[cat].length < MIN) { tail = tail.concat(groups[cat]); return; }
      rows.push({ cat: cat, list: groups[cat] });
    });
    if (tail.length) rows.push({ cat: "Other", list: tail });

    var idx = solvedIndex(state.items);
    rows.forEach(function (row) {
      row.list.sort(function (a, b) { return (+a.id || 1e9) - (+b.id || 1e9); });
      row.solved = row.list.filter(function (p) { return idx["problem:" + p.id]; }).length;
    });
    rows.sort(function (a, b) { return b.solved - a.solved || b.list.length - a.list.length; });

    state.rows = rows;
    /* A category finished outright is the one number here that reads as an
       achievement rather than a running total, so the share text leads on it. */
    state.completedCategories = rows
      .filter(function (r) { return r.list.length >= 5 && r.solved === r.list.length; })
      .map(function (r) { return { category: r.cat, total: r.list.length }; });

    $("map").innerHTML = rows.map(function (row, i) {
      // Cell size scales down for the giant categories, so a 300-problem row
      // stays two lines instead of swamping the page.
      var size = row.list.length > 240 ? 7 : row.list.length > 140 ? 8 : row.list.length > 70 ? 9 : 11;
      var cells = row.list.map(function (p) {
        return '<a class="c" data-k="problem:' + esc(p.id) + '" data-d="' + esc(p.diff) +
          '" data-t="' + esc(p.title) + '" data-u="https://www.deep-ml.com/problems/' + esc(p.id) +
          '" href="#"></a>';
      }).join("");
      var complete = row.list.length >= 5 && row.solved === row.list.length;
      return '<section class="crow' + (complete ? " complete" : "") +
        '" style="--i:' + i + ';--s:' + size + 'px">' +
        '<div class="crow-head"><h3>' + esc(row.cat) +
        (complete ? ' <span class="tick" title="Every problem in this category">&#10003;</span>' : "") +
        "</h3>" +
        '<span class="n" data-row="' + i + '"><b>' + n(row.solved) + "</b>/" + n(row.list.length) + "</span>" +
        '<span class="pct" data-pct="' + i + '"></span></div>' +
        '<div class="cells">' + cells + "</div></section>";
    }).join("");

    paint(state);
  }

  /**
   * Repaint every cell for the current cutoff. One pass over the DOM with no
   * re-render, which is what lets the scrubber animate smoothly across 1400
   * cells.
   */
  function paint(state) {
    var idx = solvedIndex(upTo(state));
    var counts = {};
    var nodes = $("map").querySelectorAll(".c");
    for (var i = 0; i < nodes.length; i++) {
      var el = nodes[i];
      var rec = idx[el.getAttribute("data-k")];
      var cls = "c";
      if (rec) {
        cls += " " + (DCLASS[(el.getAttribute("data-d") || "unrated")] || "u");
        el.setAttribute("href", state.repoTree + "/" + rec.path);
      } else {
        el.setAttribute("href", el.getAttribute("data-u"));
      }
      el.className = cls;
    }
    // Row counters follow the same cutoff.
    (state.rows || []).forEach(function (row, i) {
      var solved = 0;
      row.list.forEach(function (p) { if (idx["problem:" + p.id]) solved++; });
      counts[i] = solved;
      var nEl = $("map").querySelector('[data-row="' + i + '"]');
      var pEl = $("map").querySelector('[data-pct="' + i + '"]');
      if (nEl) nEl.innerHTML = "<b>" + n(solved) + "</b>/" + n(row.list.length);
      if (pEl) pEl.textContent = Math.round((solved / row.list.length) * 100) + "%";
    });
  }

  /* ---------- momentum ---------- */
  function renderMomentum(state) {
    var months = {};
    state.items.forEach(function (r) {
      var m = r.dayStr.slice(0, 7);
      if (m) months[m] = (months[m] || 0) + 1;
    });
    var keys = Object.keys(months).sort();
    if (keys.length < 2) { $("momentum").classList.add("gone"); return; }

    // Fill the empty months, or a gap reads as a straight line through it.
    var all = [], cur = keys[0], last = keys[keys.length - 1];
    while (cur <= last) {
      all.push({ m: cur, v: months[cur] || 0 });
      var y = +cur.slice(0, 4), mo = +cur.slice(5, 7) + 1;
      if (mo > 12) { mo = 1; y++; }
      cur = y + "-" + String(mo).padStart(2, "0");
    }

    var W = 1000, H = 84, P = 6;
    var max = Math.max.apply(null, all.map(function (d) { return d.v; })) || 1;
    var pts = all.map(function (d, i) {
      var x = all.length > 1 ? (i / (all.length - 1)) * (W - P * 2) + P : W / 2;
      var y = H - P - (d.v / max) * (H - P * 2);
      return [x, y];
    });
    var line = pts.map(function (p, i) { return (i ? "L" : "M") + p[0].toFixed(1) + " " + p[1].toFixed(1); }).join(" ");
    var area = line + " L" + pts[pts.length - 1][0].toFixed(1) + " " + H + " L" + pts[0][0].toFixed(1) + " " + H + " Z";

    $("spark").innerHTML =
      '<path class="area" d="' + area + '"/><path class="line" d="' + line + '"/>' +
      '<text x="' + P + '" y="' + (H - 1) + '">' + esc(all[0].m) + "</text>" +
      '<text x="' + (W - P) + '" y="' + (H - 1) + '" text-anchor="end">' + esc(last) + "</text>";
    $("spark").setAttribute("viewBox", "0 0 " + W + " " + H);
    $("peak").textContent = "peak " + max + " in a month";
  }

  /* ---------- difficulty + language mix ---------- */
  function renderMix(state) {
    var items = state.items;
    var rows = ["easy", "medium", "hard"].map(function (d) {
      return { label: d, cls: d, count: items.filter(function (r) { return (r.difficulty || "").toLowerCase() === d; }).length };
    });
    $("difficulty").innerHTML = bars(rows, items.length);

    var langs = {};
    items.forEach(function (r) {
      if (r.language) langs[r.language] = (langs[r.language] || 0) + 1;
    });
    var lrows = Object.keys(langs).map(function (l) { return { label: l, cls: "flat", count: langs[l] }; })
      .sort(function (a, b) { return b.count - a.count; }).slice(0, 5);
    if (!lrows.length) { $("langpanel").classList.add("gone"); return; }
    $("languages").innerHTML = bars(lrows, items.length);
  }

  function bars(rows, total) {
    return rows.map(function (r) {
      var pct = total ? (r.count / total) * 100 : 0;
      return '<div class="brow"><span class="bl">' + esc(r.label) + "</span>" +
        '<span class="track"><i class="' + esc(r.cls) + '" style="width:' + pct.toFixed(1) + '%"></i></span>' +
        '<span class="bn tnum">' + n(r.count) + "</span></div>";
    }).join("");
  }

  /* ---------- next up ----------
     Suggestions come from the categories the user is already deepest in, which
     is the cheapest honest recommendation available on a static page. */
  function renderNext(state) {
    var idx = solvedIndex(state.items);
    var out = [];
    (state.rows || []).forEach(function (row) {
      if (out.length >= 6 || row.solved === 0) return;
      for (var i = 0; i < row.list.length && out.length < 6; i++) {
        var p = row.list[i];
        if (idx["problem:" + p.id]) continue;
        out.push({ p: p, cat: row.cat });
        break;
      }
    });
    if (!out.length) { $("nextpanel").classList.add("gone"); return; }

    $("next").innerHTML = out.map(function (o) {
      return '<a class="nx" href="https://www.deep-ml.com/problems/' + esc(o.p.id) + '" target="_blank" rel="noopener">' +
        '<span class="nt">' + esc(o.p.title) + "</span>" +
        '<span class="nm">' + esc(o.cat) + " &middot; " + esc(o.p.diff) + "</span></a>";
    }).join("");
  }

  /* ---------- labs and math ---------- */
  function renderLists(state) {
    ["lab", "math"].forEach(function (kind) {
      var el = $(kind + "s");
      var list = state.items.filter(function (r) { return r.kind === kind; });
      if (!list.length) { el.parentElement.classList.add("gone"); return; }
      list.sort(function (a, b) { return a.title.localeCompare(b.title); });
      $(kind + "count").textContent = n(list.length) + " of " + n(CFG.totals[kind + "s"] || list.length);
      el.innerHTML = list.map(function (r) {
        var d = (r.difficulty || "unrated").toLowerCase();
        return '<li><a class="it" href="' + esc(state.repoTree + "/" + r.path) + '">' +
          '<span class="ttl">' + esc(r.title) + "</span>" +
          '<span class="pill ' + esc(d) + '">' + esc(d) + "</span>" +
          '<time datetime="' + esc(r.first_solved_at) + '">' + esc(r.dayStr) + "</time></a></li>";
      }).join("");
    });
  }

  function renderFooter(progress) {
    $("updated").textContent = "Updated " + day(progress.generated_at || new Date().toISOString());
  }

  /* ---------- share ----------
     A share is only worth anything if it renders as a picture, which is what
     the og:image on this page is for. The button just makes the link easy to
     get at, with a pre-written line so nobody has to compose one. */
  function wireShare(state) {
    var url = "https://" + CFG.login + ".github.io/" + CFG.repo + "/";
    $("share").addEventListener("click", function () {
      var done = state.completedCategories || [];
      var text = done.length
        ? "I've finished " + done[0].category + " on Deep-ML (" +
          done[0].total + "/" + done[0].total + "), " + state.items.length +
          " problems solved by hand so far."
        : state.items.length + " machine learning problems solved by hand on Deep-ML.";

      if (navigator.share) {
        navigator.share({ title: CFG.login + " on Deep-ML", text: text, url: url })
          .catch(function () {});
        return;
      }
      var tweet = "https://twitter.com/intent/tweet?text=" +
        encodeURIComponent(text) + "&url=" + encodeURIComponent(url);
      window.open(tweet, "_blank", "noopener");
    });
  }

  /* ---------- time-lapse export ----------
     Drawn straight onto a canvas rather than screenshotting the DOM: the map is
     1400 elements, and the palette has to land on exact colours for the GIF
     encoder to quantise them without dithering. */
  var GW = 800, GH = 450;

  function drawFrame(ctx, state, cutIndex) {
    var cut = state.days[cutIndex];
    var idx = {};
    state.items.forEach(function (r) { if (r.dayStr <= cut) idx[r.key] = r; });

    var P = window.DMLGif.css;
    ctx.fillStyle = P(0);
    ctx.fillRect(0, 0, GW, GH);

    /* Header */
    ctx.fillStyle = P(6);
    ctx.font = "700 22px ui-monospace, monospace";
    ctx.fillText(CFG.login, 28, 42);
    ctx.fillStyle = P(5);
    ctx.font = "13px ui-monospace, monospace";
    ctx.fillText("machine learning, solved by hand", 28, 62);

    var count = Object.keys(idx).length;
    ctx.fillStyle = P(6);
    ctx.font = "700 30px ui-monospace, monospace";
    ctx.textAlign = "right";
    ctx.fillText(String(count), GW - 28, 46);
    ctx.fillStyle = P(5);
    ctx.font = "12px ui-monospace, monospace";
    ctx.fillText(cut, GW - 28, 64);
    ctx.textAlign = "left";

    /* Rows, sized to fit whatever is left of the canvas. */
    var rows = (state.rows || []).slice(0, 6);
    var top = 92;
    var avail = GH - top - 40;
    var per = avail / Math.max(1, rows.length);
    var cell = 6, gap = 2;
    var cols = Math.floor((GW - 56) / (cell + gap));

    rows.forEach(function (row, ri) {
      var y = top + ri * per;
      var solved = 0;
      row.list.forEach(function (p) { if (idx["problem:" + p.id]) solved++; });

      ctx.fillStyle = P(6);
      ctx.font = "11px ui-monospace, monospace";
      ctx.fillText(row.cat, 28, y);
      ctx.fillStyle = P(5);
      ctx.textAlign = "right";
      ctx.fillText(solved + "/" + row.list.length, GW - 28, y);
      ctx.textAlign = "left";

      row.list.forEach(function (p, i) {
        var rec = idx["problem:" + p.id];
        var c = 1;
        if (rec) c = p.diff === "easy" ? 2 : p.diff === "medium" ? 3 : p.diff === "hard" ? 4 : 5;
        ctx.fillStyle = P(c);
        ctx.fillRect(
          28 + (i % cols) * (cell + gap),
          y + 8 + Math.floor(i / cols) * (cell + gap),
          cell, cell
        );
      });
    });

    /* Footer: the reason this file is worth making. */
    ctx.fillStyle = P(5);
    ctx.font = "11px ui-monospace, monospace";
    ctx.fillText("deep-ml.com", 28, GH - 18);
    ctx.fillStyle = P(2);
    ctx.beginPath();
    ctx.arc(GW - 92, GH - 22, 4, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = P(6);
    ctx.font = "700 11px ui-monospace, monospace";
    ctx.fillText("DEEP-ML", GW - 82, GH - 18);
  }

  function wireExport(state) {
    var btn = $("export");
    if (!state.days.length || !window.DMLGif || !$("frame").getContext) {
      btn.classList.add("gone");
      return;
    }

    btn.addEventListener("click", function () {
      if (btn.disabled) return;
      btn.disabled = true;

      var canvas = $("frame");
      canvas.width = GW;
      canvas.height = GH;
      var ctx = canvas.getContext("2d");
      var writer = new window.DMLGif.Writer(GW, GH);

      /* At most 60 frames: a longer GIF is a bigger file for no more story. */
      var total = state.days.length;
      var step = Math.max(1, Math.ceil(total / 60));
      var frames = [];
      for (var i = 0; i < total; i += step) frames.push(i);
      if (frames[frames.length - 1] !== total - 1) frames.push(total - 1);

      var at = 0;
      function next() {
        if (at >= frames.length) {
          btn.textContent = "GIF";
          btn.disabled = false;
          var blob = writer.blob();
          var a = document.createElement("a");
          a.href = URL.createObjectURL(blob);
          a.download = CFG.login + "-deep-ml.gif";
          a.click();
          setTimeout(function () { URL.revokeObjectURL(a.href); }, 4000);
          return;
        }

        drawFrame(ctx, state, frames[at]);
        var rgba = ctx.getImageData(0, 0, GW, GH).data;
        /* The last frame holds, so the finished map is what a preview shows. */
        writer.frame(window.DMLGif.quantize(rgba), at === frames.length - 1 ? 2400 : 90);

        at++;
        btn.textContent = Math.round((at / frames.length) * 100) + "%";
        /* Yield between frames or the tab locks up for several seconds. */
        setTimeout(next, 0);
      }
      next();
    });
  }

  /* ---------- tooltip ---------- */
  function wireTooltip() {
    var tip = $("tip");
    $("map").addEventListener("mouseover", function (e) {
      var c = e.target.closest(".c");
      if (!c) return;
      var solved = c.className !== "c";
      tip.innerHTML = "#" + esc(c.getAttribute("data-k").split(":")[1]) + " " + esc(c.getAttribute("data-t")) +
        '<span class="t2">' + esc(c.getAttribute("data-d")) +
        (solved ? " &middot; solved, open the code" : " &middot; not solved yet") + "</span>";
      var r = c.getBoundingClientRect();
      tip.style.left = r.left + r.width / 2 + "px";
      tip.style.top = r.top + "px";
      tip.classList.add("on");
    });
    $("map").addEventListener("mouseout", function (e) {
      if (e.target.closest(".c")) tip.classList.remove("on");
    });
  }
})();
