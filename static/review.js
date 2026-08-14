/* Staged review decisions. No libraries, one network call per commit.
 *
 * Every action on a pending card used to be its own form post: one full rewrite of the
 * 2 MB store per click, one page reload, and the reviewer's place in the queue lost each
 * time. This intercepts those submits, holds the decisions in memory, and posts the whole
 * page of them to /commit-review when the reviewer presses Commit all.
 *
 * Three rules hold throughout:
 *
 *   1. PROGRESSIVE ENHANCEMENT. Every form still works untouched with JavaScript off, and
 *      nothing in the page claims decisions are staged until this file has run. The hint
 *      and the commit bar start hidden and are unhidden from here.
 *   2. ONE DECISION PER SKILL. Clicking a second action on a card replaces the first
 *      rather than queueing both, which matches what the server enforces on commit.
 *   3. All text is written with textContent, never innerHTML: skill names and definitions
 *      are scraped from the open web and are not trusted markup.
 */
(function () {
  "use strict";

  var section = document.querySelector(".queue-section");
  if (!section) return;

  var bar = document.querySelector(".js-commitbar");
  var countLabel = document.querySelector(".js-commit-count");
  var goButton = document.querySelector(".js-commit-go");
  var clearButton = document.querySelector(".js-commit-clear");
  var resultBox = document.querySelector(".js-commit-result");
  var hint = document.querySelector(".js-stage-hint");
  if (!bar || !countLabel || !goButton || !clearButton || !resultBox) return;

  /* Staged decisions, keyed by skill name. Each value is a CHAIN:
   *
   *     { source: decision|null, final: decision|null }
   *
   * A source action changes what the card cites (refetch a page, accept a suggested
   * page, accept a drafted definition) and leaves it pending; a final action decides it.
   * Staging one of each is the point: accepting a definition and approving it is one
   * intention, and making it two round trips through the queue was busywork. The server
   * always applies source before final, whichever order they were clicked. */
  var staged = new Map();

  function chainFor(skill) {
    if (!staged.has(skill)) staged.set(skill, { source: null, final: null });
    return staged.get(skill);
  }

  function chainParts(chain) {
    return [chain.source, chain.final].filter(Boolean);
  }

  function decisionCount() {
    var total = 0;
    staged.forEach(function (chain) { total += chainParts(chain).length; });
    return total;
  }

  /* Survives an accidental reload, and only an accidental one: the key carries the page
   * number, so paging away and back does not resurrect decisions made about cards that
   * are no longer on screen. Cleared as soon as a commit lands. */
  var STORAGE_KEY = "skilldashboard.staged.page" + (section.dataset.page || "1");

  var ACTION_LABELS = {
    "approve": "Approve and score",
    "reject": "Reject",
    "remediate": "Refetch a supplied page",
    "apply-suggestion": "Use the suggested page",
    "apply-draft": "Use the drafted definition",
    "mark-duplicate": "Merge into another skill",
    "reject-draft": "Send the draft back to be rewritten"
  };

  var SOURCE_ACTIONS = ["remediate", "apply-suggestion", "apply-draft"];

  function slotFor(action) {
    return SOURCE_ACTIONS.indexOf(action) >= 0 ? "source" : "final";
  }

  // ---------------------------------------------------------------- utilities

  /* Cards indexed once, by name, rather than looked up with an attribute selector.
   * Skill names carry slashes ("SAS/CONNECT"), plus signs ("C++") and quotes, all of
   * which would have to be escaped into a selector correctly every single time. */
  var CARDS = new Map();
  Array.prototype.forEach.call(section.querySelectorAll(".card[data-skill]"), function (card) {
    CARDS.set(card.dataset.skill, card);
  });

  /* The definition each card was rendered with, so an edit can be told from an untouched
   * box. Captured here rather than emitted into the HTML as a data attribute, which
   * would put every definition on the page twice. */
  Array.prototype.forEach.call(
    section.querySelectorAll('[data-stage-field="summary"]'),
    function (box) { box.dataset.initial = box.value; }
  );

  function cardFor(skill) {
    return CARDS.get(skill) || null;
  }

  function save() {
    try {
      var rows = [];
      staged.forEach(function (chain) {
        chainParts(chain).forEach(function (decision) { rows.push(decision); });
      });
      window.sessionStorage.setItem(STORAGE_KEY, JSON.stringify(rows));
    } catch (err) {
      /* Private mode, or a full quota. The decisions are still in memory and still
       * committable; only surviving a reload is lost, so this is not worth a warning
       * the reviewer would have to dismiss. */
    }
  }

  function restore() {
    var raw;
    try {
      raw = window.sessionStorage.getItem(STORAGE_KEY);
    } catch (err) {
      return;
    }
    if (!raw) return;

    var rows;
    try {
      rows = JSON.parse(raw);
    } catch (err) {
      return;
    }
    if (!Array.isArray(rows)) return;

    rows.forEach(function (decision) {
      /* Only restore onto a card still on the page. A decision whose card has since
       * been approved by someone else has nowhere to show itself, and committing it
       * invisibly is exactly the surprise this whole change exists to remove. */
      if (decision && decision.skill && decision.action && cardFor(decision.skill)) {
        chainFor(decision.skill)[slotFor(decision.action)] = decision;
      }
    });
  }

  // ---------------------------------------------------------------- rendering

  /* One row per staged decision, each with its own Undo. Two rows on a chained card,
   * because "undo" has to mean one of the two things the reviewer did, not both. */
  function stagedRow(skill, slot, decision) {
    var row = document.createElement("div");
    row.className = "card__stagedrow";

    var label = document.createElement("span");
    label.textContent = "Staged: " + (ACTION_LABELS[decision.action] || decision.action);
    row.appendChild(label);

    if (decision.action === "remediate") {
      var note = document.createElement("span");
      note.className = "muted";
      /* Said plainly because it is the one staged action whose effect the reviewer
       * cannot see before committing: the page is fetched at commit time. */
      note.textContent = "Fetched on commit.";
      row.appendChild(note);
    }

    if (decision.action === "mark-duplicate" && decision.target) {
      /* Names the survivor, because "Merge into another skill" alone does not say WHICH,
       * and the box it was typed into is inside a collapsed <details> by then. */
      var into = document.createElement("span");
      into.className = "muted";
      into.textContent = "into " + decision.target;
      row.appendChild(into);
    }

    var undo = document.createElement("button");
    undo.type = "button";
    undo.className = "card__undo";
    undo.textContent = "Undo";
    undo.addEventListener("click", function () {
      var chain = staged.get(skill);
      if (chain) {
        chain[slot] = null;
        if (!chainParts(chain).length) staged.delete(skill);
      }
      save();
      paintCard(skill);
      paintBar();
    });
    row.appendChild(undo);
    return row;
  }

  function paintCard(skill) {
    var card = cardFor(skill);
    if (!card) return;

    var banner = card.querySelector(".card__staged");
    var chain = staged.get(skill);
    var parts = chain ? chainParts(chain) : [];

    if (!parts.length) {
      card.classList.remove("card--staged");
      if (banner) {
        banner.hidden = true;
        banner.textContent = "";
      }
      return;
    }

    card.classList.add("card--staged");
    if (!banner) return;

    banner.textContent = "";
    if (chain.source) banner.appendChild(stagedRow(skill, "source", chain.source));
    if (chain.final) banner.appendChild(stagedRow(skill, "final", chain.final));

    /* The one thing a chain does that neither half does alone, and the reviewer cannot
     * see it from the two labels: the text being scored is the text the source action
     * is about to fetch, not the text on screen right now. */
    if (chain.source && chain.final && chain.final.action === "approve"
        && !chain.final.summary) {
      var chained = document.createElement("span");
      chained.className = "muted";
      chained.textContent = "Scores the new text, not the text shown above.";
      banner.appendChild(chained);
    }

    banner.hidden = false;
  }

  function paintBar() {
    // The bulk button's count is derived from what is still undecided, so it has to be
    // repainted wherever the staged set changes. Hooking it here covers staging, undo,
    // settle and the initial restore in one place.
    paintBulk();

    var count = decisionCount();
    countLabel.textContent =
      count === 1 ? "1 decision staged" : count + " decisions staged";
    bar.hidden = count === 0;
    goButton.disabled = count === 0;
  }

  function showResult(kind, message, failures, appliedItems) {
    resultBox.textContent = "";
    resultBox.className = "notice js-commit-result notice--" + kind;

    var line = document.createElement("div");
    line.textContent = message;
    resultBox.appendChild(line);

    /* Named one by one, with the status the SERVER read back after saving. A summary
     * line can be wrong in a way a list of names and statuses cannot.
     *
     * Grouped by skill, so a chain reads as one line naming both things that were done
     * to it rather than as the same skill twice with the same final status. */
    var order = [];
    var grouped = {};
    (appliedItems || []).forEach(function (item) {
      if (!grouped[item.skill]) {
        grouped[item.skill] = { status: item.status, actions: [] };
        order.push(item.skill);
      }
      grouped[item.skill].status = item.status;
      grouped[item.skill].actions.push(ACTION_LABELS[item.action] || item.action);
    });

    order.forEach(function (skill) {
      var row = document.createElement("div");
      row.className = "mono";
      row.textContent =
        skill + ": " + grouped[skill].actions.join(", then ") +
        " -> " + grouped[skill].status;
      resultBox.appendChild(row);
    });

    (failures || []).forEach(function (failure) {
      var row = document.createElement("div");
      row.className = "mono";
      row.textContent = (failure.skill || "?") + ": " + failure.message;
      resultBox.appendChild(row);
    });

    resultBox.hidden = false;
    resultBox.scrollIntoView({ block: "nearest" });
  }

  // ---------------------------------------------------------------- staging

  function stage(form) {
    var card = form.closest(".card");
    if (!card) return false;

    var skill = card.dataset.skill;
    var action = form.dataset.stageAction;
    if (!skill || !action) return false;

    var slot = slotFor(action);
    var chain = staged.get(skill) || { source: null, final: null };
    var decision = {
      skill: skill, action: action, summary: null, reference: null, target: null
    };

    if (action === "mark-duplicate") {
      var targetInput = form.querySelector('[data-stage-field="target"]');
      var target = targetInput ? targetInput.value.trim() : "";
      if (!target || target === skill) {
        /* Refuse locally rather than staging a decision the server would reject at
         * commit time with unknown_target or duplicate_self. The reviewer is looking at
         * the box right now, which is the cheapest moment to fix it. */
        if (targetInput) targetInput.focus();
        return false;
      }
      decision.target = target;
    }

    if (action === "approve") {
      var textarea = form.querySelector('[data-stage-field="summary"]');
      var text = textarea ? textarea.value.trim() : "";
      if (!text) {
        /* Let the browser's own required-field message fire rather than staging a
         * decision the server would reject with no_summary at commit time. */
        return false;
      }

      /* WHOSE TEXT GETS SCORED. Sending no summary tells the server to score whatever
       * the source action puts on the card. That is what the reviewer means by "use
       * this definition and approve it": the textarea still shows the OLD text, and
       * sending it would score exactly the thing they just replaced.
       *
       * An edited textarea always wins, chained or not. Someone who typed a correction
       * meant it, and silently discarding it would be the worse failure. */
      var edited = textarea ? textarea.value !== (textarea.dataset.initial || "") : false;
      decision.summary = (chain.source && !edited) ? null : text;
    }

    if (action === "remediate") {
      var input = form.querySelector('[data-stage-field="reference"]');
      var reference = input ? input.value.trim() : "";
      if (!reference) {
        if (input) input.focus();
        return false;
      }
      decision.reference = reference;
    }

    chain[slot] = decision;

    /* Staging a source action after an approve re-evaluates the same question: the
     * approve was made against text that is now going to be replaced. Re-derive it
     * rather than leaving a summary that describes the old page. */
    if (slot === "source" && chain.final && chain.final.action === "approve") {
      var box = card.querySelector('[data-stage-field="summary"]');
      var wasEdited = box ? box.value !== (box.dataset.initial || "") : false;
      chain.final.summary = wasEdited ? box.value.trim() : null;
    }

    staged.set(skill, chain);
    save();
    paintCard(skill);
    paintBar();
    return true;
  }

  // ------------------------------------------------- bulk staging of drafts

  /* Stages "use this definition, then approve it" on every card holding a written
   * definition.
   *
   * Goes through stage() one form at a time rather than building decisions of its own,
   * so the chain ordering, the summary re-derivation and the storage all behave exactly
   * as they do for a single click. A second decision-building path here is how the two
   * would drift.
   *
   * Order is load-bearing: the draft is staged as the SOURCE first, so when the approve
   * is staged after it, stage() sees chain.source and sends summary: null -- which tells
   * the server to score the definition it is about to apply rather than the boilerplate
   * still sitting in the textarea. A reviewer who edited that textarea keeps their text;
   * stage() checks dataset.initial for exactly that.
   *
   * Cards the model DECLINED to define have no drafted_definition, so draft_ready is
   * false and they are not touched. That is the "I don't know this skill" case, and it
   * stays a human's problem by construction.
   */
  var bulkRow = document.querySelector(".js-bulk-drafts");
  var bulkButton = document.querySelector(".js-stage-drafts");
  var bulkNote = document.querySelector(".js-stage-drafts-note");

  function draftReadyCards() {
    var found = [];
    CARDS.forEach(function (card, skill) {
      if (card.dataset.draftReady !== "1") return;
      // settle() hides a card that left the queue and marks the rest committed.
      if (card.hidden || card.classList.contains("card--settled")) return;
      var draftForm = card.querySelector('[data-stage-action="apply-draft"]');
      var approveForm = card.querySelector('[data-stage-action="approve"]');
      if (draftForm && approveForm) found.push([skill, draftForm, approveForm]);
    });
    return found;
  }

  function paintBulk() {
    if (!bulkRow || !bulkButton) return;
    var pending = draftReadyCards().filter(function (row) {
      var chain = staged.get(row[0]);
      return !(chain && chain.final);
    });

    if (!pending.length) {
      bulkRow.hidden = true;
      return;
    }
    bulkRow.hidden = false;
    bulkButton.textContent =
      "Stage all " + pending.length + " written definition" +
      (pending.length === 1 ? "" : "s") + " for approval";
    if (bulkNote) {
      bulkNote.textContent =
        "Nothing is written until you press Commit, and each one can be undone first.";
    }
  }

  if (bulkButton) {
    bulkButton.addEventListener("click", function () {
      draftReadyCards().forEach(function (row) {
        var chain = staged.get(row[0]);
        if (chain && chain.final) return;   // already decided, leave it alone
        if (stage(row[1])) stage(row[2]);
      });
      paintBulk();
    });
  }

  section.addEventListener("submit", function (event) {
    var form = event.target;
    if (!form || !form.dataset || !form.dataset.stageAction) return;

    /* checkValidity first, so a required textarea emptied by the reviewer still gets
     * the browser's native complaint instead of being silently dropped. */
    if (typeof form.checkValidity === "function" && !form.checkValidity()) return;

    if (stage(form)) event.preventDefault();
  });

  /* Marks a card as decided, using the status the server read back.
   *
   * A skill that left the queue is removed from the page; one that is still pending --
   * every refetching action leaves it pending on purpose -- stays, with its new state
   * named and its forms out of the way until the page is reloaded and shows the
   * refetched text. */
  function settle(skill, status) {
    var card = cardFor(skill);
    if (!card) return;

    if (status !== "pending") {
      card.hidden = true;
      return;
    }

    card.classList.remove("card--staged");
    card.classList.add("card--settled");

    var banner = card.querySelector(".card__staged");
    if (banner) {
      banner.textContent = "";
      var label = document.createElement("span");
      label.textContent = "Committed. Still pending: reload to read the refetched page.";
      banner.appendChild(label);
      banner.hidden = false;
    }
  }

  // ---------------------------------------------------------------- committing

  function commit() {
    if (!staged.size) return;

    /* Source before final within each skill. The server re-orders anyway, but sending
     * them in the order they will be applied keeps the failure list readable. */
    var payload = [];
    staged.forEach(function (chain) {
      chainParts(chain).forEach(function (decision) { payload.push(decision); });
    });

    goButton.disabled = true;
    var previous = goButton.textContent;
    goButton.textContent = "Committing " + payload.length + "...";

    fetch("/commit-review", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ decisions: payload })
    })
      .then(function (response) {
        /* Checked BEFORE parsing. A 404 body parses perfectly well as JSON and then
         * reads as a commit that applied nothing, which is how "it said it worked and
         * nothing changed" looks from here. */
        if (response.status === 401) {
          throw new Error("Your session has expired. Reload and sign in again; nothing was committed.");
        }
        if (response.status === 404) {
          throw new Error("The running server has no /commit-review route, so nothing was saved. Restart it: python3.11 main.py");
        }
        if (!response.ok) {
          throw new Error("The server answered " + response.status + " and nothing was saved. See the server log.");
        }
        return response.json();
      })
      .then(function (result) {
        if (result.applied > 0) {
          /* Only what actually landed is dropped, and it is matched by skill AND action:
           * half a chain can land -- the page is accepted, the approve then fails on a
           * stale item -- and dropping the whole skill would lose the half that did not.
           * Anything that failed stays staged for a retry. */
          var landed = {};
          (result.applied_items || []).forEach(function (item) {
            landed[item.skill + " " + item.action] = true;
          });
          staged.forEach(function (chain, skill) {
            ["source", "final"].forEach(function (slot) {
              var decision = chain[slot];
              if (decision && landed[skill + " " + decision.action]) chain[slot] = null;
            });
            if (!chainParts(chain).length) staged.delete(skill);
          });
          save();
        }

        showResult(
          result.applied > 0 && !(result.failures || []).length ? "ok"
            : result.applied > 0 ? "warn" : "error",
          result.message || "Commit finished.",
          result.failures,
          result.applied_items
        );

        /* The page is updated in place rather than reloaded. An automatic reload threw
         * away the per-skill breakdown a second after printing it, which is precisely
         * the evidence a reviewer needs to believe the commit landed. */
        (result.applied_items || []).forEach(function (item) {
          settle(item.skill, item.status);
        });

        goButton.textContent = previous;
        paintBar();
      })
      .catch(function (err) {
        showResult("error", String(err.message || "The commit could not be sent. Nothing was saved."));
        goButton.textContent = previous;
        paintBar();
      });
  }

  goButton.addEventListener("click", commit);

  clearButton.addEventListener("click", function () {
    var names = [];
    staged.forEach(function (decision, skill) { names.push(skill); });
    staged.clear();
    save();
    names.forEach(paintCard);
    paintBar();
  });

  window.addEventListener("beforeunload", function (event) {
    if (!staged.size) return;
    event.preventDefault();
    event.returnValue = "";
  });

  // ---------------------------------------------------------------- start

  if (hint) hint.hidden = false;
  restore();
  staged.forEach(function (decision, skill) { paintCard(skill); });
  paintBar();
})();
