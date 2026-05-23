// Reads board state from duotrigordle.com, sends it to the local solver,
// and overlays the top suggested guess.
//
// DOM contract (as of bundle index-CKGYoxDm.js):
//   _boards_*  — outer grid wrapper
//   _board_*   — individual board (5-col grid of cells)
//   _cell_*    — individual tile
//   _green_*, _yellow_*, _black_* — color classes
//   _letter_*  — span inside cell that contains the letter
//
// We match by class-name substring (e.g. [class*="_board_"]) so that the
// hashed suffixes don't break us across deploys.

(() => {
  const POLL_MS = 750;
  const NUM_BOARDS = 32;

  const log = (...a) => console.log("[dt-solver]", ...a);

  function classHas(el, substr) {
    if (!el || !el.className) return false;
    const cn = typeof el.className === "string" ? el.className : el.className.baseVal || "";
    return cn.includes(substr);
  }

  function detectColor(cell) {
    if (classHas(cell, "_green_")) return "G";
    if (classHas(cell, "_yellow_")) return "Y";
    if (classHas(cell, "_black_")) return "B";
    return null;
  }

  function cellLetter(cell) {
    const letter = cell.querySelector('[class*="_letter_"]');
    const txt = (letter ? letter.textContent : cell.textContent) || "";
    return txt.trim().toUpperCase().slice(0, 1);
  }

  function readState() {
    const boardsWrap = document.querySelector('[class*="_boards_"]');
    if (!boardsWrap) return null;
    const boardEls = Array.from(boardsWrap.querySelectorAll('[class*="_board_"]'));
    if (boardEls.length < NUM_BOARDS) return null;

    const boards = [];
    for (let bi = 0; bi < NUM_BOARDS; bi++) {
      const cells = Array.from(boardEls[bi].querySelectorAll('[class*="_cell_"]'));
      const guesses = [];
      const feedback = [];
      for (let i = 0; i < cells.length; i += 5) {
        const row = cells.slice(i, i + 5);
        if (row.length < 5) break;
        const colors = row.map(detectColor);
        if (colors.some((c) => c === null)) continue;
        const letters = row.map(cellLetter).join("");
        if (letters.length !== 5 || !/^[A-Z]{5}$/.test(letters)) continue;
        guesses.push(letters);
        feedback.push(colors.join(""));
      }
      boards.push({ guesses, feedback });
    }
    return { boards };
  }

  // De-bounced fetch: only ask the server when the read state actually changed.
  let lastKey = "";
  let inFlight = false;
  let lastSuggestion = null;
  let contextInvalidated = false;

  async function poll() {
    if (contextInvalidated) return;
    const state = readState();
    if (!state) return;
    const key = JSON.stringify(state);
    if (key === lastKey || inFlight) return;
    lastKey = key;
    inFlight = true;
    try {
      if (!chrome.runtime || !chrome.runtime.id) {
        contextInvalidated = true;
        renderError("Extension reloaded — refresh this page");
        return;
      }
      const resp = await chrome.runtime.sendMessage({ type: "suggest", state });
      if (resp && resp.ok) {
        lastSuggestion = resp.data;
        render(resp.data);
      } else {
        renderError(resp ? resp.error : "no response");
      }
    } catch (e) {
      const msg = String(e);
      if (msg.includes("Extension context invalidated") || msg.includes("message port closed")) {
        contextInvalidated = true;
        renderError("Extension reloaded — refresh this page");
      } else {
        renderError(msg);
      }
    } finally {
      inFlight = false;
    }
  }

  // --- Overlay UI ---

  function readAnswers() {
    const boardsWrap = document.querySelector('[class*="_boards_"]');
    if (!boardsWrap) return null;
    const boardEls = Array.from(boardsWrap.querySelectorAll('[class*="_board_"]'));
    const out = [];
    for (const boardEl of boardEls) {
      const cells = Array.from(boardEl.querySelectorAll('[class*="_cell_"]'));
      let answer = null;
      for (let i = 0; i < cells.length; i += 5) {
        const row = cells.slice(i, i + 5);
        if (row.length < 5) break;
        if (row.every((c) => classHas(c, "_green_"))) {
          answer = row.map(cellLetter).join("");
          break;
        }
      }
      out.push(answer);
    }
    return out;
  }

  function readGuessesPlayed() {
    // Find the board with the most rows — that one has all guesses played.
    const boardsWrap = document.querySelector('[class*="_boards_"]');
    if (!boardsWrap) return null;
    const boardEls = Array.from(boardsWrap.querySelectorAll('[class*="_board_"]'));
    let best = [];
    for (const boardEl of boardEls) {
      const cells = Array.from(boardEl.querySelectorAll('[class*="_cell_"]'));
      const words = [];
      for (let i = 0; i < cells.length; i += 5) {
        const row = cells.slice(i, i + 5);
        if (row.length < 5) break;
        if (row.some((c) => detectColor(c) === null)) continue;
        const w = row.map(cellLetter).join("");
        if (/^[A-Z]{5}$/.test(w)) words.push(w);
      }
      if (words.length > best.length) best = words;
    }
    return best;
  }

  function ensureOverlay() {
    let el = document.getElementById("dt-solver-overlay");
    if (el) return el;
    el = document.createElement("div");
    el.id = "dt-solver-overlay";
    el.innerHTML = `
      <div class="dt-bar">
        <span class="dt-title">Duotrigordle Bot</span>
        <span class="dt-status" id="dt-status">…</span>
        <button class="dt-min" id="dt-min" title="Minimize">_</button>
      </div>
      <div class="dt-body">
        <div class="dt-pick" id="dt-pick"></div>
        <div class="dt-meta" id="dt-meta"></div>
        <div class="dt-alts" id="dt-alts"></div>
        <div class="dt-tools">
          <button class="dt-btn" id="dt-copy">Copy answers</button>
          <button class="dt-btn" id="dt-copy-guesses">Copy guesses</button>
          <button class="dt-btn" id="dt-review">Review</button>
          <span class="dt-copy-status" id="dt-copy-status"></span>
        </div>
      </div>
    `;
    document.body.appendChild(el);

    const copyStatus = el.querySelector("#dt-copy-status");
    async function copyText(text, statusFn) {
      try {
        await navigator.clipboard.writeText(text);
        statusFn(true);
      } catch {
        const ta = document.createElement("textarea");
        ta.value = text;
        document.body.appendChild(ta);
        ta.select();
        try { document.execCommand("copy"); statusFn(true); }
        catch { statusFn(false); console.log("[dt-solver]", text); }
        ta.remove();
      }
    }
    el.querySelector("#dt-copy").addEventListener("click", async () => {
      const ans = readAnswers();
      if (!ans) { copyStatus.textContent = "no boards found"; return; }
      const solved = ans.filter((a) => a).length;
      const csv = ans.map((a) => a || "?????").join(",");
      await copyText(csv, (ok) => copyStatus.textContent = ok ? `answers: ${solved}/32` : "copy failed");
      setTimeout(() => (copyStatus.textContent = ""), 3000);
    });
    el.querySelector("#dt-copy-guesses").addEventListener("click", async () => {
      const gs = readGuessesPlayed();
      if (!gs || !gs.length) { copyStatus.textContent = "no guesses found"; return; }
      const csv = gs.join(",");
      await copyText(csv, (ok) => copyStatus.textContent = ok ? `guesses: ${gs.length}` : "copy failed");
      setTimeout(() => (copyStatus.textContent = ""), 3000);
    });
    el.querySelector("#dt-review").addEventListener("click", async () => {
      const state = readState();
      if (!state) { copyStatus.textContent = "no board state"; return; }
      copyStatus.textContent = "reviewing…";
      try {
        const resp = await chrome.runtime.sendMessage({ type: "review", state });
        if (resp && resp.ok) { renderReview(resp.data); copyStatus.textContent = ""; }
        else { copyStatus.textContent = (resp && resp.error || "review failed").slice(0, 60); }
      } catch (e) {
        copyStatus.textContent = String(e).slice(0, 60);
      }
    });

    // Drag the bar
    const bar = el.querySelector(".dt-bar");
    let drag = null;
    bar.addEventListener("mousedown", (e) => {
      if (e.target.id === "dt-min") return;
      const r = el.getBoundingClientRect();
      drag = { dx: e.clientX - r.left, dy: e.clientY - r.top };
      e.preventDefault();
    });
    window.addEventListener("mousemove", (e) => {
      if (!drag) return;
      el.style.left = e.clientX - drag.dx + "px";
      el.style.top = e.clientY - drag.dy + "px";
      el.style.right = "auto";
    });
    window.addEventListener("mouseup", () => (drag = null));

    el.querySelector("#dt-min").addEventListener("click", () => {
      el.classList.toggle("dt-collapsed");
    });
    return el;
  }

  function wordTiles(word) {
    return [...word].map((c) => `<span class="dt-tile">${c}</span>`).join("");
  }

  function render(data) {
    const el = ensureOverlay();
    const status = el.querySelector("#dt-status");
    const pick = el.querySelector("#dt-pick");
    const meta = el.querySelector("#dt-meta");
    const alts = el.querySelector("#dt-alts");

    status.textContent = `${data.active_boards} active · ${data.guesses_used} used`;

    if (data.game_over || !data.suggestions || !data.suggestions.length) {
      pick.innerHTML = `<span class="dt-done">${data.game_over ? "all 32 solved" : "no suggestion"}</span>`;
      meta.textContent = data.game_over ? `finished in ${data.guesses_used} guesses` : "";
      alts.innerHTML = "";
      return;
    }
    const top = data.suggestions[0];
    pick.innerHTML = wordTiles(top.word);
    const solveCount = (top.could_solve || []).length;
    const modeNote = data.mode === "perfect" ? " · Perfect" : "";
    meta.textContent = (solveCount
      ? `could solve ${solveCount} board${solveCount === 1 ? "" : "s"}`
      : "info-only guess") + modeNote;

    alts.innerHTML = "";
    for (const s of data.suggestions.slice(1)) {
      const row = document.createElement("div");
      row.className = "dt-alt";
      row.innerHTML = `<span class="dt-altword">${s.word}</span>` +
        `<span class="dt-altmeta">${(s.could_solve || []).length} solve${(s.could_solve||[]).length===1?"":"s"}</span>`;
      alts.appendChild(row);
    }
  }

  function renderReview(data) {
    let modal = document.getElementById("dt-review-modal");
    if (modal) modal.remove();
    modal = document.createElement("div");
    modal.id = "dt-review-modal";
    const s = data.summary || {};
    const turns = data.turns || [];
    // 1-99 scale. Green = high, red = low; uses the same color band for both metrics.
    const score99Color = (v) => v >= 80 ? "var(--dt-green)" : v >= 50 ? "var(--dt-yellow)" : "var(--dt-red)";
    const rows = turns.map((t) => `
      <tr>
        <td class="dt-r-num">${t.turn}</td>
        <td class="dt-r-w">${t.guess}</td>
        <td class="dt-r-pct" style="color:${score99Color(t.skill)}">${Math.round(t.skill)}</td>
        <td class="dt-r-pct" style="color:${score99Color(t.luck)}">${Math.round(t.luck)}</td>
        <td class="dt-r-num">${t.boards_solved_this_turn}</td>
        <td class="dt-r-w ${t.decision_matched ? '' : 'dt-r-miss'}">${t.bot_choice}</td>
        <td class="dt-r-match">${t.decision_matched ? '✓' : '✗'}</td>
      </tr>`).join("");
    modal.innerHTML = `
      <div class="dt-review-card">
        <div class="dt-review-head">
          <span class="dt-review-title">Game Review</span>
          <button class="dt-btn" id="dt-review-close">close</button>
        </div>
        <div class="dt-review-summary">
          <div><span>guesses</span><b>${s.total_guesses}</b></div>
          <div><span>solved</span><b>${s.boards_solved}/32</b></div>
          <div><span>avg skill</span><b style="color:${score99Color(s.avg_skill || 0)}">${Math.round(s.avg_skill || 0)}</b></div>
          <div><span>avg luck</span><b style="color:${score99Color(s.avg_luck || 0)}">${Math.round(s.avg_luck || 0)}</b></div>
          <div><span>bot agreement</span><b>${s.decisions_matched}/${s.decisions_total}</b></div>
        </div>
        <table class="dt-review-table">
          <thead><tr>
            <th>#</th><th>played</th><th>skill</th><th>luck</th><th>+solved</th><th>bot would play</th><th></th>
          </tr></thead>
          <tbody>${rows}</tbody>
        </table>
        <p class="dt-review-help">
          <b>skill (1–99)</b>: percentile of your move's quality across all possible words — 99 = bot's top pick, 1 = worst possible.
          <b>luck (1–99)</b>: percentile of how favorable the actual feedback was — 99 = couldn't have been luckier, 1 = couldn't have been worse.
          <b>bot would play</b>: the bot's strategic pick (forced-move aware); ✗ means you missed it.
        </p>
      </div>`;
    document.body.appendChild(modal);
    modal.querySelector("#dt-review-close").addEventListener("click", () => modal.remove());
    modal.addEventListener("click", (e) => { if (e.target === modal) modal.remove(); });
  }

  function renderError(msg) {
    const el = ensureOverlay();
    el.querySelector("#dt-status").textContent = "error";
    el.querySelector("#dt-pick").innerHTML = "";
    el.querySelector("#dt-meta").textContent = String(msg).slice(0, 200);
    el.querySelector("#dt-alts").innerHTML = "";
  }

  log("loaded");
  setInterval(poll, POLL_MS);
})();
