// One engine, four modes:
//   watch    : 32 random answers, the bot plays itself.
//   practice : 32 random answers, you play with the bot's picks shown.
//   daily    : daily #, the bot derives answers, you tell it what you played.
//   byhand   : no number, no extension. You play any duotrigordle.com game and
//              click the colors you see on each of the 32 boards.

(() => {
  const $ = (id) => document.getElementById(id);
  const grid = $("grid"), log = $("log"), status = $("status"), bar = $("bar");
  const cands = $("cands"), turnNum = $("turnNum");
  const solvedCount = $("solvedCount"), guessNum = $("guessNum");

  // watch
  const answersInput = $("answers"), playBtn = $("playBtn"), shuffleBtn = $("shuffleBtn");
  // practice — Enter on the word submits; clicking a top-5 row fills the word.
  const practiceWord = $("practiceWord"), newPracticeBtn = $("newPracticeBtn");
  // daily — auto-analyses on input; dailyNext appends one guess at a time.
  const gameIdInput = $("gameId"), playedInput = $("playedGuesses"), dailyNext = $("dailyNext");
  // by hand — word + click-color grid + one submit button. switch tabs to reset.
  const byhandWord = $("byhandWord"), byhandSubmit = $("byhandSubmit");

  const NUM_BOARDS = 32;
  const ALL_GREEN = 242;
  const patternDigits = (p) => [p%3, (p/3|0)%3, (p/9|0)%3, (p/27|0)%3, (p/81|0)%3];
  const digitsToPattern = (d) => d[0] + d[1]*3 + d[2]*9 + d[3]*27 + d[4]*81;

  let engine = null, solutions = [];
  let running = false, mode = "watch";
  // Top-5 on a fresh state is the same every time — score it once at init so
  // tab switches don't trigger a 500ms full re-rank for no reason.
  let freshTop5 = null;
  // Keyboard focus position in by-hand mode: {board: 0..31, pos: 0..4}.
  let kfocus = null;

  // Per-board "history" lives on each board: array of {word, pattern}. We add
  // a row after every guess for boards that were unsolved when it was played.
  let practiceAnswers = null, practiceState = null;
  let byhandState = null;
  let byhandDigits = null;  // per-cell digit array [32][5], current input row

  // mini DOM containers
  const cells = [];
  for (let i = 0; i < NUM_BOARDS; i++) {
    const d = document.createElement("div");
    d.className = "mini";
    d.dataset.board = i;
    d.innerHTML = `<div class="stack"></div><div class="ct">·</div>`;
    grid.appendChild(d);
    cells.push(d);
  }

  function placeholderCands() {
    cands.innerHTML = "";
    for (let i = 0; i < 5; i++) {
      const li = document.createElement("li");
      li.className = "cand placeholder";
      li.innerHTML = `<span class="rank">${String(i+1).padStart(2,"0")}</span>` +
        `<span class="word">${"<span class='ct'>—</span>".repeat(5)}</span><span class="meta"></span>`;
      cands.appendChild(li);
    }
  }
  placeholderCands();

  async function loadLines(p) {
    const r = await fetch(p);
    return (await r.text()).split("\n").map((s) => s.trim()).filter(Boolean);
  }
  async function loadBytes(p) { return new Uint8Array(await (await fetch(p)).arrayBuffer()); }
  function pickRandom() {
    const used = new Set(), out = [];
    while (out.length < NUM_BOARDS) {
      const i = (Math.random() * solutions.length) | 0;
      if (!used.has(i)) { used.add(i); out.push(solutions[i]); }
    }
    return out;
  }
  const setStatus = (html) => { status.innerHTML = html; };
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const fmtCount = (n) => n >= 1000 ? (n/1000).toFixed(n >= 10000 ? 0 : 1).replace(/\.0$/, "") + "k" : String(n);

  // --- mini-board rendering: Wordle-style stack of played rows --------------
  function digitClass(d) { return d === 2 ? "g" : d === 1 ? "y" : "b"; }
  function rowHtml(word, pattern) {
    const d = patternDigits(pattern);
    let h = `<div class="row">`;
    for (let p = 0; p < 5; p++) h += `<span class="cell ${digitClass(d[p])}">${word[p]}</span>`;
    return h + `</div>`;
  }
  function inputRowHtml(word) {
    let h = `<div class="row input">`;
    for (let p = 0; p < 5; p++) {
      const ch = word[p] || "";
      h += `<span class="cell ${ch ? "empty" : "empty"}" data-p="${p}">${ch || "·"}</span>`;
    }
    return h + `</div>`;
  }
  function emptyRowHtml() {
    return `<div class="row">${"<span class='cell empty'></span>".repeat(5)}</div>`;
  }

  function renderBoard(state, b, opts) {
    opts = opts || {};
    const el = cells[b];
    const bd = state.boards[b];
    const hist = bd.history || [];
    let stack = "";
    for (const r of hist) stack += rowHtml(r.word, r.pattern);
    if (hist.length === 0 && !opts.inputWord) stack = emptyRowHtml();
    if (opts.inputWord && !bd.solved) stack += inputRowHtml(opts.inputWord);
    el.querySelector(".stack").innerHTML = stack;
    el.querySelector(".ct").textContent = bd.solved ? "✓" : fmtCount(bd.cands.length);
    el.classList.toggle("solved", !!bd.solved);
    el.classList.toggle("locked", !!bd.solved);
    // attach input click handlers when input row present
    if (opts.inputWord && !bd.solved) {
      el.querySelectorAll(".row.input .cell").forEach((c) => {
        const p = parseInt(c.dataset.p, 10);
        c.addEventListener("click", () => { setKfocus(b, p); cycleByhandCell(b, p); });
        // Colour from the current digit. d=0 is the "absent/grey" entry of the
        // pattern triple, so we show grey by default — same as duotrigordle.
        const d = byhandDigits ? byhandDigits[b][p] : 0;
        c.classList.remove("empty", "b", "y", "g");
        c.classList.add(digitClass(d));
        if (kfocus && kfocus.board === b && kfocus.pos === p) c.classList.add("kfocus");
      });
    }
  }
  function renderAll(state, opts) {
    for (let b = 0; b < NUM_BOARDS; b++) renderBoard(state, b, opts);
    solvedCount.textContent = state.boards.filter((b) => b.solved).length;
    guessNum.textContent = state.guessesUsed;
  }
  function clearAll() {
    for (const c of cells) {
      c.className = "mini";
      c.querySelector(".stack").innerHTML = emptyRowHtml();
      c.querySelector(".ct").textContent = "·";
    }
    solvedCount.textContent = "0";
    guessNum.textContent = "0";
  }

  function renderCands(list, chosenIdx, opts) {
    opts = opts || {};
    cands.innerHTML = "";
    for (let i = 0; i < 5; i++) {
      const li = document.createElement("li");
      const has = i < list.length;
      const word = has ? list[i].word : "—————";
      const meta = has ? `<b>${(list[i].couldSolve || []).length}</b> solvable` : "";
      const clickable = opts.clickable && has;
      li.className = "cand" + (!has ? " placeholder" : "")
        + (i === chosenIdx ? " chosen" : "")
        + (opts.flashChosen && i === chosenIdx ? " flash" : "")
        + (clickable ? " clickable" : "");
      if (clickable) li.dataset.word = word;
      li.innerHTML = `<span class="rank">${String(i+1).padStart(2,"0")}</span>` +
        `<span class="word">${[...word].map(c => `<span class="ct">${c}</span>`).join("")}</span>` +
        `<span class="meta">${meta}</span>`;
      cands.appendChild(li);
    }
    if (opts.clickable) {
      cands.querySelectorAll(".cand.clickable").forEach((el) => {
        el.addEventListener("click", () => {
          if (mode === "practice") { practiceWord.value = el.dataset.word; practiceWord.focus(); }
          else if (mode === "byhand") { byhandWord.value = el.dataset.word; updateByhandRender(); }
        });
      });
    }
  }
  function logLine(html) {
    const d = document.createElement("div");
    d.innerHTML = html;
    log.appendChild(d);
    log.scrollTop = log.scrollHeight;
  }

  // After-apply bookkeeping: push the played row onto each board's history
  // (only for boards that were unsolved before this turn).
  function pushHistory(state, prevUnsolved, word, pats) {
    for (let b = 0; b < NUM_BOARDS; b++) {
      if (!prevUnsolved[b]) continue;
      state.boards[b].history = state.boards[b].history || [];
      state.boards[b].history.push({ word, pattern: pats[b] });
    }
  }
  function markJustSolvedAnims(state, prevUnsolved) {
    for (let b = 0; b < NUM_BOARDS; b++) {
      if (prevUnsolved[b] && state.boards[b].solved) {
        cells[b].classList.add("justsolved");
        setTimeout((el => () => el.classList.remove("justsolved"))(cells[b]), 420);
      }
    }
  }

  // --- mode switching -------------------------------------------------------
  function setMode(m) {
    if (running) return;
    mode = m;
    document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t.dataset.mode === m));
    document.querySelector(".mode-watch").hidden    = m !== "watch";
    document.querySelector(".mode-practice").hidden = m !== "practice";
    document.querySelector(".mode-daily").hidden    = m !== "daily";
    document.querySelector(".mode-byhand").hidden   = m !== "byhand";
    // Bigger grid + hover effects in by-hand mode.
    grid.classList.toggle("byhand", m === "byhand");
    log.innerHTML = "";
    placeholderCands();
    clearAll();
    turnNum.textContent = "—";
    if (m === "watch") setStatus("ready.");
    else if (m === "practice") newPracticeGame();
    else if (m === "daily") setStatus("");
    else if (m === "byhand") newByhandGame();
  }
  document.querySelectorAll(".tab").forEach((t) =>
    t.addEventListener("click", () => setMode(t.dataset.mode)));

  // --- init ----------------------------------------------------------------
  async function init() {
    setStatus("loading word lists…");
    const [sols, guesses, net] = await Promise.all([
      loadLines("data/solutions_default.txt"),
      loadLines("data/valid_guesses.txt"),
      loadBytes("data/value_net.bin"),
    ]);
    solutions = sols;
    setStatus("warming up the solver…");
    engine = new DtEngine.Engine(sols, guesses, { deferBuild: true, valueNetBytes: net, useExactV: true });
    await engine.buildAsync((f) => { bar.style.width = (f * 100).toFixed(0) + "%"; });
    bar.style.width = "100%";
    setStatus("ready.");
    playBtn.disabled = false;
    answersInput.value = pickRandom().join(",");
    freshTop5 = engine.suggest(engine.freshState(), 5);
    renderCands(freshTop5.suggestions, 0);
    turnNum.textContent = "01";
  }

  // --- WATCH ---------------------------------------------------------------
  async function autoSolve() {
    if (!engine || running) return;
    running = true;
    playBtn.disabled = true; shuffleBtn.disabled = true;
    const raw = answersInput.value.trim();
    let answers = raw ? raw.split(",").map((w) => w.trim().toUpperCase()) : pickRandom();
    if (answers.length !== NUM_BOARDS || answers.some((w) => !/^[A-Z]{5}$/.test(w))) {
      setStatus(`need 32 five-letter words (got ${answers.length}).`);
      running = false; playBtn.disabled = false; shuffleBtn.disabled = false; return;
    }
    const bad = answers.filter((w) => !engine.solIndex.has(w));
    if (bad.length) { setStatus(`not in solutions: ${bad.slice(0, 4).join(", ")}…`); running = false; playBtn.disabled = false; shuffleBtn.disabled = false; return; }

    log.innerHTML = "";
    clearAll();
    const state = engine.freshState();

    while (state.guessesUsed < 50 && !state.boards.every((b) => b.solved)) {
      const sug = engine.suggest(state, 5);
      const turn = state.guessesUsed + 1;
      turnNum.textContent = String(turn).padStart(2, "0");
      renderCands(sug.suggestions, 0);
      setStatus(`turn <b>${String(turn).padStart(2,"0")}</b> · <b>${sug.activeBoards}</b> active`);
      await sleep(320);

      const word = sug.suggestions[0].word;
      const g = engine.guessIndex.get(word);
      const pats = new Array(NUM_BOARDS);
      const prev = state.boards.map((b) => !b.solved);
      let plus = 0;
      for (let b = 0; b < NUM_BOARDS; b++) {
        if (state.boards[b].solved) { pats[b] = ALL_GREEN; continue; }
        const p = DtEngine.computeFeedback(word, answers[b]);
        pats[b] = p;
        if (p === ALL_GREEN) plus++;
      }
      engine.applyGuess(state, g, pats);
      pushHistory(state, prev, word, pats);
      renderCands(sug.suggestions, 0, { flashChosen: true });
      renderAll(state);
      markJustSolvedAnims(state, prev);
      const solved = state.boards.filter((b) => b.solved).length;
      logLine(`${String(turn).padStart(2,"0")}. <b>${word}</b> · ${solved}/32` + (plus ? ` <span class="plus">+${plus}</span>` : ""));
      setStatus(`turn <b>${String(turn).padStart(2,"0")}</b> · <b>${solved}</b>/32`);
      await sleep(solved === NUM_BOARDS ? 0 : 300);
    }
    setStatus(`solved in <b>${state.guessesUsed}</b> guesses.`);
    running = false; playBtn.disabled = false; shuffleBtn.disabled = false;
  }

  // --- PRACTICE ------------------------------------------------------------
  function newPracticeGame() {
    practiceAnswers = pickRandom();
    practiceState = engine.freshState();
    log.innerHTML = "";
    clearAll();
    practiceWord.value = "";
    practiceWord.disabled = false;
    // Fresh state -> reuse the cached top-5 instead of re-scoring 14857 guesses.
    if (freshTop5) renderCands(freshTop5.suggestions, 0, { clickable: true });
    turnNum.textContent = "01";
    setStatus(`new game · turn <b>1</b>/37`);
    practiceWord.focus();
  }
  function refreshPracticePicks() {
    const sug = engine.suggest(practiceState, 5);
    const turn = practiceState.guessesUsed + 1;
    turnNum.textContent = String(turn).padStart(2, "0");
    renderCands(sug.suggestions, 0, { clickable: true });
  }
  function submitPracticeGuess() {
    if (!engine || !practiceState) return;
    const word = practiceWord.value.trim().toUpperCase();
    if (!/^[A-Z]{5}$/.test(word)) { setStatus("guess must be 5 letters."); return; }
    if (!engine.guessIndex.has(word)) { setStatus(`"${word}" isn't a valid guess.`); return; }
    const g = engine.guessIndex.get(word);
    const pats = new Array(NUM_BOARDS);
    const prev = practiceState.boards.map((b) => !b.solved);
    let plus = 0;
    for (let b = 0; b < NUM_BOARDS; b++) {
      if (practiceState.boards[b].solved) { pats[b] = ALL_GREEN; continue; }
      const p = DtEngine.computeFeedback(word, practiceAnswers[b]);
      pats[b] = p;
      if (p === ALL_GREEN) plus++;
    }
    const botPick = engine.chooseGuess(practiceState);
    engine.applyGuess(practiceState, g, pats);
    pushHistory(practiceState, prev, word, pats);
    renderAll(practiceState);
    markJustSolvedAnims(practiceState, prev);
    const solved = practiceState.boards.filter((b) => b.solved).length;
    const youMark = (g === botPick) ? "" : ' <span class="you">(you)</span>';
    logLine(`${String(practiceState.guessesUsed).padStart(2,"0")}. <b>${word}</b>${youMark} · ${solved}/32` + (plus ? ` <span class="plus">+${plus}</span>` : ""));
    practiceWord.value = "";

    if (solved === NUM_BOARDS) {
      setStatus(`solved in <b>${practiceState.guessesUsed}</b>.`);
      practiceWord.disabled = true; placeholderCands(); return;
    }
    if (practiceState.guessesUsed >= 37) {
      setStatus(`out of guesses — <b>${solved}</b>/32 solved.`);
      practiceWord.disabled = true; placeholderCands(); return;
    }
    setStatus(`turn <b>${practiceState.guessesUsed + 1}</b>/37 · <b>${solved}</b>/32`);
    refreshPracticePicks();
    practiceWord.focus();
  }

  // --- DAILY ---------------------------------------------------------------
  function analyzeDaily() {
    if (!engine || running) return;
    const id = parseInt(gameIdInput.value, 10);
    if (!id || id < 1) return;  // wait quietly for a valid #
    const raw = playedInput.value.trim();
    const played = raw ? raw.split(/[\s,]+/).map((w) => w.trim().toUpperCase()).filter(Boolean) : [];
    const bad = played.filter((w) => !engine.guessIndex.has(w));
    if (bad.length) { setStatus(`not valid guesses: ${bad.slice(0,4).join(", ")}`); return; }

    const ans = DtEngine.dailyAnswers(id, engine.S).map((i) => engine.solutions[i]);
    const state = engine.freshState();
    log.innerHTML = "";
    clearAll();
    for (let i = 0; i < played.length; i++) {
      const word = played[i];
      const g = engine.guessIndex.get(word);
      const pats = new Array(NUM_BOARDS);
      const prev = state.boards.map((b) => !b.solved);
      let plus = 0;
      for (let b = 0; b < NUM_BOARDS; b++) {
        if (state.boards[b].solved) { pats[b] = ALL_GREEN; continue; }
        const p = DtEngine.computeFeedback(word, ans[b]);
        pats[b] = p;
        if (p === ALL_GREEN) plus++;
      }
      engine.applyGuess(state, g, pats);
      pushHistory(state, prev, word, pats);
      const solved = state.boards.filter((b) => b.solved).length;
      logLine(`${String(i+1).padStart(2,"0")}. <b>${word}</b> · ${solved}/32` + (plus ? ` <span class="plus">+${plus}</span>` : ""));
    }
    renderAll(state);
    if (state.boards.every((b) => b.solved)) {
      placeholderCands();
      turnNum.textContent = String(state.guessesUsed).padStart(2, "0");
      setStatus(`daily <b>#${id}</b> finished in <b>${state.guessesUsed}</b>.`);
      return;
    }
    const sug = engine.suggest(state, 5);
    turnNum.textContent = String(state.guessesUsed + 1).padStart(2, "0");
    renderCands(sug.suggestions, 0);
    setStatus(`daily <b>#${id}</b> · turn <b>${state.guessesUsed + 1}</b>`);
  }

  // --- BY HAND (click colors per board) ------------------------------------
  function newByhandGame() {
    byhandState = engine.freshState();
    byhandDigits = Array.from({ length: NUM_BOARDS }, () => [0,0,0,0,0]);
    byhandWord.value = "";
    byhandSubmit.disabled = true;
    log.innerHTML = "";
    if (freshTop5) renderCands(freshTop5.suggestions, 0, { clickable: true });
    turnNum.textContent = "01";
    setStatus("turn <b>1</b>");
    // Paint the editable input row right away so the cells are clickable before
    // the user types anything — placeholder dots will fill in with letters as
    // soon as they start typing.
    updateByhandRender();
    byhandWord.focus();
  }
  function updateByhandRender() {
    const w = byhandWord.value.trim().toUpperCase();
    byhandSubmit.disabled = w.length !== 5;
    renderAll(byhandState, { inputWord: w.padEnd(5, " ") });
  }
  function cycleByhandCell(b, p) {
    if (!byhandDigits) return;
    byhandDigits[b][p] = (byhandDigits[b][p] + 1) % 3;
    updateByhandRender();
    pulseCell(b, p);
  }
  function setByhandCellDigit(b, p, d) {
    if (!byhandDigits) return;
    byhandDigits[b][p] = d;
    updateByhandRender();
    pulseCell(b, p);
  }
  function pulseCell(b, p) {
    const el = cells[b] && cells[b].querySelector(`.row.input .cell[data-p="${p}"]`);
    if (!el) return;
    el.classList.remove("tap"); void el.offsetWidth; el.classList.add("tap");
  }
  function setKfocus(b, p) {
    kfocus = { board: b, pos: p };
    // Lightweight: toggle .kfocus on the relevant cells without a full re-render.
    document.querySelectorAll(".mini .row.input .cell.kfocus").forEach((c) => c.classList.remove("kfocus"));
    const el = cells[b] && cells[b].querySelector(`.row.input .cell[data-p="${p}"]`);
    if (el) el.classList.add("kfocus");
  }
  function moveKfocus(db, dp) {
    if (!byhandState) return;
    let b = kfocus ? kfocus.board : 0;
    let p = kfocus ? kfocus.pos : 0;
    // Step until we land on an unsolved board (input rows only exist there).
    for (let tries = 0; tries < NUM_BOARDS * 5; tries++) {
      p += dp;
      if (p > 4) { p = 0; b = (b + 1) % NUM_BOARDS; }
      else if (p < 0) { p = 4; b = (b - 1 + NUM_BOARDS) % NUM_BOARDS; }
      if (db !== 0) { b = (b + db + NUM_BOARDS) % NUM_BOARDS; if (dp === 0) break; }
      if (!byhandState.boards[b].solved) { setKfocus(b, p); return; }
    }
  }
  function submitByhandRow() {
    if (!engine || !byhandState) return;
    const word = byhandWord.value.trim().toUpperCase();
    if (!/^[A-Z]{5}$/.test(word)) { setStatus("guess must be 5 letters."); return; }
    if (!engine.guessIndex.has(word)) { setStatus(`"${word}" isn't a valid guess.`); return; }
    const g = engine.guessIndex.get(word);
    const pats = new Array(NUM_BOARDS);
    const prev = byhandState.boards.map((b) => !b.solved);
    let plus = 0;
    for (let b = 0; b < NUM_BOARDS; b++) {
      if (byhandState.boards[b].solved) { pats[b] = ALL_GREEN; continue; }
      pats[b] = digitsToPattern(byhandDigits[b]);
      if (pats[b] === ALL_GREEN) plus++;
    }
    engine.applyGuess(byhandState, g, pats);
    pushHistory(byhandState, prev, word, pats);
    // reset input row for next turn
    byhandDigits = Array.from({ length: NUM_BOARDS }, () => [0,0,0,0,0]);
    byhandWord.value = "";
    byhandSubmit.disabled = true;

    const solved = byhandState.boards.filter((b) => b.solved).length;
    logLine(`${String(byhandState.guessesUsed).padStart(2,"0")}. <b>${word}</b> · ${solved}/32` + (plus ? ` <span class="plus">+${plus}</span>` : ""));
    renderAll(byhandState);
    markJustSolvedAnims(byhandState, prev);

    if (solved === NUM_BOARDS) {
      setStatus(`solved in <b>${byhandState.guessesUsed}</b>.`);
      placeholderCands();
      return;
    }
    const sug = engine.suggest(byhandState, 5);
    turnNum.textContent = String(byhandState.guessesUsed + 1).padStart(2, "0");
    renderCands(sug.suggestions, 0, { clickable: true });
    setStatus(`turn <b>${byhandState.guessesUsed + 1}</b> · <b>${solved}</b>/32`);
    // Keep the editable row painted so cells stay clickable for the next turn.
    updateByhandRender();
    byhandWord.focus();
  }

  // --- wire up -------------------------------------------------------------
  shuffleBtn.addEventListener("click", () => { if (solutions.length && !running) answersInput.value = pickRandom().join(","); });
  playBtn.addEventListener("click", autoSolve);

  newPracticeBtn.addEventListener("click", newPracticeGame);
  practiceWord.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && /^[A-Za-z]{5}$/.test(practiceWord.value.trim())) submitPracticeGuess();
  });

  // Daily auto-analyses on input — bails harmlessly until both # and guesses
  // are valid. dailyNext appends one guess at a time (Enter to add).
  [gameIdInput, playedInput].forEach((el) =>
    el.addEventListener("input", () => analyzeDaily()));
  dailyNext.addEventListener("keydown", (e) => {
    if (e.key !== "Enter") return;
    const w = dailyNext.value.trim().toUpperCase();
    if (!/^[A-Z]{5}$/.test(w)) return;
    const cur = playedInput.value.trim();
    playedInput.value = cur ? cur + "," + w : w;
    dailyNext.value = "";
    analyzeDaily();
  });

  byhandWord.addEventListener("input", updateByhandRender);
  byhandWord.addEventListener("keydown", (e) => { if (e.key === "Enter" && !byhandSubmit.disabled) submitByhandRow(); });
  byhandSubmit.addEventListener("click", submitByhandRow);

  // Keyboard shortcuts for by-hand mode. Only fire when the mode is active and
  // focus isn't in a text input — so typing the word in the input box still
  // works normally. 1/B = grey, 2/Y = yellow, 3/G = green; arrows navigate.
  document.addEventListener("keydown", (e) => {
    if (mode !== "byhand") return;
    const t = e.target;
    if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA")) return;
    const k = e.key.toLowerCase();
    if (k === "1" || k === "b") { if (kfocus) { setByhandCellDigit(kfocus.board, kfocus.pos, 0); e.preventDefault(); } }
    else if (k === "2" || k === "y") { if (kfocus) { setByhandCellDigit(kfocus.board, kfocus.pos, 1); e.preventDefault(); } }
    else if (k === "3" || k === "g") { if (kfocus) { setByhandCellDigit(kfocus.board, kfocus.pos, 2); e.preventDefault(); } }
    else if (k === "arrowright") { moveKfocus(0, 1); e.preventDefault(); }
    else if (k === "arrowleft")  { moveKfocus(0, -1); e.preventDefault(); }
    else if (k === "arrowdown")  { moveKfocus(1, 0); e.preventDefault(); }
    else if (k === "arrowup")    { moveKfocus(-1, 0); e.preventDefault(); }
    else if (k === "enter")      { if (!byhandSubmit.disabled) submitByhandRow(); }
  });

  init().catch((e) => setStatus("failed to load: " + e));
})();
