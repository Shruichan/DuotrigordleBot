// Three modes share one engine:
//   watch    : roll/paste 32 answers, auto-solve animation.
//   practice : a random 32-board game; you type guesses, the bot suggests.
//   daily    : daily # + your played guesses -> top-5 next picks.
// Every board's mini-tile shows the letters the bot has pinned down (greens)
// plus the count of candidates left for that board.

(() => {
  const $ = (id) => document.getElementById(id);
  const grid = $("grid"), log = $("log"), status = $("status"), bar = $("bar");
  const cands = $("cands"), turnNum = $("turnNum");
  const solvedCount = $("solvedCount"), guessNum = $("guessNum");
  const modeTip = $("modeTip");

  // watch
  const answersInput = $("answers"), playBtn = $("playBtn"), shuffleBtn = $("shuffleBtn");
  // practice
  const practiceWord = $("practiceWord"), useBotBtn = $("useBotBtn"),
        submitWord = $("submitWord"), newPracticeBtn = $("newPracticeBtn");
  // daily
  const gameIdInput = $("gameId"), playedInput = $("playedGuesses"), analyzeBtn = $("analyzeBtn");

  const NUM_BOARDS = 32;
  let engine = null, solutions = [];
  let running = false, mode = "watch";
  // practice state
  let practiceAnswers = null, practiceState = null;

  // build mini-board DOM
  const cells = [];
  for (let i = 0; i < NUM_BOARDS; i++) {
    const d = document.createElement("div");
    d.className = "mini";
    d.innerHTML = `
      <div class="lk">${"<b class='q'>?</b>".repeat(5)}</div>
      <div class="ct">·</div>`;
    grid.appendChild(d);
    cells.push(d);
  }

  function placeholderCands() {
    cands.innerHTML = "";
    for (let i = 0; i < 5; i++) {
      const li = document.createElement("li");
      li.className = "cand placeholder";
      li.innerHTML = `
        <span class="rank">${String(i + 1).padStart(2, "0")}</span>
        <span class="word">${"<span class='ct'>—</span>".repeat(5)}</span>
        <span class="meta"></span>`;
      cands.appendChild(li);
    }
  }
  placeholderCands();

  // --- loaders -------------------------------------------------------------
  async function loadLines(p) {
    const r = await fetch(p);
    return (await r.text()).split("\n").map((s) => s.trim()).filter(Boolean);
  }
  async function loadBytes(p) {
    const r = await fetch(p);
    return new Uint8Array(await r.arrayBuffer());
  }

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
  const fmtCount = (n) => n >= 1000 ? (n / 1000).toFixed(n >= 10000 ? 0 : 1).replace(/\.0$/, "") + "k" : String(n);

  // --- mini-board rendering -----------------------------------------------
  function renderBoardCell(b, state) {
    const el = cells[b];
    const bd = state.boards[b];
    const locks = engine.boardLocks(bd);
    let html = `<div class="lk">`;
    if (bd.solved) {
      // show the answer fully green
      const ans = (bd.cands.length === 1 ? engine.solutions[bd.cands[0]] : (bd.lastAnswer || "")).padEnd(5, " ");
      for (let p = 0; p < 5; p++) html += `<b>${ans[p] || "?"}</b>`;
    } else {
      for (let p = 0; p < 5; p++) {
        const L = locks[p];
        html += L ? `<b class="f">${L}</b>` : `<b class="q">?</b>`;
      }
    }
    html += `</div><div class="ct">${bd.solved ? "✓" : fmtCount(bd.cands.length)}</div>`;
    if (bd.solved && !el.classList.contains("solved")) {
      el.className = "mini solved justsolved";
      setTimeout(() => el.classList.remove("justsolved"), 420);
    } else if (!bd.solved) {
      el.className = "mini";
    }
    el.innerHTML = html;
  }
  function renderAllBoards(state) {
    for (let b = 0; b < NUM_BOARDS; b++) renderBoardCell(b, state);
    solvedCount.textContent = state.boards.filter((b) => b.solved).length;
    guessNum.textContent = state.guessesUsed;
  }
  function clearBoards() {
    for (const c of cells) {
      c.className = "mini";
      c.innerHTML = `<div class="lk">${"<b class='q'>?</b>".repeat(5)}</div><div class="ct">·</div>`;
    }
    solvedCount.textContent = "0";
    guessNum.textContent = "0";
  }

  // --- top-5 candidates panel ---------------------------------------------
  function renderCands(list, chosenIdx, opts) {
    opts = opts || {};
    cands.innerHTML = "";
    for (let i = 0; i < 5; i++) {
      const li = document.createElement("li");
      const has = i < list.length;
      const word = has ? list[i].word : "—————";
      const meta = has
        ? `<b>${(list[i].couldSolve || []).length}</b> solvable`
        : "";
      const clickable = opts.clickable && has;
      li.className = "cand"
        + (!has ? " placeholder" : "")
        + (i === chosenIdx ? " chosen" : "")
        + (opts.flashChosen && i === chosenIdx ? " flash" : "")
        + (clickable ? " clickable" : "");
      if (clickable) li.dataset.word = word;
      li.innerHTML = `
        <span class="rank">${String(i + 1).padStart(2, "0")}</span>
        <span class="word">${[...word].map(c => `<span class="ct">${c}</span>`).join("")}</span>
        <span class="meta">${meta}</span>`;
      cands.appendChild(li);
    }
    if (opts.clickable) {
      cands.querySelectorAll(".cand.clickable").forEach((el) => {
        el.addEventListener("click", () => {
          practiceWord.value = el.dataset.word;
          submitWord.disabled = false;
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

  // --- mode switching ------------------------------------------------------
  function setMode(m) {
    if (running) return;
    mode = m;
    document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t.dataset.mode === m));
    document.querySelector(".mode-watch").hidden = m !== "watch";
    document.querySelector(".mode-practice").hidden = m !== "practice";
    document.querySelector(".mode-daily").hidden = m !== "daily";
    log.innerHTML = "";
    placeholderCands();
    clearBoards();
    turnNum.textContent = "—";
    if (m === "watch") {
      modeTip.textContent = "Roll random answers and watch the bot play.";
      setStatus("ready — <b>shuffle</b>, then <b>play</b>.");
    } else if (m === "practice") {
      modeTip.innerHTML = "Play against 32 hidden answers. Type a guess, hit <b>submit</b>, or click a row in the top-5 to use it.";
      newPracticeGame();
    } else {
      modeTip.innerHTML = "Type today's daily <b>#</b> (shown on duotrigordle.com) and the words you've already played.";
      setStatus("enter daily <b>#</b> + your <b>guesses</b>.");
    }
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
    setStatus("warming up the solver (one-time, takes a couple seconds)…");
    engine = new DtEngine.Engine(sols, guesses, { deferBuild: true, valueNetBytes: net, useExactV: true });
    await engine.buildAsync((f) => { bar.style.width = (f * 100).toFixed(0) + "%"; });
    bar.style.width = "100%";
    setStatus("ready — <b>shuffle</b>, then <b>play</b>.");
    playBtn.disabled = false; analyzeBtn.disabled = false;
    answersInput.value = pickRandom().join(",");
    // turn-1 candidates so the right panel isn't empty
    const fresh = engine.freshState();
    renderCands(engine.suggest(fresh, 5).suggestions, 0);
    turnNum.textContent = "01";
  }

  // --- WATCH: auto-solve ---------------------------------------------------
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
    if (bad.length) { setStatus(`not solutions: ${bad.slice(0, 4).join(", ")}…`); running = false; playBtn.disabled = false; shuffleBtn.disabled = false; return; }

    log.innerHTML = "";
    clearBoards();
    const ALL_GREEN = DtEngine.ALL_GREEN;
    const state = engine.freshState();
    for (let b = 0; b < NUM_BOARDS; b++) state.boards[b].lastAnswer = answers[b];

    while (state.guessesUsed < 50 && !state.boards.every((b) => b.solved)) {
      const sug = engine.suggest(state, 5);
      const turn = state.guessesUsed + 1;
      turnNum.textContent = String(turn).padStart(2, "0");
      renderCands(sug.suggestions, 0);
      setStatus(`turn <b>${String(turn).padStart(2,"0")}</b> · <b>${sug.activeBoards}</b> active`);
      await sleep(340);

      const word = sug.suggestions[0].word;
      const g = engine.guessIndex.get(word);
      const pats = new Array(NUM_BOARDS);
      let plus = 0;
      for (let b = 0; b < NUM_BOARDS; b++) {
        if (state.boards[b].solved) { pats[b] = ALL_GREEN; continue; }
        const p = DtEngine.computeFeedback(word, answers[b]);
        pats[b] = p;
        if (p === ALL_GREEN) plus++;
      }
      engine.applyGuess(state, g, pats);
      for (let b = 0; b < NUM_BOARDS; b++) state.boards[b].lastAnswer = answers[b];
      renderCands(sug.suggestions, 0, { flashChosen: true });
      renderAllBoards(state);
      const solved = state.boards.filter((b) => b.solved).length;
      logLine(`${String(turn).padStart(2,"0")}. <b>${word}</b> · ${solved}/32` + (plus ? ` <span class="plus">+${plus}</span>` : ""));
      setStatus(`turn <b>${String(turn).padStart(2,"0")}</b> · <b>${solved}</b>/32`);
      await sleep(solved === NUM_BOARDS ? 0 : 320);
    }
    setStatus(`solved in <b>${state.guessesUsed}</b> guesses.`);
    running = false; playBtn.disabled = false; shuffleBtn.disabled = false;
  }

  // --- PRACTICE: you play, bot suggests ------------------------------------
  function newPracticeGame() {
    practiceAnswers = pickRandom();
    practiceState = engine.freshState();
    for (let b = 0; b < NUM_BOARDS; b++) practiceState.boards[b].lastAnswer = practiceAnswers[b];
    log.innerHTML = "";
    clearBoards();
    practiceWord.value = "";
    practiceWord.disabled = false;
    submitWord.disabled = true;
    useBotBtn.disabled = false;
    refreshPracticePicks();
    setStatus(`new game — type a guess or use the bot's pick. turn <b>1</b>/37.`);
    practiceWord.focus();
  }
  function refreshPracticePicks() {
    if (!engine || !practiceState) return;
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
    const ALL_GREEN = DtEngine.ALL_GREEN;
    let plus = 0;
    for (let b = 0; b < NUM_BOARDS; b++) {
      if (practiceState.boards[b].solved) { pats[b] = ALL_GREEN; continue; }
      const p = DtEngine.computeFeedback(word, practiceAnswers[b]);
      pats[b] = p;
      if (p === ALL_GREEN) plus++;
    }
    const botPick = engine.chooseGuess(practiceState);
    engine.applyGuess(practiceState, g, pats);
    for (let b = 0; b < NUM_BOARDS; b++) practiceState.boards[b].lastAnswer = practiceAnswers[b];
    renderAllBoards(practiceState);

    const solved = practiceState.boards.filter((b) => b.solved).length;
    const youMark = (g === botPick) ? "" : ' <span class="you">(your pick)</span>';
    logLine(`${String(practiceState.guessesUsed).padStart(2,"0")}. <b>${word}</b>${youMark} · ${solved}/32` + (plus ? ` <span class="plus">+${plus}</span>` : ""));
    practiceWord.value = "";
    submitWord.disabled = true;

    if (solved === NUM_BOARDS) {
      setStatus(`solved all 32 in <b>${practiceState.guessesUsed}</b> guesses.`);
      practiceWord.disabled = true; useBotBtn.disabled = true;
      placeholderCands();
      return;
    }
    if (practiceState.guessesUsed >= 37) {
      setStatus(`out of guesses — solved <b>${solved}</b>/32.`);
      practiceWord.disabled = true; useBotBtn.disabled = true;
      placeholderCands();
      return;
    }
    setStatus(`turn <b>${practiceState.guessesUsed + 1}</b>/37 · <b>${solved}</b>/32 solved`);
    refreshPracticePicks();
    practiceWord.focus();
  }
  function useBotPick() {
    if (!practiceState) return;
    const top = engine.suggest(practiceState, 1).suggestions[0];
    if (!top) return;
    practiceWord.value = top.word;
    submitWord.disabled = false;
    submitPracticeGuess();
  }

  // --- DAILY: replay user's guesses against MT19937 daily answers ----------
  function analyzeDaily() {
    if (!engine || running) return;
    const id = parseInt(gameIdInput.value, 10);
    if (!id || id < 1) { setStatus("enter the daily <b>#</b> from duotrigordle.com."); return; }
    const raw = playedInput.value.trim();
    const played = raw
      ? raw.split(/[\s,]+/).map((w) => w.trim().toUpperCase()).filter(Boolean)
      : [];
    const bad = played.filter((w) => !engine.guessIndex.has(w));
    if (bad.length) { setStatus(`not valid guesses: ${bad.slice(0,4).join(", ")}`); return; }

    const ALL_GREEN = DtEngine.ALL_GREEN;
    const answerIdxs = DtEngine.dailyAnswers(id, engine.S);
    const answers = answerIdxs.map((i) => engine.solutions[i]);
    const state = engine.freshState();
    for (let b = 0; b < NUM_BOARDS; b++) state.boards[b].lastAnswer = answers[b];

    log.innerHTML = "";
    clearBoards();
    for (let i = 0; i < played.length; i++) {
      const word = played[i];
      const g = engine.guessIndex.get(word);
      const pats = new Array(NUM_BOARDS);
      let plus = 0;
      for (let b = 0; b < NUM_BOARDS; b++) {
        if (state.boards[b].solved) { pats[b] = ALL_GREEN; continue; }
        const p = DtEngine.computeFeedback(word, answers[b]);
        pats[b] = p;
        if (p === ALL_GREEN) plus++;
      }
      engine.applyGuess(state, g, pats);
      for (let b = 0; b < NUM_BOARDS; b++) state.boards[b].lastAnswer = answers[b];
      const solved = state.boards.filter((b) => b.solved).length;
      logLine(`${String(i+1).padStart(2,"0")}. <b>${word}</b> · ${solved}/32` + (plus ? ` <span class="plus">+${plus}</span>` : ""));
    }
    renderAllBoards(state);
    if (state.boards.every((b) => b.solved)) {
      placeholderCands();
      turnNum.textContent = String(state.guessesUsed).padStart(2, "0");
      setStatus(`daily <b>#${id}</b> finished — <b>${state.guessesUsed}</b> guesses.`);
      return;
    }
    const sug = engine.suggest(state, 5);
    const nextTurn = state.guessesUsed + 1;
    turnNum.textContent = String(nextTurn).padStart(2, "0");
    renderCands(sug.suggestions, 0);
    setStatus(`daily <b>#${id}</b> · turn <b>${String(nextTurn).padStart(2,"0")}</b> · <b>${sug.activeBoards}</b> active`);
  }

  // --- wire up -------------------------------------------------------------
  shuffleBtn.addEventListener("click", () => { if (solutions.length && !running) answersInput.value = pickRandom().join(","); });
  playBtn.addEventListener("click", autoSolve);
  newPracticeBtn.addEventListener("click", newPracticeGame);
  submitWord.addEventListener("click", submitPracticeGuess);
  useBotBtn.addEventListener("click", useBotPick);
  practiceWord.addEventListener("input", () => {
    const v = practiceWord.value.trim();
    submitWord.disabled = !/^[A-Za-z]{5}$/.test(v);
  });
  practiceWord.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !submitWord.disabled) submitPracticeGuess();
  });
  analyzeBtn.addEventListener("click", analyzeDaily);
  [gameIdInput, playedInput].forEach((el) =>
    el.addEventListener("keydown", (e) => { if (e.key === "Enter") analyzeDaily(); }));

  init().catch((e) => setStatus("failed to load: " + e));
})();
