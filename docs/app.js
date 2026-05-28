// Watch-it-solve demo + play-along helper. The engine + value net are loaded
// once and shared between modes:
//   watch:  random or pasted 32 answers -> auto-solve, animated.
//   play:   today's daily # + your played guesses -> top-5 next picks.

(() => {
  const $ = (id) => document.getElementById(id);
  const grid = $("grid"), log = $("log"), status = $("status"), bar = $("bar");
  const cands = $("cands"), turnNum = $("turnNum"), solvedCount = $("solvedCount");

  // Watch controls
  const answersInput = $("answers"), playBtn = $("playBtn"), shuffleBtn = $("shuffleBtn");
  // Play controls
  const gameIdInput = $("gameId"), playedInput = $("playedGuesses"), analyzeBtn = $("analyzeBtn");

  const NUM_BOARDS = 32;
  let engine = null;
  let solutions = [];
  let running = false;
  let mode = "watch";

  // mini-board cells
  const cells = [];
  for (let i = 0; i < NUM_BOARDS; i++) {
    const d = document.createElement("div");
    d.className = "mini";
    d.innerHTML = `<span class="cands-n">·</span>`;
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
        <span class="word">${[..."—————"].map(c => `<span class="ct">${c}</span>`).join("")}</span>
        <span class="meta">${i === 0 ? "waiting" : ""}</span>`;
      cands.appendChild(li);
    }
  }
  placeholderCands();

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

  function fmtCount(n) {
    if (n >= 1000) return (n / 1000).toFixed(n >= 10000 ? 0 : 1).replace(/\.0$/, "") + "k";
    return String(n);
  }

  function clearBoards() {
    for (const c of cells) { c.className = "mini"; c.innerHTML = `<span class="cands-n">·</span>`; }
    solvedCount.textContent = "0";
  }
  function paintBoards(state, answers) {
    let solved = 0;
    for (let b = 0; b < NUM_BOARDS; b++) {
      const bd = state.boards[b];
      const el = cells[b];
      if (bd.solved) {
        solved++;
        if (!el.classList.contains("solved")) {
          const ans = answers ? answers[b] : "";
          el.className = "mini solved justsolved";
          el.innerHTML = `<span class="ans">${ans}</span>`;
          setTimeout(() => el.classList.remove("justsolved"), 460);
        }
      } else {
        const n = bd.cands.length;
        el.className = "mini" + (n <= 3 ? " tight" : "");
        el.innerHTML = `<span class="cands-n">${fmtCount(n)}</span>`;
      }
    }
    solvedCount.textContent = solved;
  }

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
      li.className = "cand" + (!has ? " placeholder" : "")
                            + (i === chosenIdx ? " chosen" : "")
                            + (opts.flashChosen && i === chosenIdx ? " flash" : "")
                            + (opts.dim && i !== chosenIdx ? " dim" : "");
      li.innerHTML = `
        <span class="rank">${String(i + 1).padStart(2, "0")}</span>
        <span class="word">${[...word].map(c => `<span class="ct">${c}</span>`).join("")}</span>
        <span class="meta">${meta}</span>`;
      cands.appendChild(li);
    }
  }

  // ---- mode switching ----------------------------------------------------
  function setMode(m) {
    mode = m;
    document.querySelectorAll(".tab").forEach((t) =>
      t.classList.toggle("active", t.dataset.mode === m));
    document.querySelector(".mode-watch").hidden = m !== "watch";
    document.querySelector(".mode-play").hidden = m !== "play";
    log.innerHTML = "";
    placeholderCands();
    clearBoards();
    turnNum.textContent = "—";
    if (m === "watch") {
      setStatus("ready — <b>shuffle</b>, then <b>play</b>.");
    } else {
      setStatus("enter the daily <b>#</b> and your <b>guesses</b>, then <b>next 5 picks</b>.");
    }
  }
  document.querySelectorAll(".tab").forEach((t) => {
    t.addEventListener("click", () => { if (!running) setMode(t.dataset.mode); });
  });

  // ---- init --------------------------------------------------------------
  async function init() {
    setStatus("loading word lists…");
    const [sols, guesses, net] = await Promise.all([
      loadLines("data/solutions_default.txt"),
      loadLines("data/valid_guesses.txt"),
      loadBytes("data/value_net.bin"),
    ]);
    solutions = sols;
    setStatus("warming up the solver — this happens once per page…");
    engine = new DtEngine.Engine(sols, guesses, { deferBuild: true, valueNetBytes: net, useExactV: true });
    await engine.buildAsync((f) => { bar.style.width = (f * 100).toFixed(0) + "%"; });
    bar.style.width = "100%";
    setStatus("ready — <b>shuffle</b>, then <b>play</b>.");
    playBtn.disabled = false; analyzeBtn.disabled = false;
    answersInput.value = pickRandom().join(",");

    // Pre-fill turn 1 candidates so the panel never sits empty.
    const fresh = engine.freshState();
    const r0 = engine.suggest(fresh, 5);
    turnNum.textContent = "01";
    renderCands(r0.suggestions, 0);
    // populate mini-boards with starting candidate count
    for (const c of cells) c.innerHTML = `<span class="cands-n">${fmtCount(sols.length)}</span>`;
  }

  // ---- watch mode --------------------------------------------------------
  async function solve() {
    if (!engine || running) return;
    running = true;
    playBtn.disabled = true; shuffleBtn.disabled = true;

    const raw = answersInput.value.trim();
    let answers = raw ? raw.split(",").map((w) => w.trim().toUpperCase()) : pickRandom();
    if (answers.length !== NUM_BOARDS || answers.some((w) => !/^[A-Z]{5}$/.test(w))) {
      setStatus(`need exactly 32 five-letter words (got ${answers.length}).`);
      running = false; playBtn.disabled = false; shuffleBtn.disabled = false; return;
    }
    const bad = answers.filter((w) => !engine.solIndex.has(w));
    if (bad.length) { setStatus(`not in the solution list: ${bad.slice(0, 4).join(", ")}…`); running = false; playBtn.disabled = false; shuffleBtn.disabled = false; return; }

    log.innerHTML = "";
    clearBoards();
    const ALL_GREEN = DtEngine.ALL_GREEN;
    const state = engine.freshState();

    let turn = 0;
    while (state.guessesUsed < 50 && !state.boards.every((b) => b.solved)) {
      const sug = engine.suggest(state, 5);
      turn = state.guessesUsed + 1;
      turnNum.textContent = String(turn).padStart(2, "0");
      renderCands(sug.suggestions, 0);
      setStatus(`turn <b>${String(turn).padStart(2,"0")}</b> · <b>${sug.activeBoards}</b> active`);
      await sleep(360);

      const word = sug.suggestions[0].word;
      const g = engine.guessIndex.get(word);
      const pats = new Array(NUM_BOARDS);
      const solvedThisTurn = [];
      for (let b = 0; b < NUM_BOARDS; b++) {
        if (state.boards[b].solved) { pats[b] = ALL_GREEN; continue; }
        const p = DtEngine.computeFeedback(word, answers[b]);
        pats[b] = p;
        if (p === ALL_GREEN) solvedThisTurn.push(b);
      }
      engine.applyGuess(state, g, pats);
      renderCands(sug.suggestions, 0, { flashChosen: true, dim: true });
      paintBoards(state, answers);

      const solved = state.boards.filter((b) => b.solved).length;
      const line = document.createElement("div");
      const plus = solvedThisTurn.length ? ` <span class="plus">+${solvedThisTurn.length}</span>` : "";
      line.innerHTML = `${String(turn).padStart(2, "0")}. <b>${word}</b> · ${solved}/32${plus}`;
      log.appendChild(line);
      log.scrollTop = log.scrollHeight;
      setStatus(`turn <b>${String(turn).padStart(2,"0")}</b> · <b>${solved}</b>/32`);
      await sleep(solved === NUM_BOARDS ? 0 : 340);
    }
    setStatus(`solved in <b>${state.guessesUsed}</b> guesses.`);
    running = false; playBtn.disabled = false; shuffleBtn.disabled = false;
  }

  // ---- play-along mode ---------------------------------------------------
  function analyze() {
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

    log.innerHTML = "";
    clearBoards();
    const replayLine = (i, word, solvedNow, plus) => {
      const line = document.createElement("div");
      line.innerHTML = `${String(i + 1).padStart(2, "0")}. <b>${word}</b> · ${solvedNow}/32` +
        (plus ? ` <span class="plus">+${plus}</span>` : "");
      log.appendChild(line);
    };

    // Replay the user's guesses against the daily answers.
    for (let i = 0; i < played.length; i++) {
      const word = played[i];
      const g = engine.guessIndex.get(word);
      const pats = new Array(NUM_BOARDS);
      let solvedNow = 0, plus = 0;
      for (let b = 0; b < NUM_BOARDS; b++) {
        if (state.boards[b].solved) { pats[b] = ALL_GREEN; continue; }
        const p = DtEngine.computeFeedback(word, answers[b]);
        pats[b] = p;
        if (p === ALL_GREEN) plus++;
      }
      engine.applyGuess(state, g, pats);
      solvedNow = state.boards.filter((b) => b.solved).length;
      replayLine(i, word, solvedNow, plus);
    }
    log.scrollTop = log.scrollHeight;
    paintBoards(state, answers);

    const allSolved = state.boards.every((b) => b.solved);
    if (allSolved) {
      placeholderCands();
      turnNum.textContent = String(state.guessesUsed).padStart(2, "0");
      setStatus(`game finished — <b>${state.guessesUsed}</b> guesses on daily <b>#${id}</b>.`);
      return;
    }

    const sug = engine.suggest(state, 5);
    const nextTurn = state.guessesUsed + 1;
    turnNum.textContent = String(nextTurn).padStart(2, "0");
    renderCands(sug.suggestions, 0);
    setStatus(`daily <b>#${id}</b> · turn <b>${String(nextTurn).padStart(2,"0")}</b> · <b>${sug.activeBoards}</b> active boards`);
  }

  // ---- wire up -----------------------------------------------------------
  shuffleBtn.addEventListener("click", () => { if (solutions.length && !running) answersInput.value = pickRandom().join(","); });
  playBtn.addEventListener("click", solve);
  analyzeBtn.addEventListener("click", analyze);
  // Allow Enter in either play-along input to trigger analyze.
  [gameIdInput, playedInput].forEach((el) =>
    el.addEventListener("keydown", (e) => { if (e.key === "Enter") analyze(); }));

  init().catch((e) => setStatus("failed to load: " + e));
})();
