// Landing-page demo: load the engine, then animate a full 32-board solve for a
// user-supplied (or random) answer set. Same engine the extension ships.

(() => {
  const $ = (id) => document.getElementById(id);
  const grid = $("grid"), feed = $("feed"), status = $("status"), bar = $("bar");
  const answersInput = $("answers"), solveBtn = $("solveBtn"), randomBtn = $("randomBtn");

  let engine = null;
  let solutions = [];
  const NUM_BOARDS = 32;

  // Build the board cells.
  const cells = [];
  for (let i = 0; i < NUM_BOARDS; i++) {
    const d = document.createElement("div");
    d.className = "mini";
    d.textContent = "·";
    grid.appendChild(d);
    cells.push(d);
  }

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

  function setStatus(html) { status.innerHTML = html; }
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

  async function init() {
    setStatus("loading word lists…");
    const [sols, guesses, net] = await Promise.all([
      loadLines("data/solutions_default.txt"),
      loadLines("data/valid_guesses.txt"),
      loadBytes("data/value_net.bin"),
    ]);
    solutions = sols;
    setStatus("warming up solver…");
    engine = new DtEngine.Engine(sols, guesses, { deferBuild: true, valueNetBytes: net, useExactV: true });
    await engine.buildAsync((f) => { bar.style.width = (f * 100).toFixed(0) + "%"; });
    bar.style.width = "100%";
    setStatus("ready — paste answers or hit <b>Random</b>, then <b>Solve</b>.");
    solveBtn.disabled = false;
    answersInput.value = pickRandom().join(",");
  }

  function resetBoards() {
    for (const c of cells) { c.className = "mini"; c.textContent = "·"; }
    feed.innerHTML = "";
  }

  async function solve() {
    if (!engine) return;
    const raw = answersInput.value.trim();
    let answers = raw ? raw.split(",").map((w) => w.trim().toUpperCase()) : pickRandom();
    if (answers.length !== NUM_BOARDS || answers.some((w) => !/^[A-Z]{5}$/.test(w))) {
      setStatus(`need exactly 32 five-letter words (got ${answers.length}).`);
      return;
    }
    const bad = answers.filter((w) => !engine.solIndex.has(w));
    if (bad.length) { setStatus(`not in the solution list: ${bad.slice(0, 4).join(", ")}…`); return; }

    solveBtn.disabled = true; randomBtn.disabled = true;
    resetBoards();
    const ALL_GREEN = DtEngine.ALL_GREEN;
    const state = engine.freshState();
    let turn = 0;
    while (state.guessesUsed < 50 && !state.boards.every((b) => b.solved)) {
      const g = engine.chooseGuess(state);
      const word = engine.guessWord(g);
      const pats = new Array(NUM_BOARDS);
      const solvedThisTurn = [];
      for (let b = 0; b < NUM_BOARDS; b++) {
        if (state.boards[b].solved) { pats[b] = ALL_GREEN; continue; }
        const p = DtEngine.computeFeedback(word, answers[b]);
        pats[b] = p;
        if (p === ALL_GREEN) solvedThisTurn.push(b);
      }
      engine.applyGuess(state, g, pats);
      turn++;

      for (const b of solvedThisTurn) {
        cells[b].textContent = answers[b];
        cells[b].className = "mini solved justsolved";
        setTimeout(((el) => () => el.classList.remove("justsolved"))(cells[b]), 220);
      }
      const solved = state.boards.filter((b) => b.solved).length;
      const line = document.createElement("div");
      line.innerHTML = `${String(turn).padStart(2)}. <b>${word}</b> — ${solved}/32 solved` +
        (solvedThisTurn.length ? ` (+${solvedThisTurn.length})` : "");
      feed.appendChild(line);
      feed.scrollTop = feed.scrollHeight;
      setStatus(`guess <b>${turn}</b> · <b>${solved}</b>/32 solved`);
      await sleep(solved === NUM_BOARDS ? 0 : 90);
    }
    const total = state.guessesUsed;
    setStatus(`solved all 32 in <b>${total}</b> guesses.`);
    solveBtn.disabled = false; randomBtn.disabled = false;
  }

  randomBtn.addEventListener("click", () => { if (solutions.length) answersInput.value = pickRandom().join(","); });
  solveBtn.addEventListener("click", solve);

  init().catch((e) => setStatus("failed to load: " + e));
})();
