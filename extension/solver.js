// In-page solver: loads the bundled word lists + value net, builds the engine
// once (off the critical path), and answers suggest/review requests. Replaces
// the old localhost backend — the extension is now fully self-contained.

(() => {
  const url = (p) => chrome.runtime.getURL(p);
  const loadLines = async (p) => (await (await fetch(url(p))).text())
    .split("\n").map((s) => s.trim()).filter(Boolean);
  const loadBytes = async (p) => new Uint8Array(await (await fetch(url(p))).arrayBuffer());

  const st = { engine: null, phase: "starting", progress: 0, error: null };

  const ready = (async () => {
    try {
      st.phase = "loading word lists";
      const [solutions, guesses, net] = await Promise.all([
        loadLines("data/solutions_default.txt"),
        loadLines("data/valid_guesses.txt"),
        loadBytes("data/value_net.bin"),
      ]);
      st.phase = "building tables";
      const eng = new DtEngine.Engine(solutions, guesses, {
        deferBuild: true, valueNetBytes: net, useExactV: true,
      });
      await eng.buildAsync((f) => { st.progress = f; });
      st.engine = eng;
      st.phase = "ready";
      return eng;
    } catch (e) {
      st.phase = "error";
      st.error = String(e);
      throw e;
    }
  })();

  window.DTSolver = {
    ready,
    status() {
      if (st.phase === "ready") return "ready";
      if (st.phase === "building tables") return `warming up ${Math.round(st.progress * 100)}%`;
      if (st.phase === "error") return "solver error";
      return st.phase;
    },
    async suggest(boards, topN) {
      const eng = await ready;
      const state = eng.stateFromBoards(boards);
      const r = eng.suggest(state, topN || 5);
      return {
        active_boards: r.activeBoards,
        guesses_used: r.guessesUsed,
        game_over: r.gameOver,
        mode: "perfect",
        suggestions: r.suggestions.map((s) => ({ word: s.word, could_solve: s.couldSolve })),
      };
    },
    async review(boards) {
      const eng = await ready;
      return eng.reviewGame(boards);
    },
  };
})();
