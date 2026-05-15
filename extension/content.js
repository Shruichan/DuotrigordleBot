// Reads board state from duotrigordle.com and asks the local solver.
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
      const guesses = [], feedback = [];
      for (let i = 0; i < cells.length; i += 5) {
        const row = cells.slice(i, i + 5);
        if (row.length < 5) break;
        const colors = row.map(detectColor);
        if (colors.some((c) => c === null)) continue;
        const letters = row.map(cellLetter).join("");
        if (!/^[A-Z]{5}$/.test(letters)) continue;
        guesses.push(letters); feedback.push(colors.join(""));
      }
      boards.push({ guesses, feedback });
    }
    return { boards };
  }

  let lastKey = "";
  setInterval(async () => {
    const state = readState();
    if (!state) return;
    const key = JSON.stringify(state);
    if (key === lastKey) return;
    lastKey = key;
    try {
      const resp = await chrome.runtime.sendMessage({ type: "suggest", state });
      log("suggestion:", resp);
    } catch (e) { log("err:", e); }
  }, POLL_MS);

  log("loaded");
})();
