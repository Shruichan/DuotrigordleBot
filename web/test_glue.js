// Tests the browser-facing API: play a full game while recording the per-board
// (guess, feedback) rows exactly as the content script would scrape them, then
// confirm stateFromBoards() rebuilds an equivalent state (the suggestion at each
// step matches live play) and reviewGame() runs and totals up correctly.

const fs = require("fs");
const path = require("path");
const { Engine, computeFeedback, NUM_BOARDS, ALL_GREEN } = require("./engine.js");

const DATA = path.join(__dirname, "..", "data");
const read = (f) => fs.readFileSync(path.join(DATA, f), "utf8").split("\n").map((s) => s.trim()).filter(Boolean);
const solutions = read("solutions_default.txt");
const guesses = read("valid_guesses.txt");
const netBytes = new Uint8Array(fs.readFileSync(path.join(DATA, "value_net_20260527.bin")));

const eng = new Engine(solutions, guesses, { useExactV: true, valueNetBytes: netBytes });
const patStr = (p) => { let s = ""; for (let i = 0; i < 5; i++) { const d = p % 3; p = (p - d) / 3; s += d === 2 ? "G" : d === 1 ? "Y" : "B"; } return s; };

// Fixed answer set.
let seed = 4242;
const rnd = () => (seed = (seed * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff;
const used = new Set(); const answers = [];
while (answers.length < NUM_BOARDS) { const i = Math.floor(rnd() * solutions.length); if (!used.has(i)) { used.add(i); answers.push(solutions[i]); } }

// Play, recording the DOM-style readout and checking reconstruction each turn.
const domBoards = answers.map(() => ({ guesses: [], feedback: [] }));
const live = eng.freshState();
let mismatches = 0;
while (live.guessesUsed < 50 && !live.boards.every((b) => b.solved)) {
  // Reconstruction check: rebuild from the DOM rows so far, compare suggestion.
  const rebuilt = eng.stateFromBoards(domBoards);
  const sugLive = eng.chooseGuess(live);
  const sugRebuilt = eng.chooseGuess(rebuilt);
  if (sugLive !== sugRebuilt) {
    mismatches++;
    if (mismatches <= 3) console.log("turn", live.guessesUsed, "live", eng.guessWord(sugLive), "rebuilt", eng.guessWord(sugRebuilt));
  }
  const g = sugLive;
  const pats = new Array(NUM_BOARDS);
  for (let b = 0; b < NUM_BOARDS; b++) {
    if (live.boards[b].solved) { pats[b] = ALL_GREEN; continue; }
    const p = computeFeedback(eng.guessWord(g), answers[b]);
    pats[b] = p;
    domBoards[b].guesses.push(eng.guessWord(g));
    domBoards[b].feedback.push(patStr(p));
  }
  eng.applyGuess(live, g, pats);
}

console.log("reconstruction mismatches:", mismatches, mismatches === 0 ? "(PASS)" : "(FAIL)");
console.log("game total:", live.guessesUsed);

// suggest() shape
const sg = eng.suggest(eng.stateFromBoards(domBoards.map((b) => ({ guesses: b.guesses.slice(0, 2), feedback: b.feedback.slice(0, 2) }))), 5);
console.log("suggest() after 2 turns:", sg.suggestions[0].word, "active", sg.activeBoards, "alts", sg.suggestions.length);

// reviewGame()
const rv = eng.reviewGame(domBoards);
console.log("review: total", rv.summary.total_guesses, "solved", rv.summary.boards_solved,
            "avg_skill", rv.summary.avg_skill.toFixed(1), "avg_luck", rv.summary.avg_luck.toFixed(1),
            "matched", rv.summary.decisions_matched + "/" + rv.summary.decisions_total);
const skillsOk = rv.turns.every((t) => t.skill >= 1 && t.skill <= 99 && t.luck >= 1 && t.luck <= 99);
console.log("review skill/luck in [1,99]:", skillsOk ? "PASS" : "FAIL");
