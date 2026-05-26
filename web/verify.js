// Node harness: play one Duotrigordle game with the JS engine on a fixed
// answer-set and print the guess sequence + total. Compare against the C++
// bench (--answers) to confirm the port is faithful.
//
//   node web/verify.js [exactv] [ANSWERS_CSV]
// If ANSWERS_CSV omitted, picks 32 distinct solutions with a fixed seed.

const fs = require("fs");
const path = require("path");
const { Engine, computeFeedback, NUM_BOARDS, ALL_GREEN } = require("./engine.js");

const DATA = path.join(__dirname, "..", "data");
const read = (f) => fs.readFileSync(path.join(DATA, f), "utf8").split("\n").map((s) => s.trim()).filter(Boolean);
const solutions = read("solutions_default.txt");
const guesses = read("valid_guesses.txt");

const useExactV = process.argv.includes("exactv");
let netBytes = null;
if (useExactV) {
  const nb = fs.readFileSync(path.join(DATA, "value_net_20260527.bin"));
  netBytes = new Uint8Array(nb);
}

// Answer set
let answers;
const csvArg = process.argv.find((a) => a.includes(",") && a.length > 10);
if (csvArg) {
  answers = csvArg.split(",").map((w) => w.trim().toUpperCase());
} else {
  // Deterministic LCG pick of 32 distinct solutions.
  let seed = 12345;
  const rnd = () => (seed = (seed * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff;
  const used = new Set();
  answers = [];
  while (answers.length < NUM_BOARDS) {
    const i = Math.floor(rnd() * solutions.length);
    if (!used.has(i)) { used.add(i); answers.push(solutions[i]); }
  }
}

console.error("config:", useExactV ? "greedy+exactV" : "plain greedy");
console.error("answers:", answers.join(","));

const t0 = Date.now();
const eng = new Engine(solutions, guesses, { useExactV, valueNetBytes: netBytes });
console.error("feedback table built in", Date.now() - t0, "ms");

const state = eng.freshState();
const seq = [];
while (state.guessesUsed < 50) {
  const allSolved = state.boards.every((b) => b.solved);
  if (allSolved) break;
  const g = eng.chooseGuess(state);
  const pats = new Array(NUM_BOARDS);
  for (let b = 0; b < NUM_BOARDS; b++) {
    pats[b] = state.boards[b].solved ? ALL_GREEN : computeFeedback(eng.guessWord(g), answers[b]);
  }
  eng.applyGuess(state, g, pats);
  const solvedNow = state.boards.filter((b) => b.solved).length;
  seq.push(`${String(seq.length + 1).padStart(2)}. ${eng.guessWord(g)}  active=${NUM_BOARDS - solvedNow}`);
}
console.log(seq.join("\n"));
console.log("TOTAL_GUESSES", state.guessesUsed, "solved", state.boards.filter((b) => b.solved).length + "/" + NUM_BOARDS);
