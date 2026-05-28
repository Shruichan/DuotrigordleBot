// Duotrigordle solver, ported from the C++ core to plain JS so it runs in the
// browser (the extension + the web demo) and under Node for the tests.
// Same logic: entropy-greedy scoring across all 32 boards, the expected-solves
// bonus, distinct-answer propagation, the rhyme-trap override, and the value-net
// re-rank that keeps the worst case down. No deps — hand it the word lists and
// it builds the feedback table once on construction.

(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.DtEngine = factory();
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  const NUM_BOARDS = 32;
  const ALL_GREEN = 242; // 2*(1+3+9+27+81)
  const NUM_PATTERNS = 243;
  const LOG2 = Math.log(2);

  // Feedback for a single (guess, answer) pair, base-3 with position 0 as the
  // least-significant digit. Greens first, then yellows consume unmatched
  // answer letters left-to-right — matches Wordle's duplicate handling.
  function computeFeedback(guess, answer) {
    const used = [false, false, false, false, false];
    const d = [0, 0, 0, 0, 0];
    for (let i = 0; i < 5; i++) {
      if (guess[i] === answer[i]) { d[i] = 2; used[i] = true; }
    }
    for (let i = 0; i < 5; i++) {
      if (d[i] === 2) continue;
      for (let j = 0; j < 5; j++) {
        if (!used[j] && guess[i] === answer[j]) { d[i] = 1; used[j] = true; break; }
      }
    }
    return d[0] + d[1] * 3 + d[2] * 9 + d[3] * 27 + d[4] * 81;
  }

  // --- value net (tiny MLP, 25-64-64-1) ---------------------------------------
  // Binary layout written by scripts/train_value_net.py:
  //   u32 magic(0xDDDDD000), u32 feat_dim, u32 nLayers,
  //   u32 sizes[nLayers+1], then per layer: f32 W[out*in], f32 b[out],
  //   u32 magic_end(0xDDDDD001).
  function parseValueNet(bytes) {
    const dv = new DataView(bytes.buffer || bytes, bytes.byteOffset || 0);
    let o = 0;
    const u32 = () => { const v = dv.getUint32(o, true); o += 4; return v; };
    if (u32() !== 0xddddd000) throw new Error("value net: bad magic");
    const featDim = u32();
    const nLayers = u32();
    const sizes = [];
    for (let i = 0; i < nLayers + 1; i++) sizes.push(u32());
    const W = [], B = [];
    for (let l = 0; l < nLayers; l++) {
      const inF = sizes[l], outF = sizes[l + 1];
      const w = new Float32Array(inF * outF);
      for (let i = 0; i < w.length; i++) { w[i] = dv.getFloat32(o, true); o += 4; }
      const b = new Float32Array(outF);
      for (let i = 0; i < outF; i++) { b[i] = dv.getFloat32(o, true); o += 4; }
      W.push(w); B.push(b);
    }
    if (u32() !== 0xddddd001) throw new Error("value net: bad end magic");
    return { featDim, sizes, W, B };
  }

  function valueNetEval(net, feats) {
    let cur = feats;
    for (let l = 0; l < net.W.length; l++) {
      const inF = net.sizes[l], outF = net.sizes[l + 1];
      const w = net.W[l], b = net.B[l];
      const out = new Float32Array(outF);
      const lastLayer = l + 1 === net.W.length;
      for (let oi = 0; oi < outF; oi++) {
        let s = b[oi];
        const base = oi * inF;
        for (let i = 0; i < inF; i++) s += w[base + i] * cur[i];
        out[oi] = lastLayer ? s : (s > 0 ? s : 0); // ReLU on hidden layers
      }
      cur = out;
    }
    return cur[0];
  }

  // 25-dim state features from per-board post-guess candidate counts (0 = solved).
  function featuresFromCounts(guessesUsedPost, counts) {
    const f = new Float32Array(25);
    let total = 0, active = 0, solved = 0, maxC = 0, minC = 1e9, sumLog = 0;
    let c1 = 0, c2 = 0, c3 = 0, c4 = 0, c11 = 0, c30 = 0;
    const sizes = [];
    for (let b = 0; b < NUM_BOARDS; b++) {
      const k = counts[b];
      if (k <= 0) { solved++; continue; }
      active++; total += k;
      if (k > maxC) maxC = k;
      if (k < minC) minC = k;
      sumLog += Math.log(k) / LOG2;
      sizes.push(k);
      if (k === 1) c1++; else if (k === 2) c2++; else if (k === 3) c3++;
      else if (k <= 10) c4++; else if (k <= 30) c11++; else c30++;
    }
    if (active === 0) minC = 0;
    sizes.sort((a, b) => b - a);
    f[0] = guessesUsedPost / 37;
    f[1] = (37 - guessesUsedPost) / 37;
    f[2] = active / 32;
    f[3] = solved / 32;
    f[4] = total / (32 * 2653);
    f[5] = sumLog / 32;
    f[6] = active > 0 ? total / active / 2653 : 0;
    f[7] = maxC / 2653;
    f[8] = minC / 2653;
    f[9] = c1 / 32; f[10] = c2 / 32; f[11] = c3 / 32;
    f[12] = c4 / 32; f[13] = c11 / 32; f[14] = c30 / 32;
    for (let i = 0; i < 10; i++) f[15 + i] = i < sizes.length ? sizes[i] / 2653 : 0;
    return f;
  }

  function entropy(row, cands) {
    const n = cands.length;
    if (n === 0) return 0;
    const hist = new Int32Array(NUM_PATTERNS);
    for (let i = 0; i < n; i++) hist[row[cands[i]]]++;
    let h = 0;
    for (let p = 0; p < NUM_PATTERNS; p++) {
      const c = hist[p];
      if (c === 0) continue;
      const pr = c / n;
      h -= pr * (Math.log(pr) / LOG2);
    }
    return h;
  }

  class Engine {
    // solutions, guesses: arrays of 5-letter uppercase strings (solutions ⊂ guesses).
    // opts: { alpha=300, opener="LITRE", valueNetBytes=null, useExactV=true, exactK=12 }
    constructor(solutions, guesses, opts) {
      opts = opts || {};
      this.solutions = solutions;
      this.guesses = guesses;
      this.S = solutions.length;
      this.G = guesses.length;
      this.alpha = opts.alpha != null ? opts.alpha : 300;
      this.exactK = opts.exactK != null ? opts.exactK : 12;
      this.useExactV = opts.useExactV !== false;

      const gi = new Map();
      for (let i = 0; i < this.G; i++) gi.set(guesses[i], i);
      this.guessIndex = gi;
      this.solIndex = new Map();
      for (let i = 0; i < this.S; i++) this.solIndex.set(solutions[i], i);

      // guess <-> solution maps
      this.guessToSol = new Int32Array(this.G).fill(-1);
      this.solToGuess = new Int32Array(this.S);
      for (let s = 0; s < this.S; s++) {
        const g = gi.get(solutions[s]);
        this.solToGuess[s] = g;
        this.guessToSol[g] = s;
      }

      this.opener = opts.opener ? gi.get(opts.opener) : gi.get("LITRE");
      if (this.opener == null) this.opener = -1;

      this.net = opts.valueNetBytes ? parseValueNet(opts.valueNetBytes) : null;

      // Defer the heavy table build for the browser (use buildAsync); Node and
      // tests just build it inline.
      if (!opts.deferBuild) this._buildFeedbackTable();
    }

    // 14857 x 2653 Uint8Array. ~39MB; computed once (~2-5s).
    _buildFeedbackTable() {
      const G = this.G, S = this.S;
      const table = new Uint8Array(G * S);
      const sols = this.solutions, gs = this.guesses;
      for (let g = 0; g < G; g++) {
        const gw = gs[g];
        const base = g * S;
        for (let s = 0; s < S; s++) table[base + s] = computeFeedback(gw, sols[s]);
      }
      this.fb = table;
    }

    // Same table, built in chunks that yield to the event loop so the page
    // stays responsive. onProgress(fraction 0..1) is optional.
    async buildAsync(onProgress) {
      const G = this.G, S = this.S;
      const table = new Uint8Array(G * S);
      const sols = this.solutions, gs = this.guesses;
      for (let g = 0; g < G; g++) {
        const gw = gs[g];
        const base = g * S;
        for (let s = 0; s < S; s++) table[base + s] = computeFeedback(gw, sols[s]);
        if ((g & 511) === 0) {
          if (onProgress) onProgress(g / G);
          await new Promise((r) => setTimeout(r, 0));
        }
      }
      this.fb = table;
      if (onProgress) onProgress(1);
      return this;
    }
    row(g) { return this.fb.subarray(g * this.S, g * this.S + this.S); }

    freshState() {
      const boards = [];
      const all = new Array(this.S);
      for (let i = 0; i < this.S; i++) all[i] = i;
      for (let b = 0; b < NUM_BOARDS; b++) boards.push({ solved: false, cands: all.slice() });
      return { boards, guessesUsed: 0, answerUsed: new Uint8Array(this.S) };
    }

    // Apply guess g to a state given the per-board feedback patterns (ALL_GREEN
    // = solved). Mutates and returns the state. Mirrors the C++ distinct-answer
    // propagation: any unsolved |C|=1 board fixes its answer for the others.
    applyGuess(state, g, patterns) {
      const gSol = this.guessToSol[g];
      for (let b = 0; b < NUM_BOARDS; b++) {
        const bd = state.boards[b];
        if (bd.solved) continue;
        const p = patterns[b];
        if (p === ALL_GREEN) {
          bd.solved = true; bd.cands = [];
          if (gSol >= 0) state.answerUsed[gSol] = 1;
          continue;
        }
        const r = this.row(g);
        bd.cands = bd.cands.filter((s) => r[s] === p);
      }
      // Propagate distinct-answer constraint to quiescence.
      let changed = true;
      while (changed) {
        changed = false;
        for (let b = 0; b < NUM_BOARDS; b++) {
          const bd = state.boards[b];
          if (bd.solved || bd.cands.length !== 1) continue;
          const s = bd.cands[0];
          if (!state.answerUsed[s]) { state.answerUsed[s] = 1; changed = true; }
        }
        for (let b = 0; b < NUM_BOARDS; b++) {
          const bd = state.boards[b];
          if (bd.solved || bd.cands.length === 1) continue; // keep forced boards' sole cand
          const before = bd.cands.length;
          bd.cands = bd.cands.filter((s) => !state.answerUsed[s]);
          if (bd.cands.length !== before) changed = true;
        }
      }
      state.guessesUsed++;
      return state;
    }

    _activeBoards(state) {
      const a = [];
      for (let b = 0; b < NUM_BOARDS; b++) {
        const bd = state.boards[b];
        if (!bd.solved && bd.cands.length > 0) a.push(b);
      }
      return a;
    }

    _expectedSolves(active, state) {
      const es = new Float64Array(this.S).fill(1);
      for (const b of active) {
        const cs = state.boards[b].cands;
        const keep = 1 - 1 / cs.length;
        for (const s of cs) es[s] *= keep;
      }
      for (let i = 0; i < this.S; i++) es[i] = 1 - es[i];
      return es;
    }

    _scoreGuess(g, active, state, es) {
      const r = this.row(g);
      let s = 0;
      for (const b of active) s += entropy(r, state.boards[b].cands);
      const sol = this.guessToSol[g];
      if (sol >= 0) s += this.alpha * es[sol];
      return s;
    }

    // Returns the chosen guess index for the current state.
    chooseGuess(state) {
      if (state.guessesUsed === 0 && this.opener >= 0) return this.opener;

      const active = this._activeBoards(state);
      if (active.length === 0) return 0;
      const es = this._expectedSolves(active, state);

      // Forced shortcut: if some board is down to one candidate, just play it
      // (best of the forced words by score). Guaranteed solve next.
      const forced = [];
      for (const b of active) {
        if (state.boards[b].cands.length === 1) forced.push(this.solToGuess[state.boards[b].cands[0]]);
      }
      if (forced.length) {
        let best = forced[0], bestS = -1;
        for (const g of forced) {
          const sc = this._scoreGuess(g, active, state, es);
          if (sc > bestS) { bestS = sc; best = g; }
        }
        return best;
      }

      // Score every valid guess; keep the argmax and a top-K shortlist.
      const scores = new Float64Array(this.G);
      let bestG = 0, bestS = -Infinity;
      for (let g = 0; g < this.G; g++) {
        const sc = this._scoreGuess(g, active, state, es);
        scores[g] = sc;
        if (sc > bestS) { bestS = sc; bestG = g; }
      }

      // Exact expected-V re-ranking among the top-K (tail-averse). Uses the
      // deterministic expected post-state count per board, E[|C'|]=Σ|part|²/|C|.
      if (this.useExactV && this.net && active.length >= 3) {
        const K = Math.min(this.exactK, this.G);
        const order = Array.from({ length: this.G }, (_, i) => i);
        // Partial selection of top-K by score.
        order.sort((a, b) => scores[b] - scores[a]);
        const topK = order.slice(0, K);
        const postGU = state.guessesUsed + 1;
        let bestV = Infinity, bestE = bestG;
        const counts = new Int32Array(NUM_BOARDS);
        for (const g of topK) {
          const r = this.row(g);
          for (let b = 0; b < NUM_BOARDS; b++) {
            const bd = state.boards[b];
            if (bd.solved || bd.cands.length === 0) { counts[b] = 0; continue; }
            const part = new Int32Array(NUM_PATTERNS);
            for (const s of bd.cands) part[r[s]]++;
            let sumsq = 0;
            for (let p = 0; p < NUM_PATTERNS; p++) sumsq += part[p] * part[p];
            const e = sumsq / bd.cands.length;
            const rc = Math.round(e);
            counts[b] = rc <= 1 ? (bd.cands.length === 1 ? 0 : 1) : rc;
          }
          const v = valueNetEval(this.net, featuresFromCounts(postGU, counts));
          if (v < bestV) { bestV = v; bestE = g; }
        }
        bestG = bestE;
      }

      // Rhyme-trap override: a single remaining board with |C|>=3 near-rhymes.
      // A blind candidate guess averages 2 but can take |C| turns; an info word
      // that splits them into >=2 partitions is 2 turns guaranteed.
      let nActive = active.length, maxC = 0, singleB = -1;
      for (const b of active) {
        const c = state.boards[b].cands.length;
        if (c > maxC) { maxC = c; singleB = b; }
      }
      if (nActive === 1 && maxC >= 3) {
        const cs = state.boards[singleB].cands;
        let bestH = -1, bestInfo = bestG;
        for (let g = 0; g < this.G; g++) {
          const r = this.row(g);
          const h = entropy(r, cs);
          if (h > bestH) { bestH = h; bestInfo = g; }
        }
        if (bestH > 0.99) bestG = bestInfo;
      }

      return bestG;
    }

    guessWord(g) { return this.guesses[g]; }

    // "GYBBG" -> base-3 pattern (G=2, Y=1, B=0), position 0 least-significant.
    static parsePattern(str) {
      let p = 0, m = 1;
      for (let i = 0; i < 5; i++) {
        const ch = str[i];
        const d = ch === "G" || ch === "g" ? 2 : ch === "Y" || ch === "y" ? 1 : 0;
        p += d * m; m *= 3;
      }
      return p;
    }

    // Rebuild a game state from the on-screen board readout. `boards` is an
    // array of { guesses:[5-letter strings], feedback:["GYBBG",...] } — the same
    // shape the content script scrapes. All boards share the guess sequence, but
    // solved boards stop adding rows, so the longest board carries every guess.
    stateFromBoards(boards) {
      const state = { boards: [], guessesUsed: 0, answerUsed: new Uint8Array(this.S) };
      let maxRows = 0;
      const all = [];
      for (let i = 0; i < this.S; i++) all.push(i);
      for (const bd of boards) {
        const rows = Math.min(bd.guesses.length, bd.feedback.length);
        maxRows = Math.max(maxRows, rows);
        let cands = all;
        let solved = false;
        for (let i = 0; i < rows; i++) {
          const gw = bd.guesses[i];
          const pat = Engine.parsePattern(bd.feedback[i]);
          if (pat === ALL_GREEN) {
            solved = true;
            const sIdx = this.solIndex.get(gw);
            if (sIdx != null) state.answerUsed[sIdx] = 1;
            cands = [];
            break;
          }
          const gi = this.guessIndex.get(gw);
          if (gi != null) {
            const r = this.row(gi);
            cands = cands.filter((s) => r[s] === pat);
          } else {
            cands = cands.filter((s) => computeFeedback(gw, this.solutions[s]) === pat);
          }
        }
        state.boards.push({ solved, cands });
      }
      while (state.boards.length < NUM_BOARDS) state.boards.push({ solved: false, cands: all.slice() });
      state.guessesUsed = maxRows;
      // Propagate distinct-answer constraint (some boards may now be forced).
      let changed = true;
      while (changed) {
        changed = false;
        for (const bd of state.boards) {
          if (bd.solved || bd.cands.length !== 1) continue;
          if (!state.answerUsed[bd.cands[0]]) { state.answerUsed[bd.cands[0]] = 1; changed = true; }
        }
        for (const bd of state.boards) {
          if (bd.solved || bd.cands.length === 1) continue;
          const before = bd.cands.length;
          bd.cands = bd.cands.filter((s) => !state.answerUsed[s]);
          if (bd.cands.length !== before) changed = true;
        }
      }
      return state;
    }

    // Suggestion payload for the UI: the chosen guess plus a few alternatives.
    // couldSolve = active boards on which the word is still a live answer.
    suggest(state, topN) {
      topN = topN || 5;
      const active = this._activeBoards(state);
      const gameOver = active.length === 0;
      const out = { activeBoards: active.length, guessesUsed: state.guessesUsed, gameOver, suggestions: [] };
      if (gameOver) return out;

      const chosen = this.chooseGuess(state);
      const es = this._expectedSolves(active, state);

      // Rank the rest by greedy score for the alternatives list. Even on the
      // forced-opener turn we still compute the score-sorted list so the top-N
      // panel has real alternatives to show next to the chosen pick.
      const scored = [];
      for (let g = 0; g < this.G; g++) scored.push([g, this._scoreGuess(g, active, state, es)]);
      scored.sort((a, b) => b[1] - a[1]);
      const couldSolve = (g) => {
        const sol = this.guessToSol[g];
        if (sol < 0) return [];
        const bs = [];
        for (const b of active) if (state.boards[b].cands.includes(sol)) bs.push(b);
        return bs;
      };
      const seen = new Set();
      const push = (g) => { if (!seen.has(g)) { seen.add(g); out.suggestions.push({ word: this.guesses[g], couldSolve: couldSolve(g) }); } };
      push(chosen);
      for (const [g] of scored) { if (out.suggestions.length >= topN) break; push(g); }
      return out;
    }

    // Wordlebot-style review of a finished/partial game. All boards play the
    // same guess sequence in lockstep (the fullest board carries it), so we
    // replay turn by turn and score each played guess.
    //   skill (1-99): where the played guess's greedy score ranks among all words.
    //   luck  (1-99): how favorable the feedback was — small resulting partitions
    //                 averaged across the boards the guess touched.
    reviewGame(boards) {
      let seq = [];
      for (const bd of boards) if (bd.guesses.length > seq.length) seq = bd.guesses.slice();
      const turns = [];
      const state = this.freshState();
      let matched = 0;
      const all = []; for (let i = 0; i < this.S; i++) all.push(i);

      for (let t = 0; t < seq.length; t++) {
        const active = this._activeBoards(state);
        if (active.length === 0) break;
        const es = this._expectedSolves(active, state);
        const playedWord = seq[t];
        const playedG = this.guessIndex.get(playedWord);

        // skill: range-normalized rank of the played score among all guesses.
        let best = -Infinity, worst = Infinity, played = 0;
        for (let g = 0; g < this.G; g++) {
          const sc = this._scoreGuess(g, active, state, es);
          if (sc > best) best = sc;
          if (sc < worst) worst = sc;
          if (g === playedG) played = sc;
        }
        const skill = best > worst + 1e-9 ? 1 + 98 * (played - worst) / (best - worst) : 99;
        const botChoice = this.chooseGuess(state);

        // Apply the played guess and measure luck from the realized partitions.
        const pats = new Array(NUM_BOARDS);
        let luckSum = 0, luckN = 0, solvedThis = 0;
        for (let b = 0; b < NUM_BOARDS; b++) {
          const bd = state.boards[b];
          if (bd.solved) { pats[b] = ALL_GREEN; continue; }
          // Recover the feedback this board showed at turn t (if it has the row).
          const fbStr = boards[b] && boards[b].feedback[t];
          const pat = fbStr != null ? Engine.parsePattern(fbStr)
                                    : (playedG != null ? this.row(playedG)[bd.cands[0]] : 0);
          pats[b] = pat;
          if (pat === ALL_GREEN) { solvedThis++; continue; }
          if (playedG != null && bd.cands.length > 1) {
            const r = this.row(playedG);
            const part = new Int32Array(NUM_PATTERNS);
            for (const s of bd.cands) part[r[s]]++;
            const landed = part[pat];
            // probability mass of outcomes at least as large (unlucky) as landed
            let worseOrEq = 0, eq = 0;
            for (let p = 0; p < NUM_PATTERNS; p++) {
              if (part[p] === 0) continue;
              if (part[p] > landed) worseOrEq += part[p];
              else if (part[p] === landed) eq += part[p];
            }
            const pct = (worseOrEq + eq / 2) / bd.cands.length;
            luckSum += pct; luckN++;
          }
        }
        const luck = luckN > 0 ? 1 + 98 * (luckSum / luckN) : 50;
        if (playedG === botChoice) matched++;

        turns.push({
          turn: t + 1, guess: playedWord,
          skill, luck,
          boards_solved_this_turn: solvedThis,
          bot_choice: this.guesses[botChoice],
          decision_matched: playedG === botChoice,
        });
        if (playedG != null) this.applyGuess(state, playedG, pats);
      }

      const avg = (k) => turns.length ? turns.reduce((a, t) => a + t[k], 0) / turns.length : 0;
      const solved = state.boards.filter((b) => b.solved).length;
      return {
        summary: {
          total_guesses: state.guessesUsed, boards_solved: solved,
          avg_skill: avg("skill"), avg_luck: avg("luck"),
          decisions_matched: matched, decisions_total: turns.length,
        },
        turns,
      };
    }
  }

  // std::mt19937 port — exact-byte match with the C++ daily picker so the
  // website can derive the 32 daily answers from a duotrigordle game number.
  function MT19937(seed) {
    const s = new Uint32Array(624);
    s[0] = seed >>> 0;
    for (let i = 1; i < 624; i++) {
      s[i] = (Math.imul(1812433253, s[i - 1] ^ (s[i - 1] >>> 30)) + i) >>> 0;
    }
    let idx = 624;
    function twist() {
      for (let i = 0; i < 624; i++) {
        const y = ((s[i] & 0x80000000) | (s[(i + 1) % 624] & 0x7fffffff)) >>> 0;
        let v = s[(i + 397) % 624] ^ (y >>> 1);
        if (y & 1) v ^= 0x9908b0df;
        s[i] = v >>> 0;
      }
      idx = 0;
    }
    return function next() {
      if (idx >= 624) twist();
      let y = s[idx++];
      y ^= y >>> 11;
      y ^= (y << 7) & 0x9d2c5680;
      y ^= (y << 15) & 0xefc60000;
      y ^= y >>> 18;
      return y >>> 0;
    };
  }

  // Same algorithm as core/src/simulator.cpp::daily_answers — pulls 32 distinct
  // indices from std::mt19937(gameId) % numSolutions.
  function dailyAnswers(gameId, numSolutions) {
    const rng = MT19937(gameId);
    const seen = new Uint8Array(numSolutions);
    const out = [];
    while (out.length < 32) {
      const i = rng() % numSolutions;
      if (!seen[i]) { seen[i] = 1; out.push(i); }
    }
    return out;
  }

  return { Engine, computeFeedback, dailyAnswers, MT19937, NUM_BOARDS, ALL_GREEN };
});
