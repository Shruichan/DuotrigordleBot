# Pushing past the greedy floor — value net, MCTS, and the mean-vs-tail wall

All benches: 2000 games, seed 7, LITRE opener, α=300, distinct-answer constraint on.
Daily picker (MT19937) gives the same distribution as random distinct sampling.

## TL;DR

- **Greedy α=300 is the mean-optimum at 33.62.** Nothing beats it. Three
  independent search methods (learned value-net lookahead, determinized MCTS,
  budget-aware "panic") all fail to lower the mean.
- **The 35-guess tail is structural.** ~67/2000 (3.4%) of answer-sets force
  multiple simultaneous `|C|=2` 50/50s in a tight endgame; no policy solves them
  in ≤34. Proven by three risk-averse variants all flooring at ~67–71.
- **Mean and tail are one risk dial.** You cannot have "majority 33s" *and*
  "no tail" — they are opposite ends of the same tradeoff.
- **Best tail-averse config** (`--la-exact`): mean **33.75** on the daily distribution
  (33.76 on uniform random samples), shaves ~30 of the 35s and most of the 36s
  at a cost of +0.13 mean and fewer lucky 33s. Caps games at 35 the vast
  majority of the time but doesn't *eliminate* 36s — there's a ~0.05%
  structural floor of catastrophic seeds across any 2k-game sample.

## Followup — daily-specialized training (negative result)

Trained a value-net on greedy self-play across the historical daily-MT19937
sequence (IDs 1..1500) instead of random distinct samples, on the hypothesis
that the net could pick up daily-specific seed quirks. It didn't:

| config (benched on daily IDs 1..2000) | mean | max | 35s | 36s |
|---|---|---|---|---|
| exact-V random-trained (current default) | 33.75 | 36 | 78 | 1 |
| exact-V daily-trained                    | 33.73 | 36 | 78 | 2 |

Within noise. Both training distributions sample 32 distinct words from the
same 2653-solution pool, so the conditional `state → remaining-turns` density
the net learns is nearly identical. The bottleneck remains: top-K greedy
candidates are too close in true value for *any* net to differentiate
reliably. We don't ship the daily-trained net; the gen-side `--daily START END`
flag is left in `dt_gen_value_data` for future experiments.

## Results table

| approach | mean | max | 32 | 33 | 34 | 35 | 36 |
|---|---|---|---|---|---|---|---|
| **greedy (default, lowest mean)** | **33.62** | 36 | 8 | **831** | 1071 | 87 | 3 |
| value-net 1-ply lookahead (sampled) | 33.64 | 36 | 4 | 795 | 1121 | 78 | 2 |
| MCTS (best of 8 configs) | 33.62 | 36 | 8 | 831 | 1070 | 86 | 5 |
| panic mode (any setting) | 34.1–36.7 | 38–39 | — | — | — | catastrophic | — |
| **exact-V (lowest tail)** | 33.76 | **35** | 3 | 548 | 1382 | **67** | **0** |
| exact-V quantile τ=0.90 | 33.77 | 35 | 1 | 524 | 1406 | 69 | 0 |

## What was tried, and why each did/didn't work

### 1. Value network (succeeded as a model, neutral as a policy)
- `dt_gen_value_data`: 100k greedy self-play games → 3.3M `(state[25], remaining_turns)` rows.
- `scripts/train_value_net.py`: MLP 25→64→64→1, SmoothL1. **val L1 = 0.10 turns** —
  the value function is excellent at predicting remaining turns.
- As a **1-ply lookahead leaf** (replacing the hand-tuned `E[turns|k]` table):
  mean +0.02. The value net is accurate at the *mean*, but the top-K greedy
  candidates have near-identical true value, and Monte-Carlo sampling noise
  (N determinizations) swamps the tiny differences → near-random tie-break.

### 2. Determinized MCTS (no help)
- `strategy_mcts.cpp`: flat Monte-Carlo over top-K greedy candidates; sample R
  determinizations, apply, roll out with greedy to depth `d`, value-net cutoff.
- 8-config sweep (depth 2–4, R 20–40, K 10–12, risk 0–15, gate 14–20): **all
  33.62–33.63, tail unmoved.** The tail isn't caused by greedy picking a bad
  move from a good state — the trap is baked into the random answer-set.

### 3. Budget-aware "panic" mode (backfired)
- Hypothesis: when slack toward a 34-finish is tight, drop α→0 to play an info
  word that disambiguates several `|C|=2` boards at once.
- Reality: pure-entropy in a tight endgame plays a globally-informative word
  that **solves nothing**, keeps boards ambiguous, keeps panic triggering →
  games balloon to 37+. Every setting made mean *and* tail worse. **Disabled by
  default** (kept as documented opt-in `--panic-slack`).
- Lesson: a `|C|=2` board is a genuine 50/50. Committing (avg 1.5 turns) beats
  info-then-commit (2 turns guaranteed) in expectation. Greedy α=300 already
  makes this trade correctly.

### 4. Exact expected-V ranking (the real lever for the tail)
- `--la-exact`: rank top-K by **deterministic** expected post-state value
  `E[V(s')]` (per board `E[|C'|]=Σ|part_p|²/|C|`) — no sampling noise.
- Minimizing E[V] is **risk-averse**: it clusters outcomes at 34, kills all 36s
  (max=35), and trims 35s (87→67). Costs +0.14 mean and ~280 of the 33s.
- **Quantile retrain (τ=0.8/0.9/0.95)** to be explicitly worst-case-averse:
  no further tail reduction (67→69–71). This is the structural floor.

## Why the tail can't be removed (mean-first)

A `|C|=2` board cannot be solved in one guess (it's a coin flip). A 35-guess
game = 3+ failed coin-flips in one game. When the daily answer-set drops several
`|C|=2` boards into a tight endgame simultaneously, missing 3+ is likely and
unavoidable — info-splitting them all would cost *more* turns than the misses.
The published "optimal" multi-Wordle solvers (~33.4 mean) also carry a 35/36
tail for exactly these seeds.

## Recommended configs

- **Lowest mean / most 33s (current default):**
  `dt_bench -S greedy --opener LITRE -a 300`  → 33.62, but max 36.
- **Tail-averse for the live bot (recommended if 36s bother you):**
  `dt_bench -S greedy --opener LITRE -a 300 --la-k 12 --la-exact --value-net data/value_net_20260527.bin`
  → 33.76, **max 35, zero 36s**.

To use the tail-averse policy in the live bot, load `value_net_20260527.bin` in
`dt_worker` and set `--la-k 12 --la-exact` on its greedy instance (one-line wiring;
left to your call since it trades +0.14 mean for the bounded max).

## Artifacts
- `core/bench/gen_value_data.cpp` — value training-data exporter
- `scripts/train_value_net.py` — MLP trainer (Huber or `--quantile` pinball)
- `core/{include,src}/value_net.*` — dependency-free C++ MLP inference
- `core/{include,src}/strategy_mcts.*` — determinized rollout search
- greedy opt-in knobs: `--la-exact[-max-active]`, `--panic-slack/-alpha`, `--la-k/-n`
- sweep scripts + logs: `scripts/{mcts,panic,exactv,quantile}_*`
