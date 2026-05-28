# DuotrigordleBot

A solver for [duotrigordle.com](https://duotrigordle.com) — 32 Wordle boards at
once, 37 guesses. It scores every word against all the boards together and plays
the best one.

The extension and the web demo run the whole thing **in your browser** — no
server, nothing to install past the extension itself.

- **Web demo + install steps:** https://shruichan.github.io/DuotrigordleBot/
- **Extension:** load `extension/` unpacked (see below)

## How it plays

Entropy across all unsolved boards + a bonus for words that are likely to solve
a board outright. Near-tied candidates get re-ranked by a small value net that
predicts how many guesses are left, which keeps the worst games short. Plus the
usual endgame guards (distinct answers, a rhyme-trap override).

2000-game bench against the actual daily MT19937 distribution (IDs 1..2000):

| | mean | max | 35s | 36s |
|---|---|---|---|---|
| tail-averse (default) | 33.75 | 36 | 78 | 1 |
| lowest-mean greedy | 33.62 | 36 | 102 | 5 |

100% solved, every game under the 37-guess daily limit. See `EXPERIMENTS.md` for
the value-net / MCTS investigation and why ~33.6 is the floor.

## Layout

```
web/         the JS engine (source of truth) + node tests
extension/   self-contained Chrome MV3 extension (bundles the engine)
docs/        GitHub Pages site — landing + in-browser solve demo
data/        word lists + the trained value net
core/        original C++ solver (greedy / endgame / mcts) + bench
server/      optional local HTTP wrapper around the C++ worker
ml/          training scripts (value net, turn-2 specialist)
```

## Use the extension

1. Grab this repo and open `chrome://extensions`.
2. Turn on Developer mode → "Load unpacked" → pick the `extension/` folder.
3. Open duotrigordle.com. The overlay shows the next guess (first one takes a
   couple seconds while the solver warms up).

## Build the C++ core (optional)

Only needed for benching / the native worker:

```
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
ninja -C build
./build/dt_bench -n 2000 -S greedy --opener LITRE -a 300 --la-k 12 --la-exact \
    --value-net data/value_net_20260527.bin
```

Needs cmake, ninja, OpenMP, GoogleTest, nlohmann_json
(`sudo apt install cmake ninja-build libgtest-dev nlohmann-json3-dev`).
