# DuotrigordleBot

Solver bot for [duotrigordle.com](https://duotrigordle.com) 

## Layout

```
data/        wordlists scraped from the live bundle
core/        C++ solver — greedy / beam / endgame strategies
server/      Python HTTP wrappers around the C++ worker
extension/   Chrome MV3 extension that overlays suggestions on duotrigordle.com
ml/          experiments: value net + turn-2 specialist training
```

## Build

```
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
ninja -C build
```

Needs: cmake, ninja, OpenMP, GoogleTest, nlohmann_json. On Debian/Ubuntu:

```
sudo apt install cmake ninja-build libgtest-dev nlohmann-json3-dev
```

## Run

```
python3 server/serve.py          # solver on http://127.0.0.1:8765
```

Load `extension/` as an unpacked Chrome extension, open duotrigordle.com.

## Numbers

Greedy with answer-bonus, 500-game realistic (distinct-answer) bench:

- 100% solve rate (under the Daily 37-guess threshold)
- Mean: ~33.8 guesses
- Range: 33–35


Headroom over greedy at turn 2 is ~0.35 guesses, the turn-2 specialist (ml/train_turn2.py) captures a small slice of
that. Late game is essentially optimal under greedy.
