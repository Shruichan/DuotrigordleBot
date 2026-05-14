#include "strategy_endgame.hpp"

#include <algorithm>
#include <array>
#include <limits>
#include <unordered_map>
#include <vector>

namespace dt {

namespace {

using BoardCands = std::vector<WordIdx>;
using EState = std::vector<BoardCands>;

struct EStateHash {
    size_t operator()(const EState& s) const {
        size_t h = s.size();
        for (const auto& b : s) {
            h ^= b.size() + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
            for (auto x : b) {
                h ^= std::hash<WordIdx>{}(x) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
            }
        }
        return h;
    }
};

void canonicalize(EState& s) {
    for (auto& b : s) std::sort(b.begin(), b.end());
    std::sort(s.begin(), s.end());
}

class Solver {
public:
    explicit Solver(const Wordlists& w) : w_(w) {}

    struct Result { WordIdx best_g; double cost; };

    Result solve(EState state) {
        canonicalize(state);
        if (state.empty()) return {INVALID_WORD, 0.0};

        // If exactly one board with one candidate, just play it.
        if (state.size() == 1 && state[0].size() == 1) {
            return {w_.sol_to_guess(state[0][0]), 1.0};
        }

        auto it = memo_.find(state);
        if (it != memo_.end()) return it->second;

        // Candidate guesses: words in any board's candidate set.
        // Adding all 14857 valid guesses would be optimal but slow; for endgame
        // (small state), info-only guesses don't help — the optimal first move
        // is always among the candidates.
        std::vector<WordIdx> guess_pool = collect_candidate_guesses(state);

        double best_cost = std::numeric_limits<double>::infinity();
        WordIdx best_g = guess_pool.empty() ? INVALID_WORD : guess_pool[0];

        for (WordIdx g : guess_pool) {
            double c = 1.0 + expected_cost(g, state);
            if (c < best_cost) { best_cost = c; best_g = g; }
        }

        Result r{best_g, best_cost};
        memo_.emplace(std::move(state), r);
        return r;
    }

private:
    std::vector<WordIdx> collect_candidate_guesses(const EState& s) {
        std::vector<WordIdx> out;
        std::vector<uint8_t> seen(w_.num_solutions(), 0);
        for (const auto& b : s) {
            for (WordIdx sol : b) {
                if (!seen[sol]) {
                    seen[sol] = 1;
                    out.push_back(w_.sol_to_guess(sol));
                }
            }
        }
        return out;
    }

    double expected_cost(WordIdx g, const EState& state) {
        const Pattern* row = w_.feedback_row(g);
        const size_t K = state.size();

        // For each board, build partition (pattern -> candidates).
        // Use compact array of (pattern, candidates) for present patterns only.
        struct BPart { Pattern pattern; BoardCands cands; };
        std::vector<std::vector<BPart>> parts(K);
        for (size_t i = 0; i < K; ++i) {
            std::array<int, NUM_PATTERNS> idx{};
            std::fill(idx.begin(), idx.end(), -1);
            for (WordIdx s : state[i]) {
                Pattern p = row[s];
                if (idx[p] < 0) {
                    idx[p] = static_cast<int>(parts[i].size());
                    parts[i].push_back({p, {}});
                }
                parts[i][idx[p]].cands.push_back(s);
            }
        }

        // Cartesian product over per-board partitions.
        std::vector<size_t> ix(K, 0);
        const double inv_sizes_prod = [&]() {
            double v = 1.0;
            for (size_t i = 0; i < K; ++i) v *= static_cast<double>(state[i].size());
            return v;
        }();

        double total = 0.0;
        while (true) {
            // Compute new state and outcome size product.
            EState new_state;
            new_state.reserve(K);
            double size_prod = 1.0;
            for (size_t i = 0; i < K; ++i) {
                const auto& bp = parts[i][ix[i]];
                size_prod *= static_cast<double>(bp.cands.size());
                if (bp.pattern != PATTERN_ALL_GREEN) {
                    new_state.push_back(bp.cands);
                }
            }
            const double prob = size_prod / inv_sizes_prod;
            double child_cost = 0.0;
            if (!new_state.empty()) {
                Result sub = solve(std::move(new_state));
                child_cost = sub.cost;
            }
            total += prob * child_cost;

            // Increment multi-index.
            size_t d = K;
            while (d-- > 0) {
                if (++ix[d] < parts[d].size()) break;
                ix[d] = 0;
                if (d == 0) return total;
            }
        }
    }

    const Wordlists& w_;
    std::unordered_map<EState, Result, EStateHash> memo_;
};

}

WordIdx EndgameStrategy::choose_guess(const GameState& state) {
    int total = 0;
    int active = 0;
    for (const auto& b : state.boards) {
        if (!b.solved) { total += static_cast<int>(b.candidates.size()); ++active; }
    }
    if (active == 0) return 0;
    if (total > max_total_) return greedy_.choose_guess(state);

    EState es;
    es.reserve(active);
    for (const auto& b : state.boards) {
        if (!b.solved && !b.candidates.empty()) es.push_back(b.candidates);
    }

    Solver solver(w_);
    auto r = solver.solve(es);
    if (r.best_g == INVALID_WORD) return greedy_.choose_guess(state);
    return r.best_g;
}

}
