#include "strategy.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <vector>

namespace dt {

namespace {

inline double entropy_on_board(const Pattern* row, const std::vector<WordIdx>& cs) {
    if (cs.empty()) return 0.0;
    std::array<int, NUM_PATTERNS> count{};
    for (WordIdx s : cs) count[row[s]] += 1;
    const double n = static_cast<double>(cs.size());
    double H = 0.0;
    for (int c : count) {
        if (c == 0) continue;
        const double p = c / n;
        H -= p * std::log2(p);
    }
    return H;
}

}

WordIdx GreedyStrategy::choose_guess(const GameState& state) {
    if (state.guesses_used == 0 && opener_cache_ != INVALID_WORD) {
        return opener_cache_;
    }
    std::vector<int> active;
    for (int i = 0; i < NUM_BOARDS; ++i) {
        if (!state.boards[i].solved && !state.boards[i].candidates.empty()) {
            active.push_back(i);
        }
    }
    if (active.empty()) return 0;

    const size_t G = w_.num_guesses();
    std::vector<double> scores(G);
    #pragma omp parallel for schedule(static)
    for (long long g = 0; g < static_cast<long long>(G); ++g) {
        const Pattern* row = w_.feedback_row(static_cast<WordIdx>(g));
        double s = 0.0;
        for (int b : active) {
            s += entropy_on_board(row, state.boards[b].candidates);
        }
        int32_t sol = w_.guess_to_sol(static_cast<WordIdx>(g));
        if (sol >= 0) {
            for (int b : active) {
                for (WordIdx c : state.boards[b].candidates) {
                    if (c == static_cast<WordIdx>(sol)) { s += answer_bonus_; goto have_bonus; }
                }
            }
            have_bonus:;
        }
        scores[g] = s;
    }
    auto it = std::max_element(scores.begin(), scores.end());
    WordIdx best_g = static_cast<WordIdx>(std::distance(scores.begin(), it));
    if (state.guesses_used == 0) opener_cache_ = best_g;
    return best_g;
}

}
