#include "strategy.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
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

struct GuessSetup {
    std::vector<int> active;
    std::vector<float> expected_solves;
};

GuessSetup build_setup(const Wordlists& w, const GameState& state) {
    GuessSetup gs;
    for (int i = 0; i < NUM_BOARDS; ++i) {
        const auto& b = state.boards[i];
        if (!b.solved && !b.candidates.empty()) gs.active.push_back(i);
    }
    gs.expected_solves.assign(w.num_solutions(), 0.0f);
    for (int b : gs.active) {
        const auto& cands = state.boards[b].candidates;
        if (cands.empty()) continue;
        const float inv = 1.0f / static_cast<float>(cands.size());
        for (WordIdx s : cands) gs.expected_solves[s] += inv;
    }
    return gs;
}

}

WordIdx GreedyStrategy::choose_guess(const GameState& state) {
    if (state.guesses_used == 0 && opener_cache_ != INVALID_WORD) {
        return opener_cache_;
    }

    auto gs = build_setup(w_, state);
    if (gs.active.empty()) return 0;

    const size_t G = w_.num_guesses();
    std::vector<double> scores(G);
    #pragma omp parallel for schedule(static)
    for (long long g = 0; g < static_cast<long long>(G); ++g) {
        const Pattern* row = w_.feedback_row(static_cast<WordIdx>(g));
        double s = 0.0;
        for (int b : gs.active) {
            s += entropy_on_board(row, state.boards[b].candidates);
        }
        int32_t sol = w_.guess_to_sol(static_cast<WordIdx>(g));
        if (sol >= 0) s += answer_bonus_ * gs.expected_solves[sol];
        scores[g] = s;
    }
    auto it = std::max_element(scores.begin(), scores.end());
    WordIdx best_g = static_cast<WordIdx>(std::distance(scores.begin(), it));
    if (state.guesses_used == 0) opener_cache_ = best_g;
    return best_g;
}

std::vector<WordIdx> GreedyStrategy::top_k_guesses(const GameState& state, int k) const {
    auto gs = build_setup(w_, state);
    const size_t G = w_.num_guesses();
    std::vector<double> scores(G);
    #pragma omp parallel for schedule(static)
    for (long long g = 0; g < static_cast<long long>(G); ++g) {
        const Pattern* row = w_.feedback_row(static_cast<WordIdx>(g));
        double s = 0.0;
        for (int b : gs.active) {
            s += entropy_on_board(row, state.boards[b].candidates);
        }
        int32_t sol = w_.guess_to_sol(static_cast<WordIdx>(g));
        if (sol >= 0) s += answer_bonus_ * gs.expected_solves[sol];
        scores[g] = s;
    }
    std::vector<WordIdx> idx(G);
    std::iota(idx.begin(), idx.end(), WordIdx{0});
    const int kk = std::min<int>(k, static_cast<int>(G));
    std::partial_sort(idx.begin(), idx.begin() + kk, idx.end(),
                      [&](WordIdx a, WordIdx b) { return scores[a] > scores[b]; });
    idx.resize(kk);
    return idx;
}

}
