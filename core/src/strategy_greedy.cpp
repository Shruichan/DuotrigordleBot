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
    std::vector<WordIdx> forced;
};

GuessSetup build_setup(const Wordlists& w, const GameState& state) {
    GuessSetup gs;
    gs.active.reserve(NUM_BOARDS);
    for (int i = 0; i < NUM_BOARDS; ++i) {
        const auto& b = state.boards[i];
        if (!b.solved && !b.candidates.empty()) gs.active.push_back(i);
    }
    // Probability of solving at least one board if this word is played.
    // Under uniform per-board prior: 1 - prod_{b: s in C_b}(1 - 1/|C_b|).
    // Build as: track prod_{b: s in C_b}(1 - 1/|C_b|), then take 1 - that.
    // (For most words s is only in a few boards, so this is sparse.)
    gs.expected_solves.assign(w.num_solutions(), 1.0f);
    for (int b : gs.active) {
        const auto& cands = state.boards[b].candidates;
        if (cands.empty()) continue;
        const float keep = 1.0f - 1.0f / static_cast<float>(cands.size());
        for (WordIdx s : cands) gs.expected_solves[s] *= keep;
    }
    for (float& v : gs.expected_solves) v = 1.0f - v;
    for (int b : gs.active) {
        if (state.boards[b].candidates.size() == 1) {
            gs.forced.push_back(w.sol_to_guess(state.boards[b].candidates[0]));
        }
    }
    return gs;
}

}

double GreedyStrategy::score_guess(WordIdx g,
                                   const std::vector<int>& active,
                                   const GameState& state,
                                   const std::vector<float>& expected_solves) const {
    const Pattern* row = w_.feedback_row(g);
    double s = 0.0;
    for (int b : active) {
        s += entropy_on_board(row, state.boards[b].candidates);
    }
    int32_t sol = w_.guess_to_sol(g);
    if (sol >= 0) s += answer_bonus_ * expected_solves[sol];
    return s;
}

WordIdx GreedyStrategy::choose_guess(const GameState& state) {
    if (state.guesses_used == 0 && opener_cache_ != INVALID_WORD) {
        return opener_cache_;
    }

    auto gs = build_setup(w_, state);
    if (gs.active.empty()) return 0;

    if (!gs.forced.empty()) {
        double best = -1.0;
        WordIdx best_g = gs.forced[0];
        for (WordIdx g : gs.forced) {
            double s = score_guess(g, gs.active, state, gs.expected_solves);
            if (s > best) { best = s; best_g = g; }
        }
        return best_g;
    }

    const size_t G = w_.num_guesses();
    std::vector<double> scores(G);
    #pragma omp parallel for schedule(static)
    for (long long g = 0; g < static_cast<long long>(G); ++g) {
        scores[g] = score_guess(static_cast<WordIdx>(g), gs.active, state, gs.expected_solves);
    }
    auto it = std::max_element(scores.begin(), scores.end());
    WordIdx best_g = static_cast<WordIdx>(std::distance(scores.begin(), it));

    if (state.guesses_used == 0) opener_cache_ = best_g;
    return best_g;
}

std::vector<WordIdx> GreedyStrategy::top_k_guesses(const GameState& state, int k,
                                                   const std::vector<WordIdx>& pool) const {
    auto gs = build_setup(w_, state);
    if (gs.active.empty()) return {};

    // Forced moves: if any board has |C|=1, that word MUST be played eventually,
    // so it should rank at the top. Match choose_guess's behavior.
    if (!gs.forced.empty() && pool.empty()) {
        std::vector<std::pair<double, WordIdx>> forced_scored;
        forced_scored.reserve(gs.forced.size());
        for (WordIdx g : gs.forced) {
            forced_scored.push_back({score_guess(g, gs.active, state, gs.expected_solves), g});
        }
        std::sort(forced_scored.begin(), forced_scored.end(),
                  [](const auto& a, const auto& b) {
                      if (a.first != b.first) return a.first > b.first;
                      return a.second < b.second;
                  });
        std::vector<WordIdx> out;
        out.reserve(std::min<int>(k, static_cast<int>(forced_scored.size())));
        for (auto& p : forced_scored) {
            out.push_back(p.second);
            if (static_cast<int>(out.size()) >= k) return out;
        }
        // If forced doesn't fill k, fall through and append top non-forced.
        const size_t G = w_.num_guesses();
        std::vector<double> scores(G);
        std::vector<uint8_t> is_forced(G, 0);
        for (WordIdx g : gs.forced) is_forced[g] = 1;
        #pragma omp parallel for schedule(static)
        for (long long g = 0; g < static_cast<long long>(G); ++g) {
            scores[g] = is_forced[g] ? -1e18
                                     : score_guess(static_cast<WordIdx>(g), gs.active, state, gs.expected_solves);
        }
        std::vector<WordIdx> idx(G);
        std::iota(idx.begin(), idx.end(), WordIdx{0});
        const int need = k - static_cast<int>(out.size());
        std::partial_sort(idx.begin(), idx.begin() + need, idx.end(),
                          [&](WordIdx a, WordIdx b) {
                              if (scores[a] != scores[b]) return scores[a] > scores[b];
                              return a < b;
                          });
        for (int i = 0; i < need; ++i) out.push_back(idx[i]);
        return out;
    }

    if (pool.empty()) {
        const size_t G = w_.num_guesses();
        std::vector<double> scores(G);
        #pragma omp parallel for schedule(static)
        for (long long g = 0; g < static_cast<long long>(G); ++g) {
            scores[g] = score_guess(static_cast<WordIdx>(g), gs.active, state, gs.expected_solves);
        }
        std::vector<WordIdx> idx(G);
        std::iota(idx.begin(), idx.end(), WordIdx{0});
        const int kk = std::min<int>(k, static_cast<int>(G));
        std::partial_sort(idx.begin(), idx.begin() + kk, idx.end(),
                          [&](WordIdx a, WordIdx b) {
                              if (scores[a] != scores[b]) return scores[a] > scores[b];
                              return a < b;  // deterministic tie-break by index (alphabetical)
                          });
        idx.resize(kk);
        return idx;
    }

    const size_t P = pool.size();
    std::vector<double> scores(P);
    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < static_cast<long long>(P); ++i) {
        scores[i] = score_guess(pool[i], gs.active, state, gs.expected_solves);
    }
    std::vector<size_t> idx(P);
    std::iota(idx.begin(), idx.end(), size_t{0});
    const int kk = std::min<int>(k, static_cast<int>(P));
    std::partial_sort(idx.begin(), idx.begin() + kk, idx.end(),
                      [&](size_t a, size_t b) {
                          if (scores[a] != scores[b]) return scores[a] > scores[b];
                          return pool[a] < pool[b];  // tie-break by underlying word index
                      });
    std::vector<WordIdx> out;
    out.reserve(kk);
    for (int i = 0; i < kk; ++i) out.push_back(pool[idx[i]]);
    return out;
}

}
