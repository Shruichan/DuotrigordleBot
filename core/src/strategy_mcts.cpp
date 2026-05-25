#include "strategy_mcts.hpp"

#include <array>
#include <limits>
#include <random>
#include <vector>

namespace dt {

WordIdx MctsStrategy::choose_guess(const GameState& state) {
    // Opener / fresh-state shortcut.
    if (state.guesses_used == 0 && opener_ != INVALID_WORD) return opener_;

    // Count active boards; defer to greedy when the state is still wide.
    int n_active = 0;
    for (const auto& b : state.boards) if (!b.solved && !b.candidates.empty()) ++n_active;
    if (n_active == 0) return 0;
    if (n_active > max_active_) {
        greedy_.set_opener(INVALID_WORD);
        return greedy_.choose_guess(state);
    }

    // Candidate actions: top-K by greedy score.
    greedy_.set_opener(INVALID_WORD);
    std::vector<WordIdx> cands = greedy_.top_k_guesses(state, top_k_);
    if (cands.empty()) return greedy_.choose_guess(state);
    if (cands.size() == 1) return cands[0];

    const int K = static_cast<int>(cands.size());
    const bool use_net = (value_net_ != nullptr && value_net_->loaded());

    std::vector<double> cand_cost(K, std::numeric_limits<double>::infinity());

    #pragma omp parallel
    {
        // Per-thread greedy rollout policy + RNG.
        GreedyStrategy rollout_policy(w_, 300.0);
        rollout_policy.set_opener(INVALID_WORD);
        std::array<float, ValueNet::FEATURE_DIM> feats{};

        #pragma omp for schedule(dynamic, 1)
        for (int ci = 0; ci < K; ++ci) {
            WordIdx g = cands[ci];
            double sum = 0.0;
            // Deterministic per-(turn,candidate) seed for reproducible benches.
            std::mt19937_64 rng(0xC0FFEEULL
                                ^ (static_cast<uint64_t>(state.guesses_used) << 32)
                                ^ static_cast<uint64_t>(g) * 0x9E3779B97F4A7C15ULL);

            for (int r = 0; r < rollouts_; ++r) {
                // Sample a determinization: one concrete answer per unsolved board.
                std::array<WordIdx, NUM_BOARDS> answers{};
                for (int b = 0; b < NUM_BOARDS; ++b) {
                    const auto& cs = state.boards[b].candidates;
                    if (state.boards[b].solved || cs.empty()) { answers[b] = 0; continue; }
                    std::uniform_int_distribution<size_t> pick(0, cs.size() - 1);
                    answers[b] = cs[pick(rng)];
                }

                // Apply the candidate guess under this determinization.
                GameState s = state;
                {
                    std::array<Pattern, NUM_BOARDS> pats{};
                    for (int b = 0; b < NUM_BOARDS; ++b) {
                        pats[b] = s.boards[b].solved
                            ? PATTERN_ALL_GREEN
                            : w_.feedback(g, answers[b]);
                    }
                    s.apply_guess(w_, g, pats, true);
                }

                // Roll out with greedy until terminal or cutoff depth.
                int depth = 0;
                while (!s.game_over() && depth < cutoff_depth_) {
                    WordIdx ng = rollout_policy.choose_guess(s);
                    std::array<Pattern, NUM_BOARDS> pats{};
                    for (int b = 0; b < NUM_BOARDS; ++b) {
                        pats[b] = s.boards[b].solved
                            ? PATTERN_ALL_GREEN
                            : w_.feedback(ng, answers[b]);
                    }
                    s.apply_guess(w_, ng, pats, true);
                    ++depth;
                }

                double total;
                if (s.game_over()) {
                    total = static_cast<double>(s.guesses_used);
                } else if (use_net) {
                    ValueNet::compute_features(s, feats.data());
                    total = static_cast<double>(s.guesses_used)
                          + static_cast<double>(value_net_->eval(feats.data()));
                } else {
                    // No net + not terminal: fall back to a coarse remaining estimate.
                    int rem_active = 0, rem_cands = 0;
                    for (const auto& b : s.boards) {
                        if (!b.solved && !b.candidates.empty()) {
                            ++rem_active;
                            rem_cands += static_cast<int>(b.candidates.size());
                        }
                    }
                    total = static_cast<double>(s.guesses_used)
                          + rem_active
                          + 0.05 * static_cast<double>(rem_cands);
                }

                // Risk-adjusted: penalize rollouts that blow past 34.
                if (risk_lambda_ > 0.0 && total > 34.0) {
                    total += risk_lambda_ * (total - 34.0);
                }
                sum += total;
            }
            cand_cost[ci] = sum / rollouts_;
        }
    }

    // Pick the candidate with the lowest expected (risk-adjusted) total.
    int best = 0;
    for (int ci = 1; ci < K; ++ci) {
        if (cand_cost[ci] < cand_cost[best]) best = ci;
    }
    return cands[best];
}

}
