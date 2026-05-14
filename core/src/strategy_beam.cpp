#include "strategy_beam.hpp"

#include <algorithm>
#include <atomic>
#include <iostream>
#include <limits>
#include <random>

namespace dt { extern std::atomic<int> g_beam_triggered; extern std::atomic<int> g_beam_skipped; }
namespace dt { std::atomic<int> g_beam_triggered{0}; std::atomic<int> g_beam_skipped{0}; }

namespace dt {

double BeamStrategy::evaluate_candidate_paired(
    WordIdx g, const GameState& state,
    const std::vector<std::array<WordIdx, NUM_BOARDS>>& sampled_answers) const {
    double total = 0.0;
    constexpr int max_total_guesses = 50;

    for (const auto& answers : sampled_answers) {
        GameState sim = state;
        std::array<Pattern, NUM_BOARDS> pats{};
        for (int b = 0; b < NUM_BOARDS; ++b) {
            pats[b] = sim.boards[b].solved
                ? PATTERN_ALL_GREEN
                : w_.feedback(g, answers[b]);
        }
        sim.apply_guess(w_, g, pats);

        GreedyStrategy local_greedy(w_, 1.0);
        while (!sim.game_over() && sim.guesses_used < max_total_guesses) {
            WordIdx ng = local_greedy.choose_guess(sim);
            for (int b = 0; b < NUM_BOARDS; ++b) {
                pats[b] = sim.boards[b].solved
                    ? PATTERN_ALL_GREEN
                    : w_.feedback(ng, answers[b]);
            }
            sim.apply_guess(w_, ng, pats);
        }
        total += static_cast<double>(sim.guesses_used);
    }
    return total / sampled_answers.size();
}

WordIdx BeamStrategy::choose_guess(const GameState& state) {
    if (state.guesses_used == 0 && opener_cache_ != INVALID_WORD) {
        return opener_cache_;
    }

    int active = state.active_boards();
    if (active < 2 || active > max_active_for_beam_) {
        g_beam_skipped.fetch_add(1);
        WordIdx g = greedy_.choose_guess(state);
        if (state.guesses_used == 0) opener_cache_ = g;
        return g;
    }
    g_beam_triggered.fetch_add(1);

    auto top_k = greedy_.top_k_guesses(state, k_);

    for (int b = 0; b < NUM_BOARDS; ++b) {
        const auto& bd = state.boards[b];
        if (!bd.solved && bd.candidates.size() == 1) {
            WordIdx forced_g = w_.sol_to_guess(bd.candidates[0]);
            if (std::find(top_k.begin(), top_k.end(), forced_g) == top_k.end()) {
                top_k.push_back(forced_g);
            }
        }
    }

    // Generate sample answer tuples ONCE — all candidates evaluated against the same
    // samples for paired-comparison variance reduction.
    std::vector<std::array<WordIdx, NUM_BOARDS>> sampled_answers(samples_);
    {
        // paired sampling — same answer tuples across candidates
    std::mt19937_64 rng(std::hash<uint64_t>{}(state.guesses_used) ^ 0xD06EE0FFEE5DULL);
        for (int s = 0; s < samples_; ++s) {
            for (int b = 0; b < NUM_BOARDS; ++b) {
                const auto& cands = state.boards[b].candidates;
                if (state.boards[b].solved || cands.empty()) {
                    sampled_answers[s][b] = 0;
                } else {
                    std::uniform_int_distribution<size_t> dist(0, cands.size() - 1);
                    sampled_answers[s][b] = cands[dist(rng)];
                }
            }
        }
    }

    std::vector<double> scores(top_k.size(), 0.0);
    #pragma omp parallel for schedule(dynamic, 1)
    for (long long i = 0; i < static_cast<long long>(top_k.size()); ++i) {
        scores[i] = evaluate_candidate_paired(top_k[i], state, sampled_answers);
    }

    auto it = std::min_element(scores.begin(), scores.end());
    WordIdx best = top_k[std::distance(scores.begin(), it)];

    if (state.guesses_used == 0) opener_cache_ = best;
    return best;
}

}

// note: beam ties greedy on bench. keeping the implementation but not used by default.
