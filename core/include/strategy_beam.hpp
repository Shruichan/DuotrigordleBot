#pragma once

#include "strategy.hpp"

#include <random>

namespace dt {

class BeamStrategy : public Strategy {
public:
    BeamStrategy(const Wordlists& w, int k = 8, int samples = 3,
                 double answer_bonus = 1.0, int max_active_for_beam = 26)
        : w_(w), k_(k), samples_(samples),
          max_active_for_beam_(max_active_for_beam),
          greedy_(w, answer_bonus),
          opener_cache_(INVALID_WORD) {}

    WordIdx choose_guess(const GameState& state) override;
    const char* name() const override { return "beam"; }

private:
    double evaluate_candidate_paired(
        WordIdx g, const GameState& state,
        const std::vector<std::array<WordIdx, NUM_BOARDS>>& sampled_answers) const;

    const Wordlists& w_;
    int k_;
    int samples_;
    int max_active_for_beam_;
    GreedyStrategy greedy_;
    WordIdx opener_cache_;
};

}
