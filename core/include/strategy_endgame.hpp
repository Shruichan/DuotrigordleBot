#pragma once

#include "strategy.hpp"

namespace dt {

// Endgame solver: when sum of active candidates is small, enumerates the optimal
// play tree (memoized DP) and returns the provably-best guess under independent
// uniform priors per board. Falls back to greedy when state is too large.
class EndgameStrategy : public Strategy {
public:
    EndgameStrategy(const Wordlists& w, int max_total_candidates = 25,
                    double answer_bonus = 1.0)
        : w_(w), max_total_(max_total_candidates), greedy_(w, answer_bonus) {}

    WordIdx choose_guess(const GameState& state) override;
    const char* name() const override { return "endgame"; }

    int max_total_candidates() const { return max_total_; }
    void set_max_total_candidates(int v) { max_total_ = v; }

private:
    const Wordlists& w_;
    int max_total_;
    GreedyStrategy greedy_;
};

}
