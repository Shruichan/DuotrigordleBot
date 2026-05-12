#pragma once

#include "game_state.hpp"
#include "wordlists.hpp"

namespace dt {

class Strategy {
public:
    virtual ~Strategy() = default;
    virtual WordIdx choose_guess(const GameState& state) = 0;
    virtual const char* name() const = 0;
};

class GreedyStrategy : public Strategy {
public:
    explicit GreedyStrategy(const Wordlists& w, double answer_bonus = 0.1)
        : w_(w), answer_bonus_(answer_bonus), opener_cache_(INVALID_WORD) {}

    WordIdx choose_guess(const GameState& state) override;
    const char* name() const override { return "greedy"; }

    void clear_cache() { opener_cache_ = INVALID_WORD; }

private:
    const Wordlists& w_;
    double answer_bonus_;
    WordIdx opener_cache_;
};

}
