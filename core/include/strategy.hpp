#pragma once

#include "game_state.hpp"
#include "wordlists.hpp"

#include <vector>

namespace dt {

class Strategy {
public:
    virtual ~Strategy() = default;
    virtual WordIdx choose_guess(const GameState& state) = 0;
    virtual const char* name() const = 0;
};

class GreedyStrategy : public Strategy {
public:
    explicit GreedyStrategy(const Wordlists& w, double answer_bonus = 1.0)
        : w_(w), answer_bonus_(answer_bonus), opener_cache_(INVALID_WORD) {}

    WordIdx choose_guess(const GameState& state) override;
    const char* name() const override { return "greedy"; }

    // Returns top-K guess indices ranked by greedy score (descending).
    // If pool is non-empty, the search universe is restricted to those guess indices
    // (used for Perfect mode: only score candidate-only guesses).
    std::vector<WordIdx> top_k_guesses(const GameState& state, int k,
                                       const std::vector<WordIdx>& pool = {}) const;

    void clear_cache() { opener_cache_ = INVALID_WORD; }
    // Force a specific opener; useful for benching opener candidates.
    void set_opener(WordIdx g) { opener_cache_ = g; }
    // Force an entire opening sequence (overrides set_opener while in effect).
    void set_forced_prefix(std::vector<WordIdx> seq) { forced_prefix_ = std::move(seq); }
    // Per-turn alpha schedule; if non-empty, overrides answer_bonus_ based on
    // current turn index (state.guesses_used; clamped to last entry).
    void set_alpha_schedule(std::vector<double> sched) { alpha_schedule_ = std::move(sched); }

private:
    double score_guess(WordIdx g,
                       const std::vector<int>& active,
                       const GameState& state,
                       const std::vector<float>& expected_solves) const;

    const Wordlists& w_;
    double answer_bonus_;
    WordIdx opener_cache_;
    std::vector<WordIdx> forced_prefix_;
    std::vector<double> alpha_schedule_;
};

}
