#pragma once

#include "game_state.hpp"
#include "value_net.hpp"
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
    // Enable 2-step lookahead: for each top-K_la greedy candidate at turn t,
    // sample n_la answer tuples, apply, then evaluate via greedy's next-step
    // score on the resulting state. Pick the candidate maximizing the 2-step
    // total. 0 disables.
    void set_lookahead(int k_la, int n_la) { lookahead_k_ = k_la; lookahead_n_ = n_la; }
    // Exact expected-feature value ranking instead of Monte-Carlo sampling.
    // max_active gates it to states with <= that many active boards (the late
    // game, where the tail forms); large value = always on.
    void set_lookahead_exact(bool e, int max_active = 999) {
        lookahead_exact_ = e;
        lookahead_exact_max_active_ = max_active;
    }
    // Attach a value net for lookahead leaf eval. Pointer must outlive `*this`.
    void set_value_net(const ValueNet* vn) { value_net_ = vn; }
    // Budget-aware tail avoidance. When slack (= 34 - guesses_used -
    // active_boards) <= slack_threshold and >=2 boards are still ambiguous,
    // use `alpha` (typically ~0, info-favoring) for that turn instead of the
    // normal answer_bonus_. -1 disables (default).
    void set_panic(int slack_threshold, double alpha = 0.0) {
        panic_slack_ = slack_threshold;
        panic_alpha_ = alpha;
    }

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
    int lookahead_k_ = 0;  // 0 = no lookahead
    int lookahead_n_ = 0;
    bool lookahead_exact_ = false;
    int lookahead_exact_max_active_ = 999;
    const ValueNet* value_net_ = nullptr;
    int panic_slack_ = -1;       // -1 = disabled
    double panic_alpha_ = 0.0;
};

}
