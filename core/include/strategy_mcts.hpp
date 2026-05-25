#pragma once

#include "strategy.hpp"
#include "value_net.hpp"
#include "wordlists.hpp"

namespace dt {

// Determinized-rollout search ("flat Monte Carlo") over the top-K greedy
// candidate guesses. For each candidate we sample R determinizations (concrete
// answer per board drawn from its candidate set), apply the candidate, then
// roll out with greedy to either terminal or a depth-cutoff where a value net
// estimates the remaining turns. The candidate with the lowest expected total
// guesses is played.
//
// Gated by active-board count: when more than `max_active` boards remain we
// defer to plain greedy (search matters little there and greedy is the
// dominant cost). The tail risk we care about lives in small-state endgames.
class MctsStrategy : public Strategy {
public:
    MctsStrategy(const Wordlists& w, double alpha,
                 int top_k = 10, int rollouts = 20,
                 int cutoff_depth = 2, int max_active = 16)
        : w_(w), greedy_(w, alpha),
          top_k_(top_k), rollouts_(rollouts),
          cutoff_depth_(cutoff_depth), max_active_(max_active),
          opener_(INVALID_WORD), value_net_(nullptr) {}

    WordIdx choose_guess(const GameState& state) override;
    const char* name() const override { return "mcts"; }

    void set_opener(WordIdx g) { opener_ = g; }
    void set_value_net(const ValueNet* vn) { value_net_ = vn; }
    // Risk-adjusted objective: minimize  E[turns] + risk_lambda * tail_penalty,
    // where tail_penalty counts rollouts whose total exceeds 34. 0 = pure mean.
    void set_risk_lambda(double l) { risk_lambda_ = l; }

private:
    const Wordlists& w_;
    GreedyStrategy greedy_;
    int top_k_;
    int rollouts_;
    int cutoff_depth_;
    int max_active_;
    WordIdx opener_;
    const ValueNet* value_net_;
    double risk_lambda_ = 0.0;
};

}
