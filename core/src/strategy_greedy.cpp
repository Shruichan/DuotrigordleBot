#include "strategy.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
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
    if (static_cast<size_t>(state.guesses_used) < forced_prefix_.size()) {
        return forced_prefix_[state.guesses_used];
    }
    if (state.guesses_used == 0 && opener_cache_ != INVALID_WORD) {
        return opener_cache_;
    }
    // Apply turn-dependent alpha (if a schedule is set) for this call.
    const double orig_alpha = answer_bonus_;
    if (!alpha_schedule_.empty()) {
        size_t idx = std::min<size_t>(alpha_schedule_.size() - 1,
                                       static_cast<size_t>(state.guesses_used));
        const_cast<GreedyStrategy*>(this)->answer_bonus_ = alpha_schedule_[idx];
    }
    struct AlphaRestore {
        double* p; double v;
        ~AlphaRestore() { *p = v; }
    } _restore{ &(const_cast<GreedyStrategy*>(this)->answer_bonus_), orig_alpha };

    auto gs = build_setup(w_, state);
    if (gs.active.empty()) return 0;

    // Budget-aware "panic mode" for the 34-guess tail. The 35+ tail comes from
    // committing on multiple |C|>=2 boards in a tight endgame and missing the
    // 50/50s. One info word can disambiguate several |C|>=2 boards at once so
    // they all solve next turn (2 turns guaranteed, no gamble). Greedy at high
    // alpha is commit-biased and won't play it — so when slack toward a
    // 34-finish is tight AND >=2 boards still have ambiguity, drop alpha for
    // this turn to favor that disambiguating word, and skip the forced-commit
    // shortcut. slack = 34 - guesses_used - active_boards (each board needs >=1
    // more guess; negative/small slack = a single miss risks exceeding 34).
    bool panic = false;
    if (panic_slack_ >= 0) {
        int n_active = static_cast<int>(gs.active.size());
        int n_ambig = 0;
        for (int b : gs.active) if (state.boards[b].candidates.size() >= 2) ++n_ambig;
        int slack = 34 - state.guesses_used - n_active;
        if (slack <= panic_slack_ && n_ambig >= 2) {
            const_cast<GreedyStrategy*>(this)->answer_bonus_ = panic_alpha_;
            panic = true;
        }
    }

    if (!gs.forced.empty() && !panic) {
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

    // 2-step lookahead: among the top-K greedy candidates, simulate forward 1
    // step with paired-sampled answers and pick the one whose post-state minimizes
    // total candidate count. This rewards setups that cascade well even when the
    // raw 1-step score is slightly lower.
    if (lookahead_k_ > 0 && lookahead_n_ > 0 && gs.active.size() >= 3) {
        // Take top-K by score.
        const int K = std::min<int>(lookahead_k_, static_cast<int>(scores.size()));
        std::vector<WordIdx> top_k(scores.size());
        std::iota(top_k.begin(), top_k.end(), WordIdx{0});
        std::partial_sort(top_k.begin(), top_k.begin() + K, top_k.end(),
                          [&](WordIdx a, WordIdx b) {
                              if (scores[a] != scores[b]) return scores[a] > scores[b];
                              return a < b;
                          });
        top_k.resize(K);

        // Exact expected-feature value ranking (no sampling noise). For each
        // candidate g, compute the deterministic EXPECTED post-guess candidate
        // count per board:  E[|C'_b|] = (sum_p |part_p|^2) / |C_b|. Build the
        // value-net feature vector from those expected counts and evaluate V
        // once. Pick the candidate minimizing 1 + E[V(s')]. Removes the
        // Monte-Carlo variance that made the sampled lookahead pick near-randomly
        // among near-tied top-K.
        if (lookahead_exact_ && value_net_ != nullptr && value_net_->loaded()
            && static_cast<int>(gs.active.size()) <= lookahead_exact_max_active_) {
            const int post_guesses_used = state.guesses_used + 1;
            double best_v = std::numeric_limits<double>::infinity();
            WordIdx best_e = best_g;
            #pragma omp parallel
            {
                double local_best = std::numeric_limits<double>::infinity();
                WordIdx local_arg = best_g;
                std::array<int, NUM_BOARDS> post_cnts{};
                std::array<float, ValueNet::FEATURE_DIM> feats{};
                #pragma omp for nowait schedule(dynamic, 1)
                for (int ki = 0; ki < K; ++ki) {
                    WordIdx g = top_k[ki];
                    const Pattern* row = w_.feedback_row(g);
                    for (int b = 0; b < NUM_BOARDS; ++b) {
                        const auto& cs = state.boards[b].candidates;
                        if (state.boards[b].solved || cs.empty()) { post_cnts[b] = 0; continue; }
                        std::array<int, NUM_PATTERNS> part{};
                        for (WordIdx s : cs) part[row[s]] += 1;
                        long sumsq = 0;
                        for (int c : part) sumsq += static_cast<long>(c) * c;
                        double e_cnt = static_cast<double>(sumsq) / static_cast<double>(cs.size());
                        // All-green partition (size 1 once solved) contributes; if the
                        // board is certainly solved (|C|==1) expected count -> 0.
                        int rc = static_cast<int>(e_cnt + 0.5);
                        post_cnts[b] = (rc <= 1) ? (cs.size() == 1 ? 0 : 1) : rc;
                    }
                    ValueNet::compute_features_post(post_guesses_used, post_cnts.data(), feats.data());
                    double v = static_cast<double>(value_net_->eval(feats.data()));
                    if (v < local_best) { local_best = v; local_arg = g; }
                }
                #pragma omp critical
                { if (local_best < best_v) { best_v = local_best; best_e = local_arg; } }
            }
            return best_e;
        }

        // Sample N answer tuples once (paired across candidates).
        std::mt19937_64 rng(0xD06EE05ULL ^ static_cast<uint64_t>(state.guesses_used));
        std::vector<std::array<WordIdx, NUM_BOARDS>> samples(lookahead_n_);
        for (auto& s : samples) {
            for (int b = 0; b < NUM_BOARDS; ++b) {
                const auto& cands = state.boards[b].candidates;
                if (state.boards[b].solved || cands.empty()) { s[b] = 0; continue; }
                std::uniform_int_distribution<size_t> dist(0, cands.size() - 1);
                s[b] = cands[dist(rng)];
            }
        }

        // Leaf eval. Two modes:
        //  - ValueNet attached: featurize the post-state and call V(s').
        //    Trained from greedy self-play (label = remaining_turns), so much
        //    more accurate than any hand-tuned formula at distinguishing
        //    near-tied top-K candidates.
        //  - Fallback (no net): per-board E[turns | |C|] table:
        //    k=1: 1.0  k=2: 1.50  k=3: 1.83  k>=4: 1.5 + 0.4*log2(k) (cap 3.5)
        auto turns_for_k = [](int k) -> double {
            if (k <= 1) return 1.0;
            if (k == 2) return 1.50;
            if (k == 3) return 1.83;
            double v = 1.5 + 0.4 * std::log2(static_cast<double>(k));
            return v > 3.5 ? 3.5 : v;
        };
        const bool use_net = (value_net_ != nullptr && value_net_->loaded());
        const int post_guesses_used = state.guesses_used + 1;
        double best_total = std::numeric_limits<double>::infinity();
        WordIdx best_la = best_g;
        #pragma omp parallel
        {
            double local_best = std::numeric_limits<double>::infinity();
            WordIdx local_arg = best_g;
            std::array<int, NUM_BOARDS> post_cnts{};
            std::array<float, ValueNet::FEATURE_DIM> feats{};
            #pragma omp for nowait schedule(dynamic, 1)
            for (int ki = 0; ki < K; ++ki) {
                WordIdx g = top_k[ki];
                double sum_cost = 0.0;
                const Pattern* row = w_.feedback_row(g);
                for (const auto& sample : samples) {
                    if (use_net) {
                        // Build post-board counts then eval the net.
                        for (int b = 0; b < NUM_BOARDS; ++b) {
                            const auto& cands = state.boards[b].candidates;
                            if (state.boards[b].solved || cands.empty()) {
                                post_cnts[b] = 0;
                                continue;
                            }
                            Pattern p = w_.feedback(g, sample[b]);
                            if (p == PATTERN_ALL_GREEN) { post_cnts[b] = 0; continue; }
                            int cnt = 0;
                            for (WordIdx s : cands) if (row[s] == p) ++cnt;
                            post_cnts[b] = cnt;
                        }
                        ValueNet::compute_features_post(post_guesses_used,
                                                       post_cnts.data(),
                                                       feats.data());
                        sum_cost += static_cast<double>(value_net_->eval(feats.data()));
                    } else {
                        double cost = 0.0;
                        for (int b = 0; b < NUM_BOARDS; ++b) {
                            const auto& cands = state.boards[b].candidates;
                            if (state.boards[b].solved || cands.empty()) continue;
                            Pattern p = w_.feedback(g, sample[b]);
                            if (p == PATTERN_ALL_GREEN) continue;
                            int cnt = 0;
                            for (WordIdx s : cands) if (row[s] == p) ++cnt;
                            cost += turns_for_k(cnt);
                        }
                        sum_cost += cost;
                    }
                }
                double avg = sum_cost / lookahead_n_;
                if (avg < local_best) { local_best = avg; local_arg = g; }
            }
            #pragma omp critical
            {
                if (local_best < best_total) { best_total = local_best; best_la = local_arg; }
            }
        }
        best_g = best_la;
    }

    // "Rhyme trap" override: when only one or two active boards remain and at
    // least one has |C|>=3 with the candidates all nearly-rhymes (the partition
    // of the candidate-only score wouldn't distinguish them), a candidate guess
    // averages 2 turns but worst-cases (k) turns. An info word that splits the
    // candidate set into k partitions is exactly 2 turns guaranteed. Same mean,
    // bounded max — eliminates 36+ outcomes.
    int n_active = 0, max_c = 0, single_b = -1;
    for (int b = 0; b < NUM_BOARDS; ++b) {
        if (state.boards[b].solved || state.boards[b].candidates.empty()) continue;
        ++n_active;
        int c = static_cast<int>(state.boards[b].candidates.size());
        if (c > max_c) { max_c = c; single_b = b; }
    }
    if (n_active == 1 && max_c >= 3) {
        const auto& cands = state.boards[single_b].candidates;
        double best_h = -1.0;
        WordIdx best_info = best_g;
        const size_t G = w_.num_guesses();
        #pragma omp parallel
        {
            double local_h = -1.0;
            WordIdx local_arg = best_g;
            #pragma omp for nowait schedule(static)
            for (long long g = 0; g < static_cast<long long>(G); ++g) {
                const Pattern* row = w_.feedback_row(static_cast<WordIdx>(g));
                std::array<int, NUM_PATTERNS> count{};
                for (WordIdx s : cands) count[row[s]] += 1;
                const double n = static_cast<double>(cands.size());
                double H = 0.0;
                for (int c : count) {
                    if (c == 0) continue;
                    const double p = c / n;
                    H -= p * std::log2(p);
                }
                if (H > local_h) { local_h = H; local_arg = static_cast<WordIdx>(g); }
            }
            #pragma omp critical
            { if (local_h > best_h) { best_h = local_h; best_info = local_arg; } }
        }
        // Only override if the info word actually splits things further than the
        // candidate guess would (i.e., entropy >= log2(2) = 1 — partition into >=2).
        if (best_h > 0.99) return best_info;
    }

    if (state.guesses_used == 0) opener_cache_ = best_g;
    return best_g;
}

std::vector<WordIdx> GreedyStrategy::top_k_guesses(const GameState& state, int k,
                                                   const std::vector<WordIdx>& pool) const {
    auto gs = build_setup(w_, state);
    if (gs.active.empty()) return {};

    // "Rhyme trap" override: single active board with |C|>=3. Picking a
    // candidate averages 2 turns but worst-cases (k) turns. An info word that
    // splits the candidate set bounds the worst-case to 2. Match choose_guess.
    if (gs.active.size() == 1 && pool.empty()) {
        int b = gs.active[0];
        const auto& cands = state.boards[b].candidates;
        if (cands.size() >= 3) {
            const size_t G = w_.num_guesses();
            double best_h = -1.0;
            WordIdx best_info = static_cast<WordIdx>(0);
            #pragma omp parallel
            {
                double local_h = -1.0;
                WordIdx local_arg = static_cast<WordIdx>(0);
                #pragma omp for nowait schedule(static)
                for (long long g = 0; g < static_cast<long long>(G); ++g) {
                    const Pattern* row = w_.feedback_row(static_cast<WordIdx>(g));
                    std::array<int, NUM_PATTERNS> count{};
                    for (WordIdx s : cands) count[row[s]] += 1;
                    const double n = static_cast<double>(cands.size());
                    double H = 0.0;
                    for (int c : count) {
                        if (c == 0) continue;
                        const double p = c / n;
                        H -= p * std::log2(p);
                    }
                    if (H > local_h) { local_h = H; local_arg = static_cast<WordIdx>(g); }
                }
                #pragma omp critical
                { if (local_h > best_h) { best_h = local_h; best_info = local_arg; } }
            }
            if (best_h > 0.99) {
                // Prepend info word; fill rest from candidates by alphabetical order.
                std::vector<WordIdx> out{best_info};
                for (WordIdx s : cands) {
                    WordIdx cg = w_.sol_to_guess(s);
                    if (cg != best_info) out.push_back(cg);
                    if (static_cast<int>(out.size()) >= k) break;
                }
                return out;
            }
        }
    }

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
