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

        // For each candidate, evaluate average post-state total candidate count.
        // Lower = better.
        double best_total = std::numeric_limits<double>::infinity();
        WordIdx best_la = best_g;
        #pragma omp parallel
        {
            double local_best = std::numeric_limits<double>::infinity();
            WordIdx local_arg = best_g;
            #pragma omp for nowait schedule(dynamic, 1)
            for (int ki = 0; ki < K; ++ki) {
                WordIdx g = top_k[ki];
                double sum_cost = 0.0;
                const Pattern* row = w_.feedback_row(g);
                for (const auto& sample : samples) {
                    // Cost = remaining_candidates - SOLVES_WEIGHT * boards_solved.
                    // Solving a board saves ~1 future guess. Weight tuned by hand;
                    // larger weight = more solve-biased (matches the user's goal of
                    // "guarantee solves from turn 2"). 30 picked because the typical
                    // alternative info-only word can narrow ~30 candidates total.
                    constexpr double SOLVES_WEIGHT = 30.0;
                    int total = 0;
                    int solves = 0;
                    for (int b = 0; b < NUM_BOARDS; ++b) {
                        const auto& cands = state.boards[b].candidates;
                        if (state.boards[b].solved || cands.empty()) continue;
                        Pattern p = w_.feedback(g, sample[b]);
                        if (p == PATTERN_ALL_GREEN) { ++solves; continue; }
                        int cnt = 0;
                        for (WordIdx s : cands) if (row[s] == p) ++cnt;
                        total += cnt;
                    }
                    sum_cost += total - SOLVES_WEIGHT * solves;
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
