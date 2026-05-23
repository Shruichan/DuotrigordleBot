#include "feedback.hpp"
#include "game_state.hpp"
#include "strategy.hpp"
#include "wordlists.hpp"

#include <nlohmann/json.hpp>

#include <array>
#include <cmath>
#include <iostream>
#include <optional>
#include <string>

using json = nlohmann::json;

namespace {

json error(const std::string& msg) {
    return json{{"error", msg}};
}

// Per-board metrics for one guess applied to one board state.
struct BoardMetrics {
    double entropy_bits;       // -sum p_i log2 p_i over partition by this guess
    double expected_remaining; // sum_i |partition_i|^2 / |C| — expected size of bucket we'll land in
    int actual_remaining;      // size of the partition we actually observed
};

BoardMetrics board_metrics(const dt::Wordlists& w, const std::vector<dt::WordIdx>& cands,
                           dt::WordIdx g, dt::Pattern observed_pattern) {
    BoardMetrics m{0.0, 0.0, 0};
    if (cands.empty()) return m;
    const dt::Pattern* row = w.feedback_row(g);
    std::array<int, dt::NUM_PATTERNS> count{};
    for (dt::WordIdx s : cands) count[row[s]] += 1;
    const double n = static_cast<double>(cands.size());
    for (int p = 0; p < dt::NUM_PATTERNS; ++p) {
        if (count[p] == 0) continue;
        const double pi = count[p] / n;
        m.entropy_bits -= pi * std::log2(pi);
        m.expected_remaining += pi * count[p];  // sum_p P(p) * |p| = sum_p |p|^2 / n
    }
    m.actual_remaining = count[observed_pattern];
    return m;
}

// Total greedy score (entropy + answer-bonus) for one guess on a state.
double score_guess_total(const dt::Wordlists& w, const dt::GameState& state,
                          dt::WordIdx g, double alpha,
                          const std::vector<float>& expected_solves) {
    double s = 0.0;
    for (int b = 0; b < dt::NUM_BOARDS; ++b) {
        const auto& bd = state.boards[b];
        if (bd.solved || bd.candidates.empty()) continue;
        const dt::Pattern* row = w.feedback_row(g);
        std::array<int, dt::NUM_PATTERNS> count{};
        for (dt::WordIdx c : bd.candidates) count[row[c]] += 1;
        const double n = static_cast<double>(bd.candidates.size());
        for (int c : count) {
            if (c == 0) continue;
            const double p = c / n;
            s -= p * std::log2(p);
        }
    }
    int32_t sol = w.guess_to_sol(g);
    if (sol >= 0) s += alpha * expected_solves[sol];
    return s;
}

json process_request(const dt::Wordlists& w, const json& req) {
    if (!req.contains("boards") || !req["boards"].is_array()) {
        return error("missing or invalid 'boards' array");
    }
    const auto& boards = req["boards"];
    if (boards.size() != dt::NUM_BOARDS) {
        return error("'boards' must have exactly 32 entries");
    }

    int top_k = req.value("top_k", 5);
    // Default alpha tuned via 2000-game opener×alpha sweep. α=300 with the
    // hardcoded LITRE opener (below) gives mean 33.61, 42% of games at 33,
    // and only 4.25% at 35+ — best tradeoff for "more 33s, almost no 35s".
    double alpha = req.value("alpha", 300.0);
    std::string mode = req.value("mode", "auto");
    // Perfect mode: maximize P(solve at least one board each turn). Push alpha
    // very high so expected_solves dominates raw entropy. Trades a bit of tail
    // (more 35s) for more 33s and the occasional 32 (Perfect Challenge).
    if (mode == "perfect") {
        alpha = req.value("perfect_alpha", 1000.0);
    }

    dt::GameState state = dt::GameState::fresh(w);

    int max_guesses_seen = 0;
    for (size_t b = 0; b < boards.size(); ++b) {
        const auto& bj = boards[b];
        if (!bj.contains("guesses") || !bj.contains("feedback")) {
            return error("board " + std::to_string(b) + " missing guesses/feedback");
        }
        auto guesses = bj["guesses"].get<std::vector<std::string>>();
        auto feedbacks = bj["feedback"].get<std::vector<std::string>>();
        if (guesses.size() != feedbacks.size()) {
            return error("board " + std::to_string(b) + ": guesses/feedback size mismatch");
        }
        if (static_cast<int>(guesses.size()) > max_guesses_seen) {
            max_guesses_seen = static_cast<int>(guesses.size());
        }
    }

    // Replay history globally. The SAME guess was applied to all boards on a turn;
    // each board has its own feedback. A board with fewer recorded rows than max is
    // EITHER already solved (its row history truncates after the all-green) OR the
    // scraper missed a row. We must distinguish these: for already-solved, the board
    // stays solved (the apply on the still-active boards is enough); for a miss, we
    // must NOT touch that board this turn (must not auto-mark it as solved).
    for (int turn = 0; turn < max_guesses_seen; ++turn) {
        std::optional<std::string> turn_guess;
        std::array<dt::Pattern, dt::NUM_BOARDS> pats{};
        std::array<bool, dt::NUM_BOARDS> board_has_row{};

        for (size_t b = 0; b < boards.size(); ++b) {
            const auto& bj = boards[b];
            const auto& gs = bj["guesses"];
            const auto& fb = bj["feedback"];
            if (turn < static_cast<int>(gs.size())) {
                std::string gword = gs[turn].get<std::string>();
                std::string fword = fb[turn].get<std::string>();
                if (!turn_guess) turn_guess = gword;
                else if (*turn_guess != gword) {
                    return error("turn " + std::to_string(turn) +
                                 ": guess words differ across boards (must match)");
                }
                pats[b] = dt::parse_pattern(fword);
                board_has_row[b] = true;
            }
        }

        if (!turn_guess) continue;
        auto gi = w.guess_index(*turn_guess);
        if (!gi) return error("turn " + std::to_string(turn) + ": guess '" + *turn_guess + "' not in dictionary");

        // Custom apply: only touch boards that have row data this turn.
        // Already-solved boards stay solved (their apply would no-op anyway).
        // Boards without row data and not yet solved are LEFT ALONE (scraper miss).
        int32_t g_sol = w.guess_to_sol(*gi);
        for (int b = 0; b < dt::NUM_BOARDS; ++b) {
            if (!board_has_row[b]) continue;
            if (!state.boards[b].solved && pats[b] == dt::PATTERN_ALL_GREEN && g_sol >= 0) {
                state.answer_used[g_sol] = 1;
            }
        }
        for (int b = 0; b < dt::NUM_BOARDS; ++b) {
            if (board_has_row[b]) state.boards[b].apply(w, *gi, pats[b]);
        }
        for (int b = 0; b < dt::NUM_BOARDS; ++b) {
            if (state.boards[b].solved) continue;
            auto& cs = state.boards[b].candidates;
            auto it = cs.begin();
            for (auto s : cs) if (!state.answer_used[s]) *it++ = s;
            cs.erase(it, cs.end());
        }
        state.guess_history.push_back(*gi);
        ++state.guesses_used;
    }

    dt::GreedyStrategy strat(w, alpha);

    std::vector<int> active_idx;
    for (int b = 0; b < dt::NUM_BOARDS; ++b) {
        if (!state.boards[b].solved && !state.boards[b].candidates.empty()) {
            active_idx.push_back(b);
        }
    }

    std::vector<dt::WordIdx> pool;
    if (mode == "perfect" && !active_idx.empty()) {
        std::vector<uint8_t> active_sol(w.num_solutions(), 0);
        for (int b : active_idx) {
            for (auto s : state.boards[b].candidates) active_sol[s] = 1;
        }
        for (size_t s = 0; s < w.num_solutions(); ++s) {
            if (active_sol[s]) pool.push_back(w.sol_to_guess(static_cast<dt::WordIdx>(s)));
        }
    }
    auto top = active_idx.empty()
        ? std::vector<dt::WordIdx>{}
        : strat.top_k_guesses(state, top_k, pool);

    // Forced opener at fresh state: LITRE wins the 2000-game opener×alpha sweep.
    // Mean 33.61, 42.3% at 33, 4.25% at 35+ — the best single-config combo for
    // "majority 33s, near-zero 35s" while still scoring an occasional Perfect.
    if (state.guesses_used == 0 && !top.empty()) {
        auto litre = w.guess_index("LITRE");
        if (litre) {
            auto it = std::find(top.begin(), top.end(), *litre);
            if (it != top.end()) top.erase(it);
            top.insert(top.begin(), *litre);
            if (static_cast<int>(top.size()) > top_k) top.resize(top_k);
        }
    }

    json suggestions = json::array();
    for (dt::WordIdx g : top) {
        json sug;
        sug["word"] = std::string(w.guess(g));
        std::vector<int> could_solve;
        int32_t sol = w.guess_to_sol(g);
        if (sol >= 0) {
            for (int b : active_idx) {
                for (dt::WordIdx s : state.boards[b].candidates) {
                    if (s == static_cast<dt::WordIdx>(sol)) { could_solve.push_back(b); break; }
                }
            }
        }
        sug["could_solve"] = could_solve;
        suggestions.push_back(sug);
    }

    json boards_out = json::array();
    for (int b = 0; b < dt::NUM_BOARDS; ++b) {
        json bj;
        bj["solved"] = state.boards[b].solved;
        bj["candidates_remaining"] = static_cast<int>(state.boards[b].candidates.size());
        if (state.boards[b].candidates.size() <= 10) {
            std::vector<std::string> cs;
            for (dt::WordIdx s : state.boards[b].candidates) cs.push_back(std::string(w.solution(s)));
            bj["candidates"] = cs;
        }
        boards_out.push_back(bj);
    }

    return json{
        {"suggestions", suggestions},
        {"active_boards", state.active_boards()},
        {"guesses_used", state.guesses_used},
        {"game_over", state.active_boards() == 0},
        {"mode", mode},
        {"boards", boards_out}
    };
}

}

json process_review(const dt::Wordlists& w, const json& req) {
    if (!req.contains("boards") || !req["boards"].is_array()) {
        return error("missing or invalid 'boards'");
    }
    const auto& boards = req["boards"];
    if (boards.size() != dt::NUM_BOARDS) {
        return error("'boards' must have exactly 32 entries");
    }
    const double alpha = req.value("alpha", 300.0);

    int max_turns = 0;
    for (const auto& bj : boards) {
        if (!bj.contains("guesses") || !bj.contains("feedback")) {
            return error("missing guesses/feedback on a board");
        }
        max_turns = std::max<int>(max_turns, bj["guesses"].size());
    }

    dt::GameState state = dt::GameState::fresh(w);
    json turns = json::array();
    double skill_sum = 0.0, luck_sum = 0.0;
    int scored_turns = 0;

    for (int t = 0; t < max_turns; ++t) {
        // Determine the shared guess for this turn (from the first board with a row).
        std::optional<std::string> turn_guess;
        std::array<dt::Pattern, dt::NUM_BOARDS> pats{};
        std::array<bool, dt::NUM_BOARDS> board_has_row{};
        for (size_t b = 0; b < boards.size(); ++b) {
            const auto& gs_arr = boards[b]["guesses"];
            const auto& fb_arr = boards[b]["feedback"];
            if (t < static_cast<int>(gs_arr.size())) {
                std::string gword = gs_arr[t].get<std::string>();
                std::string fword = fb_arr[t].get<std::string>();
                if (!turn_guess) turn_guess = gword;
                pats[b] = dt::parse_pattern(fword);
                board_has_row[b] = true;
            }
        }
        if (!turn_guess) continue;
        auto gi_opt = w.guess_index(*turn_guess);
        if (!gi_opt) return error("turn " + std::to_string(t + 1) + ": guess '" + *turn_guess + "' not in dictionary");
        dt::WordIdx g = *gi_opt;

        // Compute expected_solves for this state (for the alpha bonus term).
        std::vector<float> expected_solves(w.num_solutions(), 0.0f);
        for (int b = 0; b < dt::NUM_BOARDS; ++b) {
            const auto& bd = state.boards[b];
            if (bd.solved || bd.candidates.empty()) continue;
            const float inv = 1.0f / static_cast<float>(bd.candidates.size());
            for (dt::WordIdx s : bd.candidates) expected_solves[s] += inv;
        }

        // Skill (1-99, NYT-style): normalize the player's move score against the
        // full range of all possible moves' scores. 99 = picked the best, 1 = picked
        // the worst, 50 = picked a median-quality move.
        double played_score = score_guess_total(w, state, g, alpha, expected_solves);
        double best_score = played_score;
        double worst_score = played_score;
        dt::WordIdx best_g = g;
        if (state.active_boards() > 0) {
            const size_t G = w.num_guesses();
            std::vector<double> all_scores(G);
            #pragma omp parallel for schedule(static)
            for (long long gg = 0; gg < static_cast<long long>(G); ++gg) {
                all_scores[gg] = score_guess_total(w, state, static_cast<dt::WordIdx>(gg),
                                                    alpha, expected_solves);
            }
            for (size_t gg = 0; gg < G; ++gg) {
                if (all_scores[gg] > best_score) {
                    best_score = all_scores[gg];
                    best_g = static_cast<dt::WordIdx>(gg);
                }
                if (all_scores[gg] < worst_score) worst_score = all_scores[gg];
            }
        }
        double skill = 99.0;
        if (best_score > worst_score + 1e-9) {
            skill = 1.0 + 98.0 * (played_score - worst_score) / (best_score - worst_score);
        }
        if (skill < 1.0) skill = 1.0;
        if (skill > 99.0) skill = 99.0;

        // Bot's actual decision (forced-aware): top_k_guesses with forced moves
        // promoted to the top. Distinct from best_g (entropy-max).
        dt::WordIdx bot_g = g;
        bool decision_matched = true;
        if (state.active_boards() > 0) {
            dt::GreedyStrategy strat(w, alpha);
            auto top = strat.top_k_guesses(state, 1);
            if (!top.empty()) {
                bot_g = top[0];
                decision_matched = (bot_g == g);
            }
        }

        // Count of boards where the played word was a candidate (could have solved).
        int played_could_solve = 0;
        int32_t played_sol = w.guess_to_sol(g);
        if (played_sol >= 0) {
            for (int b = 0; b < dt::NUM_BOARDS; ++b) {
                const auto& bd = state.boards[b];
                if (bd.solved) continue;
                for (dt::WordIdx s : bd.candidates) {
                    if (s == static_cast<dt::WordIdx>(played_sol)) { ++played_could_solve; break; }
                }
            }
        }

        // Luck (1-99, NYT-style): for each active board, compute the percentile of
        // the actual partition size against the probability-weighted distribution
        // of possible partition sizes. Smaller partition = more info = luckier.
        // Aggregate by averaging the per-board luck across active boards.
        double exp_total = 0.0, act_total = 0.0;
        double board_entropy_sum = 0.0;
        double luck_per_board_sum = 0.0;
        int luck_boards = 0;
        for (int b = 0; b < dt::NUM_BOARDS; ++b) {
            const auto& bd = state.boards[b];
            if (bd.solved || bd.candidates.empty() || !board_has_row[b]) continue;
            auto bm = board_metrics(w, bd.candidates, g, pats[b]);
            exp_total += bm.expected_remaining;
            act_total += bm.actual_remaining;
            board_entropy_sum += bm.entropy_bits;

            // Probability-weighted distribution of partition sizes for guess g on this board.
            const dt::Pattern* row = w.feedback_row(g);
            std::array<int, dt::NUM_PATTERNS> counts{};
            for (dt::WordIdx s : bd.candidates) counts[row[s]] += 1;
            const double n = static_cast<double>(bd.candidates.size());
            const int actual_size = bm.actual_remaining;
            // P(random outcome leaves a STRICTLY LARGER partition than ours).
            // Higher = lucky (most other outcomes would have been worse).
            double prob_worse = 0.0;
            double prob_equal = 0.0;
            for (int c : counts) {
                if (c == 0) continue;
                if (c > actual_size) prob_worse += c / n;
                else if (c == actual_size) prob_equal += c / n;
            }
            // Tied outcomes split evenly (standard percentile-with-ties).
            double pct = prob_worse + prob_equal / 2.0;
            double board_luck = 1.0 + 98.0 * pct;
            if (board_luck < 1.0) board_luck = 1.0;
            if (board_luck > 99.0) board_luck = 99.0;
            luck_per_board_sum += board_luck;
            ++luck_boards;
        }
        double luck = (luck_boards > 0) ? luck_per_board_sum / luck_boards : 50.0;

        // Apply the guess (with scraper-miss-safe semantics).
        int32_t g_sol = w.guess_to_sol(g);
        for (int b = 0; b < dt::NUM_BOARDS; ++b) {
            if (!board_has_row[b]) continue;
            if (!state.boards[b].solved && pats[b] == dt::PATTERN_ALL_GREEN && g_sol >= 0) {
                state.answer_used[g_sol] = 1;
            }
        }
        int solved_before = dt::NUM_BOARDS - state.active_boards();
        for (int b = 0; b < dt::NUM_BOARDS; ++b) {
            if (board_has_row[b]) state.boards[b].apply(w, g, pats[b]);
        }
        for (int b = 0; b < dt::NUM_BOARDS; ++b) {
            if (state.boards[b].solved) continue;
            auto& cs = state.boards[b].candidates;
            auto it = cs.begin();
            for (auto s : cs) if (!state.answer_used[s]) *it++ = s;
            cs.erase(it, cs.end());
        }
        state.guesses_used += 1;
        int solved_after = dt::NUM_BOARDS - state.active_boards();

        json turn_obj;
        turn_obj["turn"] = t + 1;
        turn_obj["guess"] = *turn_guess;
        turn_obj["best_info_word"] = std::string(w.guess(best_g));
        turn_obj["bot_choice"] = std::string(w.guess(bot_g));
        turn_obj["decision_matched"] = decision_matched;
        turn_obj["played_could_solve"] = played_could_solve;
        turn_obj["skill"] = skill;
        turn_obj["luck"] = luck;
        turn_obj["played_score"] = played_score;
        turn_obj["best_score"] = best_score;
        turn_obj["total_entropy_bits"] = board_entropy_sum;
        turn_obj["expected_remaining"] = exp_total;
        turn_obj["actual_remaining"] = act_total;
        turn_obj["boards_solved_this_turn"] = solved_after - solved_before;
        turn_obj["active_after"] = state.active_boards();
        turns.push_back(turn_obj);

        skill_sum += skill;
        luck_sum += luck;
        ++scored_turns;
    }

    int decisions_matched = 0;
    for (const auto& t : turns) {
        if (t.value("decision_matched", false)) ++decisions_matched;
    }
    json summary;
    summary["total_guesses"] = max_turns;
    summary["all_solved"] = state.game_over();
    summary["boards_solved"] = dt::NUM_BOARDS - state.active_boards();
    summary["avg_skill"] = scored_turns ? skill_sum / scored_turns : 0.0;
    summary["avg_luck"] = scored_turns ? luck_sum / scored_turns : 0.0;
    summary["decisions_matched"] = decisions_matched;
    summary["decisions_total"] = scored_turns;
    return json{{"turns", turns}, {"summary", summary}};
}

int main() {
    dt::Wordlists w(DT_DATA_DIR, "default");
    std::cerr << "dt_worker ready\n";
    std::cout << json{{"ready", true}}.dump() << std::endl;

    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;
        json resp;
        try {
            json req = json::parse(line);
            std::string cmd = req.value("command", "suggest");
            if (cmd == "review") {
                resp = process_review(w, req);
            } else {
                resp = process_request(w, req);
            }
        } catch (const std::exception& e) {
            resp = error(std::string("exception: ") + e.what());
        }
        std::cout << resp.dump() << std::endl;
    }
    return 0;
}
