#include "feedback.hpp"
#include "game_state.hpp"
#include "strategy.hpp"
#include "wordlists.hpp"

#include <nlohmann/json.hpp>

#include <iostream>
#include <optional>
#include <string>

using json = nlohmann::json;

namespace {

json error(const std::string& msg) {
    return json{{"error", msg}};
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
    double alpha = req.value("alpha", 1.0);
    std::string mode = req.value("mode", "auto");

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
            resp = process_request(w, req);
        } catch (const std::exception& e) {
            resp = error(std::string("exception: ") + e.what());
        }
        std::cout << resp.dump() << std::endl;
    }
    return 0;
}
