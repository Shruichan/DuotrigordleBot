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

    // Replay history. If a board lacks a row for some turn, treat it as already-solved
    // (default pattern = all green).
    for (int turn = 0; turn < max_guesses_seen; ++turn) {
        std::optional<std::string> turn_guess;
        std::array<dt::Pattern, dt::NUM_BOARDS> pats{};
        for (auto& p : pats) p = dt::PATTERN_ALL_GREEN;

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
            }
        }

        if (!turn_guess) continue;
        auto gi = w.guess_index(*turn_guess);
        if (!gi) return error("turn " + std::to_string(turn) + ": guess '" + *turn_guess + "' not in dictionary");
        state.apply_guess(w, *gi, pats);
    }

    dt::GreedyStrategy strat(w, alpha);
    auto top = strat.top_k_guesses(state, top_k);

    std::vector<int> active_idx;
    for (int b = 0; b < dt::NUM_BOARDS; ++b) {
        if (!state.boards[b].solved && !state.boards[b].candidates.empty()) {
            active_idx.push_back(b);
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

    return json{
        {"suggestions", suggestions},
        {"active_boards", state.active_boards()},
        {"guesses_used", state.guesses_used},
        {"boards", json::array()}
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
