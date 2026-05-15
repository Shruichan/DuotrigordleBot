#include "feedback.hpp"
#include "game_state.hpp"
#include "strategy.hpp"
#include "wordlists.hpp"

#include <iostream>
#include <sstream>
#include <string>

int main() {
    dt::Wordlists w(DT_DATA_DIR, "default");
    dt::GreedyStrategy strat(w);
    dt::GameState state = dt::GameState::fresh(w);

    std::cout << "Duotrigordle solver CLI. Commands:\n";
    std::cout << "  suggest        -> print best guess\n";
    std::cout << "  play <WORD> <p1> <p2> ... <p32>   (p = 5-char G/Y/B)\n";
    std::cout << "  status         -> board summary\n";
    std::cout << "  quit\n";

    std::string line;
    while (std::cout << "> " << std::flush, std::getline(std::cin, line)) {
        std::istringstream iss(line);
        std::string cmd;
        iss >> cmd;
        if (cmd == "quit" || cmd == "exit") break;
        if (cmd == "suggest") {
            auto g = strat.choose_guess(state);
            std::cout << w.guess(g) << "  (active=" << state.active_boards()
                      << ", used=" << state.guesses_used << ")\n";
        } else if (cmd == "play") {
            std::string word;
            iss >> word;
            auto gi = w.guess_index(word);
            if (!gi) { std::cout << "unknown guess\n"; continue; }
            std::array<dt::Pattern, dt::NUM_BOARDS> pats{};
            bool ok = true;
            for (int i = 0; i < dt::NUM_BOARDS; ++i) {
                std::string p;
                if (!(iss >> p)) { std::cout << "need 32 patterns\n"; ok = false; break; }
                pats[i] = dt::parse_pattern(p);
            }
            if (!ok) continue;
            state.apply_guess(w, *gi, pats);
            std::cout << "applied. active=" << state.active_boards()
                      << " used=" << state.guesses_used << "\n";
        } else if (cmd == "status") {
            for (int i = 0; i < dt::NUM_BOARDS; ++i) {
                std::cout << "  board " << i << ": ";
                if (state.boards[i].solved) std::cout << "SOLVED";
                else std::cout << state.boards[i].candidates.size() << " candidates";
                std::cout << "\n";
            }
        } else if (!cmd.empty()) {
            std::cout << "unknown command\n";
        }
    }
    return 0;
}
