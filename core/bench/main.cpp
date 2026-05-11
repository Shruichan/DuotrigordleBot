#include "simulator.hpp"
#include "strategy.hpp"
#include "wordlists.hpp"

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
    int num_games = 200;
    uint64_t seed = 42;
    std::string pool = "default";
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "-n" && i + 1 < argc) num_games = std::atoi(argv[++i]);
        else if (a == "-s" && i + 1 < argc) seed = std::strtoull(argv[++i], nullptr, 10);
        else if (a == "-p" && i + 1 < argc) pool = argv[++i];
        else { std::cerr << "usage: dt_bench [-n games] [-s seed] [-p pool]\n"; return 1; }
    }
    dt::Wordlists w(DT_DATA_DIR, pool);
    dt::GreedyStrategy strat(w);
    std::cerr << "Running " << num_games << " games...\n";
    auto stats = dt::run_benchmark(w, strat, num_games, seed);
    std::cout << "\nGames: " << stats.games << "  Solved: " << stats.solved << "\n";
    if (stats.solved) {
        std::cout << "Mean: " << std::setprecision(2) << stats.mean_guesses
                  << "  Min/Max: " << stats.min_guesses << "/" << stats.max_guesses << "\n";
    }
    return 0;
}
