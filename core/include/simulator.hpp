#pragma once

#include "game_state.hpp"
#include "strategy.hpp"
#include "wordlists.hpp"

#include <array>
#include <cstdint>
#include <vector>

namespace dt {

struct GameResult {
    bool all_solved;
    int guesses_used;
    int boards_solved;
    std::array<WordIdx, NUM_BOARDS> answers;
    std::vector<WordIdx> guess_history;
};

struct BenchStats {
    int games;
    int solved;
    double mean_guesses;
    int min_guesses;
    int max_guesses;
    double pct_under_37;
    double pct_under_32;
    std::vector<int> guess_distribution;
};

GameResult run_one_game(const Wordlists& w, Strategy& strat,
                        const std::array<WordIdx, NUM_BOARDS>& answers,
                        int max_guesses = 50);

BenchStats run_benchmark(const Wordlists& w, Strategy& strat,
                         int num_games, uint64_t seed = 42,
                         int max_guesses = 50,
                         bool verbose = true);

}
