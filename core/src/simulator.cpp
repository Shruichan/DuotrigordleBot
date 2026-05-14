#include "simulator.hpp"

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <numeric>

namespace dt {

GameResult run_one_game(const Wordlists& w, Strategy& strat,
                        const std::array<WordIdx, NUM_BOARDS>& answers,
                        int max_guesses,
                        bool use_distinct_constraint) {
    GameState state = GameState::fresh(w);
    GameResult r{};
    r.answers = answers;

    while (state.guesses_used < max_guesses && !state.game_over()) {
        WordIdx g = strat.choose_guess(state);
        std::array<Pattern, NUM_BOARDS> patterns{};
        for (int i = 0; i < NUM_BOARDS; ++i) {
            patterns[i] = state.boards[i].solved
                ? PATTERN_ALL_GREEN
                : w.feedback(g, answers[i]);
        }
        state.apply_guess(w, g, patterns, use_distinct_constraint);
    }

    r.all_solved = state.game_over();
    r.guesses_used = state.guesses_used;
    r.boards_solved = NUM_BOARDS - state.active_boards();
    r.guess_history = std::move(state.guess_history);
    return r;
}

BenchStats run_benchmark(const Wordlists& w, Strategy& strat,
                         int num_games, uint64_t seed, int max_guesses,
                         bool verbose,
                         bool distinct_answers,
                         bool use_distinct_constraint) {
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<WordIdx> pick(0, static_cast<WordIdx>(w.num_solutions() - 1));
    std::vector<uint8_t> picked_buf(w.num_solutions(), 0);

    BenchStats stats{};
    stats.games = num_games;
    stats.min_guesses = INT32_MAX;
    stats.max_guesses = 0;
    stats.guess_distribution.assign(max_guesses + 2, 0);

    int solved = 0;
    long long total_guesses = 0;
    int under_37 = 0;
    int under_32 = 0;

    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < num_games; ++i) {
        std::array<WordIdx, NUM_BOARDS> answers{};
        if (distinct_answers) {
            std::fill(picked_buf.begin(), picked_buf.end(), 0);
            for (auto& a : answers) {
                WordIdx x;
                do { x = pick(rng); } while (picked_buf[x]);
                picked_buf[x] = 1;
                a = x;
            }
        } else {
            for (auto& a : answers) a = pick(rng);
        }
        GameResult r = run_one_game(w, strat, answers, max_guesses, use_distinct_constraint);

        if (r.all_solved) {
            ++solved;
            total_guesses += r.guesses_used;
            stats.min_guesses = std::min(stats.min_guesses, r.guesses_used);
            stats.max_guesses = std::max(stats.max_guesses, r.guesses_used);
            stats.guess_distribution[r.guesses_used] += 1;
            if (r.guesses_used <= 37) ++under_37;
            if (r.guesses_used <= 32) ++under_32;
        } else {
            stats.guess_distribution.back() += 1;
        }

        if (verbose && (i + 1) % 50 == 0) {
            auto t = std::chrono::steady_clock::now();
            double s = std::chrono::duration<double>(t - t0).count();
            std::cerr << "  [" << (i + 1) << "/" << num_games << "] "
                      << "solved=" << solved
                      << " mean=" << std::fixed << std::setprecision(2)
                      << (solved ? static_cast<double>(total_guesses) / solved : 0.0)
                      << " elapsed=" << std::setprecision(1) << s << "s\n";
        }
    }

    stats.solved = solved;
    stats.mean_guesses = solved ? static_cast<double>(total_guesses) / solved : 0.0;
    stats.pct_under_37 = 100.0 * under_37 / num_games;
    stats.pct_under_32 = 100.0 * under_32 / num_games;
    return stats;
}

}
