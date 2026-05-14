#pragma once

#include "feedback.hpp"
#include "wordlists.hpp"

#include <array>
#include <cstdint>
#include <vector>

namespace dt {

constexpr int NUM_BOARDS = 32;

struct Board {
    std::vector<WordIdx> candidates;
    bool solved = false;

    void apply(const Wordlists& w, WordIdx guess_idx, Pattern p) {
        if (solved) return;
        if (p == PATTERN_ALL_GREEN) { solved = true; candidates.clear(); return; }
        const Pattern* row = w.feedback_row(guess_idx);
        auto it = candidates.begin();
        for (auto s : candidates) {
            if (row[s] == p) *it++ = s;
        }
        candidates.erase(it, candidates.end());
    }
};

struct GameState {
    std::array<Board, NUM_BOARDS> boards;
    int guesses_used = 0;
    std::vector<WordIdx> guess_history;
    // Distinct-answer constraint: Duotrigordle picks 32 distinct words from the pool.
    // Once a board is solved, its answer can be removed from all other boards' candidate sets.
    std::vector<uint8_t> answer_used;

    static GameState fresh(const Wordlists& w) {
        GameState s;
        std::vector<WordIdx> all(w.num_solutions());
        for (WordIdx i = 0; i < w.num_solutions(); ++i) all[i] = i;
        for (auto& b : s.boards) b.candidates = all;
        s.answer_used.assign(w.num_solutions(), 0);
        return s;
    }

    int active_boards() const {
        int n = 0;
        for (const auto& b : boards) if (!b.solved) ++n;
        return n;
    }

    bool game_over() const { return active_boards() == 0; }

    void apply_guess(const Wordlists& w, WordIdx g, const std::array<Pattern, NUM_BOARDS>& patterns,
                     bool use_distinct_constraint = true) {
        int32_t g_sol = w.guess_to_sol(g);
        if (use_distinct_constraint) {
            for (int i = 0; i < NUM_BOARDS; ++i) {
                if (!boards[i].solved && patterns[i] == PATTERN_ALL_GREEN && g_sol >= 0) {
                    answer_used[g_sol] = 1;
                }
            }
        }
        for (int i = 0; i < NUM_BOARDS; ++i) boards[i].apply(w, g, patterns[i]);
        if (use_distinct_constraint) {
            for (int i = 0; i < NUM_BOARDS; ++i) {
                if (boards[i].solved) continue;
                auto& cs = boards[i].candidates;
                auto it = cs.begin();
                for (auto s : cs) if (!answer_used[s]) *it++ = s;
                cs.erase(it, cs.end());
            }
        }
        guess_history.push_back(g);
        ++guesses_used;
    }
};

}
