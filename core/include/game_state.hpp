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

    static GameState fresh(const Wordlists& w) {
        GameState s;
        std::vector<WordIdx> all(w.num_solutions());
        for (WordIdx i = 0; i < w.num_solutions(); ++i) all[i] = i;
        for (auto& b : s.boards) b.candidates = all;
        return s;
    }

    int active_boards() const {
        int n = 0;
        for (const auto& b : boards) if (!b.solved) ++n;
        return n;
    }

    bool game_over() const { return active_boards() == 0; }

    void apply_guess(const Wordlists& w, WordIdx g, const std::array<Pattern, NUM_BOARDS>& patterns) {
        for (int i = 0; i < NUM_BOARDS; ++i) boards[i].apply(w, g, patterns[i]);
        guess_history.push_back(g);
        ++guesses_used;
    }
};

}
