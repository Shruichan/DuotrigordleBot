#include "feedback.hpp"
#include "game_state.hpp"
#include "wordlists.hpp"

#include <gtest/gtest.h>

class GameStateFixture : public ::testing::Test {
protected:
    static dt::Wordlists& W() {
        static dt::Wordlists w(DT_DATA_DIR, "default");
        return w;
    }
};

TEST_F(GameStateFixture, FreshStateFullCandidates) {
    auto& w = W();
    auto s = dt::GameState::fresh(w);
    EXPECT_EQ(s.active_boards(), dt::NUM_BOARDS);
    EXPECT_FALSE(s.game_over());
    for (const auto& b : s.boards) {
        EXPECT_EQ(b.candidates.size(), w.num_solutions());
        EXPECT_FALSE(b.solved);
    }
}

TEST_F(GameStateFixture, AllGreenSolvesBoard) {
    auto& w = W();
    auto state = dt::GameState::fresh(w);
    auto gi = w.guess_index("CRANE");
    ASSERT_TRUE(gi.has_value());

    std::array<dt::Pattern, dt::NUM_BOARDS> pats{};
    pats.fill(dt::parse_pattern("BBBBB"));
    pats[5] = dt::PATTERN_ALL_GREEN;
    state.apply_guess(w, *gi, pats);

    EXPECT_TRUE(state.boards[5].solved);
    EXPECT_FALSE(state.boards[0].solved);
    EXPECT_EQ(state.active_boards(), dt::NUM_BOARDS - 1);
    EXPECT_LT(state.boards[0].candidates.size(), w.num_solutions());
}

TEST_F(GameStateFixture, FilteringNarrowsToCorrectCandidate) {
    auto& w = W();
    auto state = dt::GameState::fresh(w);
    auto answer_idx = w.solution_index("PIZZA");
    ASSERT_TRUE(answer_idx.has_value());

    // Simulate several real guesses; board 0's answer is PIZZA.
    for (const char* word : {"CRANE", "SLOTH", "MIGHT", "PIZZA"}) {
        auto gi = w.guess_index(word);
        ASSERT_TRUE(gi.has_value());
        std::array<dt::Pattern, dt::NUM_BOARDS> pats{};
        pats.fill(0);
        pats[0] = dt::compute_feedback(w.guess(*gi), "PIZZA");
        state.apply_guess(w, *gi, pats);
    }
    EXPECT_TRUE(state.boards[0].solved);
}
