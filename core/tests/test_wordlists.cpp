#include "feedback.hpp"
#include "wordlists.hpp"

#include <gtest/gtest.h>
#include <random>

class WordlistsFixture : public ::testing::Test {
protected:
    static dt::Wordlists& W() {
        static dt::Wordlists w(DT_DATA_DIR, "default");
        return w;
    }
};

TEST_F(WordlistsFixture, LoadsCorrectCounts) {
    auto& w = W();
    EXPECT_EQ(w.num_solutions(), 2653);
    EXPECT_EQ(w.num_guesses(), 14857);
}

TEST_F(WordlistsFixture, EverySolutionMapsToGuess) {
    auto& w = W();
    for (dt::WordIdx s = 0; s < w.num_solutions(); ++s) {
        auto gi = w.guess_index(w.solution(s));
        ASSERT_TRUE(gi.has_value()) << "Solution " << w.solution(s) << " not in guesses";
        EXPECT_EQ(*gi, w.sol_to_guess(s));
        EXPECT_EQ(w.guess_to_sol(*gi), static_cast<int32_t>(s));
    }
}

TEST_F(WordlistsFixture, FeedbackTableMatchesComputed) {
    auto& w = W();
    // spot-check 200 random pairs match compute_feedback
    std::mt19937_64 rng(123);
    std::uniform_int_distribution<dt::WordIdx> pg(0, w.num_guesses() - 1);
    std::uniform_int_distribution<dt::WordIdx> ps(0, w.num_solutions() - 1);
    for (int i = 0; i < 200; ++i) {
        auto g = pg(rng);
        auto s = ps(rng);
        EXPECT_EQ(w.feedback(g, s), dt::compute_feedback(w.guess(g), w.solution(s)));
    }
}

TEST_F(WordlistsFixture, FeedbackRowMatchesTable) {
    auto& w = W();
    auto gi = w.guess_index("CRANE");
    ASSERT_TRUE(gi.has_value());
    const dt::Pattern* row = w.feedback_row(*gi);
    for (dt::WordIdx s = 0; s < w.num_solutions(); ++s) {
        EXPECT_EQ(row[s], w.feedback(*gi, s));
    }
}
