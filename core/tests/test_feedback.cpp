#include "feedback.hpp"

#include <gtest/gtest.h>

using dt::compute_feedback;
using dt::parse_pattern;
using dt::pattern_to_string;
using dt::PATTERN_ALL_GREEN;

TEST(Feedback, ExactMatchAllGreen) {
    EXPECT_EQ(compute_feedback("CRANE", "CRANE"), PATTERN_ALL_GREEN);
    EXPECT_EQ(pattern_to_string(PATTERN_ALL_GREEN), "GGGGG");
}

TEST(Feedback, NoOverlapAllGray) {
    EXPECT_EQ(compute_feedback("BLOCK", "GUEST"), parse_pattern("BBBBB"));
    EXPECT_EQ(pattern_to_string(parse_pattern("BBBBB")), "BBBBB");
}

TEST(Feedback, SimpleYellow) {
    EXPECT_EQ(compute_feedback("ABCDE", "EABCD"), parse_pattern("YYYYY"));
}

TEST(Feedback, DuplicateLetterInGuess_OneInAnswer) {
    // ALLEY vs APPLY: A green, L yellow (one L in answer), L gray (used), E gray, Y green
    EXPECT_EQ(pattern_to_string(compute_feedback("ALLEY", "APPLY")), "GYBBG");
}

TEST(Feedback, DuplicateLetterInGuess_NoneInAnswer) {
    EXPECT_EQ(pattern_to_string(compute_feedback("SISSY", "PRANK")), "BBBBB");
}

TEST(Feedback, GreenConsumesPreventsDuplicateYellow) {
    // ROBOT vs ROAST: R green, O green (consumes only O in answer),
    // B gray, O at pos 3 -> no unconsumed O remains -> gray, T green.
    EXPECT_EQ(pattern_to_string(compute_feedback("ROBOT", "ROAST")), "GGBBG");
}

TEST(Feedback, GreenConsumesAnswerLetter_StopsYellowFromMatching) {
    // GUESS=EERIE ANSWER=EAGER:
    // Pass 1: pos 0 E==E green; pos 1 E vs A no; pos 2 R vs G no; pos 3 I vs E no; pos 4 E vs R no
    // Pass 2 yellows:
    //   pos 1 E: answer has E at pos 3 unconsumed -> yellow
    //   pos 2 R: answer has R at pos 4 unconsumed -> yellow
    //   pos 3 I: not in answer -> gray
    //   pos 4 E: any unconsumed E? pos 0 consumed (green), pos 3 consumed (yellow) -> gray
    EXPECT_EQ(pattern_to_string(compute_feedback("EERIE", "EAGER")), "GYYBB");
}

TEST(Feedback, ParseRoundTrip) {
    for (int p = 0; p < dt::NUM_PATTERNS; ++p) {
        EXPECT_EQ(parse_pattern(pattern_to_string(static_cast<dt::Pattern>(p))), p);
    }
}

TEST(Feedback, ParseAcceptsLowercase) {
    EXPECT_EQ(parse_pattern("gybBg"), parse_pattern("GYBBG"));
}
