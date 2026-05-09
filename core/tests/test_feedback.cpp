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





TEST(Feedback, ParseRoundTrip) {
    for (int p = 0; p < dt::NUM_PATTERNS; ++p) {
        EXPECT_EQ(parse_pattern(pattern_to_string(static_cast<dt::Pattern>(p))), p);
    }
}

TEST(Feedback, ParseAcceptsLowercase) {
    EXPECT_EQ(parse_pattern("gybBg"), parse_pattern("GYBBG"));
}
