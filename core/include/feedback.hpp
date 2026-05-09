#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <string_view>

namespace dt {

using Pattern = uint8_t;
constexpr Pattern PATTERN_ALL_GREEN = 242;
constexpr int NUM_PATTERNS = 243;

Pattern compute_feedback(std::string_view guess, std::string_view answer);

Pattern parse_pattern(std::string_view s);
std::string pattern_to_string(Pattern p);

constexpr std::array<uint8_t, 5> pattern_digits(Pattern p) {
    std::array<uint8_t, 5> d{};
    for (int i = 0; i < 5; ++i) { d[i] = p % 3; p /= 3; }
    return d;
}

}
