#include "feedback.hpp"

#include <array>
#include <stdexcept>

namespace dt {

Pattern compute_feedback(std::string_view guess, std::string_view answer) {
    if (guess.size() != 5 || answer.size() != 5) {
        throw std::invalid_argument("compute_feedback: words must be 5 letters");
    }
    std::array<uint8_t, 5> digits{};
    std::array<bool, 5> answer_used{};

    for (int i = 0; i < 5; ++i) {
        if (guess[i] == answer[i]) {
            digits[i] = 2;
            answer_used[i] = true;
        }
    }
    for (int i = 0; i < 5; ++i) {
        if (digits[i] == 2) continue;
        for (int j = 0; j < 5; ++j) {
            if (!answer_used[j] && guess[i] == answer[j]) {
                digits[i] = 1;
                answer_used[j] = true;
                break;
            }
        }
    }
    Pattern p = 0;
    Pattern mult = 1;
    for (int i = 0; i < 5; ++i) {
        p = static_cast<Pattern>(p + digits[i] * mult);
        mult = static_cast<Pattern>(mult * 3);
    }
    return p;
}

Pattern parse_pattern(std::string_view s) {
    if (s.size() != 5) throw std::invalid_argument("parse_pattern: expected 5 chars");
    Pattern p = 0;
    Pattern mult = 1;
    for (int i = 0; i < 5; ++i) {
        uint8_t d;
        switch (s[i]) {
            case 'G': case 'g': d = 2; break;
            case 'Y': case 'y': d = 1; break;
            case 'B': case 'b': case '.': case '-': d = 0; break;
            default: throw std::invalid_argument("parse_pattern: char must be G/Y/B");
        }
        p = static_cast<Pattern>(p + d * mult);
        mult = static_cast<Pattern>(mult * 3);
    }
    return p;
}

std::string pattern_to_string(Pattern p) {
    std::string s(5, 'B');
    for (int i = 0; i < 5; ++i) {
        uint8_t d = p % 3;
        p = static_cast<Pattern>(p / 3);
        s[i] = d == 2 ? 'G' : (d == 1 ? 'Y' : 'B');
    }
    return s;
}

}
