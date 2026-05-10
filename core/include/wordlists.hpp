#pragma once

#include "feedback.hpp"

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace dt {

using WordIdx = uint16_t;
constexpr WordIdx INVALID_WORD = UINT16_MAX;

class Wordlists {
public:
    Wordlists(const std::filesystem::path& data_dir,
              std::string_view solution_pool = "default");

    size_t num_guesses() const { return guesses_.size(); }
    size_t num_solutions() const { return solutions_.size(); }

    std::string_view guess(WordIdx i) const { return guesses_[i]; }
    std::string_view solution(WordIdx i) const { return solutions_[i]; }

    std::optional<WordIdx> guess_index(std::string_view w) const;
    std::optional<WordIdx> solution_index(std::string_view w) const;

    int32_t guess_to_sol(WordIdx g) const { return guess_to_sol_[g]; }
    WordIdx sol_to_guess(WordIdx s) const { return sol_to_guess_[s]; }

    Pattern feedback(WordIdx guess_idx, WordIdx sol_idx) const {
        return feedback_table_[static_cast<size_t>(guess_idx) * solutions_.size() + sol_idx];
    }
    const Pattern* feedback_row(WordIdx guess_idx) const {
        return feedback_table_.data() + static_cast<size_t>(guess_idx) * solutions_.size();
    }

    const std::vector<std::string>& solutions() const { return solutions_; }
    const std::vector<std::string>& guesses() const { return guesses_; }

private:
    std::vector<std::string> guesses_;
    std::vector<std::string> solutions_;
    std::unordered_map<std::string, WordIdx> guess_idx_;
    std::unordered_map<std::string, WordIdx> sol_idx_;
    std::vector<int32_t> guess_to_sol_;
    std::vector<WordIdx> sol_to_guess_;
    std::vector<Pattern> feedback_table_;
};

}
