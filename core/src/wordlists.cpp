#include "wordlists.hpp"

#include <chrono>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace dt {

namespace {

std::vector<std::string> load_word_file(const std::filesystem::path& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Cannot open wordlist: " + path.string());
    std::vector<std::string> out;
    std::string word;
    while (std::getline(in, word)) {
        while (!word.empty() && (word.back() == '\r' || word.back() == '\n' || word.back() == ' ')) {
            word.pop_back();
        }
        if (word.empty()) continue;
        if (word.size() != 5) {
            throw std::runtime_error("Bad word in " + path.string() + ": '" + word + "'");
        }
        out.push_back(std::move(word));
    }
    return out;
}

}

Wordlists::Wordlists(const std::filesystem::path& data_dir,
                     std::string_view solution_pool) {
    const std::string sol_file = std::string("solutions_") + std::string(solution_pool) + ".txt";
    solutions_ = load_word_file(data_dir / sol_file);
    guesses_ = load_word_file(data_dir / "valid_guesses.txt");

    for (WordIdx i = 0; i < guesses_.size(); ++i) guess_idx_[guesses_[i]] = i;
    for (WordIdx i = 0; i < solutions_.size(); ++i) sol_idx_[solutions_[i]] = i;

    guess_to_sol_.assign(guesses_.size(), -1);
    sol_to_guess_.assign(solutions_.size(), INVALID_WORD);
    for (WordIdx s = 0; s < solutions_.size(); ++s) {
        auto it = guess_idx_.find(solutions_[s]);
        if (it == guess_idx_.end()) {
            throw std::runtime_error("Solution '" + solutions_[s] + "' is not in valid_guesses");
        }
        sol_to_guess_[s] = it->second;
        guess_to_sol_[it->second] = static_cast<int32_t>(s);
    }

    const size_t G = guesses_.size();
    const size_t S = solutions_.size();
    std::cerr << "Wordlists: " << G << " guesses, " << S << " solutions; "
              << "precomputing feedback table (" << (G * S / (1024 * 1024)) << " MiB)... " << std::flush;
    auto t0 = std::chrono::steady_clock::now();
    feedback_table_.resize(G * S);
    for (size_t g = 0; g < G; ++g) {
        Pattern* row = feedback_table_.data() + g * S;
        std::string_view gw = guesses_[g];
        for (size_t s = 0; s < S; ++s) {
            row[s] = compute_feedback(gw, solutions_[s]);
        }
    }
    auto t1 = std::chrono::steady_clock::now();
    std::cerr << "done in "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count()
              << " ms\n";
}

std::optional<WordIdx> Wordlists::guess_index(std::string_view w) const {
    auto it = guess_idx_.find(std::string(w));
    if (it == guess_idx_.end()) return std::nullopt;
    return it->second;
}

std::optional<WordIdx> Wordlists::solution_index(std::string_view w) const {
    auto it = sol_idx_.find(std::string(w));
    if (it == sol_idx_.end()) return std::nullopt;
    return it->second;
}

}
