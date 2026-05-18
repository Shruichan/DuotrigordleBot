// Generates hindsight-labeled training data for a specific turn (default: turn 2).
// For each of N games:
//   1. Play opener (greedy) so the state has T-1 guesses applied
//   2. Snapshot the state and features
//   3. For each of top-K candidate guesses at this state:
//        Apply candidate, then run greedy to game end
//        Record total game guesses from that branch
//   4. Write (state_features, K candidate word indices, K total_guesses) to disk
//
// Output:
//   features.bin     : float32 [N x FEATURE_DIM]
//   candidates.bin   : int32   [N x K]
//   totals.bin       : float32 [N x K]
//   meta.json
//
// Built as a separate binary (dt_export_labels) to keep main bench clean.

#include "feedback.hpp"
#include "game_state.hpp"
#include "simulator.hpp"
#include "strategy.hpp"
#include "wordlists.hpp"

#include <array>
#include <cmath>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

constexpr int FEATURES_PER_BOARD = 5 * 26 + 2;
constexpr int FEATURE_DIM = dt::NUM_BOARDS * FEATURES_PER_BOARD;

static void compute_features(const dt::Wordlists& w, const dt::GameState& state, float* out) {
    std::fill(out, out + FEATURE_DIM, 0.0f);
    for (int b = 0; b < dt::NUM_BOARDS; ++b) {
        float* f = out + b * FEATURES_PER_BOARD;
        const auto& board = state.boards[b];
        if (board.solved) {
            f[FEATURES_PER_BOARD - 1] = 1.0f;
            continue;
        }
        f[FEATURES_PER_BOARD - 2] = static_cast<float>(board.candidates.size()) / 2653.0f;
        for (dt::WordIdx sol_idx : board.candidates) {
            std::string_view word = w.solution(sol_idx);
            for (int p = 0; p < 5; ++p) {
                f[p * 26 + (word[p] - 'A')] = 1.0f;
            }
        }
    }
}

int main(int argc, char** argv) {
    int num_games = 1000;
    int target_turn = 2;     // generate label for this turn (1-indexed)
    int K = 50;               // top-K candidates to evaluate
    uint64_t seed = 42;
    std::string pool = "default";
    double alpha = 1.0;
    std::string out_dir = "labels";
    int max_total_guesses = 50;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "-n" && i + 1 < argc) num_games = std::atoi(argv[++i]);
        else if (a == "-t" && i + 1 < argc) target_turn = std::atoi(argv[++i]);
        else if (a == "-K" && i + 1 < argc) K = std::atoi(argv[++i]);
        else if (a == "-s" && i + 1 < argc) seed = std::strtoull(argv[++i], nullptr, 10);
        else if (a == "-o" && i + 1 < argc) out_dir = argv[++i];
        else if (a == "-a" && i + 1 < argc) alpha = std::atof(argv[++i]);
        else if (a == "-p" && i + 1 < argc) pool = argv[++i];
        else {
            std::cerr << "usage: dt_export_labels [-n games] [-t turn] [-K k] [-s seed] [-o out] [-a alpha] [-p pool]\n";
            return 1;
        }
    }

    dt::Wordlists w(DT_DATA_DIR, pool);
    namespace fs = std::filesystem;
    fs::create_directories(out_dir);

    std::ofstream feat_file(fs::path(out_dir) / "features.bin", std::ios::binary);
    std::ofstream cand_file(fs::path(out_dir) / "candidates.bin", std::ios::binary);
    std::ofstream tot_file(fs::path(out_dir) / "totals.bin", std::ios::binary);

    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<dt::WordIdx> pick(0, static_cast<dt::WordIdx>(w.num_solutions() - 1));
    std::vector<uint8_t> picked_buf(w.num_solutions(), 0);

    long long ok_examples = 0;
    auto t_start = std::chrono::steady_clock::now();

    std::cerr << "Generating " << num_games << " labeled examples for turn " << target_turn
              << " (K=" << K << ", alpha=" << alpha << ")\n";

    for (int g = 0; g < num_games; ++g) {
        // Pick distinct answers
        std::array<dt::WordIdx, dt::NUM_BOARDS> answers{};
        std::fill(picked_buf.begin(), picked_buf.end(), 0);
        for (auto& a : answers) {
            dt::WordIdx x;
            do { x = pick(rng); } while (picked_buf[x]);
            picked_buf[x] = 1; a = x;
        }

        // Play target_turn-1 guesses with greedy to set up the state
        dt::GreedyStrategy greedy(w, alpha);
        dt::GameState state = dt::GameState::fresh(w);
        bool game_ended_early = false;
        for (int t = 0; t < target_turn - 1; ++t) {
            if (state.game_over()) { game_ended_early = true; break; }
            dt::WordIdx gi = greedy.choose_guess(state);
            std::array<dt::Pattern, dt::NUM_BOARDS> pats{};
            for (int b = 0; b < dt::NUM_BOARDS; ++b) {
                pats[b] = state.boards[b].solved
                    ? dt::PATTERN_ALL_GREEN
                    : w.feedback(gi, answers[b]);
            }
            state.apply_guess(w, gi, pats, true);
        }
        if (game_ended_early) continue;

        // Snapshot state features at this turn
        std::vector<float> feats(FEATURE_DIM);
        compute_features(w, state, feats.data());

        // Get top-K candidates (greedy ranking)
        auto top_k = greedy.top_k_guesses(state, K);
        if (static_cast<int>(top_k.size()) < K) continue;

        std::vector<float> totals(K);
        std::vector<int32_t> cand_idx(K);
        for (int c = 0; c < K; ++c) {
            dt::WordIdx gi = top_k[c];
            cand_idx[c] = static_cast<int32_t>(gi);


            dt::GameState branch = state;
            std::array<dt::Pattern, dt::NUM_BOARDS> pats{};
            for (int b = 0; b < dt::NUM_BOARDS; ++b) {
                pats[b] = branch.boards[b].solved
                    ? dt::PATTERN_ALL_GREEN
                    : w.feedback(gi, answers[b]);
            }
            branch.apply_guess(w, gi, pats, true);

            // Play out with greedy (fresh instance so opener-cache from outer doesn't bleed in)
            dt::GreedyStrategy branch_greedy(w, alpha);
            while (!branch.game_over() && branch.guesses_used < max_total_guesses) {
                dt::WordIdx ng = branch_greedy.choose_guess(branch);
                std::array<dt::Pattern, dt::NUM_BOARDS> npats{};
                for (int b = 0; b < dt::NUM_BOARDS; ++b) {
                    npats[b] = branch.boards[b].solved
                        ? dt::PATTERN_ALL_GREEN
                        : w.feedback(ng, answers[b]);
                }
                branch.apply_guess(w, ng, npats, true);
            }
            totals[c] = static_cast<float>(branch.guesses_used);
        }

        feat_file.write(reinterpret_cast<const char*>(feats.data()), FEATURE_DIM * sizeof(float));
        cand_file.write(reinterpret_cast<const char*>(cand_idx.data()), K * sizeof(int32_t));
        tot_file.write(reinterpret_cast<const char*>(totals.data()), K * sizeof(float));
        ++ok_examples;

        if ((g + 1) % 50 == 0) {
            auto el = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t_start).count();
            std::cerr << "  [" << (g + 1) << "/" << num_games << "] examples=" << ok_examples
                      << " (" << std::fixed << std::setprecision(1) << el << "s)\n";
        }
    }

    std::ofstream meta(fs::path(out_dir) / "meta.json");
    meta << "{\"num_examples\":" << ok_examples
         << ",\"feature_dim\":" << FEATURE_DIM
         << ",\"K\":" << K
         << ",\"target_turn\":" << target_turn
         << ",\"alpha\":" << alpha
         << ",\"cand_extras_dim\":3"
         << ",\"has_cand_extras\":true}\n";
    std::cerr << "Done. Wrote " << ok_examples << " examples.\n";
    return 0;
}
