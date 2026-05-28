// Generates (state, remaining_turns_to_solve) training data for a value net.
// For each of N games we run greedy self-play with LITRE opener + alpha=300,
// and at every non-terminal turn we dump:
//   - 25-dim state features (per-board candidate-count distribution + turn pos)
//   - label = final_game_guesses - guesses_used_at_this_state
// Output:
//   features.bin : float32 [num_rows x FEATURE_DIM]
//   labels.bin   : float32 [num_rows]
//   meta.json
//
// Run: dt_gen_value_data -n 50000 -s 42 -o data/value_train

#include "feedback.hpp"
#include "game_state.hpp"
#include "simulator.hpp"
#include "strategy.hpp"
#include "wordlists.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

constexpr int FEATURE_DIM = 25;

// Compact state features:
//  0: guesses_used / 37
//  1: guesses_remaining / 37
//  2: num_active / 32
//  3: num_solved / 32
//  4: total_candidates / (32*2653)
//  5: sum log2(|C|) / 32
//  6: mean |C| over active boards / 2653
//  7: max |C| / 2653
//  8: min |C| over active / 2653
//  9: count |C|=1 / 32
// 10: count |C|=2 / 32
// 11: count |C|=3 / 32
// 12: count |C| in [4,10] / 32
// 13: count |C| in [11,30] / 32
// 14: count |C| > 30 / 32
// 15..24: top-10 sorted |C| values (descending) / 2653 (pad with 0)
static void compute_features(const dt::GameState& state, float* out) {
    std::fill(out, out + FEATURE_DIM, 0.0f);
    int total_cands = 0;
    int num_active = 0;
    int num_solved = 0;
    int max_c = 0, min_c = INT32_MAX;
    double sum_log = 0.0;
    int c1 = 0, c2 = 0, c3 = 0, c4_10 = 0, c11_30 = 0, c30p = 0;
    std::vector<int> sizes;
    sizes.reserve(dt::NUM_BOARDS);
    for (int b = 0; b < dt::NUM_BOARDS; ++b) {
        const auto& bd = state.boards[b];
        if (bd.solved) { ++num_solved; continue; }
        int k = static_cast<int>(bd.candidates.size());
        if (k == 0) { ++num_solved; continue; }
        ++num_active;
        total_cands += k;
        if (k > max_c) max_c = k;
        if (k < min_c) min_c = k;
        sum_log += std::log2(static_cast<double>(k));
        sizes.push_back(k);
        if (k == 1) ++c1;
        else if (k == 2) ++c2;
        else if (k == 3) ++c3;
        else if (k <= 10) ++c4_10;
        else if (k <= 30) ++c11_30;
        else ++c30p;
    }
    if (num_active == 0) min_c = 0;
    std::sort(sizes.begin(), sizes.end(), std::greater<int>());

    out[0] = static_cast<float>(state.guesses_used) / 37.0f;
    out[1] = static_cast<float>(37 - state.guesses_used) / 37.0f;
    out[2] = static_cast<float>(num_active) / 32.0f;
    out[3] = static_cast<float>(num_solved) / 32.0f;
    out[4] = static_cast<float>(total_cands) / (32.0f * 2653.0f);
    out[5] = static_cast<float>(sum_log) / 32.0f;
    out[6] = (num_active > 0)
        ? static_cast<float>(total_cands) / static_cast<float>(num_active) / 2653.0f
        : 0.0f;
    out[7] = static_cast<float>(max_c) / 2653.0f;
    out[8] = static_cast<float>(min_c) / 2653.0f;
    out[9]  = static_cast<float>(c1)     / 32.0f;
    out[10] = static_cast<float>(c2)     / 32.0f;
    out[11] = static_cast<float>(c3)     / 32.0f;
    out[12] = static_cast<float>(c4_10)  / 32.0f;
    out[13] = static_cast<float>(c11_30) / 32.0f;
    out[14] = static_cast<float>(c30p)   / 32.0f;
    for (int i = 0; i < 10; ++i) {
        out[15 + i] = (i < static_cast<int>(sizes.size()))
            ? static_cast<float>(sizes[i]) / 2653.0f
            : 0.0f;
    }
}

int main(int argc, char** argv) {
    int num_games = 50000;
    uint64_t seed = 42;
    std::string pool = "default";
    double alpha = 300.0;
    std::string opener = "LITRE";
    std::string out_dir = "data/value_train";
    int max_total_guesses = 50;
    int log_every = 500;
    // Daily mode: iterate over the actual MT19937(id) historical-daily sequence
    // rather than random distinct samples. When daily_start > 0, num_games is
    // overridden by (daily_end - daily_start + 1).
    int daily_start = 0, daily_end = 0;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "-n" && i + 1 < argc) num_games = std::atoi(argv[++i]);
        else if (a == "-s" && i + 1 < argc) seed = std::strtoull(argv[++i], nullptr, 10);
        else if (a == "-o" && i + 1 < argc) out_dir = argv[++i];
        else if (a == "-a" && i + 1 < argc) alpha = std::atof(argv[++i]);
        else if (a == "-p" && i + 1 < argc) pool = argv[++i];
        else if (a == "--opener" && i + 1 < argc) opener = argv[++i];
        else if (a == "--log-every" && i + 1 < argc) log_every = std::atoi(argv[++i]);
        else if (a == "--daily" && i + 2 < argc) {
            daily_start = std::atoi(argv[++i]);
            daily_end = std::atoi(argv[++i]);
        }
        else {
            std::cerr << "usage: dt_gen_value_data [-n games] [-s seed] [-o out]"
                         " [-a alpha] [-p pool] [--opener WORD] [--log-every N]"
                         " [--daily START END]\n";
            return 1;
        }
    }
    if (daily_start > 0 && daily_end >= daily_start) {
        num_games = daily_end - daily_start + 1;
    }

    dt::Wordlists w(DT_DATA_DIR, pool);
    namespace fs = std::filesystem;
    fs::create_directories(out_dir);
    std::ofstream feat_file(fs::path(out_dir) / "features.bin", std::ios::binary);
    std::ofstream label_file(fs::path(out_dir) / "labels.bin", std::ios::binary);

    std::transform(opener.begin(), opener.end(), opener.begin(), ::toupper);
    auto opener_idx = w.guess_index(opener);
    if (!opener_idx) { std::cerr << "opener not in dict: " << opener << "\n"; return 1; }

    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<dt::WordIdx> pick(0, static_cast<dt::WordIdx>(w.num_solutions() - 1));
    std::vector<uint8_t> picked_buf(w.num_solutions(), 0);

    long long total_rows = 0;
    long long completed_games = 0;
    auto t_start = std::chrono::steady_clock::now();

    std::cerr << "Generating value-net training data: " << num_games << " games, "
              << "opener=" << opener << " alpha=" << alpha;
    if (daily_start > 0) std::cerr << " (daily IDs " << daily_start << ".." << daily_end << ")";
    std::cerr << "\n";

    for (int g = 0; g < num_games; ++g) {
        // Pick distinct answers — either from the daily MT19937 sequence or
        // by random uniform sampling.
        std::array<dt::WordIdx, dt::NUM_BOARDS> answers{};
        if (daily_start > 0) {
            answers = dt::daily_answers(w, static_cast<uint32_t>(daily_start + g));
        } else {
            std::fill(picked_buf.begin(), picked_buf.end(), 0);
            for (auto& a : answers) {
                dt::WordIdx x;
                do { x = pick(rng); } while (picked_buf[x]);
                picked_buf[x] = 1; a = x;
            }
        }

        // Play full game with greedy, recording each non-terminal state and its final length.
        dt::GreedyStrategy greedy(w, alpha);
        greedy.set_opener(*opener_idx);
        dt::GameState state = dt::GameState::fresh(w);

        struct Row { std::vector<float> f; int turn; };
        std::vector<Row> rows;
        rows.reserve(40);

        while (!state.game_over() && state.guesses_used < max_total_guesses) {
            // Snapshot features BEFORE choosing/applying guess
            Row r;
            r.f.assign(FEATURE_DIM, 0.0f);
            compute_features(state, r.f.data());
            r.turn = state.guesses_used;
            rows.push_back(std::move(r));

            dt::WordIdx gi = greedy.choose_guess(state);
            std::array<dt::Pattern, dt::NUM_BOARDS> pats{};
            for (int b = 0; b < dt::NUM_BOARDS; ++b) {
                pats[b] = state.boards[b].solved
                    ? dt::PATTERN_ALL_GREEN
                    : w.feedback(gi, answers[b]);
            }
            state.apply_guess(w, gi, pats, true);
        }
        if (!state.game_over()) continue;  // Skipping incomplete games.

        const int final_turns = state.guesses_used;
        for (auto& r : rows) {
            float label = static_cast<float>(final_turns - r.turn);
            feat_file.write(reinterpret_cast<const char*>(r.f.data()), FEATURE_DIM * sizeof(float));
            label_file.write(reinterpret_cast<const char*>(&label), sizeof(float));
            ++total_rows;
        }
        ++completed_games;

        if ((g + 1) % log_every == 0) {
            auto el = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t_start).count();
            std::cerr << "  [" << (g + 1) << "/" << num_games << "] games_ok="
                      << completed_games << " rows=" << total_rows
                      << " (" << std::fixed << std::setprecision(1) << el << "s)\n";
        }
    }

    std::ofstream meta(fs::path(out_dir) / "meta.json");
    meta << "{\"num_games\":" << completed_games
         << ",\"num_rows\":" << total_rows
         << ",\"feature_dim\":" << FEATURE_DIM
         << ",\"alpha\":" << alpha
         << ",\"opener\":\"" << opener << "\""
         << "}\n";
    std::cerr << "Done. games_ok=" << completed_games << " rows=" << total_rows << "\n";
    return 0;
}
