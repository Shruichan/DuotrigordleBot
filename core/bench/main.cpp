#include "simulator.hpp"
#include "strategy.hpp"
#include "strategy_beam.hpp"
#include "strategy_endgame.hpp"
#include "strategy_mcts.hpp"
#include "value_net.hpp"
#include "wordlists.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <iomanip>
#include <memory>

namespace dt { extern std::atomic<int> g_beam_triggered; extern std::atomic<int> g_beam_skipped; }

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>

// 5x26 letter-position incidence + (|C|/2653) + solved-flag per board
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
                int letter = word[p] - 'A';
                f[p * 26 + letter] = 1.0f;
            }
        }
    }
}

int main(int argc, char** argv) {
    int num_games = 200;
    uint64_t seed = 42;
    std::string pool = "default";
    double alpha = 1.0;
    std::string strat_name = "greedy";
    int beam_k = 8;
    int beam_samples = 3;
    int endgame_threshold = 25;
    bool distinct_answers = true;
    bool use_distinct_constraint = true;
    std::string answers_csv;
    bool trace_mode = false;
    std::string export_dir;
    std::string force_opener;
    std::string force_prefix_csv;
    std::string alpha_schedule_csv;
    int lookahead_k = 0;
    int lookahead_n = 5;
    bool lookahead_exact = false;
    int lookahead_exact_max_active = 999;
    std::string value_net_path;
    int mcts_k = 10;
    int mcts_r = 20;
    int mcts_depth = 2;
    int mcts_max_active = 16;
    double mcts_risk = 0.0;
    int panic_slack = -1;
    double panic_alpha = 0.0;
    bool daily_mode = false;
    int daily_start = 1;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "-n" && i + 1 < argc) num_games = std::atoi(argv[++i]);
        else if (a == "-s" && i + 1 < argc) seed = std::strtoull(argv[++i], nullptr, 10);
        else if (a == "-p" && i + 1 < argc) pool = argv[++i];
        else if (a == "-a" && i + 1 < argc) alpha = std::atof(argv[++i]);
        else if (a == "-S" && i + 1 < argc) strat_name = argv[++i];
        else if (a == "-k" && i + 1 < argc) beam_k = std::atoi(argv[++i]);
        else if (a == "-N" && i + 1 < argc) beam_samples = std::atoi(argv[++i]);
        else if (a == "--no-distinct-answers") distinct_answers = false;
        else if (a == "--no-distinct-constraint") use_distinct_constraint = false;
        else if (a == "--answers" && i + 1 < argc) answers_csv = argv[++i];
        else if (a == "--trace") trace_mode = true;
        else if (a == "-T" && i + 1 < argc) endgame_threshold = std::atoi(argv[++i]);
        else if (a == "--export-features" && i + 1 < argc) export_dir = argv[++i];
        else if (a == "--opener" && i + 1 < argc) force_opener = argv[++i];
        else if (a == "--openers" && i + 1 < argc) force_prefix_csv = argv[++i];
        else if (a == "--alphas" && i + 1 < argc) alpha_schedule_csv = argv[++i];
        else if (a == "--la-k" && i + 1 < argc) lookahead_k = std::atoi(argv[++i]);
        else if (a == "--la-n" && i + 1 < argc) lookahead_n = std::atoi(argv[++i]);
        else if (a == "--la-exact") lookahead_exact = true;
        else if (a == "--la-exact-max-active" && i + 1 < argc) lookahead_exact_max_active = std::atoi(argv[++i]);
        else if (a == "--value-net" && i + 1 < argc) value_net_path = argv[++i];
        else if (a == "--mcts-k" && i + 1 < argc) mcts_k = std::atoi(argv[++i]);
        else if (a == "--mcts-r" && i + 1 < argc) mcts_r = std::atoi(argv[++i]);
        else if (a == "--mcts-depth" && i + 1 < argc) mcts_depth = std::atoi(argv[++i]);
        else if (a == "--mcts-max-active" && i + 1 < argc) mcts_max_active = std::atoi(argv[++i]);
        else if (a == "--mcts-risk" && i + 1 < argc) mcts_risk = std::atof(argv[++i]);
        else if (a == "--panic-slack" && i + 1 < argc) panic_slack = std::atoi(argv[++i]);
        else if (a == "--panic-alpha" && i + 1 < argc) panic_alpha = std::atof(argv[++i]);
        else if (a == "--find-worst") trace_mode = true;  // alias to print bad games
        else if (a == "--daily") { daily_mode = true; daily_start = (i + 1 < argc && argv[i+1][0] != '-') ? std::atoi(argv[++i]) : 1; }
        else if (a == "--dump-feedback-table" && i + 1 < argc) {
            std::ofstream out(argv[++i], std::ios::binary);
            // Dummy load to access feedback table
            dt::Wordlists wl(DT_DATA_DIR, pool);
            // Feedback table is internal; reconstruct via public iface
            std::vector<dt::Pattern> tbl(wl.num_guesses() * wl.num_solutions());
            for (size_t g = 0; g < wl.num_guesses(); ++g) {
                const dt::Pattern* row = wl.feedback_row(static_cast<dt::WordIdx>(g));
                std::copy(row, row + wl.num_solutions(),
                          tbl.data() + g * wl.num_solutions());
            }
            out.write(reinterpret_cast<const char*>(tbl.data()), tbl.size());
            std::cerr << "Dumped " << tbl.size() << " bytes (" << wl.num_guesses() << "x"
                      << wl.num_solutions() << ") to " << argv[i] << "\n";
            return 0;
        }
        else {
            std::cerr << "usage: dt_bench [-n games] [-s seed] [-p pool] [-a alpha]"
                      << " [-S greedy|beam] [-k beam_k] [-N beam_samples]"
                      << " [--no-distinct-answers] [--no-distinct-constraint]"
                      << " [--answers W1,W2,...,W32] [--trace]\n";
            return 1;
        }
    }

    dt::Wordlists w(DT_DATA_DIR, pool);
    std::unique_ptr<dt::Strategy> strat;
    dt::GreedyStrategy* greedy_ptr = nullptr;
    dt::MctsStrategy* mcts_ptr = nullptr;
    if (strat_name == "greedy") {
        auto g = std::make_unique<dt::GreedyStrategy>(w, alpha);
        greedy_ptr = g.get();
        strat = std::move(g);
    } else if (strat_name == "beam") {
        strat = std::make_unique<dt::BeamStrategy>(w, beam_k, beam_samples, alpha);
        std::cerr << "beam k=" << beam_k << " samples=" << beam_samples << "\n";
    } else if (strat_name == "endgame") {
        strat = std::make_unique<dt::EndgameStrategy>(w, endgame_threshold, alpha);
        std::cerr << "endgame threshold=" << endgame_threshold << "\n";
    } else if (strat_name == "mcts") {
        auto m = std::make_unique<dt::MctsStrategy>(w, alpha, mcts_k, mcts_r,
                                                    mcts_depth, mcts_max_active);
        m->set_risk_lambda(mcts_risk);
        mcts_ptr = m.get();
        strat = std::move(m);
        std::cerr << "mcts K=" << mcts_k << " R=" << mcts_r << " depth=" << mcts_depth
                  << " max_active=" << mcts_max_active << " risk=" << mcts_risk << "\n";
    } else {
        std::cerr << "unknown strategy: " << strat_name << "\n";
        return 1;
    }
    std::cerr << "alpha (answer_bonus) = " << alpha << "\n";

    if (!force_opener.empty() && (greedy_ptr || mcts_ptr)) {
        std::transform(force_opener.begin(), force_opener.end(), force_opener.begin(), ::toupper);
        auto oi = w.guess_index(force_opener);
        if (!oi) { std::cerr << "opener '" << force_opener << "' not in dictionary\n"; return 1; }
        if (greedy_ptr) greedy_ptr->set_opener(*oi);
        if (mcts_ptr) mcts_ptr->set_opener(*oi);
        std::cerr << "forced opener: " << force_opener << "\n";
    }
    if (lookahead_k > 0 && greedy_ptr) {
        greedy_ptr->set_lookahead(lookahead_k, lookahead_n);
        if (lookahead_exact) greedy_ptr->set_lookahead_exact(true, lookahead_exact_max_active);
        std::cerr << "2-step lookahead: K=" << lookahead_k << " N=" << lookahead_n
                  << (lookahead_exact ? " [exact expected-V, max_active="
                                        + std::to_string(lookahead_exact_max_active) + "]" : "")
                  << "\n";
    }
    if (panic_slack >= 0 && greedy_ptr) {
        greedy_ptr->set_panic(panic_slack, panic_alpha);
        std::cerr << "panic mode: slack<=" << panic_slack << " -> alpha=" << panic_alpha << "\n";
    }
    dt::ValueNet value_net;
    if (!value_net_path.empty() && (greedy_ptr || mcts_ptr)) {
        if (!value_net.load(value_net_path)) {
            std::cerr << "Failed to load value net from " << value_net_path << "\n";
            return 1;
        }
        if (greedy_ptr) greedy_ptr->set_value_net(&value_net);
        if (mcts_ptr) mcts_ptr->set_value_net(&value_net);
        std::cerr << "value net loaded: feat_dim=" << value_net.feature_dim() << "\n";
    }
    if (!alpha_schedule_csv.empty() && greedy_ptr) {
        std::vector<double> sched;
        std::string cur;
        alpha_schedule_csv += ",";
        for (char c : alpha_schedule_csv) {
            if (c == ',') { if (!cur.empty()) { sched.push_back(std::atof(cur.c_str())); cur.clear(); } }
            else cur.push_back(c);
        }
        greedy_ptr->set_alpha_schedule(sched);
        std::cerr << "alpha schedule (" << sched.size() << "): ";
        for (double a : sched) std::cerr << a << " ";
        std::cerr << "\n";
    }
    if (!force_prefix_csv.empty() && greedy_ptr) {
        std::vector<dt::WordIdx> seq;
        std::string cur;
        force_prefix_csv += ",";
        for (char c : force_prefix_csv) {
            if (c == ',') {
                if (cur.empty()) continue;
                std::transform(cur.begin(), cur.end(), cur.begin(), ::toupper);
                auto oi = w.guess_index(cur);
                if (!oi) { std::cerr << "prefix word '" << cur << "' not in dictionary\n"; return 1; }
                seq.push_back(*oi);
                cur.clear();
            } else cur.push_back(c);
        }
        greedy_ptr->set_forced_prefix(seq);
        std::cerr << "forced prefix (" << seq.size() << " words): ";
        for (auto g : seq) std::cerr << w.guess(g) << " ";
        std::cerr << "\n";
    }

    if (!answers_csv.empty()) {
        std::array<dt::WordIdx, dt::NUM_BOARDS> answers{};
        size_t pos = 0; int idx = 0;
        while (pos <= answers_csv.size() && idx < dt::NUM_BOARDS) {
            size_t comma = answers_csv.find(',', pos);
            std::string word = answers_csv.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
            std::transform(word.begin(), word.end(), word.begin(), ::toupper);
            auto si = w.solution_index(word);
            if (!si) {
                std::cerr << "answer #" << (idx + 1) << " '" << word << "' is not in solution pool '" << pool << "'\n";
                return 1;
            }
            answers[idx++] = *si;
            if (comma == std::string::npos) break;
            pos = comma + 1;
        }
        if (idx != dt::NUM_BOARDS) {
            std::cerr << "need exactly " << dt::NUM_BOARDS << " answers, got " << idx << "\n";
            return 1;
        }
        std::cerr << "Replaying single game with provided answers...\n";
        auto r = dt::run_one_game(w, *strat, answers, 50, use_distinct_constraint);
        std::cout << "Game result: " << (r.all_solved ? "SOLVED" : "FAILED")
                  << " in " << r.guesses_used << " guesses, "
                  << r.boards_solved << "/" << dt::NUM_BOARDS << " boards solved\n";
        if (trace_mode || !r.all_solved) {
            std::cout << "\nGuess history:\n";
            // Replay to show each guess and which boards it solved
            dt::GameState s = dt::GameState::fresh(w);
            for (size_t i = 0; i < r.guess_history.size(); ++i) {
                dt::WordIdx g = r.guess_history[i];
                int solved_before = dt::NUM_BOARDS - s.active_boards();
                std::array<dt::Pattern, dt::NUM_BOARDS> pats{};
                for (int b = 0; b < dt::NUM_BOARDS; ++b) {
                    pats[b] = s.boards[b].solved ? dt::PATTERN_ALL_GREEN : w.feedback(g, answers[b]);
                }
                s.apply_guess(w, g, pats, use_distinct_constraint);
                int solved_now = dt::NUM_BOARDS - s.active_boards();
                std::cout << "  " << std::setw(2) << (i + 1) << ". " << w.guess(g)
                          << "  active=" << std::setw(2) << s.active_boards()
                          << "  +" << (solved_now - solved_before) << " solved this turn\n";
            }
        }
        return r.all_solved ? 0 : 2;
    }

    if (!export_dir.empty()) {
        namespace fs = std::filesystem;
        fs::create_directories(export_dir);
        std::ofstream feat_file(fs::path(export_dir) / "features.bin", std::ios::binary);
        std::ofstream target_file(fs::path(export_dir) / "targets.bin", std::ios::binary);
        if (!feat_file || !target_file) {
            std::cerr << "Cannot open export files in " << export_dir << "\n";
            return 1;
        }

        std::mt19937_64 rng(seed);
        std::uniform_int_distribution<dt::WordIdx> pick(0, static_cast<dt::WordIdx>(w.num_solutions() - 1));
        std::vector<uint8_t> picked_buf(w.num_solutions(), 0);

        std::vector<float> feat_buf(FEATURE_DIM);
        long long total_examples = 0;
        long long games_solved = 0;
        long long sum_guesses = 0;

        std::cerr << "Exporting features for " << num_games << " games to " << export_dir
                  << " (feature_dim=" << FEATURE_DIM << ")...\n";
        auto t_start = std::chrono::steady_clock::now();

        for (int g = 0; g < num_games; ++g) {
            std::array<dt::WordIdx, dt::NUM_BOARDS> answers{};
            std::fill(picked_buf.begin(), picked_buf.end(), 0);
            for (auto& a : answers) {
                dt::WordIdx x;
                do { x = pick(rng); } while (picked_buf[x]);
                picked_buf[x] = 1;
                a = x;
            }

            dt::GameState state = dt::GameState::fresh(w);
            std::vector<std::vector<float>> game_features;
            game_features.reserve(40);

            while (state.guesses_used < 50 && !state.game_over()) {
                std::vector<float> ft(FEATURE_DIM);
                compute_features(w, state, ft.data());
                game_features.push_back(std::move(ft));

                dt::WordIdx gi = strat->choose_guess(state);
                std::array<dt::Pattern, dt::NUM_BOARDS> pats{};
                for (int b = 0; b < dt::NUM_BOARDS; ++b) {
                    pats[b] = state.boards[b].solved
                        ? dt::PATTERN_ALL_GREEN
                        : w.feedback(gi, answers[b]);
                }
                state.apply_guess(w, gi, pats, use_distinct_constraint);
            }

            if (state.game_over()) {
                int total = state.guesses_used;
                ++games_solved;
                sum_guesses += total;
                for (size_t t = 0; t < game_features.size(); ++t) {
                    float remaining = static_cast<float>(total) - static_cast<float>(t);
                    feat_file.write(reinterpret_cast<const char*>(game_features[t].data()),
                                    FEATURE_DIM * sizeof(float));
                    target_file.write(reinterpret_cast<const char*>(&remaining), sizeof(float));
                    ++total_examples;
                }
            }

            if ((g + 1) % 200 == 0) {
                auto el = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - t_start).count();
                std::cerr << "  [" << (g + 1) << "/" << num_games << "] "
                          << total_examples << " examples, mean="
                          << (games_solved ? double(sum_guesses) / games_solved : 0.0)
                          << ", " << std::fixed << std::setprecision(1) << el << "s\n";
            }
        }

        std::ofstream meta(fs::path(export_dir) / "meta.json");
        meta << "{\"num_examples\":" << total_examples
             << ",\"feature_dim\":" << FEATURE_DIM
             << ",\"games_solved\":" << games_solved
             << ",\"mean_guesses\":" << (games_solved ? double(sum_guesses) / games_solved : 0.0)
             << "}\n";
        std::cerr << "Done. Wrote " << total_examples << " examples.\n";
        return 0;
    }

    std::cerr << "Running " << num_games << " games (seed=" << seed << ", pool=" << pool
              << ", strategy=" << strat->name() << ")...\n";
    auto stats = dt::run_benchmark(w, *strat, num_games, seed, 50, true,
                                   distinct_answers, use_distinct_constraint,
                                   daily_mode, daily_start);
    if (daily_mode) std::cerr << "daily mode: ids " << daily_start
                              << "-" << (daily_start + num_games - 1) << "\n";
    std::cerr << "distinct_answers=" << distinct_answers
              << " use_distinct_constraint=" << use_distinct_constraint << "\n";

    std::cout << "\n=== Benchmark Results ===\n";
    std::cout << "Strategy:        " << strat->name() << "\n";
    std::cout << "Games:           " << stats.games << "\n";
    std::cout << "Solved:          " << stats.solved
              << " (" << std::fixed << std::setprecision(1)
              << (100.0 * stats.solved / stats.games) << "%)\n";
    if (stats.solved) {
        std::cout << "Mean guesses:    " << std::setprecision(2) << stats.mean_guesses << "\n";
        std::cout << "Min / Max:       " << stats.min_guesses << " / " << stats.max_guesses << "\n";
        std::cout << "Pct <=37 (Daily): " << std::setprecision(1) << stats.pct_under_37 << "%\n";
        std::cout << "Pct <=32 (Perfect): " << stats.pct_under_32 << "%\n";

        std::cout << "\nDistribution (guesses used -> count):\n";
        for (size_t i = 0; i < stats.guess_distribution.size() - 1; ++i) {
            if (stats.guess_distribution[i] > 0) {
                std::cout << "  " << std::setw(3) << i << ": " << stats.guess_distribution[i] << "\n";
            }
        }
        if (stats.guess_distribution.back() > 0) {
            std::cout << "  FAIL: " << stats.guess_distribution.back() << "\n";
        }
    }
    if (strat_name == "beam") {
        std::cout << "\nBeam triggered: " << dt::g_beam_triggered.load()
                  << "  Beam skipped (used greedy): " << dt::g_beam_skipped.load() << "\n";
    }
    return 0;
}
