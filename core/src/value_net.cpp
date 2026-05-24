#include "value_net.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>

namespace dt {

namespace {
constexpr uint32_t MAGIC_START = 0xDDDDD000u;
constexpr uint32_t MAGIC_END   = 0xDDDDD001u;
}

bool ValueNet::load(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    uint32_t magic = 0, feat_dim = 0, num_layers = 0;
    f.read(reinterpret_cast<char*>(&magic),      4);
    f.read(reinterpret_cast<char*>(&feat_dim),   4);
    f.read(reinterpret_cast<char*>(&num_layers), 4);
    if (!f || magic != MAGIC_START) {
        std::cerr << "ValueNet: bad magic at start: 0x" << std::hex << magic << std::dec << "\n";
        return false;
    }
    if (feat_dim != FEATURE_DIM) {
        std::cerr << "ValueNet: feature_dim mismatch: file=" << feat_dim
                  << " expected=" << FEATURE_DIM << "\n";
        return false;
    }
    feature_dim_ = static_cast<int>(feat_dim);
    layer_sizes_.assign(num_layers + 1, 0);
    for (uint32_t i = 0; i < num_layers + 1; ++i) {
        uint32_t s = 0;
        f.read(reinterpret_cast<char*>(&s), 4);
        layer_sizes_[i] = static_cast<int>(s);
    }
    Ws_.clear(); bs_.clear();
    Ws_.reserve(num_layers); bs_.reserve(num_layers);
    for (uint32_t li = 0; li < num_layers; ++li) {
        const int in_f  = layer_sizes_[li];
        const int out_f = layer_sizes_[li + 1];
        std::vector<float> W(static_cast<size_t>(in_f) * out_f);
        std::vector<float> b(out_f);
        f.read(reinterpret_cast<char*>(W.data()), W.size() * sizeof(float));
        f.read(reinterpret_cast<char*>(b.data()), b.size() * sizeof(float));
        if (!f) {
            std::cerr << "ValueNet: short read on layer " << li << "\n";
            layer_sizes_.clear(); Ws_.clear(); bs_.clear(); return false;
        }
        Ws_.push_back(std::move(W));
        bs_.push_back(std::move(b));
    }
    uint32_t end_magic = 0;
    f.read(reinterpret_cast<char*>(&end_magic), 4);
    if (end_magic != MAGIC_END) {
        std::cerr << "ValueNet: bad magic at end\n";
        layer_sizes_.clear(); Ws_.clear(); bs_.clear();
        return false;
    }
    return true;
}

void ValueNet::compute_features(const GameState& state, float* out) {
    std::fill(out, out + FEATURE_DIM, 0.0f);
    int total_cands = 0;
    int num_active = 0;
    int num_solved = 0;
    int max_c = 0, min_c = INT32_MAX;
    double sum_log = 0.0;
    int c1 = 0, c2 = 0, c3 = 0, c4_10 = 0, c11_30 = 0, c30p = 0;
    std::vector<int> sizes;
    sizes.reserve(NUM_BOARDS);
    for (int b = 0; b < NUM_BOARDS; ++b) {
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

void ValueNet::compute_features_post(int guesses_used_post,
                                     const int* post_cnts,
                                     float* out) {
    std::fill(out, out + FEATURE_DIM, 0.0f);
    int total_cands = 0;
    int num_active = 0;
    int num_solved = 0;
    int max_c = 0, min_c = INT32_MAX;
    double sum_log = 0.0;
    int c1 = 0, c2 = 0, c3 = 0, c4_10 = 0, c11_30 = 0, c30p = 0;
    std::vector<int> sizes;
    sizes.reserve(NUM_BOARDS);
    for (int b = 0; b < NUM_BOARDS; ++b) {
        int k = post_cnts[b];
        if (k <= 0) { ++num_solved; continue; }
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

    out[0] = static_cast<float>(guesses_used_post) / 37.0f;
    out[1] = static_cast<float>(37 - guesses_used_post) / 37.0f;
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

float ValueNet::eval(const float* features) const {
    // Two ping-pong buffers; max width of any layer in [in, h1, h2, out]
    // is the largest middle hidden, plus the input dim. Use stack-allocated
    // arrays sized generously; nets are small.
    constexpr int MAX_WIDTH = 512;
    std::array<float, MAX_WIDTH> cur{}, nxt{};
    const int num_layers = static_cast<int>(Ws_.size());
    std::copy(features, features + layer_sizes_[0], cur.begin());

    for (int li = 0; li < num_layers; ++li) {
        const int in_f  = layer_sizes_[li];
        const int out_f = layer_sizes_[li + 1];
        const float* W = Ws_[li].data();
        const float* b = bs_[li].data();
        for (int o = 0; o < out_f; ++o) {
            float s = b[o];
            const float* w_row = W + static_cast<size_t>(o) * in_f;
            for (int i = 0; i < in_f; ++i) s += w_row[i] * cur[i];
            // ReLU on all but the final layer.
            if (li + 1 < num_layers) s = s > 0.0f ? s : 0.0f;
            nxt[o] = s;
        }
        std::swap(cur, nxt);
    }
    return cur[0];
}

}
