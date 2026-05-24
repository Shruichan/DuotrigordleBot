#pragma once

#include "game_state.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace dt {

// Tiny MLP value-function inference. Loads weights produced by
// scripts/train_value_net.py and evaluates V(state) = expected remaining
// turns to fully solve the game from this state.
//
// Layout (must match train_value_net.py dump_weights):
//   uint32 magic = 0xDDDDD000
//   uint32 feature_dim
//   uint32 num_layers
//   uint32 layer_sizes[num_layers + 1]
//   per layer: float32 W[out * in] (row-major, out-major), float32 b[out]
//   uint32 magic_end = 0xDDDDD001
class ValueNet {
public:
    ValueNet() = default;

    // Returns true on success, false if file missing / corrupt.
    bool load(const std::string& path);

    bool loaded() const { return !layer_sizes_.empty(); }
    int feature_dim() const { return feature_dim_; }

    // Compute the 25-dim compact feature vector for `state`.
    // Layout matches core/bench/gen_value_data.cpp::compute_features.
    static void compute_features(const GameState& state, float* out);

    // Same layout, but computed from per-board post-guess candidate counts
    // (0 if board is solved). guesses_used_post = guesses_used + 1.
    // Used by lookahead leaf eval to avoid materializing a GameState.
    static void compute_features_post(int guesses_used_post,
                                      const int* post_cnts,
                                      float* out);

    static constexpr int FEATURE_DIM = 25;

    // Forward pass: writes prediction to result; thread-safe (no shared state).
    // `scratch` must be at least max_layer_width floats; pass nullptr to use
    // an internal stack buffer (small networks only).
    float eval(const float* features) const;

private:
    int feature_dim_ = 0;
    std::vector<int> layer_sizes_;        // size = num_layers + 1
    std::vector<std::vector<float>> Ws_;  // Ws_[i]: out_features rows, in_features cols (row-major)
    std::vector<std::vector<float>> bs_;  // bs_[i]: out_features
};

}
