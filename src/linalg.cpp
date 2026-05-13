#include "tensor.h"
#include <stdexcept>
#include <numeric>

// ── Tensor operations ─────────────────────────────

// Reshape to an arbitrary new shape
// Throws if total number of elements does not match
Tensor Tensor::reshape(std::vector<int> new_shape) const {
    int total_new_shape = std::accumulate(
        new_shape.begin(), 
        new_shape.end(), 1, 
        std::multiplies<int>());

    if (total_new_shape != size()) {
        throw std::invalid_argument(
            "Cannot reshape tensor of size " + std::to_string(size()) +
            " into shape of size " + std::to_string(total_new_shape)
        );
    }

    return Tensor(data, new_shape);
}

// Flatten to 1D
Tensor Tensor::flatten() const {
    std::vector<int> new_shape = {size()};
    return Tensor(data, new_shape);
}

// Transpose (2D only)
// Throws if tensor is not 2D
Tensor Tensor::transpose() const {
    if(ndim() != 2) {
        throw std::invalid_argument("Transpose only supported for 2D tensors");
    }
    std::vector<double> result(size(), 0);
    for(int i = 0; i < shape_[0]; i++) {
        for(int j = 0; j < shape_[1]; j++) {
            result[j * shape_[0] + i] = data[i * shape_[1] + j];
        }
    }
    std::vector<int> new_shape = {shape_[1], shape_[0]};
    return Tensor(result, new_shape);
}