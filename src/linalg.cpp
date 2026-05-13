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

Tensor Tensor::dot(const Tensor& other) const {
    if(ndim() == 1 && other.ndim() == 1) {
        if(size() != other.size()) {
            throw std::invalid_argument(
                "Shape mismatch: " + shape_to_str(shape_) + " != " + shape_to_str(other.shape_)
            );
        } 
        double result = 0;
        for(int i = 0; i < size(); i++) {
            result += data[i] * other.data[i];
        }

        return Tensor({result}, {1});
    }

    if(ndim() == 2 && other.ndim() == 2) {
        if(shape_[1] != other.shape_[0]) {
            throw std::invalid_argument(
                "Shape mismatch: " + shape_to_str(shape_) + " != " + shape_to_str(other.shape_)
            );
        }

        int rows = shape_[0];
        int cols = other.shape_[1];
        int inner = shape_[1];

        std::vector<double> result(rows * cols, 0.0);

        for(int i = 0; i < rows; i++) {
            for(int j = 0; j < cols; j++) {
                for(int k = 0; k < inner; k++) {
                    result[i * cols + j] += data[i * inner + k] * other.data[k * cols + j];
                }
            }
        }

        return Tensor(result, {rows, cols});
    } 

    
    throw std::invalid_argument("dot() only supported for 1D and 2D tensors");
    
}