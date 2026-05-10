#include "tensor.h"
#include <vector>
#include <stdexcept>
#include <string>

// Constructor: from flat data + shape
Tensor::Tensor(std::vector<double> data_tensor, std::vector<int> shape_tensor) {
    data = data_tensor;
    shape_ = shape_tensor;
}

// Constructor: from 1D list
Tensor::Tensor(std::vector<double> input) {
    data = input;
    shape_.push_back(input.size());
}

// Constructor: from 2D list, stored flat
Tensor::Tensor(std::vector<std::vector<double>> input) {
    for(int i = 0; i < input.size(); i++) {
        for(int j = 0; j < input[0].size(); j++) {
            data.push_back(input[i][j]);
        }
    }
    shape_.push_back(input.size());    
    shape_.push_back(input[0].size());
}

// Get element by flat index
double Tensor::get(int a) const {
    return data[a];
}

// Total number of elements
int Tensor::size() const {
    return data.size();
}

// Size of each dimension
std::vector<int> Tensor::shape() const {
    return shape_;
}

// Number of dimensions
int Tensor::ndim() const {
    return shape_.size();
}

// ── Math ──────────────────────────────────────────

// Sum of all elements
double Tensor::sum() const {
    double sum = 0;
    for(int i = 0; i < size(); i++) {
        sum += data[i];
    }
    return sum;
}

// Average of all elements
double Tensor::mean() const {
    return Tensor::sum() / size();
}

// Maximum value
double Tensor::max() const {
    double max = data[0];
    for(int i = 0; i < size(); i++) {
        if(data[i] > max) {
            max = data[i];
        }
    }
    return max;
}

// Minimum value
double Tensor::min() const {
    double min = data[0];
    for(int i = 0; i < size(); i++) {
        if(data[i] < min) {
            min = data[i];
        }
    }
    return min;
}

// ── Operators (tensor) ────────────────────────────

Tensor Tensor::operator+(const Tensor& other) const {
    std::vector<double> new_tensor;
    for(int i = 0; i < size(); i++) {
        new_tensor.push_back(data[i] + other.data[i]);
    }
    return Tensor(new_tensor, shape_);
}

Tensor Tensor::operator-(const Tensor& other) const {
    std::vector<double> new_tensor;
    for(int i = 0; i < size(); i++) {
        new_tensor.push_back(data[i] - other.data[i]);
    }
    return Tensor(new_tensor, shape_);
}

Tensor Tensor::operator*(const Tensor& other) const {
    std::vector<double> new_tensor;
    for(int i = 0; i < size(); i++) {
        new_tensor.push_back(data[i] * other.data[i]);
    }
    return Tensor(new_tensor, shape_);
}

Tensor Tensor::operator/(const Tensor& other) const {
    std::vector<double> new_tensor;
    for(int i = 0; i < size(); i++) {
        new_tensor.push_back(data[i] / other.data[i]);
    }
    return Tensor(new_tensor, shape_);
}

// ── Operators (scalar) ────────────────────────────

Tensor Tensor::operator+(double scalar) const {
    std::vector<double> new_tensor;
    for(int i = 0; i < size(); i++) {
        new_tensor.push_back(data[i] + scalar);
    }
    return Tensor(new_tensor, shape_);
}

Tensor Tensor::operator-(double scalar) const {
    std::vector<double> new_tensor;
    for(int i = 0; i < size(); i++) {
        new_tensor.push_back(data[i] - scalar);
    }
    return Tensor(new_tensor, shape_);
}

Tensor Tensor::operator*(double scalar) const {
    std::vector<double> new_tensor;
    for(int i = 0; i < size(); i++) {
        new_tensor.push_back(data[i] * scalar);
    }
    return Tensor(new_tensor, shape_);
}

Tensor Tensor::operator/(double scalar) const {
    std::vector<double> new_tensor;
    for(int i = 0; i < size(); i++) {
        new_tensor.push_back(data[i] / scalar);
    }
    return Tensor(new_tensor, shape_);
}

// ── Tensor operations ─────────────────────────────

// Reshape to new_rows x new_cols
Tensor Tensor::reshape(int new_rows, int new_cols) const {
    if(new_rows * new_cols != size()) {
        throw std::invalid_argument(
            "Cannot reshape tensor of size " + std::to_string(size()) + 
            " into shape (" + std::to_string(new_rows) + 
            ", " + std::to_string(new_cols) + ")"
        );
    }
    std::vector<int> new_shape = {new_rows, new_cols};
    return Tensor(data, new_shape);
}

// Flatten to 1D
Tensor Tensor::flatten() const {
    std::vector<int> new_shape = {size()};
    return Tensor(data, new_shape);
}

// Transpose (2D only)
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

// ── Utility functions ─────────────────────────────

// Create tensor filled with zeros
Tensor zeros(std::vector<int> shape) {
    int total = 1;
    for(int i = 0; i < shape.size(); i++) {
        total *= shape[i];
    }
    std::vector<double> data(total, 0.0);
    return Tensor(data, shape);
}

// Create tensor filled with ones
Tensor ones(std::vector<int> shape) {
    int total = 1;
    for(int i = 0; i < shape.size(); i++) {
        total *= shape[i];
    }
    std::vector<double> data(total, 1.0);
    return Tensor(data, shape);
}