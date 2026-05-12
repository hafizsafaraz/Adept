#include "tensor.h"
#include <vector>
#include <stdexcept>
#include <string>
#include <numeric>

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

// Constructor: from 2D list, stored flat row-major
// Throws if rows have different lengths (jagged array)
Tensor::Tensor(std::vector<std::vector<double>> input) {
    for(int i = 0; i < input.size(); i++) {
        if(input[i].size() != input[0].size()) {
            throw std::invalid_argument("All rows must have the same length");
        }
        for(int j = 0; j < input[0].size(); j++) {
            data.push_back(input[i][j]);
        }
    }
    shape_.push_back(input.size());    
    shape_.push_back(input[0].size());
}

// Get element by flat index
double Tensor::get(int a) const {
    if(a < 0 || a >= size()) {
        throw std::out_of_range(
            "Index " + std::to_string(a) + " out of range for tensor of size " + std::to_string(size())
        );
    }
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
    if(data.empty()) {
        throw std::invalid_argument("Tensor is empty");
    }
    return Tensor::sum() / size();
}

// Maximum value
// Throws if tensor is empty
double Tensor::max() const {
    if(data.empty()) {
        throw std::invalid_argument("Tensor is empty");
    }
    double max = data[0];
    for(int i = 0; i < size(); i++) {
        if(data[i] > max) {
            max = data[i];
        }
    }
    return max;
}

// Minimum value
// Throws if tensor is empty
double Tensor::min() const {
    if(data.empty()) {
        throw std::invalid_argument("Tensor is empty");
    }    
    double min = data[0];
    for(int i = 0; i < size(); i++) {
        if(data[i] < min) {
            min = data[i];
        }
    }
    return min;
}

// ── Helper functions ──────────────────────────────

std::string Tensor::shape_to_str(std::vector<int> s) const {
    std::string str = "(";
    for (int i = 0; i < s.size(); i++) {
        str += std::to_string(s[i]);
        if (i < s.size() - 1) str += ", ";
    }
    str += ")";
    return str;
}

// ── Operators (tensor) ────────────────────────────
// Element-wise operations; throws if tensors have different sizes

Tensor Tensor::operator+(const Tensor& other) const {
    if (size() != other.size()) {
        throw std::invalid_argument(
            "Shape mismatch: " + shape_to_str(shape_) + " != " + shape_to_str(other.shape_)
        );
    }
    std::vector<double> new_tensor;
    for(int i = 0; i < size(); i++) {
        new_tensor.push_back(data[i] + other.data[i]);
    }
    return Tensor(new_tensor, shape_);
}

Tensor Tensor::operator-(const Tensor& other) const {
    if (size() != other.size()) {
        throw std::invalid_argument(
            "Shape mismatch: " + shape_to_str(shape_) + " != " + shape_to_str(other.shape_)
        );
    }
    std::vector<double> new_tensor;
    for(int i = 0; i < size(); i++) {
        new_tensor.push_back(data[i] - other.data[i]);
    }
    return Tensor(new_tensor, shape_);
}

Tensor Tensor::operator*(const Tensor& other) const {
    if (size() != other.size()) {
        throw std::invalid_argument(
            "Shape mismatch: " + shape_to_str(shape_) + " != " + shape_to_str(other.shape_)
        );
    }  
    std::vector<double> new_tensor;
    for(int i = 0; i < size(); i++) {
        new_tensor.push_back(data[i] * other.data[i]);
    }
    return Tensor(new_tensor, shape_);
}

Tensor Tensor::operator/(const Tensor& other) const {
    if (size() != other.size()) {
        throw std::invalid_argument(
            "Shape mismatch: " + shape_to_str(shape_) + " != " + shape_to_str(other.shape_)
        );
    }
    std::vector<double> new_tensor;
    for(int i = 0; i < size(); i++) {
        if(other.data[i] == 0) {
            throw std::invalid_argument("Division by zero");
        }
        new_tensor.push_back(data[i] / other.data[i]);
    }
    return Tensor(new_tensor, shape_);
}

// ── Operators (scalar) ────────────────────────────
// Apply scalar to every element

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

// ── Utility functions ─────────────────────────────

// Create tensor filled with zeros given shape
Tensor zeros(std::vector<int> shape) {
    int total = 1;
    for(int i = 0; i < shape.size(); i++) {
        total *= shape[i];
    }
    std::vector<double> data(total, 0.0);
    return Tensor(data, shape);
}

// Create tensor filled with ones given shape
Tensor ones(std::vector<int> shape) {
    int total = 1;
    for(int i = 0; i < shape.size(); i++) {
        total *= shape[i];
    }
    std::vector<double> data(total, 1.0);
    return Tensor(data, shape);
}