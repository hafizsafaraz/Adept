#include "tensor.h"
#include <stdexcept>
#include <cmath>
#include <vector>

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

int Tensor::argmax() const {
    if(data.empty()) {
        throw std::invalid_argument("Tensor is empty");
    }

    int max_index = 0;
    for(int i = 0; i < size(); i++) {
        if(data[i] > data[max_index]) {
            max_index = i;
        }
    }
    return max_index;
}

int Tensor::argmin() const {
    if(data.empty()) {
        throw std::invalid_argument("Tensor is empty");
    }

    int min_index = 0;
    for(int i = 0; i < size(); i++) {
        if(data[i] < data[min_index]) {
            min_index = i;
        }
    }
    return min_index;
}

Tensor Tensor::abs() const {
    if(data.empty()) {
        throw std::invalid_argument("Tensor is empty");
    }

    std::vector<double> new_tensor;
    new_tensor.reserve(size());
    for(int i = 0; i < size(); i++) { 
        new_tensor.push_back(std::abs(data[i]));
    } 
    return Tensor(new_tensor, shape_);
}

Tensor Tensor::clip(double min, double max) const {
    if(data.empty()) {
        throw std::invalid_argument("Tensor is empty");
    }

    if(min > max) {
        throw std::invalid_argument("min cannot be greater than max");
    }

    std::vector<double> new_tensor;
    new_tensor.reserve(size());
    for(int i = 0; i < size(); i++) {
        if(data[i] < min) {
            new_tensor.push_back(min);
        } else if(data[i] > max) {
            new_tensor.push_back(max);
        } else {
            new_tensor.push_back(data[i]);
        }
    }
    return Tensor(new_tensor, shape_);
}

Tensor Tensor::pow(int n) const {
    if(data.empty()) {
        throw std::invalid_argument("Tensor is empty");
    }

    std::vector<double> new_tensor;
    new_tensor.reserve(size());
    for(int i = 0; i < size(); i++) {
        if(n < 0 && data[i] == 0) {
            throw std::invalid_argument("Cannot raise zero to a negative power");
        }
        new_tensor.push_back(std::pow(data[i], n));
    }
    return Tensor(new_tensor, shape_);

}

Tensor Tensor::sqrt() const {
    if(data.empty()) {
        throw std::invalid_argument("Tensor is empty");
    }

    std::vector<double> new_tensor;
    new_tensor.reserve(size());
    for(int i = 0; i < size(); i++) {
        if(data[i] < 0) {
            throw std::invalid_argument("Cannot take square root of a negative number");
        }
        new_tensor.push_back(std::sqrt(data[i]));
    }
    return Tensor(new_tensor, shape_);
}

double Tensor::stddev() const {
    if(data.empty()) {
        throw std::invalid_argument("Tensor is empty");
    }

    double mean_val = mean();
    double variance = 0;
    for(int i = 0; i < size(); i++) {
        variance += std::pow((data[i] - mean_val), 2);
    }

    variance = variance / size();

    return std::sqrt(variance);
}

double Tensor::var() const {
    if(data.empty()) {
        throw std::invalid_argument("Tensor is empty");
    }

    return std::pow(stddev(), 2);
}

Tensor Tensor::normalize() const {
    if(data.empty()) {
        throw std::invalid_argument("Tensor is empty");
    }

    if(max() == min()) {
        throw std::invalid_argument("Cannot normalize tensor with all equal elements");
    }

    double max_val = max();
    double min_val = min();
    double range = max_val - min_val;

    std::vector<double> new_tensor;
    new_tensor.reserve(size());
    for(int i = 0; i < size(); i++) {

        new_tensor.push_back((data[i] - min_val) / range);
    }
    return Tensor(new_tensor, shape_);
}

Tensor Tensor::log() const {
    if(data.empty()) {
        throw std::invalid_argument("Tensor is empty");
    }
    
    std::vector<double> new_tensor;
    new_tensor.reserve(size());
    for(int i = 0; i < size(); i++) {
        if(data[i] <= 0) {
            throw std::invalid_argument("Cannot take logarithm of a non-positive number");
        }
        new_tensor.push_back(std::log(data[i]));
    }

    return Tensor(new_tensor, shape_);
}