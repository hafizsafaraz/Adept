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