#include "tensor.h"
#include <stdexcept>

// ── Math ──────────────────────────────────────────

// Sum of all elements
double Tensor::sum() const {
    if(data.empty()) {
        throw std::invalid_argument("Tensor is empty");
    }
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