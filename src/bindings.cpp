#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>
#include <string>
#include "tensor.h"

namespace py = pybind11;

PYBIND11_MODULE(adept, m) {
    m.doc() = "Adept - Fast numerical library for Python";

    py::class_<Tensor>(m, "Tensor")
        .def(py::init<std::vector<double>>())
        .def(py::init<std::vector<std::vector<double>>>())

        // ── Info ──────────────────────────────────────────
        .def("get", &Tensor::get, "Get element by flat index")
        .def("size", &Tensor::size, "Return total number of elements")
        .def("shape", &Tensor::shape, "Return size of each dimension")
        .def("ndim", &Tensor::ndim, "Return number of dimensions")

        // ── Math ──────────────────────────────────────────
        .def("sum", &Tensor::sum, "Return sum of all elements. Returns 0.0 if empty")
        .def("mean", &Tensor::mean, "Return mean of all elements. Raises if tensor is empty")
        .def("max", &Tensor::max, "Return maximum value. Raises if tensor is empty")
        .def("min", &Tensor::min, "Return minimum value. Raises if tensor is empty")
        .def("argmax", &Tensor::argmax, "Return index of maximum value. Raises if tensor is empty")
        .def("argmin", &Tensor::argmin, "Return index of minimum value. Raises if tensor is empty")
        .def("abs", &Tensor::abs, "Return absolute value of every element. Raises if tensor is empty")
        .def("clip", &Tensor::clip, "Clip every element to [min, max] range. Raises if tensor is empty or min > max")
        .def("pow", &Tensor::pow, "Raise every element to the power of n. Raises if tensor is empty or n < 0 and any element is zero")
        .def("sqrt", &Tensor::sqrt, "Square root of every element. Raises if tensor is empty or any element is negative")
        .def("stddev", &Tensor::stddev, "Return standard deviation of all elements. Raises if tensor is empty")
        .def("var", &Tensor::var, "Return variance of all elements. Raises if tensor is empty")
        .def("normalize", &Tensor::normalize, "Normalize every element to [0, 1] range. Raises if tensor is empty or all elements are equal")
        .def("log", &Tensor::log, "Natural log of every element. Raises if tensor is empty or any element is <= 0")

        // ── Operators (tensor) ────────────────────────────
        .def(py::self + py::self)
        .def(py::self - py::self)
        .def(py::self * py::self)
        .def(py::self / py::self)

        // ── Operators (scalar) ────────────────────────────
        .def(py::self + double())
        .def(py::self - double())
        .def(py::self * double())
        .def(py::self / double())

        // ── Operators (scalar reverse) ────────────────────
        .def("__radd__", [](const Tensor& t, double s) { return t + s; })
        .def("__rsub__", [](const Tensor& t, double s) { return t * -1.0 + s; })
        .def("__rmul__", [](const Tensor& t, double s) { return t * s; })
        .def("__rtruediv__", [](const Tensor& t, double s) {
            // s / t, bukan t / s
            std::vector<double> result;
            for (int i = 0; i < t.size(); i++) {
                if (t.get(i) == 0) throw std::invalid_argument("Division by zero");
                result.push_back(s / t.get(i));
            }
            return Tensor(result, t.shape());
        })

        // ── Tensor Operations ─────────────────────────────
        .def("reshape", &Tensor::reshape, "Reshape tensor to a new shape. Raises if ndim > 2 or element count mismatch")
        .def("flatten", &Tensor::flatten, "Flatten tensor to 1D")
        .def("transpose", &Tensor::transpose, "Transpose 2D tensor. Raises if tensor is not 2D")
        .def("__repr__", [](const Tensor& t) {
            // Helper: format satu angka, hapus trailing zeros
            auto fmt = [](double v) -> std::string {
                std::string s = std::to_string(v);
                // Hapus trailing zeros, tapi sisakan minimal satu desimal (1.0 bukan 1.)
                size_t dot = s.find('.');
                if (dot != std::string::npos) {
                    size_t last = s.find_last_not_of('0');
                    if (last == dot) last++;  // jaga minimal "1.0"
                    s = s.substr(0, last + 1);
                }
                return s;
            };

            std::vector<int> sh = t.shape();

            if (t.ndim() == 1) {
                std::string s = "Tensor([";
                for (int i = 0; i < t.size(); i++) {
                    s += fmt(t.get(i));
                    if (i < t.size() - 1) s += ", ";
                }
                s += "])";
                return s;
            } else {
                // 2D: print nested [[row0], [row1], ...]
                int rows = sh[0], cols = sh[1];
                std::string s = "Tensor([";
                for (int i = 0; i < rows; i++) {
                    s += "[";
                    for (int j = 0; j < cols; j++) {
                        s += fmt(t.get(i * cols + j));
                        if (j < cols - 1) s += ", ";
                    }
                    s += "]";
                    if (i < rows - 1) s += ", ";
                }
                s += "])";
                return s;
            }
        })
        .def("dot", &Tensor::dot, "Dot product for 1D tensors. Raises if not 1D or size mismatch")
        .def("matmul", &Tensor::matmul, "Matrix multiplication for 2D tensors. Raises if not 2D or inner dim mismatch");

    // ── Utility Functions ─────────────────────────────
    m.def("zeros", &zeros, "Create tensor filled with zeros given shape");
    m.def("ones", &ones, "Create tensor filled with ones given shape");

}