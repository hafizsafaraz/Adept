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
        .def("sum", &Tensor::sum, "Return sum of all elements")
        .def("mean", &Tensor::mean, "Return mean of all elements")
        .def("max", &Tensor::max, "Return maximum value. Raises if tensor is empty")
        .def("min", &Tensor::min, "Return minimum value. Raises if tensor is empty")

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

        // ── Tensor Operations ─────────────────────────────
        .def("reshape", &Tensor::reshape, "Reshape tensor to a new shape. Raises if total elements do not match")
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
        });

    // ── Utility Functions ─────────────────────────────
    m.def("zeros", &zeros, "Create tensor filled with zeros given shape");
    m.def("ones", &ones, "Create tensor filled with ones given shape");

}