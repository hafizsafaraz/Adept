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
            std::string s = "Tensor([";
            for (int i = 0; i < t.size(); i++) {
                s += std::to_string(t.get(i));
                if (i < t.size() - 1) s += ", ";
            }
            s += "], shape=[";
            for (int i = 0; i < t.shape().size(); i++) {
                s += std::to_string(t.shape()[i]);
                if (i < t.shape().size() - 1) s += ", ";
            }
            s += "])";
            return s;
        });

    // ── Utility Functions ─────────────────────────────
    m.def("zeros", &zeros, "Create tensor filled with zeros given shape");
    m.def("ones", &ones, "Create tensor filled with ones given shape");

}