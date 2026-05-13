from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension

compile_args = ["-02"]

ext_modules = [
    Pybind11Extension(
        "adept",
        ["src/tensor.cpp", "src/bindings.cpp"],
        include_dirs=["include"], 
        extra_compile_args=compile_args,
        cxx_std=17,
    ),
]

setup(
    ext_modules=ext_modules,
)