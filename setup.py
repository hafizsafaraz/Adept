from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension

ext_modules = [
    Pybind11Extension(
        "adept",
        ["src/tensor.cpp", "src/bindings.cpp"],
        include_dirs=["include"],
        extra_compile_args=["/O2", "/arch:AVX2"],
    ),
]

setup(
    name="adept",
    version="0.1.1",
    ext_modules=ext_modules,
)