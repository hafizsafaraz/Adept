from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension
import sys

if sys.platform == "win32":
    compile_args = ["/O2", "/arch:AVX2"]
else:
    compile_args = ["-O2", "-mavx2"]   

ext_modules = [
    Pybind11Extension(
        "adept",
        ["src/tensor.cpp", "src/bindings.cpp"],
        include_dirs=["include"], 
        extra_compile_args=compile_args,
    ),
]

setup(
    name="adept",
    version="0.1.1",
    ext_modules=ext_modules,
)