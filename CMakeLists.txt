cmake_minimum_required(VERSION 3.15)
project(acumen)

find_package(Python REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)

pybind11_add_module(acumen 
    src/tensor.cpp
    src/math.cpp
    src/linalg.cpp
    src/bindings.cpp
)

target_include_directories(acumen PRIVATE include)