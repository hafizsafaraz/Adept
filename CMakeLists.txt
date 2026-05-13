cmake_minimum_required(VERSION 3.15)
project(adept)

find_package(Python REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)

pybind11_add_module(adept 
    src/tensor.cpp
    src/math.cpp
    src/linalg.cpp
    src/bindings.cpp
)

target_include_directories(adept PRIVATE include)