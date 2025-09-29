cmake_minimum_required(VERSION 3.17)
project(nvshmem_comm_cuda LANGUAGES CXX CUDA)


set(CMAKE_CUDA_ARCHITECTURES 80)



# 1) Find Python interpreter + development headers (for <Python.h>)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

# 2) Ask Python where torch is installed
execute_process(
  COMMAND "${Python3_EXECUTABLE}" -c
    "import torch, os; print(os.path.dirname(torch.__file__))"
  OUTPUT_VARIABLE TORCH_ROOT
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
message(STATUS "Detected Torch root at: ${TORCH_ROOT}")

# 3) Tell CMake to search TorchConfig.cmake under Torch/share/cmake,
#    and also to look for libraries under Torch/lib
list(APPEND CMAKE_PREFIX_PATH    "${TORCH_ROOT}")
list(APPEND CMAKE_LIBRARY_PATH  "${TORCH_ROOT}/lib")

# 4) Pull in the Torch CMake config (libtorch, libc10, etc.)
find_package(Torch REQUIRED CONFIG)

# 5) Find the Python‐binding library that carries pybind11 casters
find_library(
  TORCH_PYTHON_LIBRARY
  NAMES torch_python torch_python3
  HINTS "${TORCH_ROOT}/lib"
  REQUIRED
)
message(STATUS "Found torch_python library at ${TORCH_PYTHON_LIBRARY}")

# 6) Find MPI (for mpi.h + MPI::MPI_CXX)
find_package(MPI REQUIRED COMPONENTS CXX)

# 7) Pick up NVSHMEM_HOME from environment (or via -DNVSHMEM_ROOT)
if(NOT DEFINED NVSHMEM_ROOT)
  if(DEFINED ENV{NVSHMEM_HOME})
    set(NVSHMEM_ROOT $ENV{NVSHMEM_HOME})
  else()
    message(FATAL_ERROR
      "NVSHMEM_ROOT not set. Either export NVSHMEM_HOME or pass -DNVSHMEM_ROOT=<path>.")
  endif()
endif()
message(STATUS "Using NVSHMEM_ROOT = ${NVSHMEM_ROOT}")

# 8) Build exactly one Python extension module
add_library(nvshmem_comm_cuda MODULE
    src/nvshmem_comm/nvshmem_comm.cpp
    src/nvshmem_comm/ll8_coll.cu
    src/nvshmem_comm/simple_coll.cu
    src/nvshmem_comm/nvshmem_comm_bindings.cpp   # your PYBIND11_MODULE(...)
)

# define TORCH_EXTENSION_NAME so pybind11 sees the right name:
target_compile_definitions(nvshmem_comm_cuda PRIVATE
  TORCH_EXTENSION_NAME=nvshmem_comm_cuda
)

set_target_properties(nvshmem_comm_cuda PROPERTIES
  PREFIX ""                              # no “lib” prefix
  SUFFIX ".so"                           # .so suffix for Python
  LINKER_LANGUAGE CUDA                   # let nvcc drive the link
  CXX_STANDARD 17
  CUDA_SEPARABLE_COMPILATION ON          # for NVSHMEM device calls
)

# 9) Include paths for Python, Torch, MPI, NVSHMEM
target_include_directories(nvshmem_comm_cuda PRIVATE
  ${Python3_INCLUDE_DIRS}                             # <Python.h>
  ${TORCH_INCLUDE_DIRS}                               # Torch C++ API
  ${MPI_CXX_INCLUDE_DIRS}                             # mpi.h
  ${NVSHMEM_ROOT}/include                             # NVSHMEM headers
  ${CMAKE_CURRENT_SOURCE_DIR}/include                 # Our headers
)

# 10) Link against Torch, Torch’s Python binding lib, CUDA, NVSHMEM, MPI
target_link_libraries(nvshmem_comm_cuda PRIVATE
  ${TORCH_LIBRARIES}                     # libtorch.so, libc10.so, …
  ${TORCH_PYTHON_LIBRARY}                # libtorch_python.so (pybind11 casters)
  ${CUDA_CUDART_LIBRARY}                 # shared libcudart.so
  ${NVSHMEM_ROOT}/lib/libnvshmem_host.so
  ${NVSHMEM_ROOT}/lib/libnvshmem_device.a
  MPI::MPI_CXX                           # MPI C++ library
)

# 11) At runtime look next to the .so for libc10.so, libmpi, etc.
set_target_properties(nvshmem_comm_cuda PROPERTIES
  INSTALL_RPATH "$ORIGIN"
)
