# ═══════════════════════════════════════════════════════════════════════
# CMake build for PINN Burgers 1D — C++ Inference
# ═══════════════════════════════════════════════════════════════════════
#
# Builds the pinn_inference executable, which loads a TorchScript model
# via LibTorch and runs inference on (x, t) grids.
#
# Build on Aurora:
#   module load frameworks cmake
#   mkdir -p build && cd build
#   cmake .. \
#       -DCMAKE_CXX_COMPILER=icpx \
#       -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')"
#   make -j 4
#
# Build on a workstation with LibTorch installed:
#   mkdir -p build && cd build
#   cmake .. -DCMAKE_PREFIX_PATH=/path/to/libtorch
#   make -j 4

cmake_minimum_required(VERSION 3.18)
project(pinn_burgers_inference LANGUAGES CXX)

# LibTorch requires C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── Find LibTorch ────────────────────────────────────────────────────
# On Aurora after `module load frameworks`, pass:
#   -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')"
find_package(Torch REQUIRED)

message(STATUS "LibTorch found:     ${TORCH_INSTALL_PREFIX}")
message(STATUS "Torch libraries:    ${TORCH_LIBRARIES}")
message(STATUS "Torch CXX flags:    ${TORCH_CXX_FLAGS}")

# Apply Torch CXX flags — critical for ABI compatibility
# (includes _GLIBCXX_USE_CXX11_ABI setting that must match LibTorch)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# ── Build the inference executable ───────────────────────────────────
add_executable(pinn_inference src/main.cpp)
target_link_libraries(pinn_inference "${TORCH_LIBRARIES}")
target_include_directories(pinn_inference PRIVATE "${TORCH_INCLUDE_DIRS}")

# ── RPATH handling (Linux) ───────────────────────────────────────────
# Embed the path to LibTorch's shared libraries into the executable
# so it finds them at runtime without requiring LD_LIBRARY_PATH.
if(UNIX AND NOT APPLE)
    set_target_properties(pinn_inference PROPERTIES
        INSTALL_RPATH_USE_LINK_PATH TRUE
        BUILD_RPATH "${TORCH_INSTALL_PREFIX}/lib"
    )
endif()
