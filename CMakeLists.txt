cmake_minimum_required(VERSION 3.1.0)
project(polyscope-py)

# Gather dependencies
set(EIGEN3_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/deps/eigen/")

# Recurse in to pybind
add_subdirectory(deps/pybind11)

# We need polyscope
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_subdirectory(deps/polyscope)

pybind11_add_module(polyscope_bindings 
  src/cpp/core.cpp
  src/cpp/surface_mesh.cpp
  src/cpp/point_cloud.cpp
  src/cpp/curve_network.cpp
  src/cpp/volume_mesh.cpp
  src/cpp/volume_grid.cpp
  src/cpp/floating_quantities.cpp
  src/cpp/implicit_surface.cpp
  src/cpp/imgui.cpp
)

target_include_directories(polyscope_bindings PUBLIC "${EIGEN3_INCLUDE_DIR}")
target_link_libraries(polyscope_bindings PRIVATE polyscope)


# To run with the address sanitizer, compile with these flags, then call (via GCC) like:
# LD_PRELOAD=$(gcc -print-file-name=libasan.so) python3 polyscope_demo.py ~/mesh/spot.obj
# (there is probably a better way to do this, but this one works)
#target_compile_options(polyscope_bindings PUBLIC -fsanitize=address)
#target_compile_options(polyscope PUBLIC -fsanitize=address)
#target_link_options(polyscope_bindings PUBLIC -fsanitize=address -static-libasan)
#target_link_options(polyscope PUBLIC -fsanitize=address -static-libasan)

# make the resulting target actually be called polyscope, without the py
# set_target_properties(polyscope_py PROPERTIES OUTPUT_NAME polyscope)
