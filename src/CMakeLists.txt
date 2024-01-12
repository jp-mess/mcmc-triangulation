cmake_minimum_required(VERSION 3.10)
project(MCMC)

# C++ Standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Find the Eigen library
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# Add executable
add_executable(MCMC metropolis.cpp)

# Link Eigen to the executable
target_link_libraries(MCMC Eigen3::Eigen)
