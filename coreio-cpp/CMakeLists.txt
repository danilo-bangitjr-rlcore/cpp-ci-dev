cmake_minimum_required(VERSION 3.25)
project(coreio-cpp CXX)

option(ENABLE_COVERAGE "Build with gcovr instrumentation" OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True) # Optional, but recommended
set(CMAKE_CXX_EXTENSIONS OFF) # Optional, for stricter standard compliance

if(ENABLE_COVERAGE)
  message(STATUS "Coverage instrumentation ENABLED")

  add_library(coverage_flags INTERFACE)
  target_compile_options(coverage_flags INTERFACE -O0 -g --coverage)
  target_compile_options(coverage_flags INTERFACE -fprofile-arcs)
  target_compile_options(coverage_flags INTERFACE -ftest-coverage)
  target_link_options(coverage_flags INTERFACE --coverage)
endif()

include(FetchContent)
FetchContent_Declare(
  GoogleTest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        v1.17.0
)

# Source: https://google.github.io/googletest/quickstart-cmake.html
# For Windows: Prevent overriding the parent project's compiler/linker settings
# TODO: Fix lint error about this line
# set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(GoogleTest)

enable_testing()

include_directories(${CMAKE_SOURCE_DIR}/include)

add_subdirectory(coreio)
add_subdirectory(tests)
