cmake_minimum_required(VERSION 3.6.0...3.25.0)
project(coreio-cpp CXX)

option(ENABLE_COVERAGE "Build with gcovr instrumentation" OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True) # Optional, but recommended
set(CMAKE_CXX_EXTENSIONS OFF) # Optional, for stricter standard compliance

if(ENABLE_COVERAGE)
  message(STATUS "Coverage instrumentation ENABLED")
  add_compile_options(-O0 -g --coverage -fprofile-arcs -ftest-coverage)
  add_link_options(--coverage)
endif()

include(FetchContent)
FetchContent_Declare(
  GoogleTest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        v1.17.0 # Or a specific tag like 'release-1.12.1' for a stable version  
)

# Source: https://google.github.io/googletest/quickstart-cmake.html
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(GoogleTest)

enable_testing()

add_subdirectory(coreio)
add_subdirectory(tests)
