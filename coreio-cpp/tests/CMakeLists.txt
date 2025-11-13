
# Add the test executable
add_executable(
    hello_test
    hello_test.cpp
    ../coreio/hello.cpp
)
target_link_libraries(hello_test PRIVATE GTest::gtest GTest::gtest_main)

# Necessary to call include before using gtest_discover_tests
include(GoogleTest)

# Register tests with CTest
gtest_discover_tests(hello_test)

# Register the test with CTest
# add_test(NAME HelloWorldTest COMMAND hello_test)
