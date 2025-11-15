#include <gtest/gtest.h>

#include "../include/hello.h"

// Define a test case: Suite Name "HelloTest", Test Name "BasicAssertions"
TEST(HelloTest, BasicAssertions) {
  std::string expected = "Hello, World!";
  std::string actual = hello_world_message();

  // Google Test assertion for equality
  EXPECT_EQ(expected, actual);
}