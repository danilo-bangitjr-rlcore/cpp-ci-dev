
# Add the test executable
add_executable(
    coreio_test
    hello_test.cpp
    ../coreio/hello.cpp
)
target_link_libraries(coreio_test PRIVATE GTest::gtest GTest::gtest_main)
if(ENABLE_COVERAGE)
  target_link_libraries(coreio_test PRIVATE coverage_flags)
endif()

# Necessary to call include before using gtest_discover_tests
include(GoogleTest)

# Register tests with CTest
gtest_discover_tests(coreio_test)

# Add a custom target for generating coverage reports (if ENABLE_COVERAGE is ON)
if(ENABLE_COVERAGE)
    find_program(GCOVR gcovr)
    if(GCOVR)
        add_custom_target(coverage_report
            COMMAND ${CMAKE_CTEST_COMMAND} 
                --output-on-failure 
                --test-dir ${CMAKE_CURRENT_BINARY_DIR}

            # Make output dir
            COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/report

            # Generate reports
            COMMAND ${GCOVR}
                    --root=${CMAKE_SOURCE_DIR}
                    --object-directory=${CMAKE_BINARY_DIR}
                    --exclude="${CMAKE_BINARY_DIR}/_deps/googletest-src/*"
                    --exclude="${CMAKE_SOURCE_DIR}/tests"
                    --exclude-unreachable-branches
                    --print-summary
                    --html-details
                    --html-title "${PROJECT_NAME} Coverage"
                    --output ${CMAKE_BINARY_DIR}/report/index.html

            COMMAND ${GCOVR}
                    --root=${CMAKE_SOURCE_DIR}
                    --object-directory=${CMAKE_BINARY_DIR}
                    --exclude="${CMAKE_BINARY_DIR}/_deps/googletest-src/*"
                    --exclude="${CMAKE_SOURCE_DIR}/tests"
                    --exclude-unreachable-branches
                    --print-summary
                    --xml
                    --output ${CMAKE_BINARY_DIR}/report/coverage.xml

            COMMENT "Generating code coverage report..."
        )
    else()
        message(WARNING "gcovr not found. Code coverage report generation will be skipped.")
    endif()

    # TODO: Do we also want to add a target to generate LCOV reports?
    # Alternative coverage target using lcov and genhtml
    # find_program(LCOV NAMES lcov)
    # find_program(GENHTML NAMES genhtml)

    # if(LCOV AND GENHTML)
    # add_custom_target(coverage
    #     COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/coverage

    #     # 1. Zero counters
    #     COMMAND ${LCOV} --directory ${CMAKE_BINARY_DIR} --zerocounters

    #     # 2. Run all tests (via CTest)
    #     COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure

    #     # 3. Capture coverage info
    #     COMMAND ${LCOV} --directory ${CMAKE_BINARY_DIR} --capture
    #                     --output-file ${CMAKE_BINARY_DIR}/coverage.info
    #                     --rc lcov_branch_coverage=1

    #     # 4. Remove system / test files
    #     COMMAND ${LCOV} --remove ${CMAKE_BINARY_DIR}/coverage.info
    #                     '/usr/*' '${CMAKE_SOURCE_DIR}/tests/*' '${CMAKE_BINARY_DIR}/*'
    #                     --output-file ${CMAKE_BINARY_DIR}/coverage.filtered.info
    #                     --rc lcov_branch_coverage=1

    #     # 5. Generate HTML
    #     COMMAND ${GENHTML} ${CMAKE_BINARY_DIR}/coverage.filtered.info
    #                     --branch-coverage
    #                     --output-directory ${CMAKE_BINARY_DIR}/coverage
    #                     --title "MyProject Code Coverage"

    #     COMMENT "Generating HTML coverage report in ${CMAKE_BINARY_DIR}/coverage"
    # )
    # else()
    # message(WARNING "lcov or genhtml not found – coverage target disabled")
    # endif()

endif()
