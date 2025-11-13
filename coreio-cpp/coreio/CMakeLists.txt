add_executable(
  ${PROJECT_NAME}
  main.cpp
  hello.cpp
)

set_property(
  TARGET
  ${PROJECT_NAME}
  PROPERTY VS_DEBUGGER_WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)

target_include_directories(
  ${PROJECT_NAME}
  PUBLIC
  ../include/
)

target_link_libraries(
  ${PROJECT_NAME}
)

if(ENABLE_COVERAGE)
  target_link_libraries(${PROJECT_NAME} PRIVATE coverage_flags)
endif()
