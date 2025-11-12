cmake_minimum_required(VERSION 3.6.0...3.25.0)
project(coreio-cpp)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True) # Optional, but recommended
set(CMAKE_CXX_EXTENSIONS OFF) # Optional, for stricter standard compliance

enable_testing()

add_executable(
	${PROJECT_NAME}
	coreio/main.cpp
	coreio/hello.cpp
)

set_property(TARGET ${PROJECT_NAME} PROPERTY VS_DEBUGGER_WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

target_include_directories(
	${PROJECT_NAME}
	PUBLIC
	coreio/
)

target_include_directories(
	${PROJECT_NAME}
	PRIVATE
	"include/"
)

target_link_libraries(
	${PROJECT_NAME}
)

