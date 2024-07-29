cmake_minimum_required(VERSION 3.16.0)
project(hops LANGUAGES C CXX)

include(cmake/CPM.cmake)

# library
add_library(hops src/hops.cpp src/kernel.cpp src/memory.cpp)
target_compile_features(hops PUBLIC cxx_std_23)
set_target_properties(hops PROPERTIES CXX_EXTENSIONS ON)
target_include_directories(hops PUBLIC include)
target_link_libraries(hops PUBLIC nvrtc_static nvrtc-builtins_static nvptxcompiler_static cuda)

# unittests
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
	CPMAddPackage(
		NAME Catch2
		GITHUB_REPOSITORY catchorg/Catch2
		GIT_TAG 2ab20a0e008845e02bd06248e61ca6e5ad1aba33 # v3.3.1
	)

	# add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
	# add_link_options(-fsanitize=address)
	add_executable(tests tests/saxpy.cpp)
	target_compile_features(tests PRIVATE cxx_std_20)
	target_link_libraries(tests PRIVATE Catch2::Catch2WithMain hops)

	# target_compile_options(tests PUBLIC ${UTIL_WARNING_OPTIONS} -g -Wno-parentheses -march=native)
endif()
