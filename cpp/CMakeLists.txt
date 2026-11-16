cmake_minimum_required(VERSION 3.20)
project(regbie_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
# Keep IEEE semantics: no -ffast-math, so results agree with the Python code to roundoff.
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

find_package(OpenMP REQUIRED)

# FFTW 3: honour FFTW3_HOME (OSC module) before the system locations.
find_path(FFTW3_INCLUDE_DIR fftw3.h HINTS $ENV{FFTW3_HOME}/include $ENV{FFTW_HOME}/include)
find_library(FFTW3_LIBRARY fftw3 HINTS $ENV{FFTW3_HOME}/lib $ENV{FFTW_HOME}/lib)
if(NOT FFTW3_INCLUDE_DIR OR NOT FFTW3_LIBRARY)
  message(FATAL_ERROR "FFTW 3 not found; load the fftw module or set FFTW3_HOME")
endif()

# Eigen 3.4 (header only): use an installed copy if present, else download it.
find_package(Eigen3 3.4 QUIET NO_MODULE)
if(NOT TARGET Eigen3::Eigen)
  include(FetchContent)
  FetchContent_Declare(eigen
    URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
    URL_HASH SHA256=8586084f71f9bde545ee7fa6d00288b264a2b7ac3607b974e54d13e7162c1c72)
  FetchContent_GetProperties(eigen)
  if(NOT eigen_POPULATED)
    FetchContent_Populate(eigen)
  endif()
  add_library(Eigen3::Eigen INTERFACE IMPORTED)
  set_target_properties(Eigen3::Eigen PROPERTIES INTERFACE_INCLUDE_DIRECTORIES ${eigen_SOURCE_DIR})
endif()

# Every file in src/ belongs to the library.
file(GLOB LIBRARY_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
add_library(regbie ${LIBRARY_SOURCES})
target_include_directories(regbie PUBLIC include ${FFTW3_INCLUDE_DIR})
target_link_libraries(regbie PUBLIC Eigen3::Eigen ${FFTW3_LIBRARY} OpenMP::OpenMP_CXX)

# One executable per driver; every file in drivers/ is a program.
file(GLOB DRIVER_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/drivers/*.cpp)
foreach(source ${DRIVER_SOURCES})
  get_filename_component(name ${source} NAME_WE)
  add_executable(${name} ${source})
  target_link_libraries(${name} PRIVATE regbie)
endforeach()

enable_testing()
file(GLOB TEST_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cpp)
foreach(source ${TEST_SOURCES})
  get_filename_component(name ${source} NAME_WE)
  add_executable(${name} ${source})
  target_link_libraries(${name} PRIVATE regbie)
  add_test(NAME ${name} COMMAND ${name} WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/..)
endforeach()
