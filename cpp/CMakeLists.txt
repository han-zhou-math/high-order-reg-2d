cmake_minimum_required(VERSION 3.20)
project(regbie VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
# Keep IEEE semantics: no -ffast-math, so results agree with the Python code to roundoff.
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(REGBIE_BUILD_EXPERIMENTS "Build the experiment library, drivers and tests" ON)

include(GNUInstallDirs)
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
    URL_HASH SHA256=8586084f71f9bde545ee7fa6d00288b264a2b7ac3607b974e54d13e7162c1c72
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
  FetchContent_GetProperties(eigen)
  if(NOT eigen_POPULATED)
    FetchContent_Populate(eigen)
  endif()
  set(REGBIE_BUNDLED_EIGEN ON)
endif()

# ---------------------------------------------------------------- core library
add_library(regbie
  src/special.cpp src/kernels.cpp src/numerics.cpp src/geometry.cpp src/periodic.cpp
  src/layers.cpp src/grid.cpp src/io.cpp src/curve.cpp src/potentials.cpp)
add_library(regbie::regbie ALIAS regbie)
target_include_directories(regbie PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_include_directories(regbie PRIVATE ${FFTW3_INCLUDE_DIR})
target_link_libraries(regbie PUBLIC OpenMP::OpenMP_CXX PRIVATE ${FFTW3_LIBRARY})
if(REGBIE_BUNDLED_EIGEN)
  # The downloaded Eigen headers are installed with the library.
  target_include_directories(regbie PUBLIC
    $<BUILD_INTERFACE:${eigen_SOURCE_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/regbie/eigen3>)
  install(DIRECTORY ${eigen_SOURCE_DIR}/Eigen DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/regbie/eigen3)
else()
  target_link_libraries(regbie PUBLIC Eigen3::Eigen)
endif()

install(TARGETS regbie EXPORT regbieTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(DIRECTORY include/regbie DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  PATTERN experiments EXCLUDE)
install(EXPORT regbieTargets NAMESPACE regbie:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/regbie)
include(CMakePackageConfigHelpers)
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/regbieConfig.cmake.in [=[
@PACKAGE_INIT@
include(CMakeFindDependencyMacro)
find_dependency(OpenMP)
if(NOT @REGBIE_BUNDLED_EIGEN@)
  find_dependency(Eigen3 3.4 NO_MODULE)
endif()
include("${CMAKE_CURRENT_LIST_DIR}/regbieTargets.cmake")
]=])
configure_package_config_file(${CMAKE_CURRENT_BINARY_DIR}/regbieConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/regbieConfig.cmake INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/regbie)
write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/regbieConfigVersion.cmake
  COMPATIBILITY SameMajorVersion)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/regbieConfig.cmake ${CMAKE_CURRENT_BINARY_DIR}/regbieConfigVersion.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/regbie)

# Examples use only the installed interface.
file(GLOB EXAMPLE_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/examples/*.cpp)
foreach(source ${EXAMPLE_SOURCES})
  get_filename_component(name ${source} NAME_WE)
  add_executable(${name} ${source})
  target_link_libraries(${name} PRIVATE regbie::regbie)
endforeach()

# ---------------------------------------------------------------- experiments
if(REGBIE_BUILD_EXPERIMENTS)
  file(GLOB EXPERIMENT_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/experiments/*.cpp)
  add_library(regbie_experiments ${EXPERIMENT_SOURCES})
  target_include_directories(regbie_experiments PUBLIC ${FFTW3_INCLUDE_DIR})
  target_link_libraries(regbie_experiments PUBLIC regbie ${FFTW3_LIBRARY})

  # One executable per driver: the programs that write results/.
  file(GLOB DRIVER_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/drivers/*.cpp)
  foreach(source ${DRIVER_SOURCES})
    get_filename_component(name ${source} NAME_WE)
    add_executable(${name} ${source})
    target_link_libraries(${name} PRIVATE regbie_experiments)
  endforeach()

  enable_testing()
  file(GLOB TEST_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cpp)
  foreach(source ${TEST_SOURCES})
    get_filename_component(name ${source} NAME_WE)
    add_executable(${name} ${source})
    target_link_libraries(${name} PRIVATE regbie_experiments)
    add_test(NAME ${name} COMMAND ${name} WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/..)
  endforeach()
endif()
