cmake_minimum_required(VERSION 3.14)
project(minpmac_cpp VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -ffast-math")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -funroll-loops -fno-math-errno")
endif()

find_package(Eigen3 3.3 QUIET NO_MODULE)

if(NOT Eigen3_FOUND)

    set(EIGEN_SEARCH_PATHS
        /usr/include/eigen3
        /usr/local/include/eigen3
        /usr/local/MATLAB/R2025b/toolbox/shared/robotics/externalDependency/eigen/include/eigen3
        /usr/local/MATLAB/R2024b/toolbox/shared/robotics/externalDependency/eigen/include/eigen3
        /opt/homebrew/include/eigen3
    )

    find_path(EIGEN3_INCLUDE_DIR
        NAMES Eigen/Core
        PATHS ${EIGEN_SEARCH_PATHS}
    )

    if(EIGEN3_INCLUDE_DIR)
        message(STATUS "Found Eigen3 at: ${EIGEN3_INCLUDE_DIR}")
        add_library(Eigen3::Eigen INTERFACE IMPORTED)
        set_target_properties(Eigen3::Eigen PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${EIGEN3_INCLUDE_DIR}"
        )
    else()
        message(FATAL_ERROR "Eigen3 not found. Install with: sudo apt install libeigen3-dev")
    endif()
endif()

find_package(OpenMP REQUIRED)

find_library(GLPK_LIBRARY glpk)
find_path(GLPK_INCLUDE_DIR glpk.h)

if(NOT GLPK_LIBRARY OR NOT GLPK_INCLUDE_DIR)
    message(FATAL_ERROR "GLPK not found. Install with: sudo apt install libglpk-dev")
endif()

message(STATUS "Found GLPK: ${GLPK_LIBRARY}")

add_library(minpmac_lib STATIC
    solve_tone.cpp
    minpmac.cpp
)

target_include_directories(minpmac_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${GLPK_INCLUDE_DIR}
)

target_link_libraries(minpmac_lib PUBLIC
    Eigen3::Eigen
    OpenMP::OpenMP_CXX
    ${GLPK_LIBRARY}
)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(minpmac_lib PRIVATE -fsanitize=address -g)
    target_link_options(minpmac_lib PRIVATE -fsanitize=address)
endif()

add_executable(test_minpmac test_minpmac.cpp)
target_link_libraries(test_minpmac PRIVATE minpmac_lib)

install(TARGETS minpmac_lib
    ARCHIVE DESTINATION lib
)

install(FILES
    minpmac.hpp
    solve_tone.hpp
    linalg_utils.hpp
    DESTINATION include/minpmac
)
