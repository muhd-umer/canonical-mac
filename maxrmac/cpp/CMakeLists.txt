cmake_minimum_required(VERSION 3.14)
project(maxrmac_cpp VERSION 1.0.0 LANGUAGES CXX)

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

add_library(maxrmac_lib STATIC
    solve_tone.cpp
    maxrmac.cpp
)

target_include_directories(maxrmac_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(maxrmac_lib PUBLIC
    Eigen3::Eigen
    OpenMP::OpenMP_CXX
)

set_target_properties(maxrmac_lib PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(maxrmac_lib PRIVATE -fsanitize=address -g)
    target_link_options(maxrmac_lib PRIVATE -fsanitize=address)
endif()

add_executable(test_maxrmac test_maxrmac.cpp)
target_link_libraries(test_maxrmac PRIVATE maxrmac_lib)

install(TARGETS maxrmac_lib
    ARCHIVE DESTINATION lib
)

install(FILES
    maxrmac_mimo.hpp
    solve_tone.hpp
    linalg_utils.hpp
    DESTINATION include/maxrmac
)

find_package(Matlab QUIET COMPONENTS MX_LIBRARY)

if(Matlab_FOUND)
    message(STATUS "MATLAB found: ${Matlab_ROOT_DIR}")
    message(STATUS "  Building MEX interface")

    matlab_add_mex(
        NAME maxrmac_mex
        SRC maxrmac_mex.cpp
        LINK_TO maxrmac_lib
        R2018a
    )

    set_target_properties(maxrmac_mex PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../"
    )

    target_link_options(maxrmac_mex PRIVATE -static-libgcc -static-libstdc++)
else()
    message(STATUS "MATLAB not found; skipping MEX build")
    message(STATUS "  To build MEX, set Matlab_ROOT_DIR or use mex command directly")
endif()
