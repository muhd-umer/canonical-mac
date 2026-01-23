cmake_minimum_required(VERSION 3.14)
project(multiuser_opt VERSION 1.0.0 LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "No build type selected, defaulting to Release (for optimizations)")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -ffast-math -funroll-loops -fno-math-errno -DNDEBUG")
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
        message(FATAL_ERROR "Eigen3 not found. Install with: brew install eigen (macOS) or sudo apt install libeigen3-dev (Linux)")
    endif()
endif()

if(APPLE)
    execute_process(
        COMMAND brew --prefix libomp
        OUTPUT_VARIABLE HOMEBREW_LIBOMP_PREFIX
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )

    if(HOMEBREW_LIBOMP_PREFIX)
        message(STATUS "Found Homebrew libomp: ${HOMEBREW_LIBOMP_PREFIX}")
        set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I${HOMEBREW_LIBOMP_PREFIX}/include")
        set(OpenMP_CXX_LIB_NAMES "omp")
        set(OpenMP_omp_LIBRARY "${HOMEBREW_LIBOMP_PREFIX}/lib/libomp.dylib")
    else()
        message(WARNING "Homebrew libomp not found. Install with: brew install libomp")
    endif()
endif()

find_package(OpenMP REQUIRED)

find_library(GLPK_LIBRARY glpk)
find_path(GLPK_INCLUDE_DIR glpk.h)

if(NOT GLPK_LIBRARY OR NOT GLPK_INCLUDE_DIR)
    message(WARNING "GLPK not found. minpmac will not be built. Install with: brew install glpk (macOS) or sudo apt install libglpk-dev (Linux)")
    set(BUILD_MINPMAC OFF)
else()
    message(STATUS "Found GLPK: ${GLPK_LIBRARY}")
    set(BUILD_MINPMAC ON)
endif()

find_package(Matlab QUIET COMPONENTS MX_LIBRARY MEX_COMPILER)

if(Matlab_FOUND)
    message(STATUS "MATLAB found: ${Matlab_ROOT_DIR}")
    if(APPLE)
        if(CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
            set(MEX_EXT "mexmaca64")
        else()
            set(MEX_EXT "mexmaci64")
        endif()
    elseif(UNIX)
        set(MEX_EXT "mexa64")
    elseif(WIN32)
        set(MEX_EXT "mexw64")
    endif()
else()
    message(STATUS "MATLAB not found; MEX interfaces will not be built")
endif()

add_library(maxrmac_lib STATIC
    maxrmac/cpp/solve_tone.cpp
    maxrmac/cpp/maxrmac.cpp
)

target_include_directories(maxrmac_lib PUBLIC
    ${CMAKE_SOURCE_DIR}/maxrmac/cpp
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

add_executable(test_maxrmac maxrmac/cpp/test_maxrmac.cpp)
target_link_libraries(test_maxrmac PRIVATE maxrmac_lib)

if(Matlab_FOUND)
    add_library(maxrmac_mex MODULE maxrmac/cpp/maxrmac_mex.cpp)

    target_include_directories(maxrmac_mex PRIVATE ${Matlab_INCLUDE_DIRS})
    target_link_libraries(maxrmac_mex PRIVATE maxrmac_lib ${Matlab_MX_LIBRARY} ${Matlab_MEX_LIBRARY})
    
    target_compile_definitions(maxrmac_mex PRIVATE MATLAB_MEX_FILE MATLAB_DEFAULT_RELEASE=R2018a)

    set_target_properties(maxrmac_mex PROPERTIES
        PREFIX ""
        SUFFIX ".${MEX_EXT}"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/maxrmac"
    )

    if(APPLE)
        set_target_properties(maxrmac_mex PROPERTIES LINK_FLAGS "-undefined dynamic_lookup -flat_namespace")
    else()
        target_link_options(maxrmac_mex PRIVATE -static-libgcc -static-libstdc++)
    endif()
endif()

add_library(minpmac_lib STATIC
    minpmac/cpp/solve_tone.cpp
    minpmac/cpp/minpmac.cpp
)

target_include_directories(minpmac_lib PUBLIC
    ${CMAKE_SOURCE_DIR}/minpmac/cpp
    ${GLPK_INCLUDE_DIR}
)

target_link_libraries(minpmac_lib PUBLIC
    Eigen3::Eigen
    OpenMP::OpenMP_CXX
    ${GLPK_LIBRARY}
)

set_target_properties(minpmac_lib PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(minpmac_lib PRIVATE -fsanitize=address -g)
    target_link_options(minpmac_lib PRIVATE -fsanitize=address)
endif()

add_executable(test_minpmac minpmac/cpp/test_minpmac.cpp)
target_link_libraries(test_minpmac PRIVATE minpmac_lib)

if(Matlab_FOUND)
    add_library(minpmac_mex MODULE minpmac/cpp/minpmac_mex.cpp)

    target_include_directories(minpmac_mex PRIVATE ${Matlab_INCLUDE_DIRS})
    target_link_libraries(minpmac_mex PRIVATE minpmac_lib ${Matlab_MX_LIBRARY} ${Matlab_MEX_LIBRARY})

    target_compile_definitions(minpmac_mex PRIVATE MATLAB_MEX_FILE MATLAB_DEFAULT_RELEASE=R2018a)

    set_target_properties(minpmac_mex PROPERTIES
        PREFIX ""
        SUFFIX ".${MEX_EXT}"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/minpmac"
    )

    if(APPLE)
        set_target_properties(minpmac_mex PROPERTIES LINK_FLAGS "-undefined dynamic_lookup -flat_namespace")
    else()
        target_link_options(minpmac_mex PRIVATE -static-libgcc -static-libstdc++)
    endif()
endif()

install(TARGETS maxrmac_lib ARCHIVE DESTINATION lib)
install(FILES
    maxrmac/cpp/maxrmac.hpp
    maxrmac/cpp/solve_tone.hpp
    maxrmac/cpp/linalg_utils.hpp
    DESTINATION include/maxrmac
)

if(BUILD_MINPMAC)
    install(TARGETS minpmac_lib ARCHIVE DESTINATION lib)
    install(FILES
        minpmac/cpp/minpmac.hpp
        minpmac/cpp/solve_tone.hpp
        minpmac/cpp/linalg_utils.hpp
        DESTINATION include/minpmac
    ) 
endif()
