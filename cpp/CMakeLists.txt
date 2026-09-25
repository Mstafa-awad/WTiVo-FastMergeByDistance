cmake_minimum_required(VERSION 3.20)
project(wtivo_fast_weld LANGUAGES CXX)

add_library(wtivo_fast_weld SHARED wtivo_fast_weld.cpp)
target_compile_features(wtivo_fast_weld PRIVATE cxx_std_17)

if(MSVC)
    target_compile_options(wtivo_fast_weld PRIVATE /O2 /EHsc /openmp)
else()
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(wtivo_fast_weld PRIVATE OpenMP::OpenMP_CXX)
    else()
        message(WARNING "OpenMP not found; native build will be single-threaded.")
    endif()
endif()

set_target_properties(wtivo_fast_weld PROPERTIES
    OUTPUT_NAME "wtivo_fast_weld_mt"
)
