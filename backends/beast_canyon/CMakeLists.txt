# Shannon-Prime Beast Canyon Backend
# Heterogeneous MoE Orchestrator: Optane + AVX-512 + dual-GPU + sidecar
#
# Build options:
#   -DSP_WITH_CUDA=ON    — Enable NVIDIA CUDA dispatch
#   -DSP_WITH_VULKAN=ON  — Enable Vulkan dispatch (Intel Xe, AMD, etc.)
#   -DSP_BEAST_STANDALONE=ON — Build standalone test binary

cmake_minimum_required(VERSION 3.18)

# Source files
set(SP_BEAST_SOURCES
    sp_optane.c
    sp_avx512_shredder.c
    sp_hetero_sync.c
    sp_beast_canyon.c
    sp_diagnostics_bc.c
)

set(SP_BEAST_HEADERS
    sp_optane.h
    sp_avx512_shredder.h
    sp_hetero_sync.h
    sp_beast_canyon.h
    sp_diagnostics_bc.h
)

# Library target
add_library(sp_beast_canyon STATIC ${SP_BEAST_SOURCES})

target_include_directories(sp_beast_canyon
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../core
)

# Platform flags
if(MSVC)
    # MSVC: AVX-512 is enabled by default on supporting CPUs.
    # /arch:AVX512 ensures the compiler can use 512-bit instructions.
    target_compile_options(sp_beast_canyon PRIVATE /arch:AVX512 /O2)
elseif(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(sp_beast_canyon PRIVATE
        -mavx512f -mavx512bw -mavx512vl -mf16c -mfma
        -O3 -march=native
    )
endif()

# Windows-specific: link ws2_32 for sidecar socket
if(WIN32)
    target_link_libraries(sp_beast_canyon PRIVATE ws2_32)
endif()

# Linux-specific: link rt for clock_gettime, pthread
if(UNIX AND NOT APPLE)
    target_link_libraries(sp_beast_canyon PRIVATE rt pthread)
endif()

# CUDA support
option(SP_WITH_CUDA "Enable NVIDIA CUDA dispatch" OFF)
if(SP_WITH_CUDA)
    find_package(CUDAToolkit REQUIRED)
    target_compile_definitions(sp_beast_canyon PRIVATE SP_WITH_CUDA)
    target_link_libraries(sp_beast_canyon PRIVATE CUDA::cudart)
endif()

# Vulkan support
option(SP_WITH_VULKAN "Enable Vulkan dispatch" OFF)
if(SP_WITH_VULKAN)
    find_package(Vulkan REQUIRED)
    target_compile_definitions(sp_beast_canyon PRIVATE SP_WITH_VULKAN)
    target_link_libraries(sp_beast_canyon PRIVATE Vulkan::Vulkan)
endif()

# Standalone test binary
option(SP_BEAST_STANDALONE "Build standalone test binary" OFF)
if(SP_BEAST_STANDALONE)
    add_executable(sp_beast_test test_beast_canyon.c)
    target_link_libraries(sp_beast_test PRIVATE sp_beast_canyon)
    if(WIN32)
        target_link_libraries(sp_beast_test PRIVATE ws2_32)
    endif()
endif()
