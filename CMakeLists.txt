cmake_minimum_required(VERSION 3.16)
project(ultratears_lifter_llvm LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Threads REQUIRED)

# Look for default installed libllvm (15+) if available, else fall back to prebuilt
find_package(LLVM CONFIG QUIET)
if(NOT LLVM_FOUND OR LLVM_VERSION_MAJOR LESS 15)
    find_program(LLVM_CONFIG_EXECUTABLE NAMES llvm-config)
    if(LLVM_CONFIG_EXECUTABLE)
        execute_process(
            COMMAND ${LLVM_CONFIG_EXECUTABLE} --cmakedir
            OUTPUT_VARIABLE LLVM_CONFIG_CMAKE_DIR
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )
        if(LLVM_CONFIG_CMAKE_DIR AND EXISTS "${LLVM_CONFIG_CMAKE_DIR}")
            find_package(LLVM CONFIG HINTS "${LLVM_CONFIG_CMAKE_DIR}" QUIET)
        endif()
    endif()
endif()

if(LLVM_FOUND AND LLVM_VERSION_MAJOR GREATER_EQUAL 15)
    message(STATUS "Using system LLVM ${LLVM_PACKAGE_VERSION} (include: ${LLVM_INCLUDE_DIRS}, lib: ${LLVM_LIBRARY_DIRS})")
    set(ULTRS_LLVM_INCLUDE_DIRS ${LLVM_INCLUDE_DIRS})
    set(ULTRS_LLVM_LIB_DIR "${LLVM_LIBRARY_DIRS}")
    add_definitions(${LLVM_DEFINITIONS})

    if(TARGET LLVM)
        add_library(UltraTearsThirdPartyLLVM INTERFACE)
        target_link_libraries(UltraTearsThirdPartyLLVM INTERFACE LLVM)
        target_include_directories(UltraTearsThirdPartyLLVM INTERFACE ${ULTRS_LLVM_INCLUDE_DIRS})
    else()
        find_library(ULTRS_SYSTEM_LLVM_LIB NAMES LLVM libLLVM PATHS ${LLVM_LIBRARY_DIRS})
        if(NOT ULTRS_SYSTEM_LLVM_LIB)
            find_library(ULTRS_SYSTEM_LLVM_LIB NAMES LLVM libLLVM)
        endif()
        add_library(UltraTearsThirdPartyLLVM UNKNOWN IMPORTED GLOBAL)
        set_target_properties(UltraTearsThirdPartyLLVM PROPERTIES
            IMPORTED_LOCATION "${ULTRS_SYSTEM_LLVM_LIB}"
            INTERFACE_INCLUDE_DIRECTORIES "${ULTRS_LLVM_INCLUDE_DIRS}"
        )
    endif()
else()
    message(STATUS "Using prebuilt LLVM from third-party/llvm")
    set(ULTRS_LLVM_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third-party/llvm")
    set(ULTRS_LLVM_INCLUDE_DIRS
        "${ULTRS_LLVM_DIR}/include"
        "${ULTRS_LLVM_DIR}/include/llvm"
    )
    set(ULTRS_LLVM_LIB_DIR "${ULTRS_LLVM_DIR}/prebuilts/x86-64")
    set(ULTRS_LLVM_SHARED_LIB "${ULTRS_LLVM_LIB_DIR}/libLLVM.so")

    add_library(UltraTearsThirdPartyLLVM SHARED IMPORTED GLOBAL)
    set_target_properties(UltraTearsThirdPartyLLVM PROPERTIES
        IMPORTED_LOCATION "${ULTRS_LLVM_SHARED_LIB}"
        IMPORTED_SONAME "libLLVM.so.21.1"
        INTERFACE_INCLUDE_DIRECTORIES "${ULTRS_LLVM_INCLUDE_DIRS}"
    )
endif()

set(ULTRS_LIFTER_PASS_SOURCES
    llvm_backend.cpp
    passes.cpp
    pass_ultratears_patch.cpp
    pass_profile_entry_hook_elim.cpp
    pass_nso_rodata_constprop_devirt.cpp
    pass_fpcr_fast_fp_lowering.cpp
    pass_simd_pairwise_conversion_inline.cpp
    pass_aarch64_native_intrinsic.cpp
    pass_x86_64_hardware_intrinsic.cpp
    pass_exclusive_monitor_lse_fastpath.cpp
    pass_horizon_lowering.cpp
    pass_horizon_tls_spinwait.cpp
    pass_guest_stack_alignment.cpp
    pass_aarch64_idiom_deconstruct.cpp
    pass_jump_table_lowering.cpp
    pass_shadow_dispatch_table.cpp
    pass_fcmp_predicate_fusion.cpp
    pass_nzcv_predicate_fusion.cpp
    pass_nzcv_dead_flag_elim.cpp
    pass_link_register_elision.cpp
    pass_tail_call_devirt.cpp
    pass_aapcs64_abi_contract.cpp
    pass_cpu_state_promotion.cpp
    pass_cross_chunk_leaf_inline.cpp
    pass_aarch64_static_nce_abi.cpp
)

add_library(ultratears_lifter_core STATIC
    ${ULTRS_LIFTER_PASS_SOURCES}
)
target_include_directories(ultratears_lifter_core PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}"
    ${ULTRS_LLVM_INCLUDE_DIRS}
)
target_compile_definitions(ultratears_lifter_core PUBLIC ULTRS_HAVE_LLVM_HEADERS=1)
target_link_libraries(ultratears_lifter_core PUBLIC
    UltraTearsThirdPartyLLVM
    Threads::Threads
    ${CMAKE_DL_LIBS}
)
if(MSVC)
    target_compile_options(ultratears_lifter_core PRIVATE /O2 /W3)
else()
    target_compile_options(ultratears_lifter_core PRIVATE -O3 -Wall -Wextra -Wno-unused-parameter)
endif()

add_executable(ultratears-lifter
    main.cpp
    llvm_pass_plugin.cpp
)
target_link_libraries(ultratears-lifter PRIVATE
    ultratears_lifter_core
)
set_target_properties(ultratears-lifter PROPERTIES
    BUILD_RPATH "${ULTRS_LLVM_LIB_DIR}"
    INSTALL_RPATH "${ULTRS_LLVM_LIB_DIR}"
)

if(MSVC)
    target_compile_options(ultratears-lifter PRIVATE /O2 /W3)
else()
    target_compile_options(ultratears-lifter PRIVATE -O3 -Wall -Wextra -Wno-unused-parameter)
endif()

option(ULTRS_BUILD_LLVM_PLUGIN "Build shared LLVM PassBuilder plugin" ON)
if(ULTRS_BUILD_LLVM_PLUGIN)
    add_library(UltraTearsPasses SHARED llvm_pass_plugin.cpp)
    target_include_directories(UltraTearsPasses PRIVATE ${ULTRS_LLVM_INCLUDE_DIRS})
    target_compile_definitions(UltraTearsPasses PRIVATE ULTRS_HAVE_LLVM_HEADERS=1)
    target_link_libraries(UltraTearsPasses PRIVATE UltraTearsThirdPartyLLVM)
    set_target_properties(UltraTearsPasses PROPERTIES
        BUILD_RPATH "${ULTRS_LLVM_LIB_DIR}"
        INSTALL_RPATH "${ULTRS_LLVM_LIB_DIR}"
    )
endif()

# if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/CMakeLists.txt")
#     enable_testing()
#     add_subdirectory(tests)
# endif()
