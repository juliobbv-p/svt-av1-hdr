#
# Copyright(c) 2019 Intel Corporation
#
# This source code is subject to the terms of the BSD 3 Clause Clear License and
# the Alliance for Open Media Patent License 1.0. If the BSD 3 Clause Clear License
# was not distributed with this source code in the LICENSE file, you can
# obtain it at https://www.aomedia.org/license/license. If the Alliance for Open
# Media Patent License 1.0 was not distributed with this source code in the
# PATENTS file, you can obtain it at https://www.aomedia.org/license/patent-license.
#

# Encoder Library directory CMakeLists.txt

add_library(common_lib INTERFACE)

if(COVERAGE)
    if(CMAKE_C_COMPILER_ID MATCHES "GNU")
        list(APPEND PC_LIBS_PRIVATE -lgcov)
    elseif(CMAKE_C_COMPILER_ID MATCHES "Clang")
        list(APPEND PC_LIBS_PRIVATE --coverage)
    endif()
endif()

include_directories(${PROJECT_SOURCE_DIR}/Source/API/
    ${PROJECT_SOURCE_DIR}/Source/Lib/Codec/
    ${PROJECT_SOURCE_DIR}/Source/Lib/C_DEFAULT/
    ${PROJECT_SOURCE_DIR}/third_party/fastfeat/)

add_library(SvtAv1Enc)
# Required for cmake to be able to tell Xcode how to link all of the object files
if(CMAKE_GENERATOR STREQUAL "Xcode")
    file(WRITE ${PROJECT_BINARY_DIR}/dummy.c "")
    target_sources(SvtAv1Enc PRIVATE ${PROJECT_BINARY_DIR}/dummy.c)
endif()

add_subdirectory(C_DEFAULT)
add_subdirectory(Codec)
add_subdirectory(Globals)
if(NOT COMPILE_C_ONLY AND HAVE_X86_PLATFORM)
    # Include Encoder Subdirectories
    target_include_directories(SvtAv1Enc PRIVATE
        ${PROJECT_SOURCE_DIR}/Source/Lib/ASM_SSE2/
        ${PROJECT_SOURCE_DIR}/Source/Lib/ASM_SSSE3/
        ${PROJECT_SOURCE_DIR}/Source/Lib/ASM_SSE4_1/
        ${PROJECT_SOURCE_DIR}/Source/Lib/ASM_AVX2/
        ${PROJECT_SOURCE_DIR}/Source/Lib/ASM_AVX512/)
    add_subdirectory(ASM_SSE2)
    add_subdirectory(ASM_SSSE3)
    add_subdirectory(ASM_SSE4_1)
    add_subdirectory(ASM_AVX2)
    if(ENABLE_AVX512)
        add_subdirectory(ASM_AVX512)
    endif()
elseif(NOT COMPILE_C_ONLY AND HAVE_ARM_PLATFORM)
    target_include_directories(SvtAv1Enc PRIVATE
        ${PROJECT_SOURCE_DIR}/Source/Lib/ASM_NEON/
        ${PROJECT_SOURCE_DIR}/Source/Lib/ASM_NEON_DOTPROD/
        ${PROJECT_SOURCE_DIR}/Source/Lib/ASM_NEON_I8MM/
        ${PROJECT_SOURCE_DIR}/Source/Lib/ASM_SVE/
        ${PROJECT_SOURCE_DIR}/Source/Lib/ASM_SVE2/)
    add_subdirectory(ASM_NEON)
    if(ENABLE_NEON_DOTPROD)
        add_subdirectory(ASM_NEON_DOTPROD)
    endif()
    if(ENABLE_NEON_I8MM)
        add_subdirectory(ASM_NEON_I8MM)
    endif()
    if(ENABLE_SVE)
      add_subdirectory(ASM_SVE)
    endif()
    if(ENABLE_SVE2)
      add_subdirectory(ASM_SVE2)
    endif()
endif()

# Encoder Lib Source Files
target_sources(SvtAv1Enc PRIVATE
    $<TARGET_OBJECTS:FASTFEAT>
    $<TARGET_OBJECTS:GLOBALS>
    $<TARGET_OBJECTS:CODEC>
    $<TARGET_OBJECTS:C_DEFAULT>)
if(NOT COMPILE_C_ONLY AND HAVE_X86_PLATFORM)
    target_sources(SvtAv1Enc PRIVATE
        $<TARGET_OBJECTS:ASM_SSE2>
        $<TARGET_OBJECTS:ASM_SSSE3>
        $<TARGET_OBJECTS:ASM_SSE4_1>
        $<TARGET_OBJECTS:ASM_AVX2>)
    if(ENABLE_AVX512)
        target_sources(SvtAv1Enc PRIVATE
            $<TARGET_OBJECTS:ASM_AVX512>)
    endif()
elseif(NOT COMPILE_C_ONLY AND HAVE_ARM_PLATFORM)
    target_sources(SvtAv1Enc PRIVATE $<TARGET_OBJECTS:ASM_NEON>)
    if(ENABLE_NEON_DOTPROD)
        target_sources(SvtAv1Enc PRIVATE $<TARGET_OBJECTS:ASM_NEON_DOTPROD>)
    endif()
    if(ENABLE_NEON_I8MM)
        target_sources(SvtAv1Enc PRIVATE $<TARGET_OBJECTS:ASM_NEON_I8MM>)
    endif()
    if(ENABLE_SVE)
        target_sources(SvtAv1Enc PRIVATE $<TARGET_OBJECTS:ASM_SVE>)
    endif()
    if(ENABLE_SVE2)
        target_sources(SvtAv1Enc PRIVATE $<TARGET_OBJECTS:ASM_SVE2>)
    endif()
endif()

get_target_property(common_lib_source common_lib INTERFACE_SOURCES)
if(common_lib_source)
    target_sources(SvtAv1Enc PRIVATE ${common_lib_source})
endif()

if(BUILD_SHARED_LIBS)
    set(SVT_AV1_TARGET SVT-AV1-shared)
else()
    set(SVT_AV1_TARGET SVT-AV1-static)
endif()

set_target_properties(SvtAv1Enc PROPERTIES VERSION ${PROJECT_VERSION})
set_target_properties(SvtAv1Enc PROPERTIES SOVERSION ${PROJECT_VERSION_MAJOR})
set_target_properties(SvtAv1Enc PROPERTIES C_VISIBILITY_PRESET hidden)
set_target_properties(SvtAv1Enc PROPERTIES EXPORT_NAME ${SVT_AV1_TARGET})

find_package(PkgConfig)
if(NOT PkgConfig_FOUND)
  message(WARNING "PkgConfig not found; disabled building with dovi & hdr10plus.")
endif()

option(EXT_LIB_STATIC "Force linking with static libraries for external dependencies" OFF)
set(SUFFIX "")
if(EXT_LIB_STATIC)
  set(SUFFIX ${CMAKE_STATIC_LIBRARY_SUFFIX})
else()
  set(SUFFIX ${CMAKE_SHARED_LIBRARY_SUFFIX})
endif()

# libdovi detection & preprocessor macro
option(LIBDOVI_FOUND "Use the dovi library" OFF)
if(LIBDOVI_FOUND)
  if(PKG_CONFIG_FOUND)
    pkg_check_modules(PC_LIBDOVI dovi REQUIRED)
  endif()
  set(LIBDOVI_LIBRARY_DIR ${PC_LIBDOVI_LIBDIR})
  set(LIBDOVI_INCLUDE_DIR ${PC_LIBDOVI_INCLUDEDIR})

  target_include_directories(SvtAv1Enc PRIVATE ${LIBDOVI_INCLUDE_DIR})

  if(EXT_LIB_STATIC OR MSVC)
    set(LIBDOVI_REQUIRED_LIBRARIES ${PC_LIBDOVI_STATIC_LIBRARIES})
  else()
    set(LIBDOVI_REQUIRED_LIBRARIES ${PC_LIBDOVI_LIBRARIES})
  endif()

  set(LIBDOVI_LIBRARY "")
  if(WIN32 AND NOT MSVC AND CMAKE_C_COMPILER_ID MATCHES "Clang" AND NOT MINGW)
    # link hack only for native windows and clang
    # because cmake pass .lib file for windows
    # but cargo-c produce .dll.lib for shared
    # and .lib for static library this force passing
    # .dll extension by creating linker flags
    foreach(BASE ${LIBDOVI_REQUIRED_LIBRARIES})
      list(APPEND LIBDOVI_LIBRARY "-l${BASE}${SUFFIX}")
    endforeach()
  else()
    foreach(BASE ${LIBDOVI_REQUIRED_LIBRARIES})
      list(APPEND LIBDOVI_LIBRARY "${BASE}${SUFFIX}")
    endforeach()
  endif()

  target_compile_definitions(SvtAv1Enc PUBLIC LIBDOVI_FOUND=1)
  list(APPEND PC_REQUIRES_PRIVATE "dovi")
  execute_process(COMMAND ${CMAKE_COMMAND} -E echo_append "-- Building with dovi support - Yes\n")
else()
  execute_process(COMMAND ${CMAKE_COMMAND} -E echo_append "-- Building with dovi support - No\n")
endif()

# libhdr10plus detection & preprocessor macro
option(LIBHDR10PLUS_RS_FOUND "Use the hdr10plus library" OFF)
if(LIBHDR10PLUS_RS_FOUND)
  if(PKG_CONFIG_FOUND)
    pkg_check_modules(PC_LIBHDR10PLUS_RS hdr10plus-rs REQUIRED)
  endif()
  set(LIBHDR10PLUS_RS_LIBRARY_DIR ${PC_LIBHDR10PLUS_RS_LIBDIR})
  set(LIBHDR10PLUS_RS_INCLUDE_DIR ${PC_LIBHDR10PLUS_RS_INCLUDEDIR})

  target_include_directories(SvtAv1Enc PRIVATE ${LIBHDR10PLUS_RS_INCLUDE_DIR})

  if(EXT_LIB_STATIC OR MSVC)
    set(LIBHDR10PLUS_RS_REQUIRED_LIBRARIES ${PC_LIBHDR10PLUS_RS_STATIC_LIBRARIES})
  else()
    set(LIBHDR10PLUS_RS_REQUIRED_LIBRARIES ${PC_LIBHDR10PLUS_RS_LIBRARIES})
  endif()

  set(LIBHDR10PLUS_RS_LIBRARY "")
  if(WIN32 AND NOT MSVC AND CMAKE_C_COMPILER_ID MATCHES "Clang" AND NOT MINGW)
    # link hack only for native windows and clang
    # because cmake pass .lib file for windows
    # on both static and shared library
    # but cargo-c produce .dll.lib for shared
    # and .lib for static library this force passing
    # .dll extension by creating linker flags
    if(NOT EXT_LIB_STATIC)
      message(WARNING "Shared version of hdr10plus-rs library is known to cause error when link")
    endif()
    foreach(BASE ${LIBHDR10PLUS_RS_REQUIRED_LIBRARIES})
      list(APPEND LIBHDR10PLUS_RS_LIBRARY "-l${BASE}${SUFFIX}")
    endforeach()
  else()
    foreach(BASE ${LIBHDR10PLUS_RS_REQUIRED_LIBRARIES})
      list(APPEND LIBHDR10PLUS_RS_LIBRARY "${BASE}${SUFFIX}")
    endforeach()
  endif()

  target_compile_definitions(SvtAv1Enc PUBLIC LIBHDR10PLUS_RS_FOUND=1)
  list(APPEND PC_REQUIRES_PRIVATE "hdr10plus-rs")
  execute_process(COMMAND ${CMAKE_COMMAND} -E echo_append "-- Building with hdr10plus support - Yes\n")
else()
  execute_process(COMMAND ${CMAKE_COMMAND} -E echo_append "-- Building with hdr10plus support - No\n")
endif()

target_link_libraries(SvtAv1Enc PUBLIC  "$<BUILD_INTERFACE:${PLATFORM_LIBS}>"
                                PRIVATE "$<INSTALL_INTERFACE:${PLATFORM_LIBS}>"
                                        ${LIBDOVI_LIBRARY}
                                        ${LIBHDR10PLUS_RS_LIBRARY})
target_link_directories(SvtAv1Enc PUBLIC ${LIBDOVI_LIBRARY_DIR} ${LIBHDR10PLUS_RS_LIBRARY_DIR})

file(GLOB API_HEADERS ${PROJECT_SOURCE_DIR}/Source/API/*.h)
set_target_properties(SvtAv1Enc PROPERTIES PUBLIC_HEADER "${API_HEADERS}")

# Alias for the exported library target for use in subproject
add_library(SVT-AV1::SVT-AV1 ALIAS SvtAv1Enc)
add_library(SVT-AV1::${SVT_AV1_TARGET} ALIAS SvtAv1Enc)

set(ENC_PKG_CONFIG_EXTRA_CFLAGS "")
if (BUILD_SHARED_LIBS)
    target_compile_definitions(SvtAv1Enc
        PUBLIC EB_DLL
        PRIVATE EB_BUILDING_SHARED_LIBS)
    set(ENC_PKG_CONFIG_EXTRA_CFLAGS " -DEB_DLL")
endif()

if(RTC_BUILD)
    target_compile_definitions(SvtAv1Enc PUBLIC RTC_BUILD=1)
    set(ENC_PKG_CONFIG_EXTRA_CFLAGS "${ENC_PKG_CONFIG_EXTRA_CFLAGS} -DRTC_BUILD=1")
else()
    target_compile_definitions(SvtAv1Enc PUBLIC RTC_BUILD=0)
    set(ENC_PKG_CONFIG_EXTRA_CFLAGS "${ENC_PKG_CONFIG_EXTRA_CFLAGS} -DRTC_BUILD=0")
endif()

install(TARGETS SvtAv1Enc
        EXPORT ${SVT_AV1_TARGET}Targets
        ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
                COMPONENT Development
        LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
                COMPONENT Runtime
                NAMELINK_COMPONENT Development
        RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
                COMPONENT Runtime
        PUBLIC_HEADER DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/svt-av1"
                COMPONENT Development
)

# pkg-config .pc file
list(JOIN PC_LIBS_PRIVATE " " LIBS_PRIVATE)
list(JOIN PC_REQUIRES_PRIVATE " " REQUIRES_PRIVATE)
configure_file(pkg-config.pc.in ${CMAKE_BINARY_DIR}/SvtAv1Enc.pc @ONLY)
install(FILES ${CMAKE_BINARY_DIR}/SvtAv1Enc.pc COMPONENT Development DESTINATION "${CMAKE_INSTALL_LIBDIR}/pkgconfig")

# CMake config file
include(CMakePackageConfigHelpers)

target_include_directories(SvtAv1Enc INTERFACE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/Source/API>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/svt-av1>)

configure_package_config_file(config.cmake.in SVT-AV1Config.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/SVT-AV1
    PATH_VARS CMAKE_INSTALL_LIBDIR CMAKE_INSTALL_INCLUDEDIR)

write_basic_package_version_file(SVT-AV1ConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion)

install(EXPORT ${SVT_AV1_TARGET}Targets
        NAMESPACE SVT-AV1::
        COMPONENT Development
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/SVT-AV1)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/SVT-AV1Config.cmake
              ${CMAKE_CURRENT_BINARY_DIR}/SVT-AV1ConfigVersion.cmake
        COMPONENT Development
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/SVT-AV1)
