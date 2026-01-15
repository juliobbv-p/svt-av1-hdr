#
# Copyright(c) 2019 Intel Corporation
#
# This source code is subject to the terms of the BSD 2 Clause License and
# the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
# was not distributed with this source code in the LICENSE file, you can
# obtain it at https://www.aomedia.org/license/software-license. If the Alliance for Open
# Media Patent License 1.0 was not distributed with this source code in the
# PATENTS file, you can obtain it at https://www.aomedia.org/license/patent-license.
#

cmake_minimum_required(VERSION 3.5...3.28)

# APP Directory CMakeLists.txt

# Include Subdirectories
include_directories(${PROJECT_SOURCE_DIR}/Source/API/)
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

  set(LIBDOVI_LIBRARY_DIR ${PC_LIBDOVI_LIBDIRS})

  set(LIBDOVI_INCLUDE_DIR ${PC_LIBDOVI_INCLUDEDIRS})
  
  include_directories(${LIBDOVI_INCLUDE_DIR})


  if(EXT_LIB_STATIC)
    set(LIBDOVI_REQUIRED_LIBRARIES ${PC_LIBDOVI_STATIC_LIBRARIES})
  else()
    set(LIBDOVI_REQUIRED_LIBRARIES ${PC_LIBDOVI_LIBRARIES})
  endif()
  
  set(LIBDOVI_LIBRARY "")
  if(NOT MSVC AND CMAKE_C_COMPILER_ID MATCHES "Clang|GNU" AND NOT MINGW)
    # link hack not working in msys2 terminal
    foreach(BASE ${LIBDOVI_REQUIRED_LIBRARIES})
      list(APPEND LIBDOVI_LIBRARY "-l${BASE}${SUFFIX}")
    endforeach()
  else()
    foreach(BASE ${LIBDOVI_REQUIRED_LIBRARIES})
      list(APPEND LIBDOVI_LIBRARY "${BASE}${SUFFIX}")
    endforeach()
  endif()

  add_definitions(-DLIBDOVI_FOUND=1)
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
  
  include_directories(${LIBHDR10PLUS_RS_INCLUDE_DIR})


  if(EXT_LIB_STATIC)
    set(LIBHDR10PLUS_RS_REQUIRED_LIBRARIES ${PC_LIBHDR10PLUS_RS_STATIC_LIBRARIES})
  else()
    set(LIBHDR10PLUS_RS_REQUIRED_LIBRARIES ${PC_LIBHDR10PLUS_RS_LIBRARIES})
  endif()
  
  set(LIBHDR10PLUS_RS_LIBRARY "")
  if(NOT MSVC AND CMAKE_C_COMPILER_ID MATCHES "Clang|GNU" AND NOT MINGW)
    # link hack not working in msys2 terminal
    foreach(BASE ${LIBHDR10PLUS_RS_REQUIRED_LIBRARIES})
      list(APPEND LIBHDR10PLUS_RS_LIBRARY "-l${BASE}${SUFFIX}")
    endforeach()
  else()
    foreach(BASE ${LIBHDR10PLUS_RS_REQUIRED_LIBRARIES})
      list(APPEND LIBHDR10PLUS_RS_LIBRARY "${BASE}${SUFFIX}")
    endforeach()
  endif()

  add_definitions(-DLIBHDR10PLUS_RS_FOUND=1)
  execute_process(COMMAND ${CMAKE_COMMAND} -E echo_append "-- Building with hdr10plus support - Yes\n")
else()
  execute_process(COMMAND ${CMAKE_COMMAND} -E echo_append "-- Building with hdr10plus support - No\n")
endif()

set(all_files
    ../API/EbConfigMacros.h
    ../API/EbDebugMacros.h
    ../API/EbSvtAv1.h
    ../API/EbSvtAv1Enc.h
    ../API/EbSvtAv1ErrorCodes.h
    ../API/EbSvtAv1Formats.h
    ../API/EbSvtAv1Metadata.h
    app_config.c
    app_config.h
    app_context.c
    app_context.h
    app_input_y4m.c
    app_input_y4m.h
    app_main.c
    app_output_ivf.c
    app_output_ivf.h
    app_process_cmd.c
    svt_time.c
    svt_time.h
    )

# App Source Files
add_executable(SvtAv1EncApp
    ${all_files})

if(TARGET safestringlib)
    target_sources(SvtAv1EncApp PRIVATE $<TARGET_OBJECTS:safestringlib>)
endif()

#********** SET COMPILE FLAGS************
# Link the Encoder App
target_link_libraries(SvtAv1EncApp PRIVATE SvtAv1Enc ${LIBDOVI_LIBRARY} ${LIBHDR10PLUS_RS_LIBRARY})
target_link_directories(SvtAv1EncApp PRIVATE SvtAv1Enc ${LIBDOVI_LIBRARY_DIR} ${LIBHDR10PLUS_RS_LIBRARY_DIR})

install(TARGETS SvtAv1EncApp RUNTIME COMPONENT Runtime DESTINATION ${CMAKE_INSTALL_BINDIR})
