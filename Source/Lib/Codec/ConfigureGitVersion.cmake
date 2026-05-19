#
# Copyright(c) 2020 Intel Corporation
#
# This source code is subject to the terms of the BSD 2 Clause License and
# the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
# was not distributed with this source code in the LICENSE file, you can
# obtain it at https://www.aomedia.org/license/software-license. If the Alliance for Open
# Media Patent License 1.0 was not distributed with this source code in the
# PATENTS file, you can obtain it at https://www.aomedia.org/license/patent-license.
#

# Get the current version from the git repo to generate a version header
#
# Following variables should be set by the caller:
# PACKAGE_VERSION_STRING: Initial value used if not a git checkout
# GIT_ROOT_DIR: Project root
# INPUT_FILE: Input file to replace version in
# OUTPUT_FILE: Output file to write the version header to
#
# The variable that should be used in the version header is PACKAGE_VERSION_STRING

# Order of tag preference:
# 1. Tags on the current HEAD. If there are multiple, it will follow the tie-breaker rules outlined later.
# 2. The most recent tag reachable from the current HEAD matching the following:
#     - v[0-9]*.[0-9]*.[0-9]
#     - v[0-9]*.[0-9]*.[0-9]-rc*
# In the case of a tie, the preference will be given to the tag with the highest version, and for
# equal versions, the stable tag will be preferred over the rc tag.

set(PACKAGE_VERSION_STRING "v${PACKAGE_VERSION_STRING}")

macro(configure_version)
    message(STATUS "Configured version: ${PACKAGE_VERSION_STRING}")
    configure_file(${INPUT_FILE} ${OUTPUT_FILE} @ONLY)
    return()
endmacro()

find_package(Git QUIET)
if(NOT Git_FOUND OR NOT EXISTS "${GIT_ROOT_DIR}/.git")
    configure_version()
endif()

function(parse_version_tag tag valid major minor patch is_stable rc_version)
    set(${valid} FALSE PARENT_SCOPE)
    set(${major} 0 PARENT_SCOPE)
    set(${minor} 0 PARENT_SCOPE)
    set(${patch} 0 PARENT_SCOPE)
    set(${is_stable} FALSE PARENT_SCOPE)
    set(${rc_version} 0 PARENT_SCOPE)

    if(tag MATCHES "^v([0-9]+)\\.([0-9]+)\\.([0-9]+)$")
        set(${valid} TRUE PARENT_SCOPE)
        set(${major} ${CMAKE_MATCH_1} PARENT_SCOPE)
        set(${minor} ${CMAKE_MATCH_2} PARENT_SCOPE)
        set(${patch} ${CMAKE_MATCH_3} PARENT_SCOPE)
        set(${is_stable} TRUE PARENT_SCOPE)
        return()
    endif()
    if(tag MATCHES "^v([0-9]+)\\.([0-9]+)\\.([0-9]+)-rc([0-9]*)$")
        set(${valid} TRUE PARENT_SCOPE)
        set(${major} ${CMAKE_MATCH_1} PARENT_SCOPE)
        set(${minor} ${CMAKE_MATCH_2} PARENT_SCOPE)
        set(${patch} ${CMAKE_MATCH_3} PARENT_SCOPE)
        if(CMAKE_MATCH_4)
            set(${rc_version} ${CMAKE_MATCH_4} PARENT_SCOPE)
        endif()
    endif()
endfunction()


function(select_preferred_tag lhs rhs lhs_preferred)
    parse_version_tag("${lhs}" l_valid l_major l_minor l_patch l_is_stable l_rc)
    parse_version_tag("${rhs}" r_valid r_major r_minor r_patch r_is_stable r_rc)

    if(l_valid AND NOT r_valid)
        set(${lhs_preferred} TRUE PARENT_SCOPE)
        return()
    endif()
    if(r_valid AND NOT l_valid)
        set(${lhs_preferred} FALSE PARENT_SCOPE)
        return()
    endif()
    if(NOT l_valid AND NOT r_valid)
        if(lhs STRGREATER rhs)
            set(${lhs_preferred} TRUE PARENT_SCOPE)
        else()
            set(${lhs_preferred} FALSE PARENT_SCOPE)
        endif()
        return()
    endif()
    set(l_version "${l_major}.${l_minor}.${l_patch}")
    set(r_version "${r_major}.${r_minor}.${r_patch}")
    if("${l_version}" VERSION_GREATER "${r_version}")
        set(${lhs_preferred} TRUE PARENT_SCOPE)
        return()
    endif()
    if("${r_version}" VERSION_GREATER "${l_version}")
        set(${lhs_preferred} FALSE PARENT_SCOPE)
        return()
    endif()

    # Both versions should be the same, so now we are comparing stable vs rc.
    if(l_is_stable AND NOT r_is_stable)
        set(${lhs_preferred} TRUE PARENT_SCOPE)
        return()
    endif()
    if(NOT l_is_stable AND r_is_stable)
        set(${lhs_preferred} FALSE PARENT_SCOPE)
        return()
    endif()
    if(l_rc GREATER r_rc)
        set(${lhs_preferred} TRUE PARENT_SCOPE)
    else()
        set(${lhs_preferred} FALSE PARENT_SCOPE)
    endif()
endfunction()

function(select_preferred_head_tag tags_output out_var)
    string(REPLACE "\r\n" "\n" tags_output "${tags_output}")
    string(REPLACE "\n" ";" tags_output "${tags_output}")

    list(LENGTH tags_output len)
    if(len EQUAL 0)
        set(${out_var} "" PARENT_SCOPE)
        return()
    endif()
    if(len EQUAL 1)
        set(${out_var} "${tags_output}" PARENT_SCOPE)
        return()
    endif()

    unset(canonical_tags)
    unset(other_tags)

    foreach(tag IN LISTS tags_output)
        if(tag STREQUAL "")
            continue()
        endif()

        if(tag MATCHES "^v[0-9]+\\.[0-9]+\\.[0-9]+(-rc[0-9]*)?$")
            list(APPEND canonical_tags "${tag}")
        else()
            list(APPEND other_tags "${tag}")
        endif()
    endforeach()

    if(canonical_tags)
        list(GET canonical_tags 0 selected)
        list(REMOVE_AT canonical_tags 0)

        foreach(candidate IN LISTS canonical_tags)
            select_preferred_tag("${candidate}" "${selected}" candidate_is_preferred)
            if(candidate_is_preferred)
                set(selected "${candidate}")
            endif()
        endforeach()
        set(${out_var} "${selected}" PARENT_SCOPE)
        return()
    endif()
    if(other_tags)
        list(SORT other_tags COMPARE STRING ORDER DESCENDING)
        list(GET other_tags 0 selected)
        set(${out_var} "${selected}" PARENT_SCOPE)
    else()
        set(${out_var} "" PARENT_SCOPE)
    endif()
endfunction()

function(extract_describe_tag describe_output out_tag)
    string(REGEX MATCH "^v[0-9]+\\.[0-9]+\\.[0-9]+(-rc[0-9]*)?" tag "${describe_output}")
    set(${out_tag} "${tag}" PARENT_SCOPE)
endfunction()

function(run_git_describe_with_match git_root_dir match_pattern out_result out_output out_error)
    execute_process(COMMAND
        ${GIT_EXECUTABLE} -C ${git_root_dir}
            describe
            --tags --dirty --abbrev
            --match "${match_pattern}"
        RESULT_VARIABLE res
        OUTPUT_VARIABLE out
        ERROR_VARIABLE err
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_STRIP_TRAILING_WHITESPACE)

    set(${out_result} ${res} PARENT_SCOPE)
    set(${out_output} "${out}" PARENT_SCOPE)
    set(${out_error} "${err}" PARENT_SCOPE)
endfunction()

function(run_git_describe_fallback git_root_dir out_result out_output out_error)
    execute_process(COMMAND
        ${GIT_EXECUTABLE} -C ${git_root_dir}
            describe
            --tags --dirty --abbrev
            --always
            --match "v[0-9]*.[0-9]*.[0-9]"
            --match "v[0-9]*.[0-9]*.[0-9]-rc*"
        RESULT_VARIABLE res
        OUTPUT_VARIABLE out
        ERROR_VARIABLE err
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_STRIP_TRAILING_WHITESPACE)

    set(${out_result} ${res} PARENT_SCOPE)
    set(${out_output} "${out}" PARENT_SCOPE)
    set(${out_error} "${err}" PARENT_SCOPE)
endfunction()

execute_process(COMMAND
    ${GIT_EXECUTABLE} -C ${GIT_ROOT_DIR}
        tag --points-at HEAD
    RESULT_VARIABLE res
    OUTPUT_VARIABLE out
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_STRIP_TRAILING_WHITESPACE)

if(NOT res AND out)
    select_preferred_head_tag("${out}" preferred_head_tag)

    if(preferred_head_tag)
        set(PACKAGE_VERSION_STRING "${preferred_head_tag}")
        configure_version()
    endif()
endif()

run_git_describe_with_match(
    "${GIT_ROOT_DIR}"
    "v[0-9]*.[0-9]*.[0-9]"
    stable_describe_result
    stable_describe_output
    stable_describe_error)

# Keep describe preference consistent with HEAD tag selection:
# compare by version first, and for equal versions prefer stable over rc.
run_git_describe_with_match(
    "${GIT_ROOT_DIR}"
    "v[0-9]*.[0-9]*.[0-9]-rc*"
    rc_describe_result
    rc_describe_output
    rc_describe_error)

if(NOT stable_describe_result AND NOT rc_describe_result)
    extract_describe_tag("${stable_describe_output}" stable_describe_tag)
    extract_describe_tag("${rc_describe_output}" rc_describe_tag)
    select_preferred_tag("${rc_describe_tag}" "${stable_describe_tag}" prefer_rc_tag)
    if(prefer_rc_tag)
        set(describe_result ${rc_describe_result})
        set(describe_output "${rc_describe_output}")
        set(describe_error "${rc_describe_error}")
    else()
        set(describe_result ${stable_describe_result})
        set(describe_output "${stable_describe_output}")
        set(describe_error "${stable_describe_error}")
    endif()
elseif(NOT stable_describe_result)
    set(describe_result ${stable_describe_result})
    set(describe_output "${stable_describe_output}")
    set(describe_error "${stable_describe_error}")
elseif(NOT rc_describe_result)
    set(describe_result ${rc_describe_result})
    set(describe_output "${rc_describe_output}")
    set(describe_error "${rc_describe_error}")
else()
    set(describe_result 1)
    set(describe_output "")
    if(stable_describe_error)
        set(describe_error "${stable_describe_error}")
    else()
        set(describe_error "${rc_describe_error}")
    endif()
endif()

if(describe_result)
    run_git_describe_fallback("${GIT_ROOT_DIR}" describe_result describe_output describe_error)
endif()

if(describe_result)
    message(WARNING "Failure to get version from Git: ${describe_error}")
else()
    set(PACKAGE_VERSION_STRING "${describe_output}")
endif()

configure_version()
