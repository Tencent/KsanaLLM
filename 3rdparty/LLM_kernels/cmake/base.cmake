# Copyright 2024 Tencent Inc.  All rights reserved.

if(NOT PYTHON_EXECUTABLE)
  find_package(PythonInterp)
  message(STATUS "found python executable: ${PYTHON_EXECUTABLE}")
endif()

function(cc_test TARGET_NAME)
  if(WITH_TESTING)
    set(oneValueArgs "")
    set(multiValueArgs SRCS DEPS LINKS ARGS)
    cmake_parse_arguments(cc_test "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    add_executable(${TARGET_NAME} ${cc_test_SRCS})
    target_link_libraries(${TARGET_NAME} PRIVATE ${cc_test_DEPS} ${cc_test_LINKS} gtest_main gtest gmock_main gmock -pthread)
    add_dependencies(${TARGET_NAME} ${cc_test_DEPS} gtest_main gtest gmock_main gmock)
    add_test(NAME ${TARGET_NAME}
      COMMAND ${TARGET_NAME} ${cc_test_ARGS}
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
  endif()
endfunction(cc_test)

function(py_test TARGET_NAME)
  if(WITH_TESTING)
    set(options "")
    set(oneValueArgs "")
    set(multiValueArgs SRCS DEPS ARGS ENVS)
    cmake_parse_arguments(py_test "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    set(working_dir ${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME})
    file(MAKE_DIRECTORY ${working_dir})
    add_test(NAME ${TARGET_NAME}
      COMMAND ${CMAKE_COMMAND}
      ${PYTHON_EXECUTABLE} -u ${py_test_SRCS} ${py_test_ARGS}
      WORKING_DIRECTORY ${working_dir})
    message(STATUS "test added ${TARGET_NAME} ${working_dir}/${py_test_SRCS}")
  endif()
endfunction()

function(subproject_version subproject_path VERSION_VAR)
  # Read CMakeLists.txt for subproject and extract project() call(s) from it.
  file(STRINGS "${subproject_path}/CMakeLists.txt" project_calls REGEX "[ \t]*project\\(")

  # For every project() call try to extract its VERSION option
  foreach(project_call ${project_calls})
    string(REGEX MATCH "VERSION[ ]+([^ )]+)" version_param "${project_call}")

    if(version_param)
      set(version_value "${CMAKE_MATCH_1}")
    endif()
  endforeach()

  if(version_value)
    set(${VERSION_VAR} "${version_value}" PARENT_SCOPE)
  else()
    message(WARNING "WARNING: Cannot extract version for subproject '${subproject_path}'")
  endif()
endfunction(subproject_version)

function(calculate_files_md5 files_list output_md5)
  set(combined_content "")
  foreach(file ${files_list})
    if(EXISTS ${file})
      file(READ ${file} content)
      string(APPEND combined_content ${content})
    endif()
  endforeach()
  string(MD5 md5sum "${combined_content}")
  set(${output_md5} ${md5sum} PARENT_SCOPE)
endfunction()