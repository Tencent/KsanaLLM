# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================

# Flash Attention 3 Auto Build Module
# This module automatically downloads, builds and installs Flash Attention 3 from source

message(STATUS "Setting up Flash Attention 3 Auto Build")
find_package(Git QUIET)

# Set the Flash Attention 3 source directory
set(FA3_SOURCE_DIR ${CMAKE_BINARY_DIR}/third_party/flash-attention)
set(FA3_HOPPER_DIR ${FA3_SOURCE_DIR}/hopper)

# Function to check if flash_attn_3 is available in Python
function(check_flash_attn_3_available result_var)
    execute_process(
        COMMAND python -c "import flash_attn_3; print('available')"
        OUTPUT_VARIABLE FA3_CHECK_OUTPUT
        ERROR_QUIET
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    
    if("${FA3_CHECK_OUTPUT}" STREQUAL "available")
        set(${result_var} TRUE PARENT_SCOPE)
        message(STATUS "Flash Attention 3 is already available in Python environment")
    else()
        set(${result_var} FALSE PARENT_SCOPE)
        message(STATUS "Flash Attention 3 is not available in Python environment")
    endif()
endfunction()

# Function to auto build Flash Attention 3
function(auto_build_flash_attn_3)
    message(STATUS "Starting Flash Attention 3 auto build process...")
    
    # Check if Git is available
    find_package(Git REQUIRED)
    
    # Check if Python is available
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    
    # Create the third_party directory if it doesn't exist
    file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/third_party)
    
    message(STATUS "Step 1: Preparing Flash Attention repository...")
    # Step 1: Clone the flash-attention repository only if it doesn't exist
    if(NOT EXISTS ${FA3_SOURCE_DIR})
        message(STATUS "Cloning Flash Attention repository...")
        execute_process(
            COMMAND ${GIT_EXECUTABLE} clone https://github.com/Dao-AILab/flash-attention.git ${FA3_SOURCE_DIR}
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/third_party
            RESULT_VARIABLE GIT_CLONE_RESULT
            OUTPUT_VARIABLE GIT_CLONE_OUTPUT
            ERROR_VARIABLE GIT_CLONE_ERROR
        )
        
        if(NOT GIT_CLONE_RESULT EQUAL 0)
            message(FATAL_ERROR "Failed to clone Flash Attention repository: ${GIT_CLONE_ERROR}")
        endif()
        message(STATUS "Flash Attention repository cloned successfully")
    else()
        message(STATUS "Flash Attention repository already exists, skipping clone")
        # Update the repository to make sure we have the latest tags
        execute_process(
            COMMAND ${GIT_EXECUTABLE} fetch --tags
            WORKING_DIRECTORY ${FA3_SOURCE_DIR}
            RESULT_VARIABLE GIT_FETCH_RESULT
            OUTPUT_QUIET
            ERROR_QUIET
        )
        if(GIT_FETCH_RESULT EQUAL 0)
            message(STATUS "Updated repository tags")
        endif()
    endif()
    
    message(STATUS "Step 2: Checking out to version 2.8.2...")
    # Step 2: Checkout to version 2.8.2
    execute_process(
        COMMAND ${GIT_EXECUTABLE} checkout v2.8.2
        WORKING_DIRECTORY ${FA3_SOURCE_DIR}
        RESULT_VARIABLE GIT_CHECKOUT_RESULT
        OUTPUT_VARIABLE GIT_CHECKOUT_OUTPUT
        ERROR_VARIABLE GIT_CHECKOUT_ERROR
    )
    
    if(NOT GIT_CHECKOUT_RESULT EQUAL 0)
        message(FATAL_ERROR "Failed to checkout Flash Attention to v2.8.2: ${GIT_CHECKOUT_ERROR}")
    endif()
    message(STATUS "Checked out to Flash Attention v2.8.2 successfully")
    
    message(STATUS "Step 3: Building and installing Flash Attention 3...")
    # Step 3: Build and install FA3 from hopper directory
    execute_process(
        COMMAND ${Python3_EXECUTABLE} setup.py install
        WORKING_DIRECTORY ${FA3_HOPPER_DIR}
        RESULT_VARIABLE FA3_BUILD_RESULT
        OUTPUT_VARIABLE FA3_BUILD_OUTPUT
        ERROR_VARIABLE FA3_BUILD_ERROR
    )
    
    if(NOT FA3_BUILD_RESULT EQUAL 0)
        message(WARNING "Flash Attention 3 build failed: ${FA3_BUILD_ERROR}")
        message(STATUS "Build output: ${FA3_BUILD_OUTPUT}")
        message(STATUS "You may need to install Flash Attention 3 manually")
    else()
        message(STATUS "Flash Attention 3 built and installed successfully")
    endif()
    
    message(STATUS "Flash Attention 3 auto build process completed")
    message(STATUS "Source directory: ${FA3_SOURCE_DIR}")
    message(STATUS "Hopper directory: ${FA3_HOPPER_DIR}")
    
    # Set a global property to indicate FA3 auto build is set up
    set_property(GLOBAL PROPERTY FA3_AUTO_BUILD_SETUP TRUE)
endfunction()

# Function to add FA3 dependency to a target
function(add_fa3_dependency target_name)
    get_property(FA3_AUTO_BUILD_SETUP GLOBAL PROPERTY FA3_AUTO_BUILD_SETUP)
    if(FA3_AUTO_BUILD_SETUP)
        add_dependencies(${target_name} fa3_installed)
        message(STATUS "Added FA3 auto build dependency to target: ${target_name}")
    endif()
endfunction()