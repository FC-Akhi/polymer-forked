# Install PLUTO as an external project.

include(ExternalProject)
project(RecursiveCopy)

set(PLUTO_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}/pluto/include")
set(PLUTO_LIB_DIR "${CMAKE_CURRENT_BINARY_DIR}/pluto/lib")
set(PLUTO_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/pluto")
set(PLUTO_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/pluto")
set(PLUTO_INSTALL_PREFIX_DIR "${PLUTO_SOURCE_DIR}/installation")
set(PLUTO_INSTALL_PREFIX_BIN_DIR "${PLUTO_INSTALL_PREFIX_DIR}/bin")
set(PLUTO_INSTALL_PREFIX_LIB_DIR "${PLUTO_INSTALL_PREFIX_DIR}/lib")
set(PLUTO_INSTALL_PREFIX_INCLUDE_DIR "${PLUTO_INSTALL_PREFIX_DIR}/include")

set(PLUTO_LIBCLANG_PREFIX "$LOCAL_EXTERNAL_SSD_COMPILER_PROJ_PATH/llvm-9-src-build/installation" CACHE STRING
    "The prefix to libclang used by Pluto (version < 10 required).")


# Function definition: to copy directories recursively (files, folders, symlinks...)
# It will automatically create destination folders, if it doesn't exist. So you donot have to create one.
# Include "project(RecursiveCopy)" at beginning to make this function to work
function(copy_directory src_path dest_path)
    # Create the destination directory
    file(MAKE_DIRECTORY ${dest_path})

    # Get all entries in the source directory
    file(GLOB ENTRIES RELATIVE "${src_path}" "${src_path}/*")

    foreach(ENTRY IN LISTS ENTRIES)
        set(SRC "${src_path}/${ENTRY}")
        set(DEST "${dest_path}/${ENTRY}")

        if(IS_DIRECTORY "${SRC}")
            # Recursive call if the entry is a directory
            copy_directory("${SRC}" "${DEST}")
        elseif(IS_SYMLINK "${SRC}")
            # Handle symlinks
            file(READ_SYMLINK "${SRC}" symlink_target)
            execute_process(
                COMMAND ${CMAKE_COMMAND} -E create_symlink "${symlink_target}" "${DEST}"
                RESULT_VARIABLE result
                ERROR_VARIABLE error_output
            )
            if(NOT result EQUAL "0")
                message(FATAL_ERROR "Failed to create symlink from ${SRC} to ${DEST}. Error: ${error_output}")
            endif()
            message(STATUS "Created symlink from ${SRC} to ${DEST}")
        else()
            # Copy files
            execute_process(
                COMMAND ${CMAKE_COMMAND} -E copy_if_different "${SRC}" "${DEST}"
                RESULT_VARIABLE result
                ERROR_VARIABLE error_output
            )
            if(NOT result EQUAL "0")
                message(FATAL_ERROR "Failed to copy from ${SRC} to ${DEST}. Error: ${error_output}")
            endif()
            message(STATUS "Copied ${SRC} to ${DEST}")
        endif()
    endforeach()
endfunction()



# If PLUTO_LIBCLANG_PREFIX is not set, we try to find a working version.
# Note that if you set this prefix to a invalid path, then that path will be cached and 
# the following code won't remedy that.
if (NOT PLUTO_LIBCLANG_PREFIX)
    message(STATUS "PLUTO_LIBCLANG_PREFIX not provided")

    # If the provided CMAKE_CXX_COMPILER is clang, we will check its version and use its prefix if version is matched.
    if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
        if (${CMAKE_CXX_COMPILER_VERSION} VERSION_LESS 10)
            execute_process(
                COMMAND bash -c "which ${CMAKE_CXX_COMPILER}"  
                OUTPUT_VARIABLE CLANG_ABSPATH
            )
            get_filename_component(CLANG_BINARY_DIR ${CLANG_ABSPATH} DIRECTORY)
            get_filename_component(CLANG_PREFIX_DIR ${CLANG_BINARY_DIR} DIRECTORY)

            message (STATUS "Provided CMAKE_CXX_COMPILER is clang of version less than 10 (${CMAKE_CXX_COMPILER_VERSION})") 
            message (STATUS "Use its prefix for PLUTO_LIBCLANG_PREFIX: ${CLANG_PREFIX_DIR}")

            set(PLUTO_LIBCLANG_PREFIX ${CLANG_PREFIX_DIR})
        endif()
    endif()

endif()

if (NOT PLUTO_LIBCLANG_PREFIX)
    set(PLUTO_LIBCLANG_PREFIX_CONFIG "")
else()
    # If a valid libclang is still not found, we try to search it on the system.
    message(STATUS "PLUTO_LIBCLANG_PREFIX: ${PLUTO_LIBCLANG_PREFIX}")
    set(PLUTO_LIBCLANG_PREFIX_CONFIG "--with-clang-prefix=${PLUTO_LIBCLANG_PREFIX}")
endif()





# Check if Pluto is already built at "polymer/pluto/installation/" (means "polymer/pluto/installation/lib/libpluto.so" exists)
if(NOT EXISTS "${PLUTO_INSTALL_PREFIX_LIB_DIR}/libpluto.so")
    
    message(STATUS "Pluto library not found or build required, configuring build process...")

    # Check if the source directory is a valid Git repository
    if(NOT EXISTS "${PLUTO_SOURCE_DIR}/.git")
        message(STATUS "Pluto not found at ${PLUTO_SOURCE_DIR}, downloading...")
        execute_process(
            COMMAND ${POLYMER_SOURCE_DIR}/scripts/update-pluto.sh
            OUTPUT_FILE ${CMAKE_CURRENT_BINARY_DIR}/update-pluto.log
        )
    endif()


    # Retrieve Pluto's git commit hash
    execute_process(
        COMMAND git rev-parse HEAD
        WORKING_DIRECTORY ${PLUTO_SOURCE_DIR}
        OUTPUT_VARIABLE PLUTO_GIT_HASH
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    message(STATUS "Pluto git hash: ${PLUTO_GIT_HASH}")


    # Retrieve all git submodules status
    execute_process(
        COMMAND git submodule status --recursive
        WORKING_DIRECTORY ${PLUTO_SOURCE_DIR}
        OUTPUT_VARIABLE PLUTO_SUBMODULE_GIT_STATUS
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    STRING(REGEX REPLACE "\n" ";" PLUTO_SUBMODULE_GIT_STATUS "${PLUTO_SUBMODULE_GIT_STATUS}")
    foreach(submodule IN LISTS PLUTO_SUBMODULE_GIT_STATUS)
        STRING(STRIP ${submodule} submodule)
        message(STATUS "${submodule}")
    endforeach()


    # Create the Pluto configuration shell script
    set(PLUTO_BUILD_COMMAND "${PLUTO_SOURCE_DIR}/configure-pluto.sh")
    file(WRITE ${PLUTO_BUILD_COMMAND}
        "#!/usr/bin/env bash\n"
        "mkdir -p ${PLUTO_INSTALL_PREFIX_DIR}\n"
        "./autogen.sh\n"
        "./configure --prefix=${PLUTO_INSTALL_PREFIX_DIR} ${PLUTO_LIBCLANG_PREFIX_CONFIG}\n"
        "make -j $(nproc)\n"
        "make install\n"
    )

    # Run the "${PLUTO_SOURCE_DIR}/configure-pluto.sh" shell file
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E env bash ${PLUTO_BUILD_COMMAND}
        WORKING_DIRECTORY ${PLUTO_SOURCE_DIR}
    ) 

else()
    message(STATUS "Pluto is already built. Skipping build process.")
endif()


# Copy all the ".so" files from "polymer/pluto/installation/lib" to "polymer/build/pluto/lib"
copy_directory("${PLUTO_INSTALL_PREFIX_LIB_DIR}" "${PLUTO_LIB_DIR}")

# RECURSIVELY copy all the header ".h" files from "polymer/pluto/installation/include" to "polymer/build/pluto/include"
copy_directory("${PLUTO_INSTALL_PREFIX_INCLUDE_DIR}" "${PLUTO_INCLUDE_DIR}")


# Add Pluto as external project
ExternalProject_Add(
    pluto
    PREFIX ${PLUTO_BIN_DIR}
    SOURCE_DIR ${PLUTO_SOURCE_DIR}
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
)






add_library(libpluto SHARED IMPORTED)
set_target_properties(libpluto PROPERTIES IMPORTED_LOCATION "${PLUTO_LIB_DIR}/libpluto.so")
add_library(libplutoosl SHARED IMPORTED)
set_target_properties(libplutoosl PROPERTIES IMPORTED_LOCATION "${PLUTO_LIB_DIR}/libosl.so")
add_library(libplutoisl SHARED IMPORTED)
set_target_properties(libplutoisl PROPERTIES IMPORTED_LOCATION "${PLUTO_LIB_DIR}/libisl.so")
add_library(libplutopip SHARED IMPORTED)
set_target_properties(libplutopip PROPERTIES IMPORTED_LOCATION "${PLUTO_LIB_DIR}/libpiplib_dp.so")
add_library(libplutopolylib SHARED IMPORTED)
set_target_properties(libplutopolylib PROPERTIES IMPORTED_LOCATION "${PLUTO_LIB_DIR}/libpolylib64.so")
add_library(libplutocloog SHARED IMPORTED)
set_target_properties(libplutocloog PROPERTIES IMPORTED_LOCATION "${PLUTO_LIB_DIR}/libcloog-isl.so")
add_library(libplutocandl STATIC IMPORTED)
set_target_properties(libplutocandl PROPERTIES IMPORTED_LOCATION "${PLUTO_LIB_DIR}/libcandl.so")

add_dependencies(libpluto pluto)
add_dependencies(libplutoisl pluto)
add_dependencies(libplutoosl pluto)
add_dependencies(libplutopip pluto)
add_dependencies(libplutopolylib pluto)
add_dependencies(libplutocloog pluto)
add_dependencies(libplutocandl pluto)











# ========== All Cmake How-tos ============

# # ===== Create directory
# execute_process(
#     COMMAND ${CMAKE_COMMAND} -E make_directory ${PLUTO_BINARY_DIR}
#     COMMAND ${CMAKE_COMMAND} -E make_directory ${PLUTO_LIB_DIR}
#     RESULT_VARIABLE result
#     ERROR_VARIABLE error_output
# )




# # ===== Copying symlink files, regular files
# Following process only supports copying symlink, regular files.
# Doesn't support creating dir, recursive file copying from different folders

# # Define filenames to copy
# set(PLUTO_LIB_SO_FILE_LIST_TO_COPY
#     "libpluto.so"
#     "libisl.so"
#     "libosl.so"
#     "libcloog-isl.so"
#     "libpiplib_dp.so"
#     "libpolylib64.so"
#     "libcandl.so"
# )


# # Get all files and symlinks in the source directory
# file(GLOB PLUTO_LIB_SO_FILE_LIST_TO_COPY RELATIVE "${PLUTO_INSTALL_PREFIX_LIB_DIR}" "${PLUTO_INSTALL_PREFIX_LIB_DIR}/*")

# # Loop through each file and copy
# foreach(FILE_NAME IN LISTS PLUTO_LIB_SO_FILE_LIST_TO_COPY)
#     # execute_process(
        
#     #     COMMAND ${CMAKE_COMMAND} -E copy -a "${PLUTO_INSTALL_PREFIX_LIB_DIR}/${FILE_NAME}" "${PLUTO_BINARY_DIR}/${FILE_NAME}"
#     #     RESULT_VARIABLE result
#     # )

#     # message(STATUS "I am hit...")
#     message(STATUS "Checking ${FILE_NAME}...")
#     if(IS_SYMLINK "${PLUTO_INSTALL_PREFIX_LIB_DIR}/${FILE_NAME}")
#         message(STATUS "${FILE_NAME} is a symlink.")
#         # Read the symlink to get the target
#         file(READ_SYMLINK "${PLUTO_INSTALL_PREFIX_LIB_DIR}/${FILE_NAME}" symlink_target)
#         # Create the same symlink in the destination
#         execute_process(
#             COMMAND ${CMAKE_COMMAND} -E create_symlink "${symlink_target}" "${PLUTO_LIB_DIR}/${FILE_NAME}"
#             RESULT_VARIABLE result
#             ERROR_VARIABLE error_output
#         )
#     else()
#         message(STATUS "${FILE_NAME} is not a symlink, copying normally.")
#         execute_process(
#             COMMAND ${CMAKE_COMMAND} -E copy_if_different
#                     "${PLUTO_INSTALL_PREFIX_LIB_DIR}/${FILE_NAME}"
#                     "${PLUTO_LIB_DIR}/${FILE_NAME}"
#             RESULT_VARIABLE result
#             ERROR_VARIABLE error_output
#         )
#     endif()


#     if(NOT result EQUAL "0")
#         message(FATAL_ERROR "Failed to copy ${FILE_NAME}: ${result}")
#     endif()
#     message(STATUS "Successfully copied ${FILE_NAME} from ${PLUTO_INSTALL_PREFIX_LIB_DIR} to ${PLUTO_BIN_DIR}")
# endforeach()


# # ===== Create directory
