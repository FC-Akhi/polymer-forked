# INSTALL & RUN POLYMER

Steps:

1. INSTALL PRE-REQUISITE FOR REGULAR BUILD + PLUTO
2. 


## 1. INSTALL PRE-REQUISITE FOR REGULAR BUILD + PLUTO


### _1.1. INSTALL PACKAGES FOR REGULAR BUILD TOOLS_

```sh
sudo apt-get install -y build-essential xutils-dev bison zlib1g-dev flex libglu1-mesa-dev
```


### _1.2. INSTALL PRE-REQUISITE PACKAGES FOR PLUTO_
PLUTO Prerequisites:
(Pluto) Automatic build tools (for Pluto), including `autoconf`, `automake`, and `libtool`.
(Pluto) Pre-built LLVM-9 tools (`clang-9` and `FileCheck-9`) and their header files are needed.
(Pluto) `libgmp` that is required by `isl`.
(Pluto) `flex` and `bison` for `clan` that Pluto depends on.
(Pluto) `texinfo` used to generate some docs.


```sh
sudo apt-get install libgmp3-dev texinfo autoconf libtool pkg-config
```

### _1.3. For seperate PLUTO build_
If you want to run Pluto seperately, it will search for `libtool` bin. Whereas, "apt-get install libtool" comes with only "libtoolize" bin. So you need to install the following

```sh
sudo apt-get install libtool-bin
```

### _1.4. For Polymer LLVM build_

If you don't want to get error like following
```sh
LLVMExports.cmake:55 (set_target_properties):
  The link interface of target "LLVMSupport" contains:

    Terminfo::terminfo

  but the target was not found.  Possible reasons include:

    * There is a typo in the target name.
    * A find_package call is missing for an IMPORTED target.
    * An ALIAS target is missing.
```
Install the `libncurses5-dev` package

```sh
sudo apt-get install libncurses5-dev
```


## 2. SETUP LLVM-9 for PLUTO (i.e. PLUTO which comes with/inside POLYMER)

**Why LLVM-9?**
- Because the `pet` lib inside the PLUTO expects LLVM-9 to build.
- `pet` also looks for `FileCheck` bin from LLVM. Doesn't matter coming from `llvm-9` or `llvm >= 10+`.

We have 2 possible options

2.1. Build LLVM-9 from source (HIGHLY Recommended)
2.2. Install LLVM-9 with ubuntu `apt-get` package installer. (Works, but not recommended. Only use it if you are in a VM to test something)





### _2.1. Build LLVM-9 from source_

- Clone the LLVM-9 repository

```sh
git clone -b release/9.x --depth 1 https://github.com/llvm/llvm-project.git llvm-9-src-build
```

- Create a `sh` file named `build-llvm-9-for-pluto-kumasento.sh` inside `llvm-9-src-build` and put the following content inside

```sh
mkdir -p build installation
cd build/

echo $PWD

cmake   \
    -G Ninja    \
    -S ../llvm  \
    -B .    \
    -DCMAKE_BUILD_TYPE=Release      \
    -DCMAKE_INSTALL_PREFIX=../installation  \
    -DLLVM_ENABLE_PROJECTS="llvm;clang;lld" \
    -DLLVM_INSTALL_UTILS=ON

cmake --build .

ninja install
```

- Run `build-llvm-9-for-pluto-kumasento.sh` and wait for build to finish

- Your final bin, header, lib will be populated in `llvm-9-src-build/installation` dir. Later we have to point this location inside the PLUTO build instructions.

- Note: (Optional) if accidentally your build got corrupted/error, then remove the `build` & `installation` dir, and run the `build-llvm-9-for-pluto-kumasento.sh` again

```sh
rm -Rf build/ installation/
```


### _2.2. Install LLVM-9 with ubuntu `apt-get` package installer_

- Install `clang-9` & `llvm-9`. **Both of the versions MUST be same.**. Actually `llvm-9` comes with `apt` package `libclang-9-dev`

```sh
sudo apt-get install clang-9 libclang-9-dev
```


- Use `update-alternatives` tool to make entries into the list for `clang-9` & `clang++-9`. **Remember, this time the `clang` we installed, is the first one in the system.**

```sh
sudo update-alternatives --install /usr/bin/clang clang /usr/bin/clang-9 9
sudo update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-9 9
```


- Now make entry for `llvm-config` which is the representative for whole `llvm` pack we installed

```sh
sudo update-alternatives --install /usr/bin/llvm-config llvm-config /usr/bin/llvm-config-9 9
```


- And we have to also make an entry for `FileCheck`. It comes with `llvm`

```sh
sudo update-alternatives --install /usr/bin/FileCheck FileCheck /usr/bin/FileCheck-9 9
```



## 3. SETUP LLVM-15 (Specific commit clone `87ec6f41bba6d72a3408e71cf19ae56feff523bc`) for POLYMER

- Polymer expects a specific LLVM build (`87ec6f41bba6d72a3408e71cf19ae56feff523bc`)


- So we have to clone the LLVM-15 first (Or you can clone the latest)


```sh
git clone -b release/15.x https://github.com/llvm/llvm-project.git llvm-15-87ec6f41bba6d-src-build
```


- Now get into `cd llvm-15-87ec6f41bba6d-src-build`


- Then we have to create a new branch to work with that specific commit hash (i.e. `87ec6f41bba6d`)

```sh
# Safe way is to, make a seperate branch with that specific clone
# Formula: git checkout -b <new_branch_name> <commit_sha_hash>
git checkout -b polymer-specific-commit-87ec6f4 87ec6f41bba6d72a3408e71cf19ae56feff523bc

# Chekch the current active branch
git branch

# Returns
* polymer-specific-commit-87ec6f4
  release/15.x
```

- Create a `sh` file named `build-llvm-15-87ec6f41bba6d72-for-polymer.sh` inside `llvm-15-87ec6f41bba6d-src-build` and put the following content inside

```sh

# git clone -b release/15.x https://github.com/llvm/llvm-project.git llvm-15-87ec6f41bba6d-src-build
# git checkout -b <new_branch_name> <commit_sha_hash>
# git checkout -b polymer-specific-commit-87ec6f4 87ec6f41bba6d72a3408e71cf19ae56feff523bc


mkdir -p build installation
cd build/

echo $PWD

cmake   \
    -G Ninja    \
    -S ../llvm  \
    -B .    \
    -DCMAKE_BUILD_TYPE=Release      \
    -DCMAKE_INSTALL_PREFIX=../installation  \
    -DLLVM_ENABLE_ASSERTIONS=ON     \
    -DLLVM_ENABLE_PROJECTS="llvm;mlir;clang;lld;openmp" \
    -DLLVM_INSTALL_UTILS=ON     \
    -DCMAKE_C_COMPILER=gcc    \
    -DCMAKE_CXX_COMPILER=g++    \
    -DLLVM_TARGETS_TO_BUILD="host"    \
    -DLLVM_ENABLE_OCAMLDOC=OFF \
    -DLLVM_ENABLE_BINDINGS=OFF \
    -DLLVM_OPTIMIZED_TABLEGEN=ON

cmake --build .
cmake --build . --target check-mlir
ninja install
```


- Run `build-llvm-15-87ec6f41bba6d72-for-polymer.sh` and wait for build to finish


- Your final bin, header, lib will be populated in `llvm-9-src-build/installation` dir. **But we will be using mainly `build` dir to build polymer**.


- Note: (Optional) if accidentally your build got corrupted/error, then remove the `build` & `installation` dir, and run the `build-llvm-15-87ec6f41bba6d72-for-polymer.sh` again.

```sh
rm -Rf build/ installation/
```



## 4. FINALLY SETUP POLYMER

We will playing with the latest commit as of today (Oct 27, 2023), because it works. The commit hash is `79652e13bda08accb5ac5de974f4bb0691038e29` in `main` branch.


### 4.1. _Clone Polymer and activate specific commit_

- Clone the POLYMER


```sh
git clone https://github.com/kumasento/polymer.git polymer-with-only-llvm

*If we clone from forked repo then*

git clone --recursive forked.git polymer-with-only-llvm
```


- Now get into `cd polymer-with-only-llvm`


- Then we have to create a new branch to work with that specific commit hash (i.e. `79652e13bda08accb5ac5de974f4bb0691038e29`)

```sh
# Safe way is to, make a seperate branch with that specific clone
# Formula: git checkout -b <new_branch_name> <commit_sha_hash>
git checkout -b polymer-specific-commit-79652e13bda08 79652e13bda08accb5ac5de974f4bb0691038e29

# Chekch the current active branch
git branch

# Returns
  main
* polymer-specific-commit-79652e13bda08
```


### 4.2. _Prepare PLUTO cmake config to work with LLVM-9 build_

- Open `polymer-with-only-llvm/cmake/AddPluto.cmake` in editor.

- Go and replace the `line 15 + line 16` with the LLVM-9 `PATH` for PLUTO build requirement

```sh
set(PLUTO_LIBCLANG_PREFIX "$LOCAL_EXTERNAL_SSD_COMPILER_PROJ_PATH/llvm-9-src-build/installation" CACHE STRING
    "The prefix to libclang used by Pluto (version < 10 required).")

OR

set(PLUTO_LIBCLANG_PREFIX "$HOME/compiler-projects/llvm-9-src-build/installation" CACHE STRING
    "The prefix to libclang used by Pluto (version < 10 required).")
```

- Go to `line 134`. You will find `STRING(REGEX REPLACE "\n" ";" PLUTO_SUBMODULE_GIT_STATUS ${PLUTO_SUBMODULE_GIT_STATUS})`. Replace it with following

```cmake
STRING(REGEX REPLACE "\n" ";" PLUTO_SUBMODULE_GIT_STATUS "${PLUTO_SUBMODULE_GIT_STATUS}")
```



### 4.3. _Prepare the `sh` file to build Polymer_

- Create a `sh` file named `build-polymer-for-specific-llvm-build.sh` inside `polymer-with-only-llvm` and put the following content inside

```sh

rm -Rf build/ installation/ pluto/

# Add LLVM-9 bins for different binaries (i.e. FileCheck)
export PATH=$HOME/compiler-projects/llvm-9-src-build/installation/bin${PATH:+:${PATH}}

# Add LLVM-9 lib to linker path (i.e. $LD_LIBRARY_PATH) so that "pet" lib inside pluto can find that
export LD_LIBRARY_PATH=$HOME/compiler-projects/llvm-9-src-build/installation/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}


# This
export LLVM_DIR=$HOME/compiler-projects/llvm-15-87ec6f41bba6d-src-build/build



# This works for polymer
# export LLVM_DIR=$HOME/compiler-projects/Polygeist-polymer/llvm-project/build

mkdir -p build installation
cd build/

echo $PWD

cmake  \
    -G Ninja    \
    -S ../  \
    -B .    \
    -DCMAKE_INSTALL_PREFIX=../installation  \
    -DMLIR_DIR=$LLVM_DIR/lib/cmake/mlir 	\
    -DLLVM_EXTERNAL_LIT=$LLVM_DIR/bin/llvm-lit \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_BUILD_TYPE=DEBUG
     

export LD_LIBRARY_PATH="${PWD}/pluto/lib"${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
cmake --build . --target check-polymer
ninja install
```

### 4.4. _Run the `sh` file to build Polymer_

```sh
*change the file permission*
chmod +x build-polymer-for-specific-llvm-build.sh

*Run the build script*
./build-polymer-for-specific-llvm-build.sh

```

- Now in terminal, add the pluto lib paths to `$LD_LIBRARY_PATH` so that `polymer-opt` can find the just built shared libraries (e.g. `libosl.so.0`) for pluto. **Remember, you have to do it everytime, if you cloze your current working terminal, and open a new terminal. If you want to make it permenant, add the following `export` to your `~/.profile` or `~/.bashrc`**

```sh
export LD_LIBRARY_PATH="$HOME/compiler-projects/polymer-with-only-llvm/build/pluto/lib"${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

or 

export LD_LIBRARY_PATH="$HOME/compiler-projects/all-polymer/polymer-with-only-llvm/build/pluto/lib"${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
```


- **FINALLY IT IS DONE**, Now run the following command to check if the POLYMER is working

```sh
./build/bin/polymer-opt -reg2mem -insert-redundant-load -extract-scop-stmt -canonicalize -pluto-opt -debug test/archive/PlutoTransform/matmul.mlir
```

This might give you following error

```sh
test/archive/PlutoTransform/matmul.mlir:6:8: error: custom op 'alloc' is unknown
  %A = alloc() : memref<64x64xf32>
```

To fix the error, go to `test/archive/PlutoTransform/matmul.mlir`, replace the `mlir` code with following one. The only change here is `alloc()` replaced with `memref.alloc()`. Then the bin should be working...

```sh
func @matmul() {
  %A = memref.alloc() : memref<64x64xf32>
  %B = memref.alloc() : memref<64x64xf32>
  %C = memref.alloc() : memref<64x64xf32>

  affine.for %i = 0 to 64 {
    affine.for %j = 0 to 64 {
      affine.for %k = 0 to 64 {
        %0 = affine.load %A[%i, %k] : memref<64x64xf32>
        %1 = affine.load %B[%k, %j] : memref<64x64xf32>
        %2 = arith.mulf %0, %1 : f32
        %3 = affine.load %C[%i, %j] : memref<64x64xf32>
        %4 = arith.addf %2, %3 : f32
        affine.store %4, %C[%i, %j] : memref<64x64xf32>
      }
    }
  }

  return
}
```

- **[IMPORTANT]:** If you close the terminal and later again run the `polymer-opt` command, you will encounter error like the following one

```sh

```

**[FIX]:** Load the `$HOME/compiler-projects/polymer-with-only-llvm/build/pluto/lib` path to your linker path `$LD_LIBRARY_PATH` like following. If you a permanent solution, add this to your OS's `$HOME/.profile`

```sh
export LD_LIBRARY_PATH="$HOME/compiler-projects/polymer-with-only-llvm/build/pluto/lib"${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

or

export LD_LIBRARY_PATH="$HOME/compiler-projects/all-polymer/polymer-with-only-llvm/build/pluto/lib"${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
```





## Avoid building `Pluto` again and again

- In `cmake/AddPluto.cmake`, replace with the following content

```cmake

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

set(PLUTO_LIBCLANG_PREFIX "$HOME/compiler-projects/llvm-9-src-build/installation" CACHE STRING
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


# # ===== Cmake command available options
# Usage: /home/user/installed-programs/cmake/bin/cmake -E <command> [arguments...]
# Available commands: 
#   capabilities              - Report capabilities built into cmake in JSON format
#   cat [--] <files>...       - concat the files and print them to the standard output
#   chdir dir cmd [args...]   - run command in a given directory
#   compare_files [--ignore-eol] file1 file2
#                               - check if file1 is same as file2
#   copy <file>... destination  - copy files to destination (either file or directory)
#   copy_directory <dir>... destination   - copy content of <dir>... directories to 'destination' directory
#   copy_directory_if_different <dir>... destination   - copy changed content of <dir>... directories to 'destination' directory
#   copy_if_different <file>... destination  - copy files if it has changed
#   echo [<string>...]        - displays arguments as text
#   echo_append [<string>...] - displays arguments as text but no new line
#   env [--unset=NAME ...] [NAME=VALUE ...] [--] <command> [<arg>...]
#                             - run command in a modified environment
#   environment               - display the current environment
#   make_directory <dir>...   - create parent and <dir> directories
#   md5sum <file>...          - create MD5 checksum of files
#   sha1sum <file>...         - create SHA1 checksum of files
#   sha224sum <file>...       - create SHA224 checksum of files
#   sha256sum <file>...       - create SHA256 checksum of files
#   sha384sum <file>...       - create SHA384 checksum of files
#   sha512sum <file>...       - create SHA512 checksum of files
#   remove [-f] <file>...     - remove the file(s), use -f to force it (deprecated: use rm instead)
#   remove_directory <dir>... - remove directories and their contents (deprecated: use rm instead)
#   rename oldname newname    - rename a file or directory (on one volume)
#   rm [-rRf] [--] <file/dir>... - remove files or directories, use -f to force it, r or R to remove directories and their contents recursively
#   sleep <number>...         - sleep for given number of seconds
#   tar [cxt][vf][zjJ] file.tar [file/dir1 file/dir2 ...]
#                             - create or extract a tar or zip archive
#   time command [args...]    - run command and display elapsed time
#   touch <file>...           - touch a <file>.
#   touch_nocreate <file>...  - touch a <file> but do not create it.
#   create_symlink old new    - create a symbolic link new -> old
#   create_hardlink old new   - create a hard link new -> old
#   true                      - do nothing with an exit code of 0
#   false                     - do nothing with an exit code of 1

```





// ==========================================================================


In Polymer:

- Polymer uses following polygeist version
https://github.com/llvm/Polygeist/tree/6ba6b7b8ac07c9d60994eb46b46682a9f76ea34e




In Polygeist:
- To use it, first clone the "Polygeist"
git clone https://github.com/llvm/Polygeist.git polygeist-specific-commit-hash

- Then checkout a new branch to work with that particular commit hash
git checkout -b polymer-specific 6ba6b7b8ac07c9d6099

- Then update the submodule (i.e. llvm-project (87ec6f41bba6d72a3408e71cf19ae56feff523bc))
git submodule update --init --recursive



In Polygeist:

- Find out, which llvm version it is using
https://github.com/llvm/llvm-project/tree/cbc378ecb87e3f31dd5aff91f2a621d500640412



Make new llvm-project:

git clone https://github.com/llvm/llvm-project.git llvm-src-87ec6f4-for-polygeist

git checkout -b polygeist-specific 87ec6f41bba6d72a3





====== Or =========

- Now clone seperate llvm-project
git clone https://github.com/llvm/llvm-project.git llvm-src-for-polygeist

- Then checkout a new branch to work with that particular commit hash
git checkout -b polygeist-specific cbc378ecb87e3f31dd5a




