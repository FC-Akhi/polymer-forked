rm -Rf build/ installation/
# export LD_LIBRARY_PATH=$HOME/compiler-projects/llvm-9-src-build/installation/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export LD_LIBRARY_PATH=$LOCAL_EXTERNAL_SSD_COMPILER_PROJ_PATH/llvm-9-src-build/installation/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

# This
# export LLVM_DIR=$HOME/compiler-projects/llvm-15-87ec6f41bba6d-src-build/build
export LLVM_DIR=$LOCAL_EXTERNAL_SSD_COMPILER_PROJ_PATH/llvm-15-87ec6f41bba6d-src-build/build



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
    -DMLIR_DIR=$LLVM_DIR/lib/cmake/mlir     \
    -DLLVM_EXTERNAL_LIT=$LLVM_DIR/bin/llvm-lit \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_BUILD_TYPE=DEBUG
     

export LD_LIBRARY_PATH="${PWD}/pluto/lib"${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

# export LD_LIBRARY_PATH=$HOME/compiler-projects/polymer-with-only-llvm/build/pluto/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
# export LD_LIBRARY_PATH=$LOCAL_EXTERNAL_SSD_COMPILER_PROJ_PATH/polymer-tests/polymer-okh-v0/build/pluto/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export LD_LIBRARY_PATH=$LOCAL_EXTERNAL_SSD_COMPILER_PROJ_PATH/polymer-tests/polymer-with-only-llvm/build/pluto/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}


cmake --build . --target check-polymer
ninja install
