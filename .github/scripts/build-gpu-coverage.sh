#!/usr/bin/env bash
set -euo pipefail

SRC="${1:?Usage: build-gpu-coverage.sh LLVM_SOURCE}"
ROOT=/home/ompc/tools/llvm

TC="$ROOT/installs/gpu/release"
HOST_BUILD="$ROOT/builds/gpu/release"
BUILD="${GITHUB_WORKSPACE:-$SRC}/.ompc-gpu-coverage-build"

CC="$TC/bin/clang"
CXX="$TC/bin/clang++"
LD="$TC/bin/ld.lld"

JOBS="${JOBS:-2}"

ISOLATION="$SRC/.github/scripts/ompc-gpu-coverage-isolation.cmake"

PROFILE_RT=/home/ompc/tools/llvm/installs/host/clang/lib/clang/21/lib/x86_64-unknown-linux-gnu/libclang_rt.profile.a

test -f "$ISOLATION"
test -f "$PROFILE_RT"
test -x "$CC"
test -x "$CXX"
test -x "$LD"

mkdir -p "$BUILD"

C_COMPILE='<CMAKE_C_COMPILER> <DEFINES> <INCLUDES> <FLAGS> -fprofile-instr-generate -fcoverage-mapping -o <OBJECT> -c <SOURCE>'
CXX_COMPILE='<CMAKE_CXX_COMPILER> <DEFINES> <INCLUDES> <FLAGS> -fprofile-instr-generate -fcoverage-mapping -o <OBJECT> -c <SOURCE>'

PROFILE_LINK="-Wl,--whole-archive $PROFILE_RT -Wl,--no-whole-archive"

cmake -S "$SRC/runtimes" -B "$BUILD" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER="$CC" \
  -DCMAKE_CXX_COMPILER="$CXX" \
  -DCMAKE_C_COMPILER_LAUNCHER=sccache \
  -DCMAKE_CXX_COMPILER_LAUNCHER=sccache \
  "-DCMAKE_C_COMPILE_OBJECT=$C_COMPILE" \
  "-DCMAKE_CXX_COMPILE_OBJECT=$CXX_COMPILE" \
  "-DCMAKE_SHARED_LINKER_FLAGS=-fuse-ld=lld $PROFILE_LINK" \
  "-DCMAKE_EXE_LINKER_FLAGS=--ld-path=$LD $PROFILE_LINK" \
  -DCMAKE_PROJECT_INCLUDE="$ISOLATION" \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DLLVM_DIR="$HOST_BUILD/lib/cmake/llvm" \
  -DLLVM_BINARY_DIR="$HOST_BUILD" \
  -DLLVM_ENABLE_RUNTIMES="openmp;offload" \
  -DLLVM_TARGETS_TO_BUILD="X86;NVPTX" \
  -DLLVM_ENABLE_PER_TARGET_RUNTIME_DIR=ON \
  -DLLVM_ENABLE_ASSERTIONS=OFF \
  -DLLVM_BUILD_TOOLS=ON \
  -DLIBOMP_USE_STDCPPLIB=ON \
  -DLIBOMP_ENABLE_SHARED=ON \
  -DLIBOMP_OMPX_TASKGRAPH=ON \
  -DOPENMP_STANDALONE_BUILD=0 \
  -DOPENMP_ENABLE_LIBOMPTARGET=ON \
  -DLIBOMPTARGET_ENABLE_DEBUG=OFF \
  "-DLIBOMPTARGET_PLUGINS_TO_BUILD=mpi;cuda;host" \
  "-DLIBOMPTARGET_DLOPEN_PLUGINS=mpi;cuda;host" \
  "-DLIBOMPTARGET_REMOTE_PLUGINS_TO_BUILD=cuda;host" \
  "-DLIBOMPTARGET_LLVM_LIBRARY_INTDIR=$BUILD/isolated/lib"

# Ne pas construire ALL : merge_runtime_commands peut écrire
# dans le répertoire partagé HOST_BUILD.
cmake --build "$BUILD" --parallel "$JOBS" \
  --target omp omptarget LLVMOffload \
           llvm-offload-mpi-proxy-device omptarget.nvptx

echo
echo "=== GPU COVERAGE BUILD ==="
ls -lh \
  "$BUILD/openmp/runtime/src/libomp.so" \
  "$BUILD/isolated/lib/libomptarget.so" \
  "$BUILD/isolated/bin/llvm-offload-mpi-proxy-device"

echo
echo "GPU_COVERAGE_BUILD=$BUILD"
