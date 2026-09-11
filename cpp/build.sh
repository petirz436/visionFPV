#!/usr/bin/env bash
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
BUILD_DIR="${SCRIPT_DIR}/build"

echo "=========================================================="
echo " Building vision_fpv3 [C++ Edition]..."
echo "=========================================================="

mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

cmake .. -DCMAKE_BUILD_TYPE=Release
make -j"$(nproc)"

echo ""
echo "=========================================================="
echo " Build Success! Executable created at:"
echo " ${BUILD_DIR}/vision_fpv3"
echo "=========================================================="
