#!/bin/bash
source /share/home/wanghongyu/learn/pexsi/build/env.sh
CC=mpicc
CXX=mpicxx
if [ -z "$PEXSI_DIR" ]; then
  PEXSI_DIR=${HOME}/Projects/pexsi
fi

PARMETIS_ROOT=${PEXSI_DIR}/external/parmetis_4.0.3
export PARMETIS_BUILD_DIR=${PARMETIS_ROOT}/build/Linux-x86_64
SuperLU_INSTALL=${PEXSI_DIR}/external/SuperLU_DIST_8.1.0
  cmake .. \
    -DTPL_PARMETIS_INCLUDE_DIRS="${PARMETIS_ROOT}/include;${PARMETIS_ROOT}/metis/include" \
    -DTPL_PARMETIS_LIBRARIES="${PARMETIS_ROOT}/libparmetis.a;${PARMETIS_ROOT}/libmetis.a" \
    -DTPL_COMBBLAS_INCLUDE_DIRS="${COMBBLAS_ROOT}/install/include;${COMBBLAS_ROOT}/Applications/BipartiteMatchings" \
    -DTPL_COMBBLAS_LIBRARIES="${COMBBLAS_BUILD_DIR}/libCombBLAS.a" \
    -DCMAKE_C_FLAGS="-std=c99 -O3" \
    -DCMAKE_C_COMPILER=${CC} \
    -DCMAKE_CXX_COMPILER=${CXX} \
    -DCMAKE_CXX_FLAGS="-std=c++11 -O3 -DPRNTlevel=0" \
    -DCMAKE_LINKER=mpicxx \
    -Denable_openmp=ON \
    -DTPL_ENABLE_INTERNAL_BLASLIB=OFF \
    -DTPL_ENABLE_COMBBLASLIB=OFF \
    -DTPL_ENABLE_LAPACKLIB=OFF \
    -DBUILD_SHARED_LIBS=OFF \
    -DXSDK_ENABLE_Fortran=OFF \
    -DCMAKE_INSTALL_PREFIX=""  

make -j 48 VERBOSE=1 && make install


