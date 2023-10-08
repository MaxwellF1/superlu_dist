#!/bin/bash

## if [ !$?NERSC_HOST ]

THISHOST=`hostname -s`
echo "host: $THISHOST"

  rm -fr ssg1-build; mkdir ssg1-build; cd ssg1-build;
  export PARMETIS_ROOT=/home/wanghongyu/learn/pexsi/parmetis-4.0.3 
#  rm -fr int64-build; mkdir int64-build; cd int64-build;
#  export PARMETIS_ROOT=~/lib/static/64-bit/parmetis-4.0.3 
  export PARMETIS_BUILD_DIR=${PARMETIS_ROOT}/build/Linux-x86_64
  echo "ParMetis root: $PARMETIS_ROOT"
  cmake .. \
    -DTPL_PARMETIS_INCLUDE_DIRS="${PARMETIS_ROOT}/include;${PARMETIS_ROOT}/metis/include" \
    -DTPL_PARMETIS_LIBRARIES="${PARMETIS_BUILD_DIR}/libparmetis/libparmetis.a;${PARMETIS_BUILD_DIR}/libmetis/libmetis.a" \
    -DTPL_COMBBLAS_INCLUDE_DIRS="${COMBBLAS_ROOT}/install/include;${COMBBLAS_ROOT}/Applications/BipartiteMatchings" \
    -DTPL_COMBBLAS_LIBRARIES="${COMBBLAS_BUILD_DIR}/libCombBLAS.a" \
    -DCMAKE_C_FLAGS="-std=c99 -O3 -g -DPRNTlevel=0 -DDEBUGlevel=0" \
    -DCMAKE_C_COMPILER=mpicc \
    -DCMAKE_CXX_COMPILER=mpicxx \
    -DCMAKE_CXX_FLAGS="-std=c++11" \
    -DCMAKE_Fortran_COMPILER=mpif90 \
    -DCMAKE_LINKER=mpicxx \
    -Denable_openmp=ON \
    -DTPL_ENABLE_INTERNAL_BLASLIB=OFF \
    -DTPL_ENABLE_COMBBLASLIB=OFF \
    -DTPL_ENABLE_LAPACKLIB=OFF \
    -DBUILD_SHARED_LIBS=OFF \
    -DCMAKE_INSTALL_PREFIX=.

#    -DXSDK_INDEX_SIZE=64 \
#    -DXSDK_ENABLE_Fortran=TRUE \
#   -DTPL_ENABLE_PARMETISLIB=OFF
#    -DCMAKE_CXX_FLAGS="-std=c++14"

# make VERBOSE=1
# make test
