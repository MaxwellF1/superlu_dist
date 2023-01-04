#!/bin/bash
spack load mpich@4.0.2%gcc@7.5.0
spack load gcc@7.5.0
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH//\/usr\/local\/cuda\/compat:/}
CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export CPATH=$CUDA_HOME/include:$CPATH
export PATH=$CUDA_HOME/bin:$PATH
NVARCH=`uname -s`_`uname -m`; export NVARCH
NVCOMPILERS=/home/wanghongyu/nvhpc/Linux_x86_64; export NVCOMPILERS
MANPATH=$MANPATH:$NVCOMPILERS/$NVARCH/22.9/compilers/man; export MANPATH
PATH=$NVCOMPILERS/$NVARCH/22.9/compilers/bin:$PATH; export PATH
spack load mpich@4.0.2%gcc@7.5.0
THISHOST=`hostname -s`
echo "host: $THISHOST"
if [ "$THISHOST" == "znkxjs981" ]
then
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
    -DCMAKE_CXX_FLAGS="-std=c++11 -L/usr/local/cuda/lib64 -lcudart -lcudadevrt" \
    -DCMAKE_Fortran_COMPILER=mpifort \
    -DCMAKE_Fortran_FLAGS="-L/usr/local/cuda/lib64 -lcudart -lcudadevrt"
    -DCMAKE_LINKER=mpicxx \
    -Denable_openmp=ON \
    -DTPL_ENABLE_INTERNAL_BLASLIB=OFF \
    -DTPL_ENABLE_COMBBLASLIB=OFF \
    -DTPL_ENABLE_LAPACKLIB=OFF \
    -DBUILD_SHARED_LIBS=OFF \
    -DCMAKE_CUDA_ARCHITECTURES=80 \
    -DCMAKE_CUDA_FLAGS="-I/usr/local/spack/opt/spack/linux-centos7-skylake_avx512/gcc-7.5.0/mpich-4.0.2-fokcm5f6j265rcne27n7bgbiv6ni3ryv/include" \
    -DTPL_ENABLE_CUDALIB=ON \
    -DCUDA_LIBRARIES="/usr/local/cuda/lib64/libcublas.so;/usr/local/cuda/lib64/libcudart.so;/usr/local/cuda/lib64/libcusparse.so" \
    -DCMAKE_INSTALL_PREFIX="/home/wanghongyu/learn/pexsi/superlu_dist"  
    make -j 48 VERBOSE=1 && make install
fi
#    -DXSDK_INDEX_SIZE=64 \
#    -DXSDK_ENABLE_Fortran=TRUE \
#   -DTPL_ENABLE_PARMETISLIB=OFF
#    -DCMAKE_CXX_FLAGS="-std=c++14"

# make VERBOSE=1
# make test
