#!/bin/bash
source /data/home/scv7260/wanghongyu/pexsi/build/env.sh
export LD_LIBRARY_PATH=/data/apps/cuda/11.4/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH//\/usr\/local\/cuda\/compat:/}
export CUDA_HOME=/data/apps/cuda/11.4
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export CPATH=$CUDA_HOME/include:$CPATH
export PATH=$CUDA_HOME/bin:$PATH
export MPICH_DIR=/data/home/scv7260/wanghongyu/spack-develop/opt/spack/linux-centos7-skylake_avx512/gcc-7.5.0/mpich-4.0.2-d6sz7o4eahwrlhiocyoxdy5flsran3e2
THISHOST=`hostname -s`
echo "host: $THISHOST"

  rm -fr ssg1-build; mkdir ssg1-build; cd ssg1-build;
  export PARMETIS_ROOT=/data/home/scv7260/wanghongyu/pexsi/parmetis-4.0.3 
#  rm -fr int64-build; mkdir int64-build; cd int64-build;
#  export PARMETIS_ROOT=~/lib/static/64-bit/parmetis-4.0.3 
  export PARMETIS_BUILD_DIR=${PARMETIS_ROOT}/build/Linux-x86_64
  echo "ParMetis root: $PARMETIS_ROOT"
  cmake .. \
    -DTPL_PARMETIS_INCLUDE_DIRS="${PARMETIS_ROOT}/include;${PARMETIS_ROOT}/metis/include" \
    -DTPL_PARMETIS_LIBRARIES="${PARMETIS_BUILD_DIR}/libparmetis/libparmetis.a;${PARMETIS_BUILD_DIR}/libmetis/libmetis.a" \
    -DTPL_COMBBLAS_INCLUDE_DIRS="${COMBBLAS_ROOT}/install/include;${COMBBLAS_ROOT}/Applications/BipartiteMatchings" \
    -DTPL_COMBBLAS_LIBRARIES="${COMBBLAS_BUILD_DIR}/libCombBLAS.a" \
    -DCMAKE_C_FLAGS="-I${CUDA_HOME}/include -std=c99 -O3 -DDEBUGlevel=0 -DPRNTlevel=2 -DOPT_CPU_UPANEL_TRSM -DOPT_GATHER_AVOID -DOPT_SCATTER -DOPT_ZGEMM_ON_GPU" \
    -DCMAKE_C_COMPILER=mpicc \
    -DCMAKE_CXX_COMPILER=mpicxx \
    -DCMAKE_CXX_FLAGS="-std=c++11 -O3 -L${CUDA_HOME}/lib64 -lcudart -lcudadevrt -lcublas -DPRNTlevel=0" \
    -DCMAKE_Fortran_COMPILER=mpif90 \
    -DCMAKE_Fortran_FLAGS="-I/usr/local/include -L/usr/local/cuda/lib64 -lcuda -lcudart -lcudadevrt -lcublas" \
    -DCMAKE_LINKER=mpicxx \
    -Denable_openmp=ON \
    -DTPL_ENABLE_INTERNAL_BLASLIB=OFF \
    -DTPL_ENABLE_COMBBLASLIB=OFF \
    -DTPL_ENABLE_LAPACKLIB=OFF \
    -DBUILD_SHARED_LIBS=OFF \
    -DXSDK_ENABLE_Fortran=OFF \
    -DCMAKE_CUDA_ARCHITECTURES=80 \
    -DCMAKE_CUDA_FLAGS="-O3 -DOPT_GATHER_AVOID -DOPT_CPU_UPANEL_TRSM -DOPT_GPU_UPANEL_TRSM -I${MPICH_DIR}/include" \
    -DTPL_ENABLE_CUDALIB=OFF \
    -DCUDA_LIBRARIES="/usr/local/cuda/lib64/libcublas.so;/usr/local/cuda/lib64/libcudart.so;/usr/local/cuda/lib64/libcusparse.so" \
    -DCMAKE_INSTALL_PREFIX="/share/home/wanghongyu/learn/pexsi/superlu_dist"  
    make -j 48 VERBOSE=1 && make install

#    -DXSDK_INDEX_SIZE=64 \
#    -DXSDK_ENABLE_Fortran=TRUE \
#   -DTPL_ENABLE_PARMETISLIB=OFF
#    -DCMAKE_CXX_FLAGS="-std=c++14"
##    -DCMAKE_CXX_FLAGS="-std=c++11 -L/usr/local/cuda/lib64 -lcudart -lcudadevrt -lcublas -DPRNTlevel=2 -DMULTI_GPU -DOPT_GPU_LPANEL_TRSM" \
 #    -DCMAKE_C_FLAGS="-std=c99 -g -DPRNTlevel=0 -DDEBUGlevel=0 -DPRNTlevel=2 -DMULTI_GP -DOPT_CPU_UPANEL_TRSM -DOPT_GPU_LPANEL_TRSM -DOPT_GPU_UPANEL_TRSM -DOPT_CPU_UPANEL_TRSM" \

# make VERBOSE=1
# make test