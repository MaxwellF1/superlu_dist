#!/bin/bash
export MY_HOME=/home/jiaweile/project/fengguofeng/SuperLU-24/fgf

export CUDA_PATH=/home/paraai_test/dat01/software/nvidia/cuda/11.6
export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH
export CPATH=$CUDA_PATH/include:$CPATH
export PATH=$CUDA_PATH/bin:$PATH

export MPI_PATH=/software/mpich/mpich-3.4.1
export MPI_LIBRARY_PATH=$MPI_PATH/lib        
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$MPI_LIBRARY_PATH
export CPATH=$MPI_PATH/include:$CPATH

THISHOST=`hostname -s`
echo "host: $THISHOST"

  rm -fr opt-gpu-build; mkdir opt-gpu-build; cd opt-gpu-build;
  export PARMETIS_ROOT=$MY_HOME/Software/ParMETIS/parmetis-4.0.3 
  export PARMETIS_BUILD_DIR=${PARMETIS_ROOT}/build/Linux-x86_64
  echo "ParMetis root: $PARMETIS_ROOT"
  cmake .. \
    -DTPL_PARMETIS_INCLUDE_DIRS="${PARMETIS_ROOT}/include;${PARMETIS_ROOT}/metis/include" \
    -DTPL_PARMETIS_LIBRARIES="${PARMETIS_BUILD_DIR}/libparmetis/libparmetis.a;${PARMETIS_BUILD_DIR}/libmetis/libmetis.a" \
    -DTPL_COMBBLAS_INCLUDE_DIRS="${COMBBLAS_ROOT}/install/include;${COMBBLAS_ROOT}/Applications/BipartiteMatchings" \
    -DTPL_COMBBLAS_LIBRARIES="${COMBBLAS_BUILD_DIR}/libCombBLAS.a" \
    -DCMAKE_C_FLAGS="-I$CUDA_PATH/include -std=c99 -O3 -DDEBUGlevel=0 -DPRNTlevel=2 -DOPT_CPU_UPANEL_TRSM -DOPT_GATHER_AVOID" \
    -DCMAKE_C_COMPILER=mpicc \
    -DCMAKE_CXX_COMPILER=mpicxx \
    -DXSDK_INDEX_SIZE=64  \
    -DCMAKE_CXX_FLAGS="-std=c++11 -O3 -L$CUDA_PATH/lib64 -lcudart -lcudadevrt -lcublas -DPRNTlevel=0" \
    -DCMAKE_Fortran_COMPILER=mpif90 \
    -DCMAKE_Fortran_FLAGS="-I/usr/local/include -L$CUDA_PATH/lib64 -lcuda -lcudart -lcudadevrt -lcublas" \
    -DCMAKE_LINKER=mpicxx \
    -Denable_openmp=ON \
    -DTPL_ENABLE_INTERNAL_BLASLIB=OFF \
    -DTPL_ENABLE_COMBBLASLIB=OFF \
    -DTPL_ENABLE_LAPACKLIB=OFF \
    -DBUILD_SHARED_LIBS=OFF \
    -DXSDK_ENABLE_Fortran=OFF \
    -DCMAKE_CUDA_ARCHITECTURES=80 \
    -DCMAKE_CUDA_FLAGS="-O3 -DOPT_GATHER_AVOID" \
    -DTPL_ENABLE_CUDALIB=ON \
    -DCUDA_LIBRARIES="$CUDA_PATH/lib64/libcublas.so;$CUDA_PATH/lib64/libcudart.so;$CUDA_PATH/lib64/libcusparse.so" \
    -DCMAKE_INSTALL_PREFIX=.  
    echo "-----------------------------------------------------------------------------------Configuration Done---------------------------------------------------------------------------------------------------------------------"
    sleep 3
    make -j 48 VERBOSE=1
    echo "-----------------------------------------------------------------------------------Buid Done---------------------------------------------------------------------------------------------------------------------"
    sleep 3
    make install
    echo "----------------------------------------------------------------------------------Library installed---------------------------------------------------------------------------------------------------------------------"



#    -DXSDK_INDEX_SIZE=64 \
#    -DXSDK_ENABLE_Fortran=TRUE \
#   -DTPL_ENABLE_PARMETISLIB=OFF
#    -DCMAKE_CXX_FLAGS="-std=c++14"
##    -DCMAKE_CXX_FLAGS="-std=c++11 -L/usr/local/cuda/lib64 -lcudart -lcudadevrt -lcublas -DPRNTlevel=2 -DMULTI_GPU -DOPT_GPU_LPANEL_TRSM" \
 #    -DCMAKE_C_FLAGS="-std=c99 -g -DPRNTlevel=0 -DDEBUGlevel=0 -DPRNTlevel=2 -DMULTI_GP -DOPT_CPU_UPANEL_TRSM -DOPT_GPU_LPANEL_TRSM -DOPT_GPU_UPANEL_TRSM -DOPT_CPU_UPANEL_TRSM" \

# make VERBOSE=1
# make test
