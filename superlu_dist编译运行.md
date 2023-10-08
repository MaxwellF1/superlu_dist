### 编译

1. superlu需要parmetis和metis做符号分解，首先安装这两个，这个官方都提供了编译的脚本，直接可以运行

2. superlu安装，有好几个脚本，分别代表cpu的版本，gpu版本，优化的版本，区别在于CMAKE_C_FLAGS 是否开启了优化的宏定义，目前scatter的优化开启之后会出现问题。

 



env的脚本

```shell
#!/bin/bash
spack load mpich@4.0.2
spack load gcc@7.5.0
module load cuda/11.6
spack load openblas@0.3.21 
export METIS_LIBRARIES=/share/home/wanghongyu/learn/pexsi/parmetis-4.0.3/lib
export  METIS_INCLUDE_DIR=/share/home/wanghongyu/learn/pexsi/parmetis-4.0.3/include

```



编译parmetis

```shell
cd /home/wanghongyu/learn/pexsi/parmetis-4.0.3/#进入parmetis的目录直接make就可以
make config metis_path=/home/wanghongyu/learn/pexsi/parmetis-4.0.3/metis openmp=set prefix=/home/wanghongyu/learn/pexsi/parmetis-4.0.3
make -j48 && make install
cd /home/wanghongyu/learn/pexsi/build

```



编译superlu----- run_cmake_build-opt-all-add-cublas.sh 在我们集群可以使用

```shell
#!/bin/bash
source /share/home/wanghongyu/learn/pexsi/build/env.sh #上面
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH//\/usr\/local\/cuda\/compat:/}
CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export CPATH=$CUDA_HOME/include:$CPATH
export PATH=$CUDA_HOME/bin:$PATH
THISHOST=`hostname -s`
echo "host: $THISHOST"

  rm -fr ssg1-build; mkdir ssg1-build; cd ssg1-build;
  export PARMETIS_ROOT=/share/home/wanghongyu/learn/pexsi/parmetis-4.0.3 
#  rm -fr int64-build; mkdir int64-build; cd int64-build;
#  export PARMETIS_ROOT=~/lib/static/64-bit/parmetis-4.0.3 
  export PARMETIS_BUILD_DIR=${PARMETIS_ROOT}/build/Linux-x86_64
  echo "ParMetis root: $PARMETIS_ROOT"
  cmake .. \
    -DTPL_PARMETIS_INCLUDE_DIRS="${PARMETIS_ROOT}/include;${PARMETIS_ROOT}/metis/include" \
    -DTPL_PARMETIS_LIBRARIES="${PARMETIS_BUILD_DIR}/libparmetis/libparmetis.a;${PARMETIS_BUILD_DIR}/libmetis/libmetis.a" \
    -DTPL_COMBBLAS_INCLUDE_DIRS="${COMBBLAS_ROOT}/install/include;${COMBBLAS_ROOT}/Applications/BipartiteMatchings" \
    -DTPL_COMBBLAS_LIBRARIES="${COMBBLAS_BUILD_DIR}/libCombBLAS.a" \
    -DCMAKE_C_FLAGS="-I/usr/local/cuda/include -std=c99 -O3 -DDEBUGlevel=0 -DPRNTlevel=2 -DOPT_CPU_UPANEL_TRSM -DOPT_GATHER_AVOID -DOPT_ZGEMM_ON_GPU" \
    -DCMAKE_C_COMPILER=mpicc \
    -DCMAKE_CXX_COMPILER=mpicxx \
    -DCMAKE_CXX_FLAGS="-std=c++11 -O3 -L/usr/local/cuda/lib64 -lcudart -lcudadevrt -lcublas -DPRNTlevel=0" \
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
    -DCMAKE_CUDA_FLAGS="-O3 -DOPT_GATHER_AVOID -I/share/home/wanghongyu/spack/opt/spack/linux-centos7-haswell/gcc-4.8.5/mpich-4.0.2-kaz3kvkaicme5wnjgd6zavsee2ybqkoa/include" \
    -DTPL_ENABLE_CUDALIB=ON \
    -DCUDA_LIBRARIES="/usr/local/cuda/lib64/libcublas.so;/usr/local/cuda/lib64/libcudart.so;/usr/local/cuda/lib64/libcusparse.so" \
    -DCMAKE_INSTALL_PREFIX="/share/home/wanghongyu/learn/pexsi/superlu_dist_opt"  
    make -j 48 VERBOSE=1 && make install

#    -DXSDK_INDEX_SIZE=64 \
#    -DXSDK_ENABLE_Fortran=TRUE \
#   -DTPL_ENABLE_PARMETISLIB=OFF
#    -DCMAKE_CXX_FLAGS="-std=c++14"
##    -DCMAKE_CXX_FLAGS="-std=c++11 -L/usr/local/cuda/lib64 -lcudart -lcudadevrt -lcublas -DPRNTlevel=2 -DMULTI_GPU -DOPT_GPU_LPANEL_TRSM" \
 #    -DCMAKE_C_FLAGS="-std=c99 -g -DPRNTlevel=0 -DDEBUGlevel=0 -DPRNTlevel=2 -DMULTI_GP -DOPT_CPU_UPANEL_TRSM -DOPT_GPU_LPANEL_TRSM -DOPT_GPU_UPANEL_TRSM -DOPT_CPU_UPANEL_TRSM" \

# make VERBOSE=1
# make test

```



### 运行

目前我只测试了（sparse.tamu.edu/Lee），他是复数的，效果还可以，优化之后的版本比baseline快了2-3倍，感觉主要是因为，supernode的划分方式不适合这个矩阵，导致非常不规则。

脚本run.sh

```shell
#!/bin/bash
#SBATCH --job-name=circuit_matrix_test
#SBATCH --partition=a100
#SBATCH -n 1
#SBATCH -w g03
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1
#SBATCH --output=./log/test_circuit_%j.out
#SBATCH --error=%j.err
#SBATCH --exclusive

module load mpich-4.0.2-gcc-4.8.5-kaz3kvk 
module load cmake-3.24.2-gcc-4.8.5-idyies2
module load cuda/11.3 
export OMP_NUM_THREADS=8 #g06 32 thread
export SUPERLU_BIND_MPI_GPU=1
export NSUP=1024
export NREL=512
export SUPERLU_MAXSUP=1024


export LD_LIBRARY_PATH=/share/home/wanghongyu/spack/opt/spack/linux-centos7-haswell/gcc-4.8.5/mpich-4.0.2-kaz3kvkaicme5wnjgd6zavsee2ybqkoa/lib:$LD_LIBRARY_PATH
# mpirun -n 8 ./pddrive -r 2 -c 2 -d 2 /share/home/wanghongyu/learn/pexsi/superlu_dist/lib64/EXAMPLE/circuit5M/circuit5M.mtx
mpirun -n 8 ./pzdrive -r 2 -c 2 -d 2 /share/home/wanghongyu/learn/pexsi/superlu_dist/lib64/EXAMPLE/data/fem_filter/fem_filter.mtx

```



### 我感觉需要做的

1. 我们只是实现了**复数的版本**，实数的版本还没有实现，需要重新写。
