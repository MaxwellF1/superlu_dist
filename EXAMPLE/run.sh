#!/bin/bash
#SBATCH --job-name=circuit_matrix_test
#SBATCH --partition=a100
#SBATCH -n 4
#SBATCH -w g04
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --output=./log/test_2880n1_%j.out
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
mpirun -n 8 ./pzdrive -r 2 -c 2 -d 2 /share/home/wanghongyu/learn/pexsi/superlu_dist/lib64/EXAMPLE/data/fem_filter/fem_filter.mtx
