
#include "superlu_zdefs_ex.h"
#include "utils.h"
#include <device_launch_parameters.h>

__global__ void zcatter_compress_kernel(int* usub, int klst, int iukp, int* d_nzvals, cuDoubleComplex* d_nzval,
    int ldv, int nbrow, cuDoubleComplex* d_tempv, int* d_segment_ptr, int* d_segment_offset,
    int non_zero_col_num, int segment_count)
{
    //printf("==================\n");
    if (blockIdx_y < non_zero_col_num)
    {
        int jj_ptr = blockIdx_y;
        cuDoubleComplex* nzval_cur = d_nzval + ldv * d_nzvals[jj_ptr];
        double* tempv_cur = (double*)(d_tempv + jj_ptr * nbrow);
        if (blockIdx_x < segment_count)
        {
            int ptr = blockIdx_x;// blockDim segment_compress.segment_count
            int i_start = d_segment_ptr[ptr];
            int i_end = d_segment_ptr[ptr + 1];
            int offset = d_segment_offset[ptr];
            double* NZVAL = (double*)(nzval_cur + offset);
            //printf("---------istart = %d, iend = %d nzval[%d] = %f -----------------------------\n", i_start, i_end);
            //printf("---------d_segment_ptr[%d] = %d istart = %d, iend = %d  -----------------------------\n",ptr, d_segment_ptr[ptr], i_start, i_end);
            if (threadIdx_x <= 2 * i_end - 2 * i_start)
            {
                int i = threadIdx_x + 2 * i_start;
                //printf("----------i = %d----------------\n", i);
                NZVAL[i] -= tempv_cur[i];
                
            }
        }
    }
}
__global__ void test(int* d_a)
{
    int idx = threadIdx.x;
    printf("--------d_a[%d] = %d------------\n", idx, d_a[idx]);
}

void zcatter_l_compress(int non_zero_col_num, int nbrow, int max_supernode_size, int nsupc, int* usub,
    int klst, int iukp, doublecomplex* nzval, int ldv,
    doublecomplex* tempv,
    indirect_index_segment_compress_t segment_compress, cuDoubleComplex* d_nzval, int* d_segment_ptr,
    int* d_segment_offset, cuDoubleComplex* d_tempv, int* d_nzvals, int temp_nbrow)
{
    int nzvals[max_supernode_size];
    for (int jj = 0; jj < nsupc; ++jj) {
        int_t segsize = klst - usub[iukp + jj];
        if (segsize) {
            //nzvals[non_zero_col_num] = nzval + jj * ldv;
            nzvals[non_zero_col_num] = jj;
            non_zero_col_num += 1;
        }
    }
    //printf("nsupc = %d-------------------------\n", nsupc);
    //printf("------------------------\n");
    //H2D
    checkGPU(cudaMemcpy(d_nzval, nzval, sizeof(cuDoubleComplex) * ldv * nsupc, cudaMemcpyHostToDevice));
    checkGPU(cudaMemcpy(d_tempv, tempv, sizeof(cuDoubleComplex) * nbrow * nsupc, cudaMemcpyHostToDevice));
    checkGPU(cudaMemcpy(d_nzvals, nzvals, sizeof(int) * max_supernode_size, cudaMemcpyHostToDevice));
    checkGPU(cudaMemcpy(d_segment_ptr, segment_compress.segment_ptr,
        sizeof(int) * (segment_compress.segment_count + 1), cudaMemcpyHostToDevice));
    checkGPU(cudaMemcpy(d_segment_offset, segment_compress.segment_offset,
        sizeof(int) * segment_compress.segment_count, cudaMemcpyHostToDevice));
    //kernel
    int* d_a, M = sizeof(int) * (segment_compress.segment_count + 1);
    cudaMalloc((void**)&d_a, M);
    cudaMemcpy(d_a, segment_compress.segment_ptr, M, cudaMemcpyHostToDevice);
    
    printf("segment_compress.segment_ptr[1] = %d\n", segment_compress.segment_ptr[1]);
    dim3 blocksize(1024, 1024);
    zcatter_compress_kernel << <64, 64>> > (usub, klst, iukp, d_nzvals, d_nzval,
        ldv, nbrow, d_tempv, d_a, d_segment_offset,
        non_zero_col_num, segment_compress.segment_count);

    //test << <1, 256 >> > (d_a);
    cudaDeviceSynchronize();
    //printf("0----------------------------------\n");
    cudaDeviceSynchronize();
    //D2H
    checkGPU(cudaMemcpy(nzval, d_nzval, sizeof(cuDoubleComplex) * ldv * nsupc, cudaMemcpyDeviceToHost));
    checkGPU(cudaMemcpy(tempv, d_tempv, sizeof(cuDoubleComplex) * nbrow * nsupc, cudaMemcpyDeviceToHost));
    //printf("0----------------------------------\n");
}

