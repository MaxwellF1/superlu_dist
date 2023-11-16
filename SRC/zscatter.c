/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required
approvals from U.S. Dept. of Energy)

All rights reserved.

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/

/*! @file
 * \brief Scatter the computed blocks into LU destination.
 *
 * <pre>
 * -- Distributed SuperLU routine (version 6.1.1) --
 * Lawrence Berkeley National Lab, Univ. of California Berkeley.
 * October 1, 2014
 *
 * Modified:
 *   September 18, 2017, enable SIMD vectorized scatter operation.
 *
 */
#include <math.h>
#include "utils.h"
#include "superlu_zdefs_ex.h"
#ifdef OPT_SCATTER
#include "utils.h"

#ifndef MAX_SUPERNODE_SIZE
#define MAX_SUPERNODE_SIZE 256
#endif
#endif

extern double Scatter_search_timer;
extern double Scatter_sub_timer;
extern double Scatter_trans_timer;
extern double Scatter_indirect_timer;
extern double Scatter_other_timer_l;
extern double Scatter_other_timer_u;

// #define SCATTER_TIMER

#ifdef OPT_SCATTER
// #define OPT_SEARCH

ptr_pair_t
zscatter_l_opt(
    int ib,         /* row block number of source block L(i,k) */
    int ljb,        /* local column block number of dest. block L(i,j) */
    int nsupc,      /* number of columns in destination supernode */
    int_t iukp,     /* point to destination supernode's index[] */
    int_t* xsup,
    int klst,
    int nbrow,      /* LDA of the block in tempv[] */
    int_t lptr,     /* Input, point to index[] location of block L(i,k) */
    int temp_nbrow, /* number of rows of source block L(i,k) */
    int_t* usub,
    int_t* lsub,
    doublecomplex* tempv,
    int* indirect_thread, int* indirect2,
    int_t** Lrowind_bc_ptr, doublecomplex** Lnzval_bc_ptr,
    gridinfo_t* grid,
    int_t lptrj_now,
    int_t luptrj_now)
{
    #ifdef SCATTER_TIMER
    double tt_st = SuperLU_timer_(); 
    #endif
    int iam = grid->iam;
    int_t* index = Lrowind_bc_ptr[ljb];
    int_t ldv = index[1];       /* LDA of the destination lusup. */
    int_t lptrj = lptrj_now;
    int_t luptrj = luptrj_now;
    // int_t lptrj = BC_HEADER;
    // int_t luptrj = 0;
    int_t ijb = index[lptrj];
    #ifdef SCATTER_TIMER
    Scatter_other_timer_l += SuperLU_timer_() - tt_st;
    #endif

    #ifdef SCATTER_TIMER
    tt_st = SuperLU_timer_();
    #endif
    #ifdef OPT_SEARCH
    int num_lb = index[0];
    int base = ijb;
    int local_lb_idx[100000];
    int_t lptrj_1 = lptrj;
    int_t luptrj_1 = 0;
    for(int i = 0; i < num_lb; i++){
        int_t gid = index[lptrj_1];
        lptrj_1 += LB_DESCRIPTOR + index[lptrj_1 + 1];
        local_lb_idx[gid-base] = i;
    }
    int matched_lb_idx = local_lb_idx[ib-base];
    lptrj_1 = lptrj;
    for(int i = 0; i < matched_lb_idx; i++){
        luptrj_1 += index[lptrj_1 + 1];
        lptrj_1 += LB_DESCRIPTOR + index[lptrj_1 + 1];
    }
    luptrj = luptrj_1;
    lptrj = lptrj_1;
    // while (ijb != ib)  /* Search for destination block L(i,j) */
    // {
    //     luptrj += index[lptrj + 1];
    //     lptrj += LB_DESCRIPTOR + index[lptrj + 1];
    //     ijb = index[lptrj];
    // }
    // assert(lptrj == lptrj_1);
    // assert(luptrj == luptrj_1);
    // if(luptrj != luptrj_1){
    //     printf("luptrj=%ld, luptrj_1=%ld\n",luptrj, luptrj_1);
    //     // exit(1);
    // }
    // if(lptrj != lptrj_1){
    //     printf("lptrj=%ld, lptrj_1=%ld\n",lptrj, lptrj_1 );
    //     printf("num_lb=%d, base=%d, offset=%d", num_lb, base, offset);
    //     exit(1);
    // }
    #else /*Traverse*/
    while (ijb != ib)  /* Search for destination block L(i,j) */
    {
        luptrj += index[lptrj + 1];
        lptrj += LB_DESCRIPTOR + index[lptrj + 1];
        ijb = index[lptrj];
    }
    ptr_pair_t tmp_pair;
    tmp_pair.ptr = lptrj;
    tmp_pair.uptr = luptrj;
    #endif /*end of OPT_SEARCH*/
    #ifdef SCATTER_TIMER
    Scatter_search_timer += SuperLU_timer_() - tt_st;
    #endif

    #ifdef SCATTER_TIMER
    tt_st = SuperLU_timer_();
    #endif
    int_t fnz = FstBlockC(ib);
    lptrj += LB_DESCRIPTOR;
    int_t dest_nbrow = index[lptrj - 1];

    assert(dest_nbrow >= temp_nbrow);
#pragma omp simd

    for (int i = 0; i < dest_nbrow; ++i) {
        int rel = index[lptrj + i] - fnz;
        indirect_thread[rel] = i;
    }

#pragma omp simd

    /* can be precalculated? */
    for (int i = 0; i < temp_nbrow; ++i) { /* Source index is a subset of dest. */
        int rel = lsub[lptr + i] - fnz;
        indirect2[i] = indirect_thread[rel];
    }

    #ifdef SCATTER_TIMER
    Scatter_indirect_timer += SuperLU_timer_() - tt_st;
    #endif

    #ifdef SCATTER_TIMER
    tt_st = SuperLU_timer_();
    #endif

    indirect_index_segment_compress_t segment_compress;
    indirect_index_segment_compress_init(&segment_compress, indirect2, temp_nbrow);

    #ifdef SCATTER_TIMER
    Scatter_trans_timer += SuperLU_timer_() - tt_st;
    #endif

    doublecomplex* nzval = Lnzval_bc_ptr[ljb] + luptrj; /* Destination block L(i,j) */

    #ifdef SCATTER_TIMER
    tt_st = SuperLU_timer_();
    #endif
    int non_zero_col_num = 0;
    //
    doublecomplex* nzvals[MAX_SUPERNODE_SIZE];
    for (int jj = 0; jj < nsupc; ++jj) {
        int_t segsize = klst - usub[iukp + jj];
        if (segsize) {
            nzvals[non_zero_col_num] = nzval + jj * ldv;
            non_zero_col_num += 1;
}
    }


    //printf("ldv = %d------------------------------------------\n", ldv);

    for (int jj_ptr = 0; jj_ptr < non_zero_col_num; ++jj_ptr) {

        //printf("------------jj_ptr = %d, non_zero_col_num = %d, segmentcount = %d-------\n", jj_ptr, non_zero_col_num, segment_compress.segment_count);
        doublecomplex* nzval_cur = nzvals[jj_ptr];
        double* tempv_cur = (double*)(tempv + jj_ptr * nbrow);
        for (int ptr = 0; ptr < segment_compress.segment_count; ++ptr) {
            int i_start = segment_compress.segment_ptr[ptr];
            int i_end = segment_compress.segment_ptr[ptr + 1];
            int offset = segment_compress.segment_offset[ptr];
            double* NZVAL = (double*)(nzval_cur + offset);
#if (_OPENMP>=201307)
#pragma omp simd
#endif
            for (int i = i_start * 2; i < i_end * 2; i++) {
                NZVAL[i] -= tempv_cur[i];
                //printf("---------istart = %d, iend = %d nzval[%d] = %f -----------------------------\n", i_start, i_end, i, NZVAL[i]);
            }
        }
    }
    //
    #ifdef SCATTER_TIMER
    Scatter_sub_timer += SuperLU_timer_() - tt_st;
    #endif
    indirect_index_segment_compress_destroy(&segment_compress);
    return tmp_pair;
} /* zscatter_l */



ptr_pair_t
zscatter_u_opt(
    int ib,
    int jb,
    int nsupc,
    int_t iukp,
    int_t* xsup,
    int klst,
    int nbrow,      /* LDA of the block in tempv[] */
    int_t lptr,     /* point to index location of block L(i,k) */
    int temp_nbrow, /* number of rows of source block L(i,k) */
    int_t* lsub,
    int_t* usub,
    doublecomplex* tempv,
    int_t** Ufstnz_br_ptr, doublecomplex** Unzval_br_ptr,
    gridinfo_t* grid,
    int_t iuip_lib_now,
    int_t ruip_lib_now)
{   
    #ifdef SCATTER_TIMER
    double tt_st = SuperLU_timer_();
    #endif

    int iam = grid->iam;
    int_t ilst = FstBlockC(ib + 1);
    int_t lib = LBi(ib, grid);
    int_t* index = Ufstnz_br_ptr[lib];

    int_t iuip_lib = iuip_lib_now;
    int_t ruip_lib = ruip_lib_now;

    int_t ijb = index[iuip_lib];
    #ifdef SCATTER_TIMER
    Scatter_other_timer_u += SuperLU_timer_() - tt_st;
    #endif

    #ifdef SCATTER_TIMER
    tt_st = SuperLU_timer_();
    #endif

    while (ijb < jb) {   /* Search for destination block. */
        ruip_lib += index[iuip_lib + 1];
        iuip_lib += UB_DESCRIPTOR + SuperSize(ijb);
        ijb = index[iuip_lib];
    }
    ptr_pair_t tmp_pair;
    tmp_pair.ptr = iuip_lib;
    tmp_pair.uptr = ruip_lib;

    #ifdef SCATTER_TIMER
    Scatter_search_timer += SuperLU_timer_() - tt_st;
    #endif

    /* Skip descriptor. Now point to fstnz index of block U(i,j). */
    iuip_lib += UB_DESCRIPTOR;

    #ifdef SCATTER_TIMER
    tt_st = SuperLU_timer_();
    #endif

    indirect_index_segment_compress_t segment_compress;
    indirect_index_segment_compress_init(&segment_compress, &lsub[lptr], temp_nbrow);

    int non_zero_col_num = 0;
    doublecomplex* non_zero_ucols[MAX_SUPERNODE_SIZE];
    int fnzs[MAX_SUPERNODE_SIZE];
    for (int jj = 0; jj < nsupc; ++jj) {
        int_t segsize = klst - usub[iukp + jj];
        int fnz = index[iuip_lib + jj];
        if (segsize) {
            fnzs[non_zero_col_num] = fnz;
            non_zero_ucols[non_zero_col_num] = &Unzval_br_ptr[lib][ruip_lib];
            non_zero_col_num += 1;
        }
        ruip_lib += ilst - fnz;
    }

    #ifdef SCATTER_TIMER
    Scatter_trans_timer += SuperLU_timer_() - tt_st;
    #endif

    #ifdef SCATTER_TIMER
    tt_st = SuperLU_timer_();
    #endif
    
    for (int jj_ptr = 0; jj_ptr < non_zero_col_num; ++jj_ptr) {
        int fnz = fnzs[jj_ptr];
        doublecomplex* ucol_cur = non_zero_ucols[jj_ptr];
        double* tempv_cur = (double*)(tempv + nbrow * jj_ptr);
        for (int ptr = 0; ptr < segment_compress.segment_count; ++ptr) {
            int i_start = segment_compress.segment_ptr[ptr];
            int i_end = segment_compress.segment_ptr[ptr + 1];
            int offset = segment_compress.segment_offset[ptr];
            int rel = offset - fnz;
            double* UCOL = (double*)&ucol_cur[rel];
            // for(int i = i_start; i < i_end; i++){
            //     z_sub(&UCOL[i], &UCOL[i], &tempv_cur[i]);
            // }
#if (_OPENMP>=201307)
#pragma omp simd
#endif
            for (int i = i_start * 2; i < i_end * 2; i++) {
                UCOL[i] -= tempv_cur[i];
            }
        }
    }

    #ifdef SCATTER_TIMER
    Scatter_sub_timer += SuperLU_timer_() - tt_st;
    #endif
    // if(segment_compress.segment_count == 1){
    //     for (int jj_ptr = 0; jj_ptr < non_zero_col_num; ++jj_ptr) {
    //     int fnz = fnzs[jj_ptr];
    //     doublecomplex *ucol_cur = non_zero_ucols[jj_ptr];
    //     double *tempv_cur = (double*)(tempv + nbrow * jj_ptr);
    //         for(int ptr = 0; ptr < segment_compress.segment_count; ++ptr){
    //             int i_start = segment_compress.segment_ptr[ptr];
    //             int i_end = segment_compress.segment_ptr[ptr+1];
    //             int offset = segment_compress.segment_offset[ptr];
    //             int rel = offset - fnz;
    //             double *UCOL = (double*)&ucol_cur[rel];
    //             // for(int i = i_start; i < i_end; i++){
    //             //     z_sub(&UCOL[i], &UCOL[i], &tempv_cur[i]);
    //             // }
    //             #pragma omp simd
    //             for(int i = i_start*2; i < i_end*2; i++){
    //                 UCOL[i] -= tempv_cur[i];
    //             }
    //         }
    //     }
    // }

    indirect_index_segment_compress_destroy(&segment_compress);
    return tmp_pair;
    // indirect_index_segment_compress_destroy(&segment_compress);
} /* zscatter_u */
#endif


void
zscatter_l_1(int ib,
    int ljb,
    int nsupc,
    int_t iukp,
    int_t* xsup,
    int klst,
    int nbrow,
    int_t lptr,
    int temp_nbrow,
    int* usub,
    int* lsub,
    doublecomplex* tempv,
    int* indirect_thread,
    int_t** Lrowind_bc_ptr, doublecomplex** Lnzval_bc_ptr,
    gridinfo_t* grid)
{
    // TAU_STATIC_TIMER_START("SCATTER_LB");
    // printf("hello\n");
    int_t rel, i, segsize, jj;
    doublecomplex* nzval;
    int_t* index = Lrowind_bc_ptr[ljb];
    int_t ldv = index[1];       /* LDA of the dest lusup. */
    int_t lptrj = BC_HEADER;
    int_t luptrj = 0;
    int_t ijb = index[lptrj];
    while (ijb != ib)
    {
        /* Search for dest block --
           blocks are not ordered! */
        luptrj += index[lptrj + 1];
        lptrj += LB_DESCRIPTOR + index[lptrj + 1];

        ijb = index[lptrj];
    }
    /*
     * Build indirect table. This is needed because the
     * indices are not sorted for the L blocks.
     */
    int_t fnz = FstBlockC(ib);
    lptrj += LB_DESCRIPTOR;
    for (i = 0; i < index[lptrj - 1]; ++i)
    {
        rel = index[lptrj + i] - fnz;
        indirect_thread[rel] = i;

    }

    nzval = Lnzval_bc_ptr[ljb] + luptrj;
    // tempv =bigV + (cum_nrow + cum_ncol*nbrow);
    for (jj = 0; jj < nsupc; ++jj)
    {
        segsize = klst - usub[iukp + jj];
        // printf("segsize %d \n",segsize);
        if (segsize) {
            /*#pragma _CRI cache_bypass nzval,tempv */
            for (i = 0; i < temp_nbrow; ++i) {
                rel = lsub[lptr + i] - fnz;
                z_sub(&nzval[indirect_thread[rel]], &nzval[indirect_thread[rel]],
                    &tempv[i]);
                // printf("i (src) %d, perm (dest) %d  \n",i,indirect_thread[rel]);
#ifdef PI_DEBUG
                double zz = 0.0;
                // if(!(*(long*)&zz == *(long*)&tempv[i]) )
                // printf("(%d %d, %0.3e, %0.3e, %3e ) ", ljb,
                //     nzval - Lnzval_bc_ptr[ljb] + indirect_thread[rel],
                //     nzval[indirect_thread[rel]] + tempv[i],
                //     nzval[indirect_thread[rel]], tempv[i]);
                //printing triplets (location??, old value, new value ) if none of them is zero
#endif
            }
            // printf("\n");
            tempv += nbrow;
#ifdef PI_DEBUG
            // printf("\n");
#endif
        }
        nzval += ldv;
        // printf("%d\n",nzval );
    }
    // TAU_STATIC_TIMER_STOP("SCATTER_LB");
} /* zscatter_l_1 */


ptr_pair_t
zscatter_l_opt_search_moveptr(
    int ib,    /* row block number of source block L(i,k) */
    int ljb,   /* local column block number of dest. block L(i,j) */
    int nsupc, /* number of columns in destination supernode */
    int_t iukp, /* point to destination supernode's index[] */
    int_t* xsup,
    int klst,
    int nbrow,  /* LDA of the block in tempv[] */
    int_t lptr, /* Input, point to index[] location of block L(i,k) */
    int temp_nbrow, /* number of rows of source block L(i,k) */
    int_t* usub,
    int_t* lsub,
    doublecomplex* tempv,
    int* indirect_thread, int* indirect2,
    int_t** Lrowind_bc_ptr, doublecomplex** Lnzval_bc_ptr,
    gridinfo_t* grid,
    int_t lptrj_now,
    int_t luptrj_now)
{
    #ifdef SCATTER_TIMER
    double tt_st = SuperLU_timer_(); 
    #endif
    int_t rel, i, segsize, jj;
    doublecomplex* nzval;
    int_t* index = Lrowind_bc_ptr[ljb];
    int_t ldv = index[1];       /* LDA of the destination lusup. */
    int_t lptrj = lptrj_now;
    int_t luptrj = luptrj_now;
    int_t ijb = index[lptrj];
    #ifdef SCATTER_TIMER
    Scatter_other_timer_l += SuperLU_timer_() - tt_st;
    #endif

    #ifdef SCATTER_TIMER
    tt_st = SuperLU_timer_();
    #endif
    while (ijb != ib)  /* Search for destination block L(i,j) */
    {
        luptrj += index[lptrj + 1];
        lptrj += LB_DESCRIPTOR + index[lptrj + 1];
        ijb = index[lptrj];
    }

    ptr_pair_t tmp_pair;
    tmp_pair.ptr = lptrj;
    tmp_pair.uptr = luptrj;
    #ifdef SCATTER_TIMER
    Scatter_search_timer += SuperLU_timer_() - tt_st;
    #endif
    /*
     * Build indirect table. This is needed because the indices are not sorted
     * in the L blocks.
     */
    #ifdef SCATTER_TIMER
    tt_st = SuperLU_timer_();
    #endif
    int_t fnz = FstBlockC(ib);
    int_t dest_nbrow;
    lptrj += LB_DESCRIPTOR;
    dest_nbrow = index[lptrj - 1];

#if (_OPENMP>=201307)
#pragma omp simd
#endif
    for (i = 0; i < dest_nbrow; ++i) {
        rel = index[lptrj + i] - fnz;
        indirect_thread[rel] = i;

    }

#if (_OPENMP>=201307)
#pragma omp simd
#endif
    /* can be precalculated? */
    for (i = 0; i < temp_nbrow; ++i) { /* Source index is a subset of dest. */
        rel = lsub[lptr + i] - fnz;
        indirect2[i] = indirect_thread[rel];
    }
    #ifdef SCATTER_TIMER
    Scatter_indirect_timer += SuperLU_timer_() - tt_st;
    #endif

    #ifdef SCATTER_TIMER
    tt_st = SuperLU_timer_();
    #endif

    nzval = Lnzval_bc_ptr[ljb] + luptrj; /* Destination block L(i,j) */
#ifdef __INTEL_COMPILER
#pragma ivdep
#endif
    for (jj = 0; jj < nsupc; ++jj) {
        segsize = klst - usub[iukp + jj];
        if (segsize) {
#if (_OPENMP>=201307)
#pragma omp simd
#endif
            for (i = 0; i < temp_nbrow; ++i) {
                z_sub(&nzval[indirect2[i]], &nzval[indirect2[i]], &tempv[i]);
            }
            tempv += nbrow;
        }
        nzval += ldv;
    }
    #ifdef SCATTER_TIMER
    Scatter_sub_timer += SuperLU_timer_() - tt_st;
    #endif   
    return tmp_pair;
} /* zscatter_l_opt_search_moveptr */


ptr_pair_t
zscatter_u_opt_search_moveptr(int ib,
    int jb,
    int nsupc,
    int_t iukp,
    int_t* xsup,
    int klst,
    int nbrow,      /* LDA of the block in tempv[] */
    int_t lptr,     /* point to index location of block L(i,k) */
    int temp_nbrow, /* number of rows of source block L(i,k) */
    int_t* lsub,
    int_t* usub,
    doublecomplex* tempv,
    int_t** Ufstnz_br_ptr, doublecomplex** Unzval_br_ptr,
    gridinfo_t* grid,
    int_t iuip_lib_now,
    int_t ruip_lib_now)
{
#ifdef PI_DEBUG
    printf("A(%d,%d) goes to U block \n", ib, jb);
#endif
    // TAU_STATIC_TIMER_START("SCATTER_U");
    // TAU_STATIC_TIMER_START("SCATTER_UB");
    #ifdef SCATTER_TIMER
    double tt_st = SuperLU_timer_();
    #endif
    int_t jj, i, fnz, rel;
    int segsize;
    doublecomplex* ucol;
    int_t ilst = FstBlockC(ib + 1);
    int_t lib = LBi(ib, grid);
    int_t* index = Ufstnz_br_ptr[lib];

    /* Reinitilize the pointers to the beginning of the k-th column/row of
     * L/U factors.
     * usub[] - index array for panel U(k,:)
     */
    int_t iuip_lib, ruip_lib;
    iuip_lib = iuip_lib_now;
    ruip_lib = ruip_lib_now;

    int_t ijb = index[iuip_lib];
    #ifdef SCATTER_TIMER
    Scatter_other_timer_u += SuperLU_timer_() - tt_st;
    #endif

    #ifdef SCATTER_TIMER
    tt_st = SuperLU_timer_();
    #endif
    while (ijb < jb) {   /* Search for destination block. */
        ruip_lib += index[iuip_lib + 1];
        // printf("supersize[%ld] \t:%ld \n",ijb,SuperSize( ijb ) );
        iuip_lib += UB_DESCRIPTOR + SuperSize(ijb);
        ijb = index[iuip_lib];
    }
    ptr_pair_t tmp_pair;
    tmp_pair.ptr = iuip_lib;
    tmp_pair.uptr = ruip_lib;
    #ifdef SCATTER_TIMER
    Scatter_search_timer += SuperLU_timer_() - tt_st;
    #endif

    #ifdef SCATTER_TIMER
    tt_st = SuperLU_timer_();
    #endif

    /* Skip descriptor. Now point to fstnz index of block U(i,j). */
    iuip_lib += UB_DESCRIPTOR;

    // tempv = bigV + (cum_nrow + cum_ncol*nbrow);
    for (jj = 0; jj < nsupc; ++jj) {
        segsize = klst - usub[iukp + jj];
        fnz = index[iuip_lib++];
        if (segsize) {          /* Nonzero segment in U(k,j). */
            ucol = &Unzval_br_ptr[lib][ruip_lib];

            // printf("========Entering loop=========\n");
#if (_OPENMP>=201307)
#pragma omp simd
#endif
            for (i = 0; i < temp_nbrow; ++i) {
                rel = lsub[lptr + i] - fnz;
                // printf("%d %d %d %d %d \n",lptr,i,fnz,temp_nbrow,nbrow );
                // printf("hello   ucol[%d] %d %d : \n",rel,lsub[lptr + i],fnz);
                z_sub(&ucol[rel], &ucol[rel], &tempv[i]);

#ifdef PI_DEBUG
                // double zz = 0.0;
                // if (!(*(long*)&zz == *(long*)&tempv[i]))
                //     printf("(%d, %0.3e, %0.3e ) ", rel, (double)ucol[rel] + (double)tempv[i],
                //         ucol[rel]);
                //printing triplets (location??, old value, new value ) if none of them is zero
#endif
            } /* for i = 0:temp_nbropw */
            tempv += nbrow; /* Jump LDA to next column */
#ifdef PI_DEBUG
            // printf("\n");
#endif
        }  /* if segsize */

        ruip_lib += ilst - fnz;

    }  /* for jj = 0:nsupc */
    #ifdef SCATTER_TIMER
    Scatter_sub_timer += SuperLU_timer_() - tt_st;
    #endif

#ifdef PI_DEBUG
    // printf("\n");
#endif
    // TAU_STATIC_TIMER_STOP("SCATTER_UB");
    return tmp_pair;
} /* zscatter_u_opt_search_moveptr */


ptr_pair_t
zscatter_l_opt_search_moveptr_recidx(
    int ib,    /* row block number of source block L(i,k) */
    int ljb,   /* local column block number of dest. block L(i,j) */
    int nsupc, /* number of columns in destination supernode */
    int_t iukp, /* point to destination supernode's index[] */
    int_t* xsup,
    int klst,
    int nbrow,  /* LDA of the block in tempv[] */
    int_t lptr, /* Input, point to index[] location of block L(i,k) */
    int temp_nbrow, /* number of rows of source block L(i,k) */
    int_t* usub,
    int_t* lsub,
    doublecomplex* tempv,
    int* indirect_thread, int* indirect2,
    int_t** Lrowind_bc_ptr, doublecomplex** Lnzval_bc_ptr,
    gridinfo_t* grid,
    int_t lptrj_now,
    int_t luptrj_now,
    int_t local_lidx_now)
{
    #ifdef SCATTER_TIMER
    double tt_st = SuperLU_timer_(); 
    #endif
    int_t rel, i, segsize, jj;
    doublecomplex* nzval;
    int_t* index = Lrowind_bc_ptr[ljb];
    int_t ldv = index[1];       /* LDA of the destination lusup. */
    int_t lptrj = lptrj_now;
    int_t luptrj = luptrj_now;
    int_t ijb = index[lptrj];
    #ifdef SCATTER_TIMER
    Scatter_other_timer_l += SuperLU_timer_() - tt_st;
    #endif

    #ifdef SCATTER_TIMER
    tt_st = SuperLU_timer_();
    #endif
    int uidx = 0;
    while (ijb != ib)  /* Search for destination block L(i,j) */
    {
        luptrj += index[lptrj + 1];
        lptrj += LB_DESCRIPTOR + index[lptrj + 1];
        ijb = index[lptrj];
        uidx++;
    }

    ptr_pair_t tmp_pair;
    tmp_pair.ptr = lptrj;
    tmp_pair.uptr = luptrj;
    tmp_pair.idx = local_lidx_now + i;
    #ifdef SCATTER_TIMER
    Scatter_search_timer += SuperLU_timer_() - tt_st;
    #endif
    /*
     * Build indirect table. This is needed because the indices are not sorted
     * in the L blocks.
     */
    #ifdef SCATTER_TIMER
    tt_st = SuperLU_timer_();
    #endif
    int_t fnz = FstBlockC(ib);
    int_t dest_nbrow;
    lptrj += LB_DESCRIPTOR;
    dest_nbrow = index[lptrj - 1];

#if (_OPENMP>=201307)
#pragma omp simd
#endif
    for (i = 0; i < dest_nbrow; ++i) {
        rel = index[lptrj + i] - fnz;
        indirect_thread[rel] = i;

    }

#if (_OPENMP>=201307)
#pragma omp simd
#endif
    /* can be precalculated? */
    for (i = 0; i < temp_nbrow; ++i) { /* Source index is a subset of dest. */
        rel = lsub[lptr + i] - fnz;
        indirect2[i] = indirect_thread[rel];
    }
    #ifdef SCATTER_TIMER
    Scatter_indirect_timer += SuperLU_timer_() - tt_st;
    #endif

    #ifdef SCATTER_TIMER
    tt_st = SuperLU_timer_();
    #endif

    nzval = Lnzval_bc_ptr[ljb] + luptrj; /* Destination block L(i,j) */
#ifdef __INTEL_COMPILER
#pragma ivdep
#endif
    for (jj = 0; jj < nsupc; ++jj) {
        segsize = klst - usub[iukp + jj];
        if (segsize) {
#if (_OPENMP>=201307)
#pragma omp simd
#endif
            for (i = 0; i < temp_nbrow; ++i) {
                z_sub(&nzval[indirect2[i]], &nzval[indirect2[i]], &tempv[i]);
            }
            tempv += nbrow;
        }
        nzval += ldv;
    }
    #ifdef SCATTER_TIMER
    Scatter_sub_timer += SuperLU_timer_() - tt_st;
    #endif   
    return tmp_pair;
} /* zscatter_l_opt_search_moveptr_recidx */

int_t
zscatter_l_table(
    int ib,    /* row block number of source block L(i,k) */
    int ljb,   /* local column block number of dest. block L(i,j) */
    int nsupc, /* number of columns in destination supernode */
    int_t iukp, /* point to destination supernode's index[] */
    int_t* xsup,
    int klst,
    int nbrow,  /* LDA of the block in tempv[] */
    int_t lptr, /* Input, point to index[] location of block L(i,k) */
    int temp_nbrow, /* number of rows of source block L(i,k) */
    int_t* usub,
    int_t* lsub,
    doublecomplex* tempv,
    int* indirect_thread, int* indirect2,
    int_t** Lrowind_bc_ptr, doublecomplex** Lnzval_bc_ptr,
    gridinfo_t* grid,
    int_t* gid_table, int_t* lptr_table, int_t* luptr_table,
    int_t lid_now)
{
    #ifdef SCATTER_TIMER
    double tt_st = SuperLU_timer_(); 
    #endif
    int_t rel, i, segsize, jj;
    doublecomplex* nzval;
    int_t* index = Lrowind_bc_ptr[ljb];
    int_t ldv = index[1];       /* LDA of the destination lusup. */
    int_t ijb = gid_table[lid_now];
    #ifdef SCATTER_TIMER
    Scatter_other_timer_l += SuperLU_timer_() - tt_st;
    #endif

    #ifdef SCATTER_TIMER
    tt_st = SuperLU_timer_();
    #endif
    while (ijb != ib)  /* Search for destination block L(i,j) */
    {
        lid_now++;
        ijb = gid_table[lid_now];
    }
    int_t lptrj = lptr_table[lid_now];
    int_t luptrj = luptr_table[lid_now];

    #ifdef SCATTER_TIMER
    Scatter_search_timer += SuperLU_timer_() - tt_st;
    #endif
    /*
     * Build indirect table. This is needed because the indices are not sorted
     * in the L blocks.
     */
    #ifdef SCATTER_TIMER
    tt_st = SuperLU_timer_();
    #endif
    int_t fnz = FstBlockC(ib);
    int_t dest_nbrow;
    lptrj += LB_DESCRIPTOR;
    dest_nbrow = index[lptrj - 1];

#if (_OPENMP>=201307)
#pragma omp simd
#endif
    for (i = 0; i < dest_nbrow; ++i) {
        rel = index[lptrj + i] - fnz;
        indirect_thread[rel] = i;

    }

#if (_OPENMP>=201307)
#pragma omp simd
#endif
    /* can be precalculated? */
    for (i = 0; i < temp_nbrow; ++i) { /* Source index is a subset of dest. */
        rel = lsub[lptr + i] - fnz;
        indirect2[i] = indirect_thread[rel];
    }
    #ifdef SCATTER_TIMER
    Scatter_indirect_timer += SuperLU_timer_() - tt_st;
    #endif

    #ifdef SCATTER_TIMER
    tt_st = SuperLU_timer_();
    #endif

    nzval = Lnzval_bc_ptr[ljb] + luptrj; /* Destination block L(i,j) */
#ifdef __INTEL_COMPILER
#pragma ivdep
#endif
    for (jj = 0; jj < nsupc; ++jj) {
        segsize = klst - usub[iukp + jj];
        if (segsize) {
#if (_OPENMP>=201307)
#pragma omp simd
#endif
            for (i = 0; i < temp_nbrow; ++i) {
                z_sub(&nzval[indirect2[i]], &nzval[indirect2[i]], &tempv[i]);
            }
            tempv += nbrow;
        }
        nzval += ldv;
    }
    #ifdef SCATTER_TIMER
    Scatter_sub_timer += SuperLU_timer_() - tt_st;
    #endif   
    return lid_now;
} /* zscatter_l_table */

void
zscatter_l(
    int ib,    /* row block number of source block L(i,k) */
    int ljb,   /* local column block number of dest. block L(i,j) */
    int nsupc, /* number of columns in destination supernode */
    int_t iukp, /* point to destination supernode's index[] */
    int_t* xsup,
    int klst,
    int nbrow,  /* LDA of the block in tempv[] */
    int_t lptr, /* Input, point to index[] location of block L(i,k) */
    int temp_nbrow, /* number of rows of source block L(i,k) */
    int_t* usub,
    int_t* lsub,
    doublecomplex* tempv,
    int* indirect_thread, int* indirect2,
    int_t** Lrowind_bc_ptr, doublecomplex** Lnzval_bc_ptr,
    gridinfo_t* grid)
{
    #ifdef SCATTER_TIMER
    double tt_st = SuperLU_timer_(); 
    #endif
    int_t rel, i, segsize, jj;
    doublecomplex* nzval;
    int_t* index = Lrowind_bc_ptr[ljb];
    int_t ldv = index[1];       /* LDA of the destination lusup. */
    int_t lptrj = BC_HEADER;
    int_t luptrj = 0;
    int_t ijb = index[lptrj];
    #ifdef SCATTER_TIMER
    Scatter_other_timer_l += SuperLU_timer_() - tt_st;
    #endif

    #ifdef SCATTER_TIMER
    tt_st = SuperLU_timer_();
    #endif
    while (ijb != ib)  /* Search for destination block L(i,j) */
    {
        luptrj += index[lptrj + 1];
        lptrj += LB_DESCRIPTOR + index[lptrj + 1];
        ijb = index[lptrj];
    }

    #ifdef SCATTER_TIMER
    Scatter_search_timer += SuperLU_timer_() - tt_st;
    #endif
    /*
     * Build indirect table. This is needed because the indices are not sorted
     * in the L blocks.
     */
    #ifdef SCATTER_TIMER
    tt_st = SuperLU_timer_();
    #endif
    int_t fnz = FstBlockC(ib);
    int_t dest_nbrow;
    lptrj += LB_DESCRIPTOR;
    dest_nbrow = index[lptrj - 1];

#if (_OPENMP>=201307)
#pragma omp simd
#endif
    for (i = 0; i < dest_nbrow; ++i) {
        rel = index[lptrj + i] - fnz;
        indirect_thread[rel] = i;

    }

#if (_OPENMP>=201307)
#pragma omp simd
#endif
    /* can be precalculated? */
    for (i = 0; i < temp_nbrow; ++i) { /* Source index is a subset of dest. */
        rel = lsub[lptr + i] - fnz;
        indirect2[i] = indirect_thread[rel];
    }
    #ifdef SCATTER_TIMER
    Scatter_indirect_timer += SuperLU_timer_() - tt_st;
    #endif

    #ifdef SCATTER_TIMER
    tt_st = SuperLU_timer_();
    #endif

    nzval = Lnzval_bc_ptr[ljb] + luptrj; /* Destination block L(i,j) */
#ifdef __INTEL_COMPILER
#pragma ivdep
#endif
    for (jj = 0; jj < nsupc; ++jj) {
        segsize = klst - usub[iukp + jj];
        if (segsize) {
#if (_OPENMP>=201307)
#pragma omp simd
#endif
            for (i = 0; i < temp_nbrow; ++i) {
                z_sub(&nzval[indirect2[i]], &nzval[indirect2[i]], &tempv[i]);
            }
            tempv += nbrow;
        }
        nzval += ldv;
    }
    #ifdef SCATTER_TIMER
    Scatter_sub_timer += SuperLU_timer_() - tt_st;
    #endif   

} /* zscatter_l */


void
zscatter_u(int ib,
    int jb,
    int nsupc,
    int_t iukp,
    int_t* xsup,
    int klst,
    int nbrow,      /* LDA of the block in tempv[] */
    int_t lptr,     /* point to index location of block L(i,k) */
    int temp_nbrow, /* number of rows of source block L(i,k) */
    int_t* lsub,
    int_t* usub,
    doublecomplex* tempv,
    int_t** Ufstnz_br_ptr, doublecomplex** Unzval_br_ptr,
    gridinfo_t* grid)
{
#ifdef PI_DEBUG
    printf("A(%d,%d) goes to U block \n", ib, jb);
#endif
    // TAU_STATIC_TIMER_START("SCATTER_U");
    // TAU_STATIC_TIMER_START("SCATTER_UB");
    #ifdef SCATTER_TIMER
    double tt_st = SuperLU_timer_();
    #endif
    int_t jj, i, fnz, rel;
    int segsize;
    doublecomplex* ucol;
    int_t ilst = FstBlockC(ib + 1);
    int_t lib = LBi(ib, grid);
    int_t* index = Ufstnz_br_ptr[lib];

    /* Reinitilize the pointers to the beginning of the k-th column/row of
     * L/U factors.
     * usub[] - index array for panel U(k,:)
     */
    int_t iuip_lib, ruip_lib;
    iuip_lib = BR_HEADER;
    ruip_lib = 0;

    int_t ijb = index[iuip_lib];
    #ifdef SCATTER_TIMER
    Scatter_other_timer_u += SuperLU_timer_() - tt_st;
    #endif

    #ifdef SCATTER_TIMER
    tt_st = SuperLU_timer_();
    #endif
    while (ijb < jb) {   /* Search for destination block. */
        ruip_lib += index[iuip_lib + 1];
        // printf("supersize[%ld] \t:%ld \n",ijb,SuperSize( ijb ) );
        iuip_lib += UB_DESCRIPTOR + SuperSize(ijb);
        ijb = index[iuip_lib];
    }
    assert(ijb == jb);
    #ifdef SCATTER_TIMER
    Scatter_search_timer += SuperLU_timer_() - tt_st;
    #endif

    #ifdef SCATTER_TIMER
    tt_st = SuperLU_timer_();
    #endif

    /* Skip descriptor. Now point to fstnz index of block U(i,j). */
    iuip_lib += UB_DESCRIPTOR;

    // tempv = bigV + (cum_nrow + cum_ncol*nbrow);
    for (jj = 0; jj < nsupc; ++jj) {
        segsize = klst - usub[iukp + jj];
        fnz = index[iuip_lib++];
        if (segsize) {          /* Nonzero segment in U(k,j). */
            ucol = &Unzval_br_ptr[lib][ruip_lib];

            // printf("========Entering loop=========\n");
#if (_OPENMP>=201307)
#pragma omp simd
#endif
            for (i = 0; i < temp_nbrow; ++i) {
                rel = lsub[lptr + i] - fnz;
                // printf("%d %d %d %d %d \n",lptr,i,fnz,temp_nbrow,nbrow );
                // printf("hello   ucol[%d] %d %d : \n",rel,lsub[lptr + i],fnz);
                z_sub(&ucol[rel], &ucol[rel], &tempv[i]);

#ifdef PI_DEBUG
                // double zz = 0.0;
                // if (!(*(long*)&zz == *(long*)&tempv[i]))
                //     printf("(%d, %0.3e, %0.3e ) ", rel, (double)ucol[rel] + (double)tempv[i],
                //         ucol[rel]);
                //printing triplets (location??, old value, new value ) if none of them is zero
#endif
            } /* for i = 0:temp_nbropw */
            tempv += nbrow; /* Jump LDA to next column */
#ifdef PI_DEBUG
            // printf("\n");
#endif
        }  /* if segsize */

        ruip_lib += ilst - fnz;

    }  /* for jj = 0:nsupc */
    #ifdef SCATTER_TIMER
    Scatter_sub_timer += SuperLU_timer_() - tt_st;
    #endif

#ifdef PI_DEBUG
    // printf("\n");
#endif
    // TAU_STATIC_TIMER_STOP("SCATTER_UB");
} /* zscatter_u */


/*Divide CPU-GPU dgemm work here*/
#ifdef PI_DEBUG
int Ngem = 2;
// int_t Ngem = 0;
int min_gpu_col = 6;
#else

    // int_t Ngem = 0;

#endif


#if 0 // Sherry: moved and corrected in util.c 
#ifdef GPU_ACC

void
gemm_division_cpu_gpu(
    int* num_streams_used,  /*number of streams that will be used */
    int* stream_end_col,    /*array holding last column blk for each partition */
    int* ncpu_blks,        /*Number of CPU dgemm blks */
    /*input */
    int nbrow,              /*number of row in A matrix */
    int ldu,                /*number of k in dgemm */
    int nstreams,
    int* full_u_cols,       /*array containing prefix sum of work load */
    int num_blks            /*Number of work load */
)
{
    int Ngem = sp_ienv_dist(7);  /*get_mnk_dgemm ();*/
    int min_gpu_col = get_cublas_nb();

    // Ngem = 1000000000;
    /*
       cpu is to gpu dgemm should be ideally 0:1 ratios to hide the total cost
       However since there is gpu latency of around 20,000 ns implying about
       200000 floating point calculation be done in that time so ~200,000/(2*nbrow*ldu)
       should be done in cpu to hide the latency; we Ngem =200,000/2
     */
    int i, j;

    // {
    //     *num_streams_used=0;
    //     *ncpu_blks = num_blks;
    //     return;
    // }

    for (int i = 0; i < nstreams; ++i)
    {
        stream_end_col[i] = num_blks;
    }

    *ncpu_blks = 0;
    /*easy returns -1 when number of column are less than threshold */
    if (full_u_cols[num_blks - 1] < (Ngem / (nbrow * ldu)) || num_blks == 1)
    {
        *num_streams_used = 0;
        *ncpu_blks = num_blks;
#ifdef PI_DEBUG
        printf("full_u_cols[num_blks-1] %d  %d \n",
            full_u_cols[num_blks - 1], (Ngem / (nbrow * ldu)));
        printf("Early return \n");
#endif
        return;

    }

    /* Easy return -2 when number of streams =0 */
    if (nstreams == 0)
    {
        *num_streams_used = 0;
        *ncpu_blks = num_blks;
        return;
        /* code */
    }
    /*find first block where count > Ngem */


    for (i = 0; i < num_blks - 1; ++i)  /*I can use binary search here */
    {
        if (full_u_cols[i + 1] > Ngem / (nbrow * ldu))
            break;
    }
    *ncpu_blks = i + 1;

    int_t cols_remain =
        full_u_cols[num_blks - 1] - full_u_cols[*ncpu_blks - 1];

#ifdef PI_DEBUG
    printf("Remaining cols %d num_blks %d cpu_blks %d \n", cols_remain,
        num_blks, *ncpu_blks);
#endif
    if (cols_remain > 0)
    {
        *num_streams_used = 1;  /* now atleast one stream would be used */

#ifdef PI_DEBUG
        printf("%d %d  %d %d \n", full_u_cols[num_blks - 1],
            full_u_cols[*ncpu_blks], *ncpu_blks, nstreams);
#endif
        int_t FP_MIN = 200000 / (nbrow * ldu);
        int_t cols_per_stream = SUPERLU_MAX(min_gpu_col, cols_remain / nstreams);
        cols_per_stream = SUPERLU_MAX(cols_per_stream, FP_MIN);
#ifdef PI_DEBUG
        printf("cols_per_stream :\t%d\n", cols_per_stream);
#endif

        int_t cutoff = cols_per_stream + full_u_cols[*ncpu_blks - 1];
        for (int_t i = 0; i < nstreams; ++i)
        {
            stream_end_col[i] = num_blks;
        }
        j = *ncpu_blks;
        for (i = 0; i < nstreams - 1; ++i)
        {
            int_t st = (i == 0) ? (*ncpu_blks) : stream_end_col[i - 1];

            for (j = st; j < num_blks - 1; ++j)
            {
#ifdef PI_DEBUG
                printf("i %d, j %d, %d  %d ", i, j, full_u_cols[j + 1],
                    cutoff);
#endif
                if (full_u_cols[j + 1] > cutoff)
                {
#ifdef PI_DEBUG
                    printf("cutoff met \n");
#endif
                    cutoff = cols_per_stream + full_u_cols[j];
                    stream_end_col[i] = j + 1;
                    *num_streams_used += 1;
                    j++;
                    break;
                }
#ifdef PI_DEBUG
                printf("\n");
#endif
            }

        }

    }
}

void
gemm_division_new(int* num_streams_used,   /*number of streams that will be used */
    int* stream_end_col, /*array holding last column blk for each partition */
    int* ncpu_blks,  /*Number of CPU dgemm blks */
    /*input */
    int nbrow,    /*number of row in A matrix */
    int ldu,  /*number of k in dgemm */
    int nstreams,
    Ublock_info_t* Ublock_info,    /*array containing prefix sum of work load */
    int num_blks  /*Number of work load */
)
{
    int Ngem = sp_ienv_dist(7); /*get_mnk_dgemm ();*/
    int min_gpu_col = get_cublas_nb();

    // Ngem = 1000000000;
    /*
       cpu is to gpu dgemm should be ideally 0:1 ratios to hide the total cost
       However since there is gpu latency of around 20,000 ns implying about
       200000 floating point calculation be done in that time so ~200,000/(2*nbrow*ldu)
       should be done in cpu to hide the latency; we Ngem =200,000/2
     */
    int_t i, j;


    for (int i = 0; i < nstreams; ++i)
    {
        stream_end_col[i] = num_blks;
    }

    *ncpu_blks = 0;
    /*easy returns -1 when number of column are less than threshold */
    if (Ublock_info[num_blks - 1].full_u_cols < (Ngem / (nbrow * ldu)) || num_blks == 1)
    {
        *num_streams_used = 0;
        *ncpu_blks = num_blks;

        return;

    }

    /* Easy return -2 when number of streams =0 */
    if (nstreams == 0)
    {
        *num_streams_used = 0;
        *ncpu_blks = num_blks;
        return;
        /* code */
    }
    /*find first block where count > Ngem */


    for (i = 0; i < num_blks - 1; ++i)  /*I can use binary search here */
    {
        if (Ublock_info[i + 1].full_u_cols > Ngem / (nbrow * ldu))
            break;
    }
    *ncpu_blks = i + 1;

    int_t cols_remain =
        Ublock_info[num_blks - 1].full_u_cols - Ublock_info[*ncpu_blks - 1].full_u_cols;

    if (cols_remain > 0)
    {
        *num_streams_used = 1;  /* now atleast one stream would be used */

        int_t FP_MIN = 200000 / (nbrow * ldu);
        int_t cols_per_stream = SUPERLU_MAX(min_gpu_col, cols_remain / nstreams);
        cols_per_stream = SUPERLU_MAX(cols_per_stream, FP_MIN);

        int_t cutoff = cols_per_stream + Ublock_info[*ncpu_blks - 1].full_u_cols;
        for (int_t i = 0; i < nstreams; ++i)
        {
            stream_end_col[i] = num_blks;
        }
        j = *ncpu_blks;
        for (i = 0; i < nstreams - 1; ++i)
        {
            int_t st = (i == 0) ? (*ncpu_blks) : stream_end_col[i - 1];

            for (j = st; j < num_blks - 1; ++j)
            {
                if (Ublock_info[j + 1].full_u_cols > cutoff)
                {

                    cutoff = cols_per_stream + Ublock_info[j].full_u_cols;
                    stream_end_col[i] = j + 1;
                    *num_streams_used += 1;
                    j++;
                    break;
                }

            }

        }

    }
}

#endif  /* defined GPU_ACC */


#endif // comment out the above code 
