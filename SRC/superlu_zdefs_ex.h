#pragma once

#include <stdbool.h>
#include "dcomplex.h"
#include "superlu_defs_ex.h"
#include "superlu_zdefs.h"
#include "utils.h"
#include "cuda_runtime.h"
// colunm
typedef struct {
    doublecomplex* val;
    int row;
    int col;
    int ld;
} zMatrix_t;

void zMatrix_init(zMatrix_t* matirx, int row, int col, int ld, doublecomplex* val);
void zMatrix_set_val(zMatrix_t* matirx, doublecomplex* val);
bool zMatrix_check_symmetry(const zMatrix_t* matirx);
bool zMatrix_check_transpose(const zMatrix_t* A, const zMatrix_t* B);

// infomation
void show_supernode_size(int_t n, Glu_persist_t* Glu_persist, gridinfo_t* grid);
void show_Lindex(int_t k, int_t* Lindex, Glu_persist_t* Glu_persist, gridinfo_t* grid);
void show_Uindex(int_t k, int_t* Uindex, Glu_persist_t* Glu_persist, gridinfo_t* grid);
void show_L_structure(int_t n, zLUstruct_t* LUstruct, gridinfo_t* grid);
void show_U_structure(int_t n, zLUstruct_t* LUstruct, gridinfo_t* grid);
void check_symmetry(int_t n, zLUstruct_t* LUstruct, gridinfo_t* grid);

void check_L_is_ordered(int_t n, zLUstruct_t* LUstruct, gridinfo_t* grid);
void check_U_is_full(int_t n, zLUstruct_t* LUstruct, gridinfo_t* grid);
#ifdef OPT_GPU_UPANEL_TRSM
void upanelfact_trsm(int_t k0, int_t k, Glu_persist_t* Glu_persist, gridinfo_t* grid, zLocalLU_t* Llu, SuperLUStat_t* stat, doublecomplex* dL, doublecomplex* dU, cudaStream_t* streams, cublasHandle_t* handle, int i);
#else
void upanelfact_trsm(int_t k0, int_t k, Glu_persist_t* Glu_persist, gridinfo_t* grid, zLocalLU_t* Llu, SuperLUStat_t* stat);



#endif
void zscatter_l_opt(
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
    superlu_dist_options_t* options);

void zscatter_u_opt(
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
    superlu_dist_options_t* options);
#ifdef GPU_ACC
#ifdef __cplusplus
extern "C" {
#endif

    void zcatter_l_compress(int non_zero_col_num, int nbrow, int max_supernode_size, int nsupc, int* usub,
        int klst, int iukp, doublecomplex* nzval, int ldv,
        doublecomplex* tempv,
        indirect_index_segment_compress_t segment_compress, cuDoubleComplex* d_nzval, int* d_segment_ptr,
        int* d_segment_offset, cuDoubleComplex* d_tempv, int* d_nzvals, int temp_nbrow);
#ifdef __cplusplus
}
#endif
#endif

