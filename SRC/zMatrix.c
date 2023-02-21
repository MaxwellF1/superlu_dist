
#include "superlu_zdefs_ex.h"

void zMatrix_init(zMatrix_t* matirx, int row, int col, int ld, doublecomplex* val) {
    matirx->row = row;
    matirx->col = col;
    matirx->ld = ld;
    matirx->val = val;
}

void zMatrix_set_val(zMatrix_t* matirx, doublecomplex* val) {
    matirx->val = val;
}

bool zMatrix_check_symmetry(const zMatrix_t* matirx) {
    if (matirx->row != matirx->col) {
        return false;
    }
    for (int r = 0; r < matirx->row; ++r) {
        for (int c = 0; c < matirx->col; ++c) {
            doublecomplex a = matirx->val[r + c * matirx->ld];
            doublecomplex b = matirx->val[c + r * matirx->ld];
            if (fabs(a.r - b.r) > 1e-12)
                return false;
            if (fabs(a.i + b.i) > 1e-12)
                return false;
        }
    }
    return true;
}

bool zMatrix_check_transpose(const zMatrix_t* A, const zMatrix_t* B) {
    if (A->row != B->col || B->row != A->col) {
        printf("(%d %d), (%d %d)\n", A->row, A->col, B->row, B->col);
        return false;
    }
    for (int ar = 0; ar < A->row; ++ar) {
        for (int ac = 0; ac < A->col; ++ac) {
            doublecomplex a = A->val[ar + ac * A->ld];
            int br = ac;
            int bc = ar;
            assert(br < B->row);
            assert(bc < B->col);
            doublecomplex b = B->val[br + bc * B->ld];
            if (fabs(a.r - b.r) > 1e-12) {
                return false;
            }
        }
    }
    return true;
}
