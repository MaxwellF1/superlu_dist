
#include "superlu_zdefs_ex.h"

void zMatrix_init(zMatrix_t* matrix, int row, int col, int ld, doublecomplex* val)
{
    matrix->row = row;
    matrix->col = col;
    matrix->ld = ld;
    matrix->val = val;
}

