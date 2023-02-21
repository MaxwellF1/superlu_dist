#include "superlu_zdefs_ex.h"

// Enumerate Type name
char* yes_no_name[] = { "NO", "YES" };
char* fact_name[] = { "DOFACT", "SamePattern", "SamePattern_SameRowPerm", "FACTORED" };
char* rowperm_name[] = { "NOROWPERM", "LargeDiag_MC64", "LargeDiag_HWPM", "MY_PERMR" };
char* colperm_name[] = { "NATURAL", "MMD_ATA", "MMD_AT_PLUS_A", "COLAMD", "METIS_AT_PLUS_A", "PARMETIS", "ZOLTAN", "MY_PERMC" };
char* trans_name[] = { "NOTRANS", "TRANS", "CONJ" };
char* DiagScale_name[] = { "NOEQUIL", "ROW", "COL", "BOTH" };
char* IterRefine_name[] = { "NOREFINE", "SLU_SINGLE", "SLU_DOUBLE", "SLU_EXTRA" };
char* norm_name[] = { "ONE_NORM", "TWO_NORM", "INF_NORM" };

void show_options(superlu_dist_options_t* options, gridinfo_t* grid) {
    int iam = grid->iam;
    if (iam == 0) {
        // ParSymbFact
        printf("SuperLU_Dist options : ---------------------\n");
        printf("Fact : %s\n", fact_name[options->Fact]);
        printf("Equil : %s\n", yes_no_name[options->Equil]);
        printf("DiagInv : %s\n", yes_no_name[options->DiagInv]);
        printf("ColPerm : %s\n", colperm_name[options->ColPerm]);
        printf("Trans : %s\n", trans_name[options->Trans]);
        printf("DiagPivotThresh : %lf\n", options->DiagPivotThresh);
        printf("IterRefine : %s\n", IterRefine_name[options->IterRefine]);
        printf("SymmetricMode : %s\n", yes_no_name[options->SymmetricMode]);
        printf("PivotGrowth : %s\n", yes_no_name[options->PivotGrowth]);
        printf("ConditionNumber : %s\n", yes_no_name[options->ConditionNumber]);
        printf("RowPerm : %s\n", rowperm_name[options->RowPerm]);
        printf("ILU_Norm : %s\n", norm_name[options->ILU_Norm]);
        printf("ParSymbFact : %s\n", yes_no_name[options->ParSymbFact]);
        printf("ReplaceTinyPivot : %s\n", yes_no_name[options->ReplaceTinyPivot]);
        printf("PrintStat : %s\n", yes_no_name[options->PrintStat]);
        printf("num_lookaheads : %d\n", options->num_lookaheads);
        printf("lookahead_etree : %s\n", yes_no_name[options->lookahead_etree]);
        printf("SymPattern : %s\n", yes_no_name[options->SymPattern]);
        printf("Algo3d : %s\n", yes_no_name[options->Algo3d]);
        printf("--------------------------------------------\n");
    }
}

static int int_t_cmp(const void* a, const void* b)
{
    return (int)(*(const int_t*)a - *(const int_t*)b);
}

void sort_xlsub_lsub(int_t n, int_t* xlsub, int_t* lsub, Glu_persist_t* Glu_persist, gridinfo_t* grid) {
    int_t* xsup = Glu_persist->xsup;
    int_t* supno = Glu_persist->supno;
    int_t nsupers = supno[n - 1] + 1;
    int_t ncb = CEILING(nsupers, grid->npcol);
    for (int_t bc = 0; bc < ncb; ++bc) {
        qsort((void*)&lsub[xlsub[bc]], xlsub[bc + 1] - xlsub[bc], sizeof(int_t), int_t_cmp);
    }
}

void sort_xusub_usub(int_t n, int_t* xusub, int_t* usub, Glu_persist_t* Glu_persist, gridinfo_t* grid) {
    int iam = grid->iam;
    if (!iam) {
        int_t* xsup = Glu_persist->xsup;
        int_t* supno = Glu_persist->supno;
        int_t nsupers = supno[n - 1] + 1;
        int_t nrb = CEILING(nsupers, grid->nprow);
        for (int_t br = 0; br < nrb; ++br) {
            qsort((void*)&usub[xusub[br]], xusub[br + 1] - xusub[br], sizeof(int_t), int_t_cmp);
        }
    }
}

void show_xlsub_lsub(int_t n, int_t* xlsub, int_t* lsub, Glu_persist_t* Glu_persist, gridinfo_t* grid) {
    int iam = grid->iam;
    if (!iam) {
        int_t* xsup = Glu_persist->xsup;
        int_t* supno = Glu_persist->supno;
        int_t nsupers = supno[n - 1] + 1;
        int_t nrb = CEILING(nsupers, grid->nprow);
        int_t ncb = CEILING(nsupers, grid->npcol);
        printf("nsupers : %d\n", nsupers);
        printf("nrb : %d\n", nrb);
        printf("ncb : %d\n", ncb);
        printf("xlsub lsub");
        for (int_t bc = 0; bc < ncb; ++bc) {
            printf("supernode : %d : %d\n\t", bc, xlsub[bc + 1] - xlsub[bc]);
            for (int_t index = xlsub[bc]; index < xlsub[bc + 1]; ++index) {
                printf("%d,", lsub[index]);
            }
            printf("\n");
        }
    }
}

void show_xusub_usub(int_t n, int_t* xusub, int_t* usub, Glu_persist_t* Glu_persist, gridinfo_t* grid) {
    int iam = grid->iam;
    if (!iam) {
        int_t* xsup = Glu_persist->xsup;
        int_t* supno = Glu_persist->supno;
        int_t nsupers = supno[n - 1] + 1;
        int_t nrb = CEILING(nsupers, grid->nprow);
        int_t ncb = CEILING(nsupers, grid->npcol);
        printf("nsupers : %d\n", nsupers);
        printf("nrb : %d\n", nrb);
        printf("ncb : %d\n", ncb);
        printf("xusub usub");
        for (int_t br = 0; br < nrb; ++br) {
            printf("supernode : %d : %d\n\t", br, xusub[br + 1] - xusub[br]);
            for (int_t index = xusub[br]; index < xusub[br + 1]; ++index) {
                printf("%d,", usub[index]);
            }
            printf("\n");
        }
    }
}

void show_Lindex(int_t k, int_t* Lindex, Glu_persist_t* Glu_persist, gridinfo_t* grid) {
    int_t* xsup = Glu_persist->xsup;
    int_t* supno = Glu_persist->supno;
    int iam = grid->iam;
    int myrow = MYROW(iam, grid);
    int mycol = MYCOL(iam, grid);
    printf("%d :", k);
    if (Lindex) {
        int_t nlb = Lindex[0]; // number of L blocks in L(:,gb)
        int_t total_rows = Lindex[1];
        int_t index = BC_HEADER;
        for (int j = 0; j < nlb; j++)
        {
            int_t supernode_id = Lindex[k];
            int_t rows = Lindex[k + 1];
            printf(" %d", supernode_id);
            index += LB_DESCRIPTOR + rows;
        }
        printf("\n");
    }    
else {
        printf(" empty column\n");
    }
}

void show_Uindex(int_t k, int_t* Uindex, Glu_persist_t* Glu_persist, gridinfo_t* grid) {
    int_t* xsup = Glu_persist->xsup;
    int_t* supno = Glu_persist->supno;
    int iam = grid->iam;
    int myrow = MYROW(iam, grid);
    int mycol = MYCOL(iam, grid);
    printf("%d :", k);
    int_t nsupr = SuperSize(k);
    int_t fstVtx = xsup[k];
    int_t lstVtx = xsup[k + 1];
    if (Uindex) {
        int_t nub = Uindex[0]; // number of U blocks in U(gb,:)
        int_t nzval_len = Uindex[1];
        int_t index_len = Uindex[2];
        int_t index = BR_HEADER;
        for (int j = 0; j < nub; j++) {
            int_t supernode_id = Uindex[index];
            int_t block_nzval_len = Uindex[index + 1];
            int_t nsupc = SuperSize(supernode_id);
            int non_empty_column_count = 0;
            /* Start fstnz in index */
            // for(int kk = 0; kk < nsupc; kk++){
            //     int_t fstnz = Uindex[index + UB_DESCRIPTOR + kk];
            //     int_t segsize = lstVtx - fstnz;
            //     if(segsize > 0){
            //         assert(segsize == nsupr);
            //         non_empty_column_count ++;
            //     }
            // }
            printf(" %d", supernode_id);
            index += UB_DESCRIPTOR + nsupc;
            // assert(non_empty_column_count * nsupr == block_nzval_len);
        }
        printf("\n");
    }
    else {
        printf(" empty row\n");
    }
}

void show_L_structure(int_t n, zLUstruct_t* LUstruct, gridinfo_t* grid) {
    int iam = grid->iam;

    Glu_persist_t* Glu_persist = LUstruct->Glu_persist;
    zLocalLU_t* Llu = LUstruct->Llu;
    int_t* xsup = Glu_persist->xsup;
    int_t* supno = Glu_persist->supno;
    int_t nsupers = supno[n - 1] + 1;
    int_t nrb = CEILING(nsupers, grid->nprow);
    int_t ncb = CEILING(nsupers, grid->npcol);
    int myrow = MYROW(iam, grid);
    int mycol = MYCOL(iam, grid);
    printf("L Structrue : \n");
    printf("ncb : %d\n", ncb);
    for (int_t lb = 0; lb < ncb; lb++) {
        int_t gb = lb * grid->npcol + mycol;
        int_t* index = Llu->Lrowind_bc_ptr[lb];
        // printf("super node : %d\n", gb);
        printf("%d :", gb);
        int_t nsupc = SuperSize(gb);
        int_t fstVtx = xsup[gb];
        int_t lstVtx = xsup[gb + 1];
        if (index) {   /* Not an empty column */
            int_t nlb = index[0]; // number of L blocks in L(:,gb)
            int_t total_rows = index[1];
            // printf("header (nlb, total_rows) :  (%d, %d)\n", nlb, total_rows);
            int_t k = BC_HEADER;
            int krow = PROW(gb, grid);
            if (krow == myrow) {  /* skip the diagonal block */
                int_t supernode_id = index[k];
                int_t rows = index[k + 1];
                int_t cols = SuperSize(supernode_id);
                // printf("diagonal block: %d, shape : (%d, %d)\n", supernode_id, rows, cols);
                printf(" %d", supernode_id);
                // printf("cols : ");
                // for(int ii = xsup[supernode_id]; ii < xsup[supernode_id+1]; ii++){
                //     printf("%d, ", ii);
                // }
                // printf("\n");
                // printf("rows : ");
                // for(int ii = 0; ii < rows; ii++){
                //     printf("%d, ", index[k + LB_DESCRIPTOR + ii]);
                // }
                // printf("\n");
                k += LB_DESCRIPTOR + index[k + 1];
                nlb--;
            }
            for (int j = 0; j < nlb; j++)
            {
                int_t supernode_id = index[k];
                int_t rows = index[k + 1];
                // printf("block(supernode_id, rows) %d, %d\n", supernode_id, rows);
                printf(" %d", supernode_id);
                // for(int i = 0; i < rows; ++i){
                //     printf("%d, ", index[k + LB_DESCRIPTOR + i]);
                // }
                // printf("\n");
                k += LB_DESCRIPTOR + rows;
            }
            printf("\n");
        }
        else {
            printf(" empty column\n");
        }
        // break; // just show one colume
    }
}

void show_U_structure(int_t n, zLUstruct_t* LUstruct, gridinfo_t* grid) {
    int iam = grid->iam;
    Glu_persist_t* Glu_persist = LUstruct->Glu_persist;
    zLocalLU_t* Llu = LUstruct->Llu;
    int_t* xsup = Glu_persist->xsup;
    int_t* supno = Glu_persist->supno;
    int_t nsupers = supno[n - 1] + 1;
    int_t nrb = CEILING(nsupers, grid->nprow);
    int_t ncb = CEILING(nsupers, grid->npcol);
    int myrow = MYROW(iam, grid);
    int mycol = MYCOL(iam, grid);
    printf("U Structrue : \n");
    printf("nrb : %d\n", nrb);
    for (int_t lb = 0; lb < nrb; lb++) {
        int_t gb = lb * grid->nprow + myrow;
        int_t* index = Llu->Ufstnz_br_ptr[lb];
        // printf("super node : %d\n", gb);
        printf("%d :", gb);
        int_t nsupr = SuperSize(gb);
        int_t fstVtx = xsup[gb];
        int_t lstVtx = xsup[gb + 1];
        if (index) {   /* Not an empty row */
            int_t nub = index[0]; // number of U blocks in U(gb,:)
            int_t nzval_len = index[1];
            int_t index_len = index[2];
            assert(index[index_len] == -1);
            // printf("header (nub, nzval_len, index_len) :  (%d, %d, %d)\n", nub, nzval_len, index_len);
            int_t k = BR_HEADER;
            for (int j = 0; j < nub; j++)
            {
                int_t supernode_id = index[k];
                int_t block_nzval_len = index[k + 1];
                int_t nsupc = SuperSize(supernode_id);

                int non_empty_column_count = 0;
                /* Start fstnz in index */
                for (int kk = 0; kk < nsupc; kk++) {
                    int_t fstnz = index[k + UB_DESCRIPTOR + kk];
                    int_t segsize = lstVtx - fstnz;
                    if (segsize > 0) {
                        assert(segsize == nsupr);
                        non_empty_column_count++;
                    }
                }

                // printf("block(supernode_id, block_nzval_len, SuperSize(supernode_id), non_empty_column_count) %d, %d, %d, %d\n", supernode_id, block_nzval_len, nsupc, non_empty_column_count);
                printf(" %d", supernode_id);
                k += UB_DESCRIPTOR + nsupc;

                assert(non_empty_column_count * nsupr == block_nzval_len);
            }
            printf(" \n");
        }
        else {
            printf(" empty row\n");
        }
        // break; // just show one colume
    }
}

void check_L_is_ordered(int_t n, zLUstruct_t* LUstruct, gridinfo_t* grid) {
    int iam = grid->iam;
    Glu_persist_t* Glu_persist = LUstruct->Glu_persist;
    zLocalLU_t* Llu = LUstruct->Llu;
    int_t* xsup = Glu_persist->xsup;
    int_t* supno = Glu_persist->supno;
    int_t nsupers = supno[n - 1] + 1;
    int_t nrb = CEILING(nsupers, grid->nprow);
    int_t ncb = CEILING(nsupers, grid->npcol);
    int myrow = MYROW(iam, grid);
    int mycol = MYCOL(iam, grid);
    for (int_t lb = 0; lb < ncb; lb++) {
        int_t gb = lb * grid->npcol + mycol;
        int_t* index = Llu->Lrowind_bc_ptr[lb];
        if (index) {   /* Not an empty column */
            int_t nlb = index[0]; // number of L blocks in L(:,gb)
            int_t total_rows = index[1];
            int_t k = BC_HEADER;
            for (int j = 0; j < nlb; j++)
            {
                int_t supernode_id = index[k];
                int_t rows = index[k + 1];
                if (rows > 1) {
                    for (int i = 1; i < rows; ++i) {
                        assert(index[k + LB_DESCRIPTOR + i] > index[k + LB_DESCRIPTOR + i - 1]);
                    }
                }
                k += LB_DESCRIPTOR + rows;
            }
        }
    }
}


void check_U_is_full(int_t n, zLUstruct_t* LUstruct, gridinfo_t* grid) {
    int iam = grid->iam;
    Glu_persist_t* Glu_persist = LUstruct->Glu_persist;
    zLocalLU_t* Llu = LUstruct->Llu;
    int_t* xsup = Glu_persist->xsup;
    int_t* supno = Glu_persist->supno;
    int_t nsupers = supno[n - 1] + 1;
    int_t nrb = CEILING(nsupers, grid->nprow);
    int_t ncb = CEILING(nsupers, grid->npcol);
    int myrow = MYROW(iam, grid);
    int mycol = MYCOL(iam, grid);
    for (int_t lb = 0; lb < nrb; lb++) {
        int_t gb = lb * grid->nprow + myrow;
        int_t* index = Llu->Ufstnz_br_ptr[lb];
        int_t nsupr = SuperSize(gb);
        int_t fstVtx = xsup[gb];
        int_t lstVtx = xsup[gb + 1];
        if (index) {   /* Not an empty row */
            int_t nub = index[0]; // number of U blocks in U(gb,:)
            int_t nzval_len = index[1];
            int_t index_len = index[2];
            assert(index[index_len] == -1);
            int_t total_non_empty_column_count = 0;
            int_t k = BR_HEADER;
            for (int j = 0; j < nub; j++)
            {
                int_t supernode_id = index[k];
                int_t block_nzval_len = index[k + 1];
                int_t nsupc = SuperSize(supernode_id);

                int non_empty_column_count = 0;
                /* Start fstnz in index */
                for (int kk = 0; kk < nsupc; kk++) {
                    int_t fstnz = index[k + UB_DESCRIPTOR + kk];
                    int_t segsize = lstVtx - fstnz;
                    if (segsize > 0) {
                        assert(segsize == nsupr);
                        non_empty_column_count++;
                    }
                }
                k += UB_DESCRIPTOR + nsupc;
                assert(non_empty_column_count * nsupr == block_nzval_len);
                total_non_empty_column_count += non_empty_column_count;
            }
            assert(nzval_len == total_non_empty_column_count * nsupr);
        }
    }
}

void show_supernode_size(int_t n, Glu_persist_t* Glu_persist, gridinfo_t* grid) {
    int iam = grid->iam;
    if (iam == 0) {
        int_t* xsup = Glu_persist->xsup;
        int_t* supno = Glu_persist->supno;
        int_t nsupers = supno[n - 1] + 1;
        printf("nsupers : %d\n", nsupers);
        for (int_t gb = 0; gb < nsupers; gb++) {
            printf("%d, ", SuperSize(gb));
        }
        printf("\n");
    }
}

void check_symmetry(int_t n, zLUstruct_t* LUstruct, gridinfo_t* grid) {
    int iam = grid->iam;
    if (grid->nprow != grid->npcol) {
        return;
    }
    Glu_persist_t* Glu_persist = LUstruct->Glu_persist;
    zLocalLU_t* Llu = LUstruct->Llu;
    int_t* xsup = Glu_persist->xsup;
    int_t* supno = Glu_persist->supno;
    int_t nsupers = supno[n - 1] + 1;
    int_t nrb = CEILING(nsupers, grid->nprow);
    int_t ncb = CEILING(nsupers, grid->npcol);
    assert(ncb == nrb);
    int_t nb = nrb;
    int myrow = MYROW(iam, grid);
    int mycol = MYCOL(iam, grid);
    //  check diagonal process
    if (myrow == mycol) {
        for (int_t lb = 0; lb < nb; lb++) {
            int_t gb = lb * grid->npcol + mycol;
            int_t* L_index = Llu->Lrowind_bc_ptr[lb];
            int_t* U_index = Llu->Ufstnz_br_ptr[lb];
            int_t nsupr = SuperSize(gb);
            int_t nsupc = SuperSize(gb);
            int_t fstVtx = xsup[gb];
            int_t lstVtx = xsup[gb + 1];
            if (L_index && U_index) {
                int_t nlb = L_index[0]; // number of L blocks in L(:,gb)
                int_t total_rows = L_index[1];
                int_t nub = U_index[0];
                int_t nzval_len = U_index[1];
                int_t index_len = U_index[2];
                int_t k;
                int_t tmp_index;
                assert(nlb - 1 == nub);
                int_t lub = nub;
                nlb -= 1;    // skip diagonal block

                // A diagonal 
                zMatrix_t D;
                // assert first L block is diagonal block
                assert(L_index[BC_HEADER] == gb);
                int_t D_rows = L_index[BC_HEADER + 1];
                zMatrix_init(&D, D_rows, nsupc, total_rows, Llu->Lnzval_bc_ptr[lb]);
                assert(zMatrix_check_symmetry(&D));

                // L all 
                zMatrix_t L, U;
                int_t total_L_rows = total_rows - D_rows;
                zMatrix_init(&L, total_L_rows, nsupc, total_rows, Llu->Lnzval_bc_ptr[lb] + D_rows);

                // get L block non empty rows
                int_t* L_row_counts = SUPERLU_MALLOC((nlb) * sizeof(int_t));
                int_t* L_row_ptr = SUPERLU_MALLOC((nlb + 1) * sizeof(int_t));
                int_t* L_rows = SUPERLU_MALLOC((total_L_rows) * sizeof(int_t));

                k = BC_HEADER + LB_DESCRIPTOR + L_index[BC_HEADER + 1];  // skip diagonal block
                tmp_index = 0;
                for (int_t j = 0; j < nlb; j++) {
                    int_t supernode_id = L_index[k];
                    int_t rows = L_index[k + 1];
                    L_row_counts[j] = rows;
                    for (int_t kk = 0; kk < rows; ++kk) {
                        L_rows[tmp_index++] = L_index[k + LB_DESCRIPTOR + kk];
                    }
                    k += LB_DESCRIPTOR + L_index[k + 1];
                }
                L_row_ptr[0] = 0;
                for (int_t j = 0; j < nlb; j++) {
                    L_row_ptr[j + 1] = L_row_ptr[j] + L_row_counts[j];
                }
                assert(L_row_ptr[nlb] == total_L_rows);

                // U all
                k = BR_HEADER;
                int_t* U_column_counts = SUPERLU_MALLOC(nub * sizeof(int_t));
                int_t* U_col_ptr = SUPERLU_MALLOC((nub + 1) * sizeof(int_t));
                for (int_t j = 0; j < nub; j++) {
                    U_column_counts[j] = 0;
                    int_t supernode_id = U_index[k];
                    int_t block_nzval_len = U_index[k + 1];
                    int_t columns = SuperSize(supernode_id);
                    for (int_t kk = 0; kk < columns; kk++) {
                        int_t fstnz = U_index[k + UB_DESCRIPTOR + kk];
                        int_t segsize = lstVtx - fstnz;
                        if (segsize > 0) {
                            assert(segsize == nsupr);
                            U_column_counts[j] += 1;
                        }
                    }
                    k += UB_DESCRIPTOR + columns;
                }

                U_col_ptr[0] = 0;
                for (int_t j = 0; j < nub; j++) {
                    U_col_ptr[j + 1] = U_col_ptr[j] + U_column_counts[j];
                }
                int_t total_U_cols = U_col_ptr[nub];

                assert(total_L_rows == total_U_cols);
                assert(total_U_cols * nsupr == nzval_len);
                zMatrix_init(&U, nsupr, total_U_cols, nsupr, Llu->Unzval_br_ptr[lb]);

                // get U block non empty cols
                int_t* U_cols = SUPERLU_MALLOC((total_U_cols) * sizeof(int_t));
                tmp_index = 0;
                k = BR_HEADER;
                for (int_t j = 0; j < nub; j++) {
                    int_t supernode_id = U_index[k];
                    int_t block_nzval_len = U_index[k + 1];
                    int_t columns = SuperSize(supernode_id);
                    int_t fstCol = FstBlockC(supernode_id);

                    for (int kk = 0; kk < columns; kk++) {
                        int_t fstnz = U_index[k + UB_DESCRIPTOR + kk];
                        int_t segsize = lstVtx - fstnz;
                        if (segsize > 0) {
                            assert(segsize == nsupr);
                            U_cols[tmp_index++] = fstCol + kk;
                        }
                    }
                    k += UB_DESCRIPTOR + columns;
                }

                // L rows is out of order.
                // for(int_t j = 0; j < lub; ++j){
                //     // print L rows:
                //     printf("block %d\n", j);
                //     printf("L : ");
                //     for(int_t kk = L_row_ptr[j]; kk < L_row_ptr[j+1]; ++kk){
                //         printf("%d, ", L_rows[kk]);
                //     }
                //     printf("\n");
                //     printf("U : ");
                //     for(int_t kk = U_col_ptr[j]; kk < U_col_ptr[j+1]; ++kk){
                //         printf("%d, ", U_cols[kk]);
                //     }
                //     printf("\n");
                // }

                // mapping from B to A
                int_t* mapUC2LR = SUPERLU_MALLOC(total_U_cols * sizeof(int_t));
                for (int_t j = 0; j < lub; ++j) {
                    for (int_t kkc = U_col_ptr[j]; kkc < U_col_ptr[j + 1]; ++kkc) {
                        int_t bcol = U_cols[kkc];
                        for (int_t kkr = L_row_ptr[j]; kkr < L_row_ptr[j + 1]; ++kkr) {
                            int_t arow = L_rows[kkr];
                            if (bcol == arow) {
                                mapUC2LR[kkc] = kkr;
                            }
                        }
                    }
                }

                // check symmetry
                for (int_t uc = 0; uc < U.col; ++uc) {
                    for (int_t ur = 0; ur < U.row; ++ur) {
                        doublecomplex u = U.val[ur + uc * U.ld];
                        int_t lr = mapUC2LR[uc];
                        int_t lc = ur;
                        doublecomplex l = L.val[lr + lc * L.ld];
                        if (fabs(l.i) > 1e-12) {
                            assert(false);
                        }
                        if (fabs(u.i) > 1e-12) {
                            assert(false);
                        }
                        if (fabs(l.r - u.r) > 1e-12 || fabs(l.i + u.i) > 1e-12) {
                            assert(false);
                        }
                    }
                }

                SUPERLU_FREE(mapUC2LR);
                SUPERLU_FREE(L_row_counts);
                SUPERLU_FREE(L_row_ptr);
                SUPERLU_FREE(L_rows);
                SUPERLU_FREE(U_column_counts);
                SUPERLU_FREE(U_col_ptr);
                SUPERLU_FREE(U_cols);
            }
            break;
        }
    }
}
