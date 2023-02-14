#pragma once
#include "superlu_defs.h"
#include <mpi.h>
#include <stdbool.h>

typedef struct {
    int_t lptr;
    int_t ib;
    int_t FullRow;
} LBlock_info_t;

// utils
int get_tag_up();

// zinfomation
void show_options(superlu_dist_options_t * options, gridinfo_t * grid);
void sort_xlsub_lsub(int_t n, int_t* xlsub,int_t* lsub, Glu_persist_t *Glu_persist, gridinfo_t * grid);
void sort_xusub_usub(int_t n, int_t* xusub,int_t* usub, Glu_persist_t *Glu_persist, gridinfo_t * grid);
void show_xlsub_lsub(int_t n, int_t* xlsub,int_t* lsub, Glu_persist_t *Glu_persist, gridinfo_t * grid);
void show_xusub_usub(int_t n, int_t* xusub,int_t* usub, Glu_persist_t *Glu_persist, gridinfo_t * grid);

typedef struct{
    int_t node;
    int_t edge;
    int_t* edge_ptr;
    int_t* edges;
} egraph_t;

void egraph_global_build(egraph_t* graph, int_t n, Glu_persist_t* Glu_persist, int_t** Lrowind_bc_ptr, gridinfo_t * grid);
void egraph_global_reverse(egraph_t* graph_r, const egraph_t* graph);
void egraph_destroy(egraph_t* graph);


typedef struct{
    MPI_Comm comm;
    int root;
    int* sendto;
    int sendto_size;
    int recvfrom; // size of recv 
    bool in;  // in dests
} bcast_t;

void bcast_init(bcast_t* bcast);
void bcast_flat_tree_build(bcast_t* bcast, int* dests, int dests_size, int root, MPI_Comm comm);
void bcast_destory(bcast_t* bcast);

void bcast_build_LB(bcast_t** LB_p, const egraph_t* graph, int_t nsupers, gridinfo_t* grid);
void bcast_build_UB(bcast_t** UB_p, const egraph_t* graph, int_t nsupers, gridinfo_t* grid);

void bcast_destory_all(bcast_t* bcasts, int_t size);

void bcast_blocked_forward(bcast_t* bcast, void* buffer, int count, MPI_Datatype datatype, int tag, int* recv_count_p);
