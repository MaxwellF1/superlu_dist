#pragma once

#ifdef __cplusplus
extern "C" {
#endif 

#include <stdlib.h>

#ifndef MAX_BLOCK_SIZE 
#define MAX_BLOCK_SIZE 256
#endif
#ifndef UTILS_H
#define UTILS_H



    typedef struct {
        int segment_ptr[MAX_BLOCK_SIZE + 1];
        int segment_offset[MAX_BLOCK_SIZE];
        int segment_count;
    } indirect_index_segment_compress_t;

    static void indirect_index_segment_compress_init(indirect_index_segment_compress_t* segment_compress, const int* indirect_index, int len) {
        if (len < 1) {
            segment_compress->segment_count = 0;
            return;
        }
        int* segment_ptr = segment_compress->segment_ptr;
        int* segment_offset = segment_compress->segment_offset;

        // segment compress
        segment_ptr[0] = 0;
        segment_offset[0] = indirect_index[0];
        int segment_prev = indirect_index[0];
        int segment_count = 1;
        for (int i = 1; i < len; i++) {
            int segment_cur = indirect_index[i] - i;//offset
            if (segment_prev != segment_cur) { // 如果跳到了下一个block， offset这个时候值变化了
                segment_ptr[segment_count] = i;//记录这个segment的ptr
                segment_offset[segment_count] = segment_cur;//记录这个segment的offset
                segment_count += 1;//下个segment
                segment_prev = segment_cur;//保存这次的offset 然后下次进行比较
            }
        }
        segment_ptr[segment_count] = len;

        // output
        segment_compress->segment_count = segment_count;
    }

    static void indirect_index_segment_compress_destroy(indirect_index_segment_compress_t* segment_compress) {
    }

    //zscatter_opt

    // void indirect_index_segment_compress_use_example(){
    //     indirect_index_segment_compress_t segment_compress;
    //     double* arr;
    //     for(int ptr = 0; ptr < segment_compress.segment_count; ++ptr){
    //         int i_start = segment_compress.segment_ptr[ptr];
    //         int i_end = segment_compress.segment_ptr[ptr+1];
    //         int offset = segment_compress.segment_offset[ptr];
    //         double *ARR = arr + offset;
    //         for(int i = i_start; i < i_end; i++){
    //             // do some thing for ARR[i]
    //         } 
    //     }
    // }

#ifdef __cplusplus
}
#endif
#endif 
