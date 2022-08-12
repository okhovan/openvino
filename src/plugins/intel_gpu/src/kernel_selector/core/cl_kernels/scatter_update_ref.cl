// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/data_types.cl"
#include "include/batch_headers/fetch_data.cl"

#define AXIS_B (0)
#define AXIS_F (1)
#define AXIS_W (2)
#define AXIS_Z (OUTPUT_DIMS - 3)
#define AXIS_Y (OUTPUT_DIMS - 2)
#define AXIS_X (OUTPUT_DIMS - 1)

#define GET_OUTPUT_INDEX(idx_order) OUTPUT_GET_INDEX(idx_order)
#define MY_GET_UPDATES_INDEX(idx_order) INPUT2_GET_INDEX(idx_order)

#if OUTPUT_DIMS == 4
    #define ORDER b,f,y,x
#elif OUTPUT_DIMS == 5
    #define ORDER b,f,z,y,x
#elif OUTPUT_DIMS == 6
    #define ORDER b,f,w,z,y,x
#endif


inline void FUNC(planar_to_bfyx)(const uint planar_index,
                                 const uint batch_num, const uint channel_num, const uint height, const uint width,
                                 uint* dst_b, uint* dst_f, uint* dst_y, uint* dst_x)
{
    const uint feature_size = height * width;
    const uint batch_size = channel_num * feature_size;

    *dst_b = planar_index / batch_size;
    const uint dst_fxy = planar_index % batch_size;
    *dst_f = dst_fxy / feature_size;
    const uint dst_xy = dst_fxy % feature_size;
    *dst_y = dst_xy / width;
    *dst_x = dst_xy % width;
}


KERNEL(scatter_update_ref)(const __global INPUT0_TYPE* dictionary,
                   const __global INPUT1_TYPE* indices,
                   const __global INPUT2_TYPE* updates,
                   __global OUTPUT_TYPE* output
#if HAS_FUSED_OPS_DECLS
                   , FUSED_OPS_DECLS
#endif
)
{
    const uint dim0 = get_global_id(0);
    const uint dim1 = get_global_id(1);
    const uint dim2 = get_global_id(2);
#ifndef IS_SECOND_ITER // First kernel
    #if OUTPUT_DIMS == 4
        const uint x = dim0;
        const uint y = dim1;
        const uint f = dim2 % OUTPUT_FEATURE_NUM;
        const uint b = dim2 / OUTPUT_FEATURE_NUM;
    #elif OUTPUT_DIMS == 5
        const uint x = dim0 % OUTPUT_SIZE_X;
        const uint y = dim0 / OUTPUT_SIZE_X;
        const uint z = dim1;
        const uint f = dim2 % OUTPUT_FEATURE_NUM;
        const uint b = dim2 / OUTPUT_FEATURE_NUM;
    #elif OUTPUT_DIMS == 6
        const uint x = dim0 % OUTPUT_SIZE_X;
        const uint y = dim0 / OUTPUT_SIZE_X;
        const uint z = dim1 % OUTPUT_SIZE_Z;
        const uint w = dim1 / OUTPUT_SIZE_Z;
        const uint f = dim2 % OUTPUT_FEATURE_NUM;
        const uint b = dim2 / OUTPUT_FEATURE_NUM;
    #endif

    const uint output_idx = GET_OUTPUT_INDEX(ORDER);

    INPUT0_TYPE val = dictionary[output_idx];
    #if HAS_FUSED_OPS
        FUSED_OPS_FIRST_KERNEL;
        output[output_idx] = TO_OUTPUT_TYPE(FUSED_OPS_RESULT_FIRST_KERNEL);
    #else
        output[output_idx] = ACTIVATION(val, ACTIVATION_PARAMS);
    #endif

#else // Second kernel
    #if (OUTPUT_DIMS == 4)
        // bf|y|x
        #if (AXIS_VALUE == AXIS_F)
            const uint b = dim2 / INDICES_SIZE;
            const uint f = dim2 % INDICES_SIZE;
        #else
            const uint b = dim2 / OUTPUT_FEATURE_NUM;
            const uint f = dim2 % OUTPUT_FEATURE_NUM;
        #endif
        const uint y = dim1;
        const uint x = dim0;
    #elif (OUTPUT_DIMS == 5)
        // bf|z|yx
        #if (AXIS_VALUE == AXIS_F)
            const uint b = dim2 / INDICES_SIZE;
            const uint f = dim2 % INDICES_SIZE;
        #else
            const uint b = dim2 / OUTPUT_FEATURE_NUM;
            const uint f = dim2 % OUTPUT_FEATURE_NUM;
        #endif
        const uint z = dim1;
        #if (AXIS_VALUE == AXIS_X)
            const uint y = dim0 / INDICES_SIZE;
            const uint x = dim0 % INDICES_SIZE;
        #else
            const uint y = dim0 / OUTPUT_SIZE_X;
            const uint x = dim0 % OUTPUT_SIZE_X;
        #endif
    #elif (OUTPUT_DIMS == 6)
        // bf|wz|yx
        #if (AXIS_VALUE == AXIS_F)
            const uint b = dim2 / INDICES_SIZE;
            const uint f = dim2 % INDICES_SIZE;
        #else
            const uint b = dim2 / OUTPUT_FEATURE_NUM;
            const uint f = dim2 % OUTPUT_FEATURE_NUM;
        #endif
        #if (AXIS_VALUE == AXIS_Z)
            const uint w = dim1 / INDICES_SIZE;
            const uint z = dim1 % INDICES_SIZE;
        #else
            const uint w = dim1 / OUTPUT_SIZE_Z;
            const uint z = dim1 % OUTPUT_SIZE_Z;
        #endif
        #if (AXIS_VALUE == AXIS_X)
            const uint y = dim0 / INDICES_SIZE;
            const uint x = dim0 % INDICES_SIZE;
        #else
            const uint y = dim0 / OUTPUT_SIZE_X;
            const uint x = dim0 % OUTPUT_SIZE_X;
        #endif
    #endif

    const uint output_idx = GET_OUTPUT_INDEX(SECOND_ITER_OUTPUT_INDEX_ORDER);
    const uint ref_updates_idx = GET_UPDATES_INDEX(UPDATES_INDEX_ORDER);
    const uint updates_idx = MY_GET_UPDATES_INDEX(UPDATES_INDEX_ORDER);
    printf("%d %d %d %d - %d %d\n", b, f, y, x, ref_updates_idx, updates_idx);
/*
    uint bb, ff, yy, xx;
    FUNC_CALL(planar_to_bfyx)(plain_output_idx, OUTPUT_BATCH_NUM, OUTPUT_FEATURE_NUM, OUTPUT_SIZE_Y, OUTPUT_SIZE_X,
                   bb, ff, yy, xx);
    //since output shape is the same dictionary shape, we can use the same index
    const uint output_idx = plain_output_idx;//OUTPUT_GET_INDEX(bb, ff, yy, xx);
*/


    INPUT2_TYPE val = updates[updates_idx];

    #if HAS_FUSED_OPS
        FUSED_OPS_SECOND_KERNEL;
        output[output_idx] = TO_OUTPUT_TYPE(FUSED_OPS_RESULT_SECOND_KERNEL);
    #else
        output[output_idx] = ACTIVATION(val, ACTIVATION_PARAMS);
    #endif
#endif
}

#undef GET_OUTPUT_INDEX
#undef AXIS_B
#undef AXIS_F
#undef AXIS_W
#undef AXIS_Z
#undef AXIS_Y
#undef AXIS_X
