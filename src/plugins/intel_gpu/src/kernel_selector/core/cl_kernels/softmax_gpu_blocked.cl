// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/common.cl"


__attribute__((intel_reqd_sub_group_size(16)))
KERNEL(softmax)(
        __global INPUT0_TYPE* input,
__global OUTPUT_TYPE* output
#if HAS_FUSED_OPS_DECLS
, FUSED_OPS_DECLS
#endif
) {
/*
#if INPUT0_DIMS == 5
    const uint other0 = (uint)get_global_id(0) % INPUT0_OTHER0_SIZE;  // y*z (x-norm), x*z (y-norm), x*y (z-norm)
    const uint other2 = (uint)get_global_id(0) / INPUT0_OTHER0_SIZE;
#else
    const uint other0 = get_global_id(0);  // y for x-norm, x for y-norm
    const uint other2 = 0;
#endif
    const uint other1 = get_global_id(1);  // feature (x,y,z-normalization), or y (b,f-normalization)
    const uint other3  = get_global_id(2); // batch, or feature for b-normalization
*/

    const uint no_offset = 0;
    uint cls = 0;
    uint *b_offset, *f_offset, *z_offset, *y_offset, *x_offset;
    b_offset = f_offset = z_offset = y_offset = x_offset = &no_offset;

#if SOFTMAX_DIM_X
#define CLASS_NUM INPUT0_SIZE_X
    const uint b = get_global_id(2);
    const uint f = get_global_id(1);
    const uint z = (uint)get_global_id(0) % INPUT0_SIZE_Y;
    const uint y = (uint)get_global_id(0) / INPUT0_SIZE_Y;
    const uint x = 0;
    x_offset = &cls;
#elif SOFTMAX_DIM_Y
#define CLASS_NUM INPUT0_SIZE_Y
    const uint b = get_global_id(2);
    const uint f = get_global_id(1);
    const uint z = (uint)get_global_id(0) / INPUT0_SIZE_X;
    const uint y = 0;
    const uint x = (uint)get_global_id(0) % INPUT0_SIZE_X;
    y_offset = &cls;
#elif SOFTMAX_DIM_Z
#define CLASS_NUM INPUT0_SIZE_Z
    const uint b = get_global_id(2);
    const uint f = get_global_id(1);
    const uint z = 0;
    const uint y = (uint)get_global_id(0) % INPUT0_SIZE_X;
    const uint x = (uint)get_global_id(0) / INPUT0_SIZE_X;
    z_offset = &cls;
#elif SOFTMAX_DIM_FEATURE
#define CLASS_NUM INPUT0_FEATURE_NUM
    const uint b = get_global_id(2);
    const uint f = 0;
    const uint z = (uint)get_global_id(0) / INPUT0_SIZE_X;
    const uint y = get_global_id(1);
    const uint x = (uint)get_global_id(0) % INPUT0_SIZE_X;
    f_offset = &cls;
#elif SOFTMAX_DIM_BATCH
#define CLASS_NUM INPUT0_BATCH_NUM
    const uint b = 0;
    const uint f = get_global_id(2);
    const uint z = (uint)get_global_id(0) / INPUT0_SIZE_X;
    const uint y = get_global_id(1);
    const uint x = (uint)get_global_id(0) % INPUT0_SIZE_X;
    b_offset = &cls;
#else
#error
#endif

    ACCUMULATOR_TYPE max_value = UNIT_VAL_MIN;
    ACCUMULATOR_TYPE data[CLASS_NUM];

    for (cls = 0; cls < CLASS_NUM; ++cls)
    {
#if INPUT0_DIMS == 5
        const uint index = INPUT0_GET_INDEX(b + *b_offset, f + *f_offset, z + *z_offset, y + *y_offset, x + *x_offset);
#else
        const uint index = INPUT0_GET_INDEX(b + *b_offset, f + *f_offset, y + *y_offset, x + *x_offset);
#endif
    ACCUMULATOR_TYPE in = input[index];
    max_value = max(max_value, in);
    data[cls] = in;
//        printf("b=%d f=%d z=%d y=%d x=%d cls=%d f_offset=%d index=%d in=%f\n",
//                b, f, z, y, x, cls, index, in);
}

    // TODO: currently we calculate on float32 because it's lot of "add" operation and it stuck on the value "8192.0f"
    ACCUMULATOR_TYPE denominator = 0.0;
    for (cls = 0; cls < CLASS_NUM; ++cls)
    {
        data[cls] = native_exp(data[cls] - max_value);;
        denominator += data[cls];
    }

    for (cls = 0; cls < CLASS_NUM; ++cls)
    {
        const ACCUMULATOR_TYPE res = data[cls] / denominator;
#if INPUT0_DIMS == 5
        const uint output_idx = OUTPUT_GET_INDEX(b + *b_offset, f + *f_offset, z + *z_offset, y + *y_offset, x + *x_offset);
#else
        const uint output_idx = OUTPUT_GET_INDEX(b + *b_offset, f + *f_offset, y + *y_offset, x + *x_offset);
#endif
#if HAS_FUSED_OPS
        FUSED_OPS;
        output[output_idx] = FUSED_OPS_RESULT;
#else
        output[output_idx] = ACTIVATION(res, ACTIVATION_PARAMS);
#endif
    }
}
