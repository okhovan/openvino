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
    const uint parallel_items = INPUT0_BATCH_NUM * INPUT0_FEATURE_NUM;
    __local INPUT0_TYPE max_values_per_batch_and_feature[parallel_items];
    __local INPUT0_TYPE denominators_per_batch_and_feature[parallel_items];

    const uint b = get_local_id(0);
    const uint f = get_local_id(1);

    ACCUMULATOR_TYPE max_value = UNIT_VAL_MIN;

#if INPUT0_DIMS == 5
    for (uint z = 0; z < INPUT0_SIZE_Z; ++z) {
#endif
        for (uint y = 0; y < INPUT0_SIZE_Y; ++y) {
            for (uint x = 0; x < INPUT0_SIZE_X; ++x) {
#if INPUT0_DIMS == 5
                const uint index = INPUT0_GET_INDEX(b, f, z, y, x);
#else
                const uint index = INPUT0_GET_INDEX(b, f, y, x);
#endif
                ACCUMULATOR_TYPE in = input[index];
                output[index] = in;
                max_value = max(max_value, in);
            }
        }
#if INPUT0_DIMS == 5
    }
#endif

    const uint parallel_idx = b * INPUT0_FEATURE_NUM + f;
    max_values_per_batch_and_feature[parallel_idx] = max_value;

    barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);

    max_value = UNIT_VAL_MIN;
    for (uint i = 0; i < parallel_items; ++i) {
        max_value = max(max_value, max_values_per_batch_and_feature[i]);
        //printf("%d %d max[%d]=%f\n", bb, ff, i, max_values_per_batch_and_feature[i]);
    }

    ACCUMULATOR_TYPE denominator = 0.0;

#if INPUT0_DIMS == 5
    for (uint z = 0; z < INPUT0_SIZE_Z; ++z) {
#endif
        for (uint y = 0; y < INPUT0_SIZE_Y; ++y) {
            for (uint x = 0; x < INPUT0_SIZE_X; ++x) {
#if INPUT0_DIMS == 5
                const uint index = INPUT0_GET_INDEX(b, f, z, y, x);
#else
                const uint index = INPUT0_GET_INDEX(b, f, y, x);
#endif
                output[index] = native_exp(output[index] - max_value);
                denominator += output[index];
            }
        }
#if INPUT0_DIMS == 5
    }
#endif

    denominators_per_batch_and_feature[parallel_idx] = denominator;

    barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);

    denominator = 0;
    for (uint i = 0; i < parallel_items; ++i) {
        denominator += denominators_per_batch_and_feature[i];
    }

#if INPUT0_DIMS == 5
    for (uint z = 0; z < INPUT0_SIZE_Z; ++z) {
#endif
        for (uint y = 0; y < INPUT0_SIZE_Y; ++y) {
            for (uint x = 0; x < INPUT0_SIZE_X; ++x) {
#if INPUT0_DIMS == 5
                const uint output_idx = OUTPUT_GET_INDEX(b, f, z, y, x);
#else
                const uint output_idx = OUTPUT_GET_INDEX(b, f, y, x);
#endif
                const ACCUMULATOR_TYPE res = output[output_idx] / denominator;

                #if HAS_FUSED_OPS
                FUSED_OPS;
                output[output_idx] = FUSED_OPS_RESULT;
                #else
                output[output_idx] = ACTIVATION(res, ACTIVATION_PARAMS);
                #endif
            }
        }
#if INPUT0_DIMS == 5
    }
#endif

    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
}
