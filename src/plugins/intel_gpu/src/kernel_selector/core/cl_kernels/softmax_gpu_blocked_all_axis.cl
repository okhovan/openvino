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
    ACCUMULATOR_TYPE max_value = UNIT_VAL_MIN;
    ACCUMULATOR_TYPE data[CLASS_NUM];
    uint cls = 0;

    __local INPUT0_TYPE max_values_per_batch_and_feature[INPUT0_BATCH_NUM*INPUT0_FEATURE_NUM];
    __local INPUT0_TYPE denominators_per_batch_and_feature[INPUT0_BATCH_NUM*INPUT0_FEATURE_NUM];

    const uint bb = get_local_id(0);
    const uint ff = get_local_id(1);

//    for (uint b=0; b < INPUT0_BATCH_NUM; ++b) {
//        for (uint f=0; f < INPUT0_FEATURE_NUM; ++f) {
            #if INPUT0_DIMS == 5
            for (uint z=0; z < INPUT0_SIZE_Z; ++z) {
            #endif
                for (uint y=0; y < INPUT0_SIZE_Y; ++y) {
                    for (uint x=0; x < INPUT0_SIZE_X; ++x) {
                        #if INPUT0_DIMS == 5
                        const uint index = INPUT0_GET_INDEX(bb, ff, z, y, x);
                        #else
                        const uint index = INPUT0_GET_INDEX(bb, ff, y, x);
                        #endif

                        ACCUMULATOR_TYPE in = input[index];
                        max_value = max(max_value, in);
                        //data[index/*cls++*/] = in;
                        output[index] = in;
                        printf("%d %d %d %d %f %f\n", bb, ff, y, x, in, max_value);

                    }
                }
            #if INPUT0_DIMS == 5
            }
            #endif
//        }
//    }
    max_values_per_batch_and_feature[bb * INPUT0_FEATURE_NUM + ff] = max_value;

    barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);

    max_value = UNIT_VAL_MIN;
    for (uint i=0; i<INPUT0_BATCH_NUM*INPUT0_FEATURE_NUM; ++i) {
        max_value = max(max_value, max_values_per_batch_and_feature[i]);
        //printf("%d %d max[%d]=%f\n", bb, ff, i, max_values_per_batch_and_feature[i]);
    }


    // TODO: currently we calculate on float32 because it's lot of "add" operation and it stuck on the value "8192.0f"
    ACCUMULATOR_TYPE denominator = 0.0;
//    for (cls = 0; cls < CLASS_NUM; ++cls) {
//        data[cls] = native_exp(data[cls] - max_value);
//        denominator += data[cls];
//    }

    #if INPUT0_DIMS == 5
    for (uint z=0; z < INPUT0_SIZE_Z; ++z) {
    #endif
        for (uint y=0; y < INPUT0_SIZE_Y; ++y) {
            for (uint x=0; x < INPUT0_SIZE_X; ++x) {
                #if INPUT0_DIMS == 5
                const uint index = INPUT0_GET_INDEX(bb, ff, z, y, x);
                #else
                const uint index = INPUT0_GET_INDEX(bb, ff, y, x);
                #endif
                output[index] = native_exp(output[index] - max_value);
                denominator += output[index];

            }
        }
    #if INPUT0_DIMS == 5
    }
    #endif

    denominators_per_batch_and_feature[bb * INPUT0_FEATURE_NUM + ff] = denominator;

    barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);

    denominator = 0;
    for (uint i=0; i<INPUT0_BATCH_NUM*INPUT0_FEATURE_NUM; ++i) {
        denominator += denominators_per_batch_and_feature[i];
    }


    cls = 0;
//    for (uint b=0; b < INPUT0_BATCH_NUM; ++b) {
//        for (uint f=0; f < INPUT0_FEATURE_NUM; ++f) {
            #if INPUT0_DIMS == 5
            for (uint z=0; z < INPUT0_SIZE_Z; ++z) {
            #endif
                for (uint y=0; y < INPUT0_SIZE_Y; ++y) {
                    for (uint x=0; x < INPUT0_SIZE_X; ++x) {
                        #if INPUT0_DIMS == 5
                        const uint output_idx = OUTPUT_GET_INDEX(bb, ff, z, y, x);
                        #else
                        const uint output_idx = OUTPUT_GET_INDEX(bb, ff, y, x);
                        #endif

                        const ACCUMULATOR_TYPE res = output[output_idx/*cls++*/] / denominator;

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
//        }
//    }
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    printf("%d %d max=%f\n", bb, ff, max_value);

}
