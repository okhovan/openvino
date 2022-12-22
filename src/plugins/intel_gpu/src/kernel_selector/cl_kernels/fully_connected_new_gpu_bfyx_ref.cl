// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/data_types.cl"
#include "include/batch_headers/fetch_data.cl"
#include "include/batch_headers/fetch_weights.cl"

inline uint FUNC(get_input_index)(uint b, uint f, uint w, uint z, uint y, uint x)
{
#if INPUT_RANK == 2
    return INPUT0_GET_INDEX(y, x, 0, 0);
#elif INPUT_RANK == 3
    #define INPUT_BATCH1_NUM INPUT0_BATCH_NUM
    return INPUT0_GET_INDEX(z % INPUT_BATCH1_NUM, y, x, 0);
#elif INPUT_RANK == 4
    #define INPUT_BATCH1_NUM INPUT0_BATCH_NUM
    #define INPUT_BATCH2_NUM INPUT0_FEATURE_NUM
    return INPUT0_GET_INDEX(w % INPUT_BATCH1_NUM, z % INPUT_BATCH2_NUM, y, x);
#elif INPUT_RANK == 5
    #define INPUT_BATCH1_NUM INPUT0_BATCH_NUM
    #define INPUT_BATCH2_NUM INPUT0_FEATURE_NUM
    #define INPUT_BATCH3_NUM INPUT0_SIZE_Z
    return INPUT0_GET_INDEX(f % INPUT_BATCH1_NUM, w % INPUT_BATCH2_NUM, z % INPUT_BATCH3_NUM, y, x);
#elif INPUT_RANK == 6
    #define INPUT_BATCH1_NUM INPUT0_BATCH_NUM
    #define INPUT_BATCH2_NUM INPUT0_FEATURE_NUM
    #define INPUT_BATCH3_NUM INPUT0_SIZE_W
    #define INPUT_BATCH4_NUM INPUT0_SIZE_Z
    return INPUT0_GET_INDEX(b % INPUT_BATCH1_NUM, f % INPUT_BATCH2_NUM, w % INPUT_BATCH3_NUM, z % INPUT_BATCH4_NUM, y, x);
#else
#error Invalid input rank
#endif
}

inline uint FUNC(get_weight_index)(uint b, uint f, uint w, uint z, uint y, uint x)
{
#if WEIGHT_RANK == 2
    return GET_FILTER_INDEX(FILTER, 0, y, x, 0, 0);
#elif WEIGHT_RANK == 3
    #define WEIGHT_BATCH1_NUM FILTER_IFM_NUM
    return GET_FILTER_INDEX(FILTER, 0, z % WEIGHT_BATCH1_NUM, y, x, 0);
#elif WEIGHT_RANK == 4
    #define WEIGHT_BATCH1_NUM FILTER_SIZE_Z
    #define WEIGHT_BATCH2_NUM FILTER_IFM_NUM
    return GET_FILTER_INDEX(FILTER, 0, w % WEIGHT_BATCH1_NUM, z % WEIGHT_BATCH2_NUM, y, x);
#elif WEIGHT_RANK == 5
    #define WEIGHT_BATCH1_NUM FILTER_IFM_NUM
    #define WEIGHT_BATCH2_NUM FILTER_OFM_NUM
    #define WEIGHT_BATCH3_NUM FILTER_SIZE_Z
    return GET_FILTER_INDEX_5D(FILTER, 0, f % WEIGHT_BATCH1_NUM, w % WEIGHT_BATCH2_NUM, z % WEIGHT_BATCH3_NUM, y, x);
#elif WEIGHT_RANK == 6
    #define WEIGHT_BATCH1_NUM FILTER_IFM_NUM
    #define WEIGHT_BATCH2_NUM FILTER_OFM_NUM
    #define WEIGHT_BATCH3_NUM FILTER_GROUPS_NUM
    #define WEIGHT_BATCH4_NUM FILTER_SIZE_Z
    return GET_FILTER_INDEX_5D(FILTER, b % WEIGHT_BATCH1_NUM, f % WEIGHT_BATCH2_NUM, w % WEIGHT_BATCH3_NUM, z % WEIGHT_BATCH4_NUM, y, x);
#else
#error Invalid weight rank
#endif
}

KERNEL(fc)(
    const __global INPUT0_TYPE* input,
    __global OUTPUT_TYPE* output,
    const __global FILTER_TYPE* weights
#if BIAS_TERM
    , const __global BIAS_TYPE* biases
#endif
    )
{
#if INPUT_RANK == 2
    #define INPUT_COL_NUMBER INPUT0_FEATURE_NUM
#elif INPUT_RANK == 3
    #define INPUT_COL_NUMBER INPUT0_SIZE_Y
#elif INPUT_RANK == 4
    #define INPUT_COL_NUMBER INPUT0_SIZE_X
#elif INPUT_RANK == 5
    #define INPUT_COL_NUMBER INPUT0_SIZE_X
#elif INPUT_RANK == 6
    #define INPUT_COL_NUMBER INPUT0_SIZE_X
#else
#error Invalid input rank
#endif

#if OUTPUT_RANK == 2
    const uint out_batch1 = 0;
    const uint out_batch2 = 0;
    const uint out_batch3 = 0;
    const uint out_batch4 = 0;
    const uint out_y = get_global_id(0);
    const uint out_x = get_global_id(1);
    const uint output_idx = OUTPUT_GET_INDEX(out_y, out_x, 0, 0);
#elif OUTPUT_RANK == 3
    const uint out_batch1 = 0;
    const uint out_batch2 = 0;
    const uint out_batch3 = 0;
    const uint out_batch4 = get_global_id(0);
    const uint out_y = get_global_id(1);
    const uint out_x = get_global_id(2);
    const uint output_idx = OUTPUT_GET_INDEX(out_batch4, out_y, out_x, 0);
#elif OUTPUT_RANK == 4
    const uint out_batch1 = 0;
    const uint out_batch2 = 0;
    const uint batch = get_global_id(0);
    const uint out_batch3 = batch / OUTPUT_FEATURE_NUM;
    const uint out_batch4 = batch % OUTPUT_FEATURE_NUM;
    const uint out_y = get_global_id(1);
    const uint out_x = get_global_id(2);
    const uint output_idx = OUTPUT_GET_INDEX(out_batch3, out_batch4, out_y, out_x);
#elif OUTPUT_RANK == 5
    const uint out_batch1 = 0;
    const uint batch = get_global_id(0);
    const uint out_batch2 = batch / OUTPUT_FEATURE_NUM;
    const uint out_batch3 = batch % OUTPUT_FEATURE_NUM;
    const uint batch_y = get_global_id(1);
    const uint out_batch4 = batch_y / OUTPUT_SIZE_Y;
    const uint out_y = batch_y % OUTPUT_SIZE_Y;
    const uint out_x = get_global_id(2);
    const uint output_idx = OUTPUT_GET_INDEX(out_batch2, out_batch3, out_batch4, out_y, out_x);
#elif OUTPUT_RANK == 6
    const uint batch12 = get_global_id(0);
    const uint out_batch1 = batch12 / OUTPUT_FEATURE_NUM;
    const uint out_batch2 = batch12 % OUTPUT_FEATURE_NUM;
    const uint batch34 = get_global_id(1);
    const uint out_batch3 = batch34 / OUTPUT_SIZE_Z;
    const uint out_batch4 = batch34 % OUTPUT_SIZE_Z;
    const uint yx = get_global_id(2);
    const uint out_y = yx % OUTPUT_SIZE_Y;
    const uint out_x = yx / OUTPUT_SIZE_Y;
    const uint output_idx = OUTPUT_GET_INDEX(out_batch1, out_batch2, out_batch3, out_batch4, out_y, out_x);
#else
#error Invalid output rank
#endif

    ACCUMULATOR_TYPE dotProd = ACCUMULATOR_VAL_ZERO;

    for (uint in_x = 0; in_x < INPUT_COL_NUMBER; ++in_x) {
        const uint input_idx = FUNC_CALL(get_input_index)(out_batch1, out_batch2, out_batch3, out_batch4, out_y, in_x);
        const uint weights_idx = FUNC_CALL(get_weight_index)(out_batch1, out_batch2, out_batch3, out_batch4, out_x, in_x);
        dotProd += input[input_idx] * weights[weights_idx];
    }

    output[output_idx] = TO_OUTPUT_TYPE(dotProd);

#if BIAS_TERM
    #if OUTPUT_RANK == 2
        const uint bias_index = GET_DATA_INDEX(BIAS, 0, out_x, 0, 0);
    #elif OUTPUT_RANK == 3
        const uint bias_index = GET_DATA_INDEX(BIAS, out_batch, out_y, 0, 0);
    #elif OUTPUT_RANK == 4
        const uint bias_index = GET_DATA_INDEX(BIAS, out_batch1, out_batch2, out_y, 0);
    #elif OUTPUT_RANK == 5
        const uint bias_index = GET_DATA_INDEX(BIAS, out_batch1, out_batch2, out_batch3, out_y);
    #elif OUTPUT_RANK == 6
        const uint bias_index = GET_DATA_INDEX(BIAS, out_batch1, out_batch2, out_batch3, out_batch4, out_y, out_x, 0, 0);
    #else
    #error Invalid rank
    #endif

    const ACTIVATION_TYPE dequantized = TO_ACTIVATION_TYPE(dotProd) + biases[bias_index];

#else
    const ACTIVATION_TYPE dequantized = TO_ACTIVATION_TYPE(dotProd);
#endif
    output[output_idx] = TO_OUTPUT_TYPE(dequantized);
}
