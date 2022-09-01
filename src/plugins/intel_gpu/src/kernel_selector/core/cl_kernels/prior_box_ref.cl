// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/data_types.cl"
#include "include/batch_headers/fetch_data.cl"

inline OUTPUT_TYPE FUNC(clip_great)(OUTPUT_TYPE x, OUTPUT_TYPE threshold) {
    return x < threshold ? x : threshold;
}

inline OUTPUT_TYPE FUNC(clip_less)(OUTPUT_TYPE x, OUTPUT_TYPE threshold) {
    return x > threshold ? x : threshold;
}

inline uint FUNC(get_index)(INPUT0_TYPE w, INPUT0_TYPE h) {
    return (w + h * PRIOR_BOX_WIDTH) * PRIOR_BOX_NUM_PRIORS_4;
}

inline void FUNC(calculate_data)(OUTPUT_TYPE center_x,
                                 OUTPUT_TYPE center_y,
                                 OUTPUT_TYPE box_width,
                                 OUTPUT_TYPE box_height,
                                 bool clip,
                                 uint output_index,
                                 __global OUTPUT_TYPE* dst_data) {
    uint idx = output_index;
    if (clip) {
        // order: xmin, ymin, xmax, ymax
        dst_data[idx] = FUNC_CALL(clip_less)((center_x - box_width) * PRIOR_BOX_IWI, 0);
        dst_data[idx + 1] = FUNC_CALL(clip_less)((center_y - box_height) * PRIOR_BOX_IHI, 0);
        dst_data[idx + 2] = FUNC_CALL(clip_great)((center_x + box_width) * PRIOR_BOX_IWI, 1);
        dst_data[idx + 3] = FUNC_CALL(clip_great)((center_y + box_height) * PRIOR_BOX_IHI, 1);
    } else {
        dst_data[idx] = (center_x - box_width) * PRIOR_BOX_IWI;
        dst_data[idx + 1] = (center_y - box_height) * PRIOR_BOX_IHI;
        dst_data[idx + 2] = (center_x + box_width) * PRIOR_BOX_IWI;
        dst_data[idx + 3] = (center_y + box_height) * PRIOR_BOX_IHI;

        #ifdef PRIOR_BOX_IS_CLUSTERED
            for(uint i = 0; i < 4; ++i) {
                dst_data[idx + i] *= 0.5;
            }
        #endif
    }
};

KERNEL(prior_box_ref)
(const __global INPUT0_TYPE* output_size, const __global INPUT1_TYPE* image_size, __global OUTPUT_TYPE* output) {
    const uint w = get_global_id(0);
    const uint h = get_global_id(1);
    const uint out_index = FUNC_CALL(get_index)(w, h);


    OUTPUT_TYPE center_x, center_y;
    #ifdef PRIOR_BOX_STEP
        center_x = (PRIOR_BOX_OFFSET + w) * PRIOR_BOX_STEP;
        center_y = (PRIOR_BOX_OFFSET + h) * PRIOR_BOX_STEP;
    #else
        center_x = (w + 0.5f) * PRIOR_BOX_STEP_X;
        center_y = (h + 0.5f) * PRIOR_BOX_STEP_Y;
    #endif
    OUTPUT_TYPE box_width, box_height;

    #ifdef PRIOR_BOX_IS_CLUSTERED
        uint num_priors = PRIOR_BOX_NUM_PRIORS_4 * 4;
        for (uint s = 0; s < num_priors; ++s) {
            box_width = PRIOR_BOX_WIDTHS[s];
            box_height = PRIOR_BOX_WIDTHS[s];
            FUNC_CALL(calculate_data)
            (center_x, center_y, box_width, box_height, true, out_index, output);
        }
    #endif

    for (uint s = 0; s < PRIOR_BOX_FIXED_SIZE_SIZE; ++s) {
        #if PRIOR_BOX_FIXED_SIZE_SIZE > 0
            OUTPUT_TYPE fixed_size_ = PRIOR_BOX_FIXED_SIZE[s];
        #else
            OUTPUT_TYPE fixed_size_ = 0;
        #endif

        box_height = box_width = fixed_size_ * 0.5f;

        #if PRIOR_BOX_FIXED_RATIO_SIZE > 0
            for (uint k = 0; k < PRIOR_BOX_ASPECT_RATIO_SIZE; ++k) {
                OUTPUT_TYPE ar = PRIOR_BOX_ASPECT_RATIO[k];
                uint density_ = PRIOR_BOX_DENSITY[s];
                uint shift = PRIOR_BOX_FIXED_SIZE[s] / density_;
                ar = sqrt(ar);
                OUTPUT_TYPE box_width_ratio = PRIOR_BOX_FIXED_SIZE[s] * 0.5f * ar;
                OUTPUT_TYPE box_height_ratio = PRIOR_BOX_FIXED_SIZE[s] * 0.5f / ar;
                for (uint r = 0; r < density_; ++r) {
                    for (uint c = 0; c < density_; ++c) {
                        OUTPUT_TYPE center_x_temp = center_x - fixed_size_ / 2 + shift / 2.f + c * shift;
                        OUTPUT_TYPE center_y_temp = center_y - fixed_size_ / 2 + shift / 2.f + r * shift;
                        FUNC_CALL(calculate_data)
                        (center_x_temp, center_y_temp, box_width_ratio, box_height_ratio, true, out_index, output);
                    }
                }
            }
        #else
            #if PRIOR_BOX_DENSITY_SIZE > 0
                uint density_ = PRIOR_BOX_DENSITY[s];
                uint shift = PRIOR_BOX_FIXED_SIZE[s] / density_;
                for (uint r = 0; r < density_; ++r) {
                    for (uint c = 0; c < density_; ++c) {
                        OUTPUT_TYPE center_x_temp = center_x - fixed_size_ / 2 + shift / 2.f + c * shift;
                        OUTPUT_TYPE center_y_temp = center_y - fixed_size_ / 2 + shift / 2.f + r * shift;
                        FUNC_CALL(calculate_data)
                        (center_x_temp, center_y_temp, box_width, box_height, true, out_index, output);
                    }
                }
            #endif
            //  Rest of priors
            for (uint k = 0; k < PRIOR_BOX_ASPECT_RATIO_SIZE; ++k) {
                OUTPUT_TYPE ar = PRIOR_BOX_ASPECT_RATIO[k];
                if (fabs(ar - 1.) < 1e-6) {
                    continue;
                }
                #if PRIOR_BOX_DENSITY_SIZE > 0 && PRIOR_BOX_FIXED_SIZE_SIZE > 0
                    uint density_ = PRIOR_BOX_DENSITY[s];
                    uint shift = PRIOR_BOX_FIXED_SIZE[s] / density_;
                    ar = sqrt(ar);
                    OUTPUT_TYPE box_width_ratio = PRIOR_BOX_FIXED_SIZE[s] * 0.5f * ar;
                    OUTPUT_TYPE box_height_ratio = PRIOR_BOX_FIXED_SIZE[s] * 0.5f / ar;
                    for (uint r = 0; r < density_; ++r) {
                        for (uint c = 0; c < density_; ++c) {
                            OUTPUT_TYPE center_x_temp = center_x - fixed_size_ / 2 + shift / 2.f + c * shift;
                            OUTPUT_TYPE center_y_temp = center_y - fixed_size_ / 2 + shift / 2.f + r * shift;
                            FUNC_CALL(calculate_data)
                            (center_x_temp, center_y_temp, box_width_ratio, box_height_ratio, true, out_index, output);
                        }
                    }
                #endif
            }
        #endif
    }

    for (uint ms_idx = 0; ms_idx < PRIOR_BOX_MIN_SIZE_SIZE; ++ms_idx) {
        box_width = PRIOR_BOX_MIN_SIZE[ms_idx] * 0.5f;
        box_height = PRIOR_BOX_MIN_SIZE[ms_idx] * 0.5f;
        FUNC_CALL(calculate_data)(center_x, center_y, box_width, box_height, false, out_index, output);
        #ifdef PRIOR_BOX_MIN_MAX_ASPECT_RATIO_ORDER
            if (PRIOR_BOX_MAX_SIZE_SIZE > ms_idx) {
                box_width = box_height = sqrt(PRIOR_BOX_MIN_SIZE[ms_idx] * PRIOR_BOX_MAX_SIZE[ms_idx]) * 0.5f;
                FUNC_CALL(calculate_data)(center_x, center_y, box_width, box_height, false, out_index + 4, output);
            }

            if (PRIOR_BOX_SCALE_ALL_SIZES || (!PRIOR_BOX_SCALE_ALL_SIZES && (ms_idx == PRIOR_BOX_MIN_SIZE_SIZE - 1))) {
                uint s_idx = PRIOR_BOX_SCALE_ALL_SIZES ? ms_idx : 0;
                for (uint k = 0; k < PRIOR_BOX_ASPECT_RATIO_SIZE; ++k) {
                    OUTPUT_TYPE ar = PRIOR_BOX_ASPECT_RATIO[k];
                    if (fabs(ar - 1.0f) < 1e-6) {
                        continue;
                    }

                    ar = sqrt(ar);
                    box_width = PRIOR_BOX_MIN_SIZE[s_idx] * 0.5f * ar;
                    box_height = PRIOR_BOX_MIN_SIZE[s_idx] * 0.5f / ar;
                    FUNC_CALL(calculate_data)(center_x, center_y, box_width, box_height, false, out_index + 8, output);
                }
            }
        #else
            if (PRIOR_BOX_SCALE_ALL_SIZES || (!PRIOR_BOX_SCALE_ALL_SIZES && (ms_idx == PRIOR_BOX_MIN_SIZE_SIZE - 1))) {
                uint s_idx = PRIOR_BOX_SCALE_ALL_SIZES ? ms_idx : 0;
                for (uint k = 0; k < PRIOR_BOX_ASPECT_RATIO_SIZE; ++k) {
                    OUTPUT_TYPE ar = PRIOR_BOX_ASPECT_RATIO[k];
                    if (fabs(ar - 1.0f) < 1e-6) {
                        continue;
                    };

                    ar = sqrt(ar);
                    box_width = PRIOR_BOX_MIN_SIZE[s_idx] * 0.5f * ar;
                    box_height = PRIOR_BOX_MIN_SIZE[s_idx] * 0.5f / ar;
                    FUNC_CALL(calculate_data)(center_x, center_y, box_width, box_height, false, out_index + 4, output);
                }
            }

            if (PRIOR_BOX_MAX_SIZE_SIZE > ms_idx) {
                box_width = box_height = sqrt(PRIOR_BOX_MIN_SIZE[ms_idx] * PRIOR_BOX_MAX_SIZE[ms_idx]) * 0.5f;
                FUNC_CALL(calculate_data)(center_x, center_y, box_width, box_height, false, out_index + 8, output);
            }
        #endif
    }

    #ifdef PRIOR_BOX_CLIP
        for (uint i = out_index; i < (out_index + PRIOR_BOX_NUM_PRIORS_4); ++i) {
            output[i] = (min)((max)(output[i], 0.0f), 1.0f);
        }
    #endif

    uint channel_size = OUTPUT_LENGTH / 2;
    #ifdef PRIOR_BOX_IS_CLUSTERED
        uint var_loop_count = PRIOR_BOX_VARIANCE_SIZE;
    #else
        uint var_loop_count = 4;
    #endif

    #if PRIOR_BOX_VARIANCE_SIZE == 1
        for (uint i = out_index; i < (out_index + PRIOR_BOX_NUM_PRIORS_4); ++i) {
            output[i + channel_size] = PRIOR_BOX_VARIANCE[0];
        }
    #else
        for (uint i = out_index / var_loop_count; i < (out_index + PRIOR_BOX_NUM_PRIORS_4) / var_loop_count; ++i) {
            for (uint j = 0; j < var_loop_count; j++) {
                output[i * (j + 1) + channel_size] = PRIOR_BOX_VARIANCE[j];
            }
        }
    #endif
}
