// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_base_opencl.h"
#include "kernel_selector_params.h"

namespace kernel_selector {

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// nms_rotated_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct nms_rotated_params : public base_params {
    nms_rotated_params() : base_params(KernelType::NMS_ROTATED)
    /*,
    box_encoding(BoxEncodingType::BOX_ENCODING_CORNER), sort_result_descending(true),
    num_select_per_class_type(base_params::ArgType::Constant), num_select_per_class(0),
    iou_threshold_type(base_params::ArgType::Constant), iou_threshold(0.0f),
    score_threshold_type(base_params::ArgType::Constant), score_threshold(0.0f),
    soft_nms_sigma_type(base_params::ArgType::Constant), soft_nms_sigma(0.0f),
    has_second_output(false), has_third_output(false),
    use_multiple_outputs(false)
     */
    {}

    bool sort_result_descending{true};
    Datatype output_data_type{Datatype::INT32};
    bool clockwise{true};
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// nms_rotated_optional_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct nms_rotated_optional_params : optional_params {
    nms_rotated_optional_params() : optional_params(KernelType::NMS_ROTATED) {}
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// NMSRotatedKernelRef
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class NMSRotatedKernelRef : public KernelBaseOpenCL {
public:
    NMSRotatedKernelRef() : KernelBaseOpenCL("nms_rotated_gpu_ref") {}

    using DispatchData = CommonDispatchData;
    KernelsData GetKernelsData(const Params& params, const optional_params& options) const override;
    KernelsPriority GetKernelsPriority(const Params& params, const optional_params& options) const override;
    ParamsKey GetSupportedKey() const override;

protected:
    Datatype GetAccumulatorType(const nms_rotated_params& params) const;
    virtual JitConstants GetJitConstants(const nms_rotated_params& params) const;
    bool Validate(const Params& p, const optional_params& o) const override;
    void SetKernelArguments(const nms_rotated_params& params, clKernelData& kernel, size_t idx) const;
};

}  // namespace kernel_selector
