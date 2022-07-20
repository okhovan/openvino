// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "softmax_kernel_blocked_all_axis.h"
#include "kernel_selector_utils.h"

namespace kernel_selector {
ParamsKey SoftmaxKernelBlockedAllAxis::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::UINT8);

    // TODO remove plain layouts after debugging
/*
    k.EnableInputLayout(DataLayout::byxf);
    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableInputLayout(DataLayout::yxfb);
    k.EnableInputLayout(DataLayout::bf);
    k.EnableInputLayout(DataLayout::fb);
    k.EnableInputLayout(DataLayout::bfzyx);
    k.EnableInputLayout(DataLayout::f);
    k.EnableOutputLayout(DataLayout::f);
    k.EnableOutputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::byxf);
    k.EnableOutputLayout(DataLayout::yxfb);
    k.EnableOutputLayout(DataLayout::bf);
    k.EnableOutputLayout(DataLayout::fb);
    k.EnableOutputLayout(DataLayout::bfzyx);
*/

    k.EnableInputLayout(DataLayout::b_fs_yx_fsv16);
    k.EnableOutputLayout(DataLayout::b_fs_yx_fsv16);
    k.EnableInputLayout(DataLayout::b_fs_yx_fsv32);
    k.EnableOutputLayout(DataLayout::b_fs_yx_fsv32);
    k.EnableInputLayout(DataLayout::bs_fs_yx_bsv16_fsv16);
    k.EnableOutputLayout(DataLayout::bs_fs_yx_bsv16_fsv16);
    k.EnableInputLayout(DataLayout::bs_fs_yx_bsv32_fsv16);
    k.EnableOutputLayout(DataLayout::bs_fs_yx_bsv32_fsv16);
    k.EnableInputLayout(DataLayout::bs_fs_yx_bsv32_fsv32);
    k.EnableOutputLayout(DataLayout::bs_fs_yx_bsv32_fsv32);
    k.EnableInputLayout(DataLayout::b_fs_zyx_fsv16);
    k.EnableInputLayout(DataLayout::bs_fs_zyx_bsv16_fsv16);
    k.EnableOutputLayout(DataLayout::b_fs_zyx_fsv16);
    k.EnableOutputLayout(DataLayout::bs_fs_zyx_bsv16_fsv16);

    k.EnableSoftmaxDim(SoftmaxDim::ALL);
    k.EnableSoftmaxDim(SoftmaxDim::FYX);
    k.EnableDifferentTypes();
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();

    return k;
}

SoftmaxKernelBlockedAllAxis::Parent::DispatchData SoftmaxKernelBlockedAllAxis::SetDefault(const softmax_params& params,
                                                                                                const optional_params& optParams) const {
    auto dispatchData = Parent::SetDefault(params, optParams);
    //***const auto& out = params.outputs[0];

    switch (params.dim) {
        case SoftmaxDim::ALL:
            dispatchData.gws = {1, 1, 1};
            break;
        case SoftmaxDim::FYX:
            dispatchData.gws = {1, 1, 1};
            break;

/*
        case SoftmaxDim::X:
            dispatchData.gws = {out.Y().v * out.Z().v, out.Feature().v, out.Batch().v};
            break;
        case SoftmaxDim::Y:
            dispatchData.gws = {out.X().v * out.Z().v, out.Feature().v, out.Batch().v};
            break;
        case SoftmaxDim::Z:
            dispatchData.gws = {out.X().v * out.Y().v, out.Feature().v, out.Batch().v};
            break;
        case SoftmaxDim::FEATURE:
            dispatchData.gws = {out.X().v * out.Z().v, out.Y().v, out.Batch().v};
            break;
        case SoftmaxDim::BATCH:
            dispatchData.gws = {out.X().v * out.Z().v, out.Y().v, out.Feature().v};
            break;
*/
        default:
            dispatchData.gws = {1, 1, 1};
    }

    dispatchData.lws = {1, 1, 1};//GetOptimalLocalWorkGroupSizes(dispatchData.gws, params.engineInfo);

    return dispatchData;
}

KernelsPriority SoftmaxKernelBlockedAllAxis::GetKernelsPriority(const Params& /*params*/, const optional_params& /*options*/) const {
    // TODO bring back DONT_USE_IF_HAVE_SOMETHING_ELSE after debugging
    return /*FORCE_PRIORITY_9*/ DONT_USE_IF_HAVE_SOMETHING_ELSE;
}

KernelsData SoftmaxKernelBlockedAllAxis::GetKernelsData(const Params& params, const optional_params& options) const {
    return GetCommonKernelsData(params, options);
}
JitConstants SoftmaxKernelBlockedAllAxis::GetJitConstants(const softmax_params& params, DispatchData dispatchData) const {
    auto jit = SoftmaxKernelBase::GetJitConstants(params, dispatchData);
    jit.AddConstant(MakeJitConstant("SOFTMAX_DIM_" + toString(params.dim), "1"));

    std::vector<std::string> idx_order;
    const auto& in = params.inputs[0];
    const auto ndims = in.GetDims().size();
    const auto class_num = in.Feature().v * in.Batch().v * in.X().v * in.Y().v * (ndims == 5 ? in.Z().v : 1);
    jit.AddConstant(MakeJitConstant("CLASS_NUM", class_num));

    switch (params.dim) {
        case SoftmaxDim::ALL:
            idx_order = {"other3", "other1", ndims == 5 ? "other2" : "0", "other0", "cls"};
            break;
        case SoftmaxDim::FYX:
            idx_order = {"other3", "other1", ndims == 5 ? "other2" : "0", "cls", "other0"};
            break;
        default:
            break;
    }

    auto acc_dt = GetAccumulatorType(params);
    jit.Merge(MakeTypeJitConstants(acc_dt, "ACCUMULATOR"));

    if (!params.fused_ops.empty()) {
        FusedOpsConfiguration conf = {"",
                                      idx_order,
                                      "res",
                                      acc_dt};
        jit.Merge(MakeFusedOpsJitConstants(params, {conf}));
    }

    return jit;
}
}  // namespace kernel_selector
