// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "softmax_kernel_ref.h"
#include "kernel_selector_utils.h"

namespace kernel_selector {
ParamsKey SoftmaxKernelRef::GetSupportedKey() const {
    ParamsKey k = GetDefaultSupportedKey();

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

/*
    k.EnableAllInputLayout();
    k.EnableAllOutputLayout();
*/
    return k;
}

SoftmaxKernelRef::Parent::DispatchData SoftmaxKernelRef::SetDefault(const softmax_params& params,
                                                                    const optional_params& optParams) const {
    auto dispatchData = Parent::SetDefault(params, optParams);

    dispatchData.gws = GetSoftmaxDimGlobalSizes(params.dim, params.outputs[0]);

    assert(dispatchData.gws.size() == 3);

    dispatchData.lws = GetOptimalLocalWorkGroupSizes(dispatchData.gws, params.engineInfo);

    return dispatchData;
}

KernelsPriority SoftmaxKernelRef::GetKernelsPriority(const Params& /*params*/, const optional_params& /*options*/) const {
    return DONT_USE_IF_HAVE_SOMETHING_ELSE;
}

KernelsData SoftmaxKernelRef::GetKernelsData(const Params& params, const optional_params& options) const {
    return GetCommonKernelsData(params, options);
}
}  // namespace kernel_selector
