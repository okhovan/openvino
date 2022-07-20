// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "softmax_kernel_base.h"

namespace kernel_selector {
class SoftmaxKernelBlockedSingleAxis : public SoftmaxKernelBase {
public:
    using Parent = SoftmaxKernelBase;
    SoftmaxKernelBlockedSingleAxis() : Parent("softmax_gpu_blocked_single_axis") {}
    virtual ~SoftmaxKernelBlockedSingleAxis() {}

    KernelsData GetKernelsData(const Params& params, const optional_params& options) const override;
    KernelsPriority GetKernelsPriority(const Params& params, const optional_params& options) const override;
    ParamsKey GetSupportedKey() const override;

protected:
    JitConstants GetJitConstants(const softmax_params& params, DispatchData dispatchData) const override;
    DispatchData SetDefault(const softmax_params& params, const optional_params& optParams) const override;
    Datatype GetAccumulatorType(const softmax_params& params) const {
        if (params.inputs[0].GetDType() == Datatype::F16)
            return Datatype::F16;
        else
            return Datatype::F32;
    }
};
}  // namespace kernel_selector