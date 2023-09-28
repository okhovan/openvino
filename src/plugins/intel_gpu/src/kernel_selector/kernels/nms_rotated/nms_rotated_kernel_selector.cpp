// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "nms_rotated_kernel_selector.h"
#include "nms_rotated_kernel_ref.h"

namespace kernel_selector {

nms_rotated_kernel_selector::nms_rotated_kernel_selector() { Attach<NMSRotatedKernelRef>(); }

KernelsData nms_rotated_kernel_selector::GetBestKernels(const Params& params, const optional_params& options) const {
    return GetNaiveBestKernel(params, options, KernelType::NMS_ROTATED);
}
}  // namespace kernel_selector
