// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_selector.h"

namespace kernel_selector {
class nms_rotated_kernel_selector : public kernel_selector_base {
public:
    static nms_rotated_kernel_selector& Instance() {
        static nms_rotated_kernel_selector instance_;
        return instance_;
    }

    nms_rotated_kernel_selector();

    KernelsData GetBestKernels(const Params& params, const optional_params& options) const override;
};
}  // namespace kernel_selector
