// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "softmax_inst.h"
#include "primitive_base.hpp"
#include "impls/implementation_map.hpp"
#include "kernel_selector_helper.h"
#include "softmax/softmax_kernel_selector.h"
#include "softmax/softmax_kernel_base.h"
#include "intel_gpu/runtime/error_handler.hpp"

namespace cldnn {
namespace ocl {

struct softmax_impl : typed_primitive_impl_ocl<softmax> {
    using parent = typed_primitive_impl_ocl<softmax>;
    using parent::parent;

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<softmax_impl>(*this);
    }

    static primitive_impl* create(const softmax_node& arg) {
        auto sm_params = get_default_params<kernel_selector::softmax_params>(arg);
        auto sm_optional_params =
            get_default_optional_params<kernel_selector::softmax_optional_params>(arg.get_program());

        auto& input = sm_params.inputs[0];
        auto& output = sm_params.outputs[0];
        const auto primitive = arg.get_primitive();
        const auto is_blocked_format = arg.input().get_output_layout().format.is_blocked();

        switch (primitive->dimension) {
            case softmax::normalize_x:
                sm_params.dim = kernel_selector::softmax_dim::X;
                break;

            case softmax::normalize_y:
                sm_params.dim = kernel_selector::softmax_dim::Y;
                break;

            case softmax::normalize_fyx:
                if (is_blocked_format) {
                    sm_params.dim = kernel_selector::softmax_dim::FYX;
                } else {
                    // Flatten fused with softmax
                    input = input.FlattenFeatureAndSpatials();
                    output = output.FlattenFeatureAndSpatials();

                    sm_params.dim = kernel_selector::softmax_dim::FEATURE;
                }
                break;

            case softmax::normalize_b:
                sm_params.dim = kernel_selector::softmax_dim::BATCH;
                break;

            case softmax::normalize_f:
                sm_params.dim = kernel_selector::softmax_dim::FEATURE;
                break;

            case softmax::normalize_z:
                sm_params.dim = kernel_selector::softmax_dim::Z;
                break;

            case softmax::normalize_all:
                if (is_blocked_format) {
                    sm_params.dim = kernel_selector::softmax_dim::ALL;
                } else {
                    input = input.FlattenEverything();
                    output = output.FlattenEverything();

                    sm_params.dim = kernel_selector::softmax_dim::FEATURE;
                }
                break;

            default:
                throw std::runtime_error("Wrong API - no such softmax");
        }

        auto& kernel_selector = kernel_selector::softmax_kernel_selector::Instance();
        auto best_kernels = kernel_selector.GetBestKernels(sm_params, sm_optional_params);

        CLDNN_ERROR_BOOL(arg.id(),
                         "Best_kernel.empty()",
                         best_kernels.empty(),
                         "Cannot find a proper kernel with this arguments");

        auto softmax_node = new softmax_impl(arg, best_kernels[0]);

        return softmax_node;
    }
};

namespace detail {

attach_softmax_impl::attach_softmax_impl() {
    const auto types = {data_types::f16, data_types::f32, data_types::i32};
    const auto formats = {
            format::bfyx,
            format::yxfb,
            format::b_fs_yx_fsv16,
            format::b_fs_yx_fsv32,
            format::bs_fs_yx_bsv16_fsv16,
            format::bs_fs_yx_bsv32_fsv16,
            format::bs_fs_yx_bsv32_fsv32,
            format::bfzyx,
            format::b_fs_zyx_fsv16,
            format::b_fs_zyx_fsv32,
            format::bs_fs_zyx_bsv16_fsv32,
            format::bs_fs_zyx_bsv16_fsv16,
            format::bs_fs_zyx_bsv32_fsv32,
            format::bs_fs_zyx_bsv32_fsv16
    };

    std::set<std::tuple<data_types, format::type>> keys;
    for (const auto& t : types) {
        for (const auto& f : formats) {
            keys.emplace(t, f);
        }
    }
    implementation_map<softmax>::add(impl_types::ocl, softmax_impl::create, keys);
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn
