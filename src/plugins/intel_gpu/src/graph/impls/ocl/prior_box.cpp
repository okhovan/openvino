// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <prior_box/prior_box_kernel_ref.h>
#include <prior_box/prior_box_kernel_selector.h>
#include <prior_box_inst.h>

#include <impls/implementation_map.hpp>
#include <vector>

#include "intel_gpu/runtime/error_handler.hpp"
#include "primitive_base.hpp"

namespace cldnn {
namespace ocl {

struct prior_box_impl : typed_primitive_impl_ocl<prior_box> {
    using parent = typed_primitive_impl_ocl<prior_box>;
    using parent::parent;

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<prior_box_impl>(*this);
    }

    static primitive_impl* create(const prior_box_node& arg, const kernel_impl_params& impl_param) {
        auto params = get_default_params<kernel_selector::prior_box_params>(impl_param);
        const auto& kernel_selector = kernel_selector::prior_box_kernel_selector::Instance();
        const auto& primitive = arg.get_primitive();
        const auto& attrs = primitive->attributes;

        params.min_size = attrs.min_sizes;
        params.max_size = attrs.max_sizes;
        params.density = attrs.densities;
        params.fixed_ratio = attrs.fixed_ratios;
        params.fixed_size = attrs.fixed_sizes;
        params.clip = attrs.clip;
        params.flip = attrs.flip;
        params.step = attrs.step;
        params.offset = attrs.offset;
        params.scale_all_sizes = attrs.scale_all_sizes;
        params.min_max_aspect_ratios_order = attrs.min_max_aspect_ratios_order;
        params.aspect_ratio = primitive->aspect_ratios;
        params.variance = primitive->variances;
        params.reverse_image_width = primitive->reverse_image_width;
        params.reverse_image_height = primitive->reverse_image_height;
        params.step_x = primitive->step_x;
        params.step_y = primitive->step_y;
        params.width = primitive->width;
        params.height = primitive->height;
        params.widths = attrs.widths;
        params.heights = attrs.heights;
        params.step_widths = attrs.step_width;
        params.step_heights = attrs.step_height;
        params.is_clustered = primitive->is_clustered();
        const auto output_shape = impl_param.output_layout.get_shape();
        params.num_priors_4 = output_shape[1] / (params.width * params.height);

        params.inputs.push_back(convert_data_tensor(impl_param.input_layouts[1]));
        const auto best_kernels = kernel_selector.GetBestKernels(params, kernel_selector::prior_box_optional_params());
        CLDNN_ERROR_BOOL(arg.id(),
                         "Best_kernel.empty()",
                         best_kernels.empty(),
                         "Cannot find a proper kernel with this arguments");
        return new prior_box_impl(arg, best_kernels[0]);
    }
};

namespace detail {

attach_prior_box_impl::attach_prior_box_impl() {
    auto types = {data_types::i32, data_types::i64};
    auto formats = {format::bfyx, format::bfzyx, format::bfwzyx};
    implementation_map<prior_box>::add(impl_types::ocl, prior_box_impl::create, types, formats);
}
}  // namespace detail

}  // namespace ocl
}  // namespace cldnn
