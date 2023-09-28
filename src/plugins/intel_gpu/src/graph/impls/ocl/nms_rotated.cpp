// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "primitive_base.hpp"

#include "nms_rotated_inst.h"
#include "data_inst.h"
#include "nms_rotated/nms_rotated_kernel_ref.h"
#include "nms_rotated/nms_rotated_kernel_selector.h"

namespace cldnn {
namespace ocl {
struct nms_rotated_impl : typed_primitive_impl_ocl<nms_rotated> {
    using parent = typed_primitive_impl_ocl<nms_rotated>;
    using parent::parent;
    using kernel_selector_t = kernel_selector::nms_rotated_kernel_selector;
    using kernel_params_t = std::pair<kernel_selector::nms_rotated_params, kernel_selector::nms_rotated_optional_params>;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::ocl::nms_rotated_impl)

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<nms_rotated_impl>(*this);
    }

protected:
    kernel_arguments_data get_arguments(const typed_primitive_inst<nms_rotated>& instance) const override {
        kernel_arguments_data args;
        for (size_t i = 0; i < instance.inputs_memory_count(); i++) {
            args.inputs.push_back(instance.input_memory_ptr(i));
        }

        for (size_t i = 0; i < instance.outputs_memory_count(); i++) {
            args.outputs.push_back(instance.output_memory_ptr(i));
        }
        // Legacy APIs using mutable inputs for multiple outputs
        args.inputs.push_back(instance.selected_scores_mem());
        args.inputs.push_back(instance.valid_outputs_mem());

        return args;
    }

public:
    static std::unique_ptr<primitive_impl> create(const nms_rotated_node& arg, const kernel_impl_params& impl_param) {
        const auto& primitive = impl_param.typed_desc<nms_rotated>();
        auto params = get_default_params<kernel_selector::nms_rotated_params>(impl_param);
        auto optional_params =
            get_default_optional_params<kernel_selector::nms_rotated_optional_params>(impl_param.get_program());

        for (size_t i = 1; i < impl_param.input_layouts.size(); i++) {
            params.inputs.push_back(convert_data_tensor(impl_param.get_input_layout(i)));
        }

        params.inputs.push_back(convert_data_tensor(impl_param.input_layouts[4]));
//???            params.has_second_output = true;
        params.inputs.push_back(convert_data_tensor(impl_param.input_layouts[5]));


/*
        if (arg.use_multiple_outputs()) {
*/
            params.outputs.push_back(convert_data_tensor(impl_param.output_layouts[1]));
            params.outputs.push_back(convert_data_tensor(impl_param.output_layouts[2]));
/*
            params.use_multiple_outputs = true;
        }
*/

        params.sort_result_descending = primitive->sort_result_descending;
        params.output_data_type = primitive->output_data_type == cldnn::data_types::i32 ?
                kernel_selector::Datatype::INT32 : kernel_selector::Datatype::INT64;
        params.clockwise = primitive->clockwise;

        if (impl_param.get_program().get_node(primitive->id).is_dynamic()) {
            int z = 0;
            ++z;
//            params.reuse_internal_buffer = true;
        }

        params.set_dynamic_shape_offsets();

        auto& kernel_selector = kernel_selector::nms_rotated_kernel_selector::Instance();
        auto best_kernel = kernel_selector.get_best_kernel(params, optional_params);

        return make_unique<nms_rotated_impl>(best_kernel);
    }

private:
    template <class T>
    static T get_value(cldnn::program_node& node) {
        T retValue;
        auto mem = node.as<data>().get_attached_memory_ptr();
        auto& stream = node.get_program().get_stream();
        switch (mem->get_layout().data_type) {
        case data_types::f16: {
            mem_lock<half_t, mem_lock_type::read> lock(mem, stream);
            auto mem_value = static_cast<half_t*>(lock.data());
            retValue = static_cast<T>(*mem_value);
        } break;
        case data_types::f32: {
            mem_lock<float, mem_lock_type::read> lock(mem, stream);
            auto mem_value = static_cast<float*>(lock.data());
            retValue = static_cast<T>(*mem_value);
        } break;
        case data_types::i32: {
            mem_lock<int32_t, mem_lock_type::read> lock(mem, stream);
            auto mem_value = static_cast<int32_t*>(lock.data());
            retValue = static_cast<T>(*mem_value);
        } break;
        case data_types::i64: {
            mem_lock<int64_t, mem_lock_type::read> lock(mem, stream);
            auto mem_value = static_cast<int64_t*>(lock.data());
            retValue = static_cast<T>(*mem_value);
        } break;
        default:
            throw std::runtime_error("Not supported data type.");
        }

        return retValue;
    }
};

namespace detail {

attach_nms_rotated_impl::attach_nms_rotated_impl() {
    implementation_map<nms_rotated>::add(impl_types::ocl,
                                                 shape_types::static_shape,
                                                 nms_rotated_impl::create,
                                                 {
                                                     std::make_tuple(data_types::i32, format::bfyx),

                                                     std::make_tuple(data_types::f16, format::bfyx),
                                                     std::make_tuple(data_types::f16, format::b_fs_yx_fsv16),
                                                     std::make_tuple(data_types::f16, format::b_fs_yx_fsv32),
                                                     std::make_tuple(data_types::f16, format::bs_fs_yx_bsv16_fsv16),
                                                     std::make_tuple(data_types::f16, format::bs_fs_yx_bsv32_fsv16),
                                                     std::make_tuple(data_types::f16, format::bs_fs_yx_bsv32_fsv32),

                                                     std::make_tuple(data_types::f32, format::bfyx),
                                                     std::make_tuple(data_types::f32, format::b_fs_yx_fsv16),
                                                     std::make_tuple(data_types::f32, format::b_fs_yx_fsv32),
                                                     std::make_tuple(data_types::f32, format::bs_fs_yx_bsv16_fsv16),
                                                     std::make_tuple(data_types::f32, format::bs_fs_yx_bsv32_fsv16),
                                                     std::make_tuple(data_types::f32, format::bs_fs_yx_bsv32_fsv32),
                                                 });
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::nms_rotated_impl)
