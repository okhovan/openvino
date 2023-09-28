// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/plugin/common_utils.hpp"

#include "openvino/op/nms_rotated.hpp"

#include "intel_gpu/primitives/mutable_data.hpp"
#include "intel_gpu/primitives/nms_rotated.hpp"


namespace ov {
namespace intel_gpu {

static void CreateNMSRotatedOp(ProgramBuilder& p, const std::shared_ptr<ov::op::v13::NMSRotated>& op) {
    validate_inputs_count(op, {5});
    auto inputs = p.GetInputInfo(op);
    std::string layerName = layer_type_name_ID(op);

    const size_t num_outputs = op->get_output_size();
    assert(num_outputs == 3);

    if (p.use_new_shape_infer()) {
        auto z = 0;
        ++z;
    }

    std::vector<cldnn::memory::ptr> shared_memory;

    const auto boxes_shape = op->get_input_partial_shape(0);
    //const auto max_number_of_boxes = (boxesShape[0] * boxesShape[1]).get_length();
    const auto max_number_of_boxes = boxes_shape[0].get_length() * boxes_shape[1].get_length();

    const auto scores_precision = op->get_output_element_type(1);
    const cldnn::layout scores_layout{
        cldnn::element_type_to_data_type(scores_precision),
        cldnn::format::bfyx,
        cldnn::tensor(static_cast<int32_t>(max_number_of_boxes), 3, 1, 1)};
    shared_memory.emplace_back(p.get_engine().allocate_memory(scores_layout));
    const cldnn::primitive_id scores_id = layer_type_name_ID(op) + "_md_write_first";
    const cldnn::mutable_data scores_prim{scores_id, shared_memory.back()};
    p.add_primitive(*op, scores_prim);
    inputs.push_back(cldnn::input_info(scores_id));

    auto valid_output_precision = op->get_output_element_type(2);
    if (valid_output_precision == ov::element::i64) {
        valid_output_precision = ov::element::i32;
    }
    const cldnn::layout valid_output_layout{
        cldnn::element_type_to_data_type(valid_output_precision),
        cldnn::format::get_default_format(op->get_output_shape(2).size()),
        tensor_from_dims(op->get_output_shape(2))};
    shared_memory.emplace_back(p.get_engine().allocate_memory(valid_output_layout));
    const cldnn::primitive_id valid_output_id = layer_type_name_ID(op) + "_md_write_second";
    const cldnn::mutable_data valid_output_prim{valid_output_id, shared_memory.back()};
    p.add_primitive(*op, valid_output_prim);
    inputs.push_back(cldnn::input_info(valid_output_id));

    const auto prim = cldnn::nms_rotated(layerName,
                                         inputs,
                                         op->get_sort_result_descending(),
                                         cldnn::element_type_to_data_type(op->get_output_type_attr()),
                                         op->get_clockwise(),
                                         max_number_of_boxes,
                                         scores_id,
                                         valid_output_id);
    p.add_primitive(*op, prim);
}

REGISTER_FACTORY_IMPL(v13, NMSRotated);

}  // namespace intel_gpu
}  // namespace ov
