// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/program.hpp"
#include "intel_gpu/plugin/common_utils.hpp"

#include "ngraph/op/generate_proposals.hpp"

#include "intel_gpu/primitives/mutable_data.hpp"
#include "intel_gpu/primitives/generate_proposals.hpp"

namespace ov {
namespace runtime {
namespace intel_gpu {

static void CreateGenerateProposalsOp(Program& p, const std::shared_ptr<ngraph::op::v9::GenerateProposals>& op) {
    p.ValidateInputs(op, {4});
    if (op->get_output_size() != 3) {
        IE_THROW() << "GenerateProposals requires 3 outputs";
    }

    auto inputs = p.GetInputPrimitiveIDs(op);
    const auto& attrs = op->get_attrs();
    const auto op_friendly_name = op->get_friendly_name();
    const auto layer_type_name = layer_type_name_ID(op);
    const auto layer_name = layer_type_name + ".0";

    // output 2 - scores
    const auto mutable_precision_1 = op->get_output_element_type(1);
    const auto output_shape_1 = op->get_output_shape(1);
    const cldnn::layout mutable_layout_1{DataTypeFromPrecision(mutable_precision_1),
                                         DefaultFormatForDims(output_shape_1.size()),
                                         tensor_from_dims(output_shape_1)};
    cldnn::memory::ptr shared_memory_1{p.GetEngine().allocate_memory(mutable_layout_1)};

    const auto mutable_id_w_1 = layer_type_name + "_md_write_1";
    const cldnn::mutable_data mutable_prim_w_1{mutable_id_w_1, shared_memory_1, op_friendly_name};
    p.primitiveIDs[mutable_id_w_1] = mutable_id_w_1;
    p.AddPrimitive(mutable_prim_w_1);
    inputs.push_back(mutable_id_w_1);

    // output 3 - roisNum
    const auto mutable_precision_2 = op->get_output_element_type(2);
    const auto output_shape_2 = op->get_output_shape(2);
    const cldnn::layout mutable_layout_2{DataTypeFromPrecision(mutable_precision_2),
                                         DefaultFormatForDims(output_shape_2.size()),
                                         tensor_from_dims(output_shape_2)};
    cldnn::memory::ptr shared_memory_2{p.GetEngine().allocate_memory(mutable_layout_2)};

    const auto mutable_id_w_2 = layer_type_name + "_md_write_2";
    const cldnn::mutable_data mutable_prim_w_2{mutable_id_w_2, shared_memory_2, op_friendly_name};
    p.primitiveIDs[mutable_id_w_2] = mutable_id_w_2;
    p.AddPrimitive(mutable_prim_w_2);
    inputs.push_back(mutable_id_w_2);

    const cldnn::generate_proposals prim{layer_name,
                                         inputs[0], inputs[1], inputs[2], inputs[3],
                                         mutable_id_w_1, mutable_id_w_2,
                                         attrs.min_size, attrs.nms_threshold, attrs.pre_nms_count, attrs.post_nms_count,
                                         attrs.normalized, attrs.nms_eta,
                                         DataTypeFromPrecision(op->get_roi_num_type()),
                                         op_friendly_name};

    p.AddPrimitive(prim);

    const auto mutable_id_r_1 = layer_type_name + ".1";
    const cldnn::mutable_data mutable_prim_r_1{mutable_id_r_1, {layer_name}, shared_memory_1, op_friendly_name};
    p.primitiveIDs[mutable_id_r_1] = mutable_id_r_1;
    p.AddPrimitive(mutable_prim_r_1);

    const auto mutable_id_r_2 = layer_type_name + ".2";
    const cldnn::mutable_data mutable_prim_r_2{mutable_id_r_2, {layer_name}, shared_memory_2, op_friendly_name};
    p.primitiveIDs[mutable_id_r_2] = mutable_id_r_2;
    p.AddPrimitive(mutable_prim_r_2);

    p.AddPrimitiveToProfiler(prim, op);
}

REGISTER_FACTORY_IMPL(v9, GenerateProposals);

}  // namespace intel_gpu
}  // namespace runtime
}  // namespace ov
