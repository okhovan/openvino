// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "primitive.hpp"
#include "intel_gpu/graph/serialization/string_serializer.hpp"

#include <vector>

namespace cldnn {

/// @brief Performs non max suppression of rotated input boxes and returns indices of selected boxes.
struct nms_rotated : public primitive_base<nms_rotated> {
    CLDNN_DECLARE_PRIMITIVE(nms_rotated)

    nms_rotated() : primitive_base("", {}) {}

    /// @brief Creates NMSRotated primitive.
    /// @param id This primitive id.
    /// @param boxes Id of input primitive with bounding boxes.
    /// @param scores Id of input primitive with boxes scores per class.
    /// @param max_output_boxes_per_class Id of input primitive specifying max number of boxes to be selected per class.
    /// @param iou_threshold Id of input primitive specifying threshold value for IOU.
    /// @param score_threshold Id of input primitive specifying minimum score for the box to be processed.
    /// @param sort_result_descending Specifies whether it is necessary to sort selected boxes across batches or not.
    /// @param output_data_type Output data type
    /// @param clockwise Specifies whether it is necessary to sort selected boxes across batches or not.
    /// @param selected_scores Id of primitive specifying output for scores for each selected box.
    /// @param valid_outputs Id of primitive specifying output for total number of selected boxes.
    nms_rotated(const primitive_id& id,
                const std::vector<input_info> inputs,
/*
                const input_info& boxes,
                const input_info& scores,
                const input_info& max_output_boxes_per_class,
                const input_info& iou_threshold,
                const input_info& score_threshold,
*/
                const bool sort_result_descending = true,
                const data_types output_data_type = data_types::i32,
                bool clockwise = true,
                const size_t max_number_of_boxes = 0,
                const primitive_id selected_scores = primitive_id{},
                const primitive_id valid_outputs = primitive_id{})
        : primitive_base{id, inputs, {padding{}}, {optional_data_type{}}, 3}
        , sort_result_descending(sort_result_descending)
        , output_data_type(output_data_type)
        , clockwise(clockwise)
        , max_number_of_boxes(max_number_of_boxes)
        , selected_scores(selected_scores)
        , valid_outputs(valid_outputs) {}

    bool sort_result_descending;
    data_types output_data_type;
    bool clockwise;
    size_t max_number_of_boxes;
    primitive_id selected_scores;
    primitive_id valid_outputs;

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, sort_result_descending);
        seed = hash_combine(seed, output_data_type);
        seed = hash_combine(seed, clockwise);
        seed = hash_combine(seed, selected_scores.empty());
        seed = hash_combine(seed, valid_outputs.empty());
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;

        const auto rhs_casted = downcast<const nms_rotated>(rhs);

        #define cmp_fields(name) name == rhs_casted.name
        return cmp_fields(sort_result_descending) &&
               cmp_fields(output_data_type) &&
               cmp_fields(clockwise);
        #undef cmp_fields
    }

    std::vector<std::reference_wrapper<const primitive_id>> get_dependencies() const override {
        std::vector<std::reference_wrapper<const primitive_id>> ret;
        ret.push_back(selected_scores);
        ret.push_back(valid_outputs);
        return ret;
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<nms_rotated>::save(ob);
        ob << sort_result_descending;
        ob << make_data(&output_data_type, sizeof(data_types));
        ob << clockwise;
        ob << selected_scores;
        ob << valid_outputs;
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<nms_rotated>::load(ib);
        ib >> sort_result_descending;
        ib >> make_data(&output_data_type, sizeof(data_types));
        ib >> clockwise;
        ib >> selected_scores;
        ib >> valid_outputs;
    }
};
}  // namespace cldnn
