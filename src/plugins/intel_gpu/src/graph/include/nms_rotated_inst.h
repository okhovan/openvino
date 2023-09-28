// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "intel_gpu/primitives/nms_rotated.hpp"
#include "primitive_inst.h"

#include <memory>
#include <string>

namespace cldnn {

template <>
struct typed_program_node<nms_rotated> : public typed_program_node_base<nms_rotated> {
    using parent = typed_program_node_base<nms_rotated>;

public:
    typed_program_node(std::shared_ptr<primitive> prim, program& prog)
        : parent(prim, prog)
    {}

    program_node& input() const { return get_dependency(0); }

    program_node& input_boxes() const {
        return get_dependency(0);
    }

    program_node& input_scores() const {
        return get_dependency(1);
    }

    program_node& max_output_boxes_per_class_node() const {
        return get_dependency(2);
    }

    program_node& iou_threshold_node() const {
        return get_dependency(3);
    }

    program_node& score_threshold_node() const {
        return get_dependency(4);
    }

    program_node& selected_scores_node() const {
        return get_dependency(5);
    }

    program_node& valid_outputs_node() const {
        return get_dependency(6);
    }
};

using nms_rotated_node = typed_program_node<nms_rotated>;

template <>
class typed_primitive_inst<nms_rotated> : public typed_primitive_inst_base<nms_rotated> {
    using parent = typed_primitive_inst_base<nms_rotated>;
    using parent::parent;

public:
    typed_primitive_inst(network& network, nms_rotated_node const& node)
        : parent(network, node)
    {}

//    template<typename ShapeType>
//    static std::vector<layout> calc_output_layouts(nms_rotated_node const& /*node*/, const kernel_impl_params& impl_param);
    static layout calc_output_layout(nms_rotated_node const& node, kernel_impl_params const& impl_param);
    static std::string to_string(nms_rotated_node const& node);

    memory::ptr input_boxes_mem() const {
        return dep_memory_ptr(0);
    }

    memory::ptr input_scores_mem() const {
        return dep_memory_ptr(1);
    }

    memory::ptr max_output_boxes_per_class_mem() const {
        return dep_memory_ptr(2);
    }
    std::shared_ptr<const primitive_inst> max_output_boxes_per_class_inst() const {
        return dependencies().at(2).first;
    }

    memory::ptr iou_threshold_mem() const {
        return dep_memory_ptr(3);
    }
    std::shared_ptr<const primitive_inst> iou_threshold_inst() const {
        return dependencies().at(3).first;
    }

    memory::ptr score_threshold_mem() const {
        return dep_memory_ptr(4);
    }
    std::shared_ptr<const primitive_inst> score_threshold_inst() const {
        return dependencies().at(4).first;
    }

    memory::ptr selected_scores_mem() const {
        return dep_memory_ptr(5);
    }

    memory::ptr valid_outputs_mem() const {
        return dep_memory_ptr(6);
    }
};

using nms_rotated_inst = typed_primitive_inst<nms_rotated>;

}  // namespace cldnn
