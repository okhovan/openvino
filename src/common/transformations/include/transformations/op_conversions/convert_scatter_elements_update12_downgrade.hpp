// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/pass/graph_rewrite.hpp>
#include <transformations_visibility.hpp>

namespace ov {
namespace pass {
/**
 * @ingroup ie_transformation_common_api
 * @brief Converts Pad v12 to Pad v1
 */
class TRANSFORMATIONS_API ConvertScatterElementsUpdate12ToScatterElementsUpdate3 : public MatcherPass {
public:
    OPENVINO_RTTI("ConvertScatterElementsUpdate12ToScatterElementsUpdate3", "0");
    ConvertScatterElementsUpdate12ToScatterElementsUpdate3();
};

}  // namespace pass
}  // namespace ov
