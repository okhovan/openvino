// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/scatter_elements_update.hpp>
#include <intel_gpu/runtime/memory.hpp>
#include <intel_gpu/graph/topology.hpp>
#include <intel_gpu/graph/network.hpp>

using namespace cldnn;
using namespace ::tests;

namespace {
template<typename T>
struct ScatterElementsUpdateParams {
    int64_t axis;
    tensor data_tensor;
    std::vector<T> data;
    tensor indices_tensor;
    std::vector<T> indices;
    std::vector<T> updates;
    std::vector<T> expected;
};

template<typename T>
using ScatterElementsUpdateParamsWithFormat = std::tuple<
    ScatterElementsUpdateParams<T>,
    format::type,     // source (plain) layout
    format::type      // target (blocked) layout
>;

const std::vector<format::type> formats2D{
        format::bfyx,
        format::b_fs_yx_fsv16,
        format::b_fs_yx_fsv32,
        format::bs_fs_yx_bsv16_fsv16,
        format::bs_fs_yx_bsv32_fsv16,
        format::bs_fs_yx_bsv32_fsv32
};

const std::vector<format::type> formats3D{
        format::bfzyx,
        format::b_fs_zyx_fsv16,
        format::bs_fs_zyx_bsv16_fsv16
};

const std::vector<format::type> formats4D{
        format::bfwzyx
};

template<typename T>
std::vector<T> getValues(const std::vector<float> &values) {
    std::vector<T> result(values.begin(), values.end());
    return result;
}

template<typename T>
std::vector<ScatterElementsUpdateParams<T>> generateScatterElementsUpdateParams2D() {
    const std::vector<ScatterElementsUpdateParams<T>> result = {
        {   1,
            tensor{2, 4, 1, 1},
            getValues<T>({ 0, 1, 2, 3, 4, 5, 6, 7 }),
            tensor{2, 2, 1, 1},
            getValues<T>({ 0, 1, 2, 3 }),
            getValues<T>({ -10, -11, -12, -13 }),
            getValues<T>({ -10, -11, 2, 3, 4, 5, -12, -13 })
        },
        {   2,
            tensor{2, 1, 2, 2},
            getValues<T>({ 0, 1, 2, 3, 4, 5, 6, 7 }),
            tensor{2, 1, 2, 1},
            getValues<T>({ 0, 1, 0, 1 }),
            getValues<T>({ -10, -11, -12, -13 }),
            getValues<T>({ -10, 1, 2, -11, -12, 5, 6, -13 })
        },
        {   3,
            tensor{2, 1, 2, 2},
            getValues<T>({ 0, 1, 2, 3, 4, 5, 6, 7 }),
            tensor{2, 1, 1, 2},
            getValues<T>({ 0, 1, 0, 1 }),
            getValues<T>({ -10, -11, -12, -13 }),
            getValues<T>({ -10, 1, 2, -11, -12, 5, 6, -13 })
        },
    };

    return result;
}

template<typename T>
std::vector<ScatterElementsUpdateParams<T>> generateScatterElementsUpdateParams3D() {
    const std::vector<ScatterElementsUpdateParams<T>> result = {
        {   1,
            tensor{2, 4, 1, 1, 3},
            getValues<T>({ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23 }),
            tensor{2, 1, 1, 1, 2},
            getValues<T>({ 0, 3, 1, 2 }),
            getValues<T>({ -100, -110, -120, -130 }),
            getValues<T>({ -100, 1, 2, 3, 4, 5, 6, 7, 8, 9, -110, 11, 12, 13, 14, -120, 16, 17, 18, -130, 20, 21, 22, 23 })
        },
        {   4,
            tensor{2, 4, 1, 1, 3},
            getValues<T>({ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23 }),
            tensor{2, 1, 1, 1, 2},
            getValues<T>({ 0, 1, 0, 1 }),
            getValues<T>({ -100, -110, -120, -130 }),
            getValues<T>({ -100, 1, -110, 3, 4, 5, 6, 7, 8, 9, 10, 11, -120, 13, -130, 15, 16, 17, 18, 19, 20, 21, 22, 23 })
        },
    };

    return result;
}

template<typename T>
std::vector<ScatterElementsUpdateParams<T>> generateScatterElementsUpdateParams4D() {
    const std::vector<ScatterElementsUpdateParams<T>> result = {
        {   5,
            tensor{2, 4, 2, 1, 1, 3},
            getValues<T>({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                          24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47}),
            tensor{2, 1, 1, 1, 1, 2},
            getValues<T>({2, 1, 1, 1, 2}),
            getValues<T>({-100, -110, -120, -130}),
            getValues<T>({0, 1, -100, -110, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                          24, -120, 26, -130, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47}),
        }
    };

    return result;
}

template<typename T>
float getError() {
    return 0.0;
}

template<>
float getError<float>() {
    return 0.001;
}

template<>
float getError<half_t>() {
    return 0.2;
}

std::string toString(const format::type format) {
    switch(format) {
        case format::bfyx:
            return "bfyx";
        case format::b_fs_yx_fsv16:
            return "b_fs_yx_fsv16";
        case format::b_fs_yx_fsv32:
            return "b_fs_yx_fsv32";
        case format::bs_fs_yx_bsv16_fsv16:
            return "bs_fs_yx_bsv16_fsv16";
        case format::bs_fs_yx_bsv32_fsv16:
            return "bs_fs_yx_bsv32_fsv16";
        case format::bs_fs_yx_bsv32_fsv32:
            return "bs_fs_yx_bsv32_fsv32";
        case format::bfzyx:
            return "bfzyx";
        case format::b_fs_zyx_fsv16:
            return "b_fs_zyx_fsv16";
        case format::bs_fs_zyx_bsv16_fsv16:
            return "bs_fs_zyx_bsv16_fsv16";
        case format::bfwzyx:
            return "bfwzyx";
        default:
            return std::to_string(format);
    }
}

struct PrintToStringParamName {
    template<class T>
    std::string operator()(const testing::TestParamInfo<ScatterElementsUpdateParamsWithFormat<T> > &param) {
        std::stringstream buf;
        ScatterElementsUpdateParams<T> p;
        format::type plain_format;
        format::type target_format;
        std::tie(p, plain_format, target_format) = param.param;
        buf << "_axis=" << p.axis
            << "_data=" << p.data_tensor.to_string()
            << "_indices=" << p.indices_tensor.to_string()
            << "_plainFormat=" << toString(plain_format)
            << "_targetFormat=" << toString(target_format);
        return buf.str();
    }
};
}; // namespace

template<typename T>
struct scatter_elements_update_gpu_formats_test
        : public ::testing::TestWithParam<ScatterElementsUpdateParamsWithFormat<T> > {
public:
    void test() {
        const auto data_type = type_to_data_type<T>::value;
        ScatterElementsUpdateParams<T> params;
        format::type plain_format;
        format::type target_format;

        std::tie(params, plain_format, target_format) = this->GetParam();

        auto& engine = get_test_engine();
        const auto data = engine.allocate_memory({data_type, plain_format, params.data_tensor});
        const auto indices = engine.allocate_memory({data_type, plain_format, params.indices_tensor});
        const auto updates = engine.allocate_memory({data_type, plain_format, params.indices_tensor});

        set_values(data, params.data);
        set_values(indices, params.indices);
        set_values(updates, params.updates);

        topology topology;
        topology.add(input_layout("Data", data->get_layout()));
        topology.add(input_layout("Indices", indices->get_layout()));
        topology.add(input_layout("Updates", updates->get_layout()));
        topology.add(reorder("DataReordered", "Data", target_format, data_type));
        topology.add(reorder("IndicesReordered", "Indices", target_format, data_type));
        topology.add(reorder("UpdatesReordered", "Updates", target_format, data_type));
        topology.add(
            scatter_elements_update("ScatterEelementsUpdate", "DataReordered", "IndicesReordered",
                                    "UpdatesReordered", params.axis)
        );
        topology.add(reorder("ScatterEelementsUpdatePlain", "ScatterEelementsUpdate", plain_format, data_type));

        network network{engine, topology};

        network.set_input_data("Data", data);
        network.set_input_data("Indices", indices);
        network.set_input_data("Updates", updates);

        const auto outputs = network.execute();
        const auto output = outputs.at("ScatterEelementsUpdatePlain").get_memory();
        const cldnn::mem_lock<T> output_ptr(output, get_test_stream());

        ASSERT_EQ(params.data.size(), output_ptr.size());
        ASSERT_EQ(params.expected.size(), output_ptr.size());
        for (uint32_t i = 0; i < output_ptr.size(); i++) {
            EXPECT_NEAR(output_ptr[i], params.expected[i], getError<T>())
                << "format=" << toString(target_format) << ", i=" << i;
        }
    }
};

using scatter_elements_update_gpu_formats_test_f32 = scatter_elements_update_gpu_formats_test<float>;
using scatter_elements_update_gpu_formats_test_f16 = scatter_elements_update_gpu_formats_test<half_t>;
using scatter_elements_update_gpu_formats_test_i32 = scatter_elements_update_gpu_formats_test<int32_t>;

TEST_P(scatter_elements_update_gpu_formats_test_f32, basic) {
    ASSERT_NO_FATAL_FAILURE(test());
}

TEST_P(scatter_elements_update_gpu_formats_test_f16, basic) {
    ASSERT_NO_FATAL_FAILURE(test());
}

TEST_P(scatter_elements_update_gpu_formats_test_i32, basic) {
    ASSERT_NO_FATAL_FAILURE(test());
}


INSTANTIATE_TEST_SUITE_P(scatter_elements_update_gpu_formats_test_f32_2d,
                         scatter_elements_update_gpu_formats_test_f32,
                         ::testing::Combine(
                                 ::testing::ValuesIn(generateScatterElementsUpdateParams2D<float>()),
                                 ::testing::Values(format::bfyx),
                                 ::testing::ValuesIn(formats2D)
                         ),
                         PrintToStringParamName());

INSTANTIATE_TEST_SUITE_P(scatter_elements_update_gpu_formats_test_f16_2d,
                         scatter_elements_update_gpu_formats_test_f16,
                         ::testing::Combine(
                                 ::testing::ValuesIn(generateScatterElementsUpdateParams2D<half_t>()),
                                 ::testing::Values(format::bfyx),
                                 ::testing::ValuesIn(formats2D)
                         ),
                         PrintToStringParamName());

INSTANTIATE_TEST_SUITE_P(scatter_elements_update_gpu_formats_test_i32_2d,
                         scatter_elements_update_gpu_formats_test_i32,
                         ::testing::Combine(
                                 ::testing::ValuesIn(generateScatterElementsUpdateParams2D<int32_t>()),
                                 ::testing::Values(format::bfyx),
                                 ::testing::ValuesIn(formats2D)
                         ),
                         PrintToStringParamName());

INSTANTIATE_TEST_SUITE_P(scatter_elements_update_gpu_formats_test_f32_3d,
                         scatter_elements_update_gpu_formats_test_f32,
                         ::testing::Combine(
                                 ::testing::ValuesIn(generateScatterElementsUpdateParams3D<float>()),
                                 ::testing::Values(format::bfzyx),
                                 ::testing::ValuesIn(formats3D)
                         ),
                         PrintToStringParamName());

INSTANTIATE_TEST_SUITE_P(scatter_elements_update_gpu_formats_test_f32_4d,
                         scatter_elements_update_gpu_formats_test_f32,
                         ::testing::Combine(
                                 ::testing::ValuesIn(generateScatterElementsUpdateParams4D<float>()),
                                 ::testing::Values(format::bfwzyx),
                                 ::testing::ValuesIn(formats4D)
                         ),
                         PrintToStringParamName());
