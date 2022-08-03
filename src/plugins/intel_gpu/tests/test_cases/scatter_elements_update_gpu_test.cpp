// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/scatter_elements_update.hpp>
#include <intel_gpu/runtime/memory.hpp>
#include <intel_gpu/graph/topology.hpp>
#include <intel_gpu/graph/network.hpp>

#include <cstddef>

using namespace cldnn;
using namespace ::tests;


const std::vector<format::type> z_formats2D{
        format::bfyx,
        format::b_fs_yx_fsv16,
        format::b_fs_yx_fsv32,
        format::bs_fs_yx_bsv16_fsv16,
        format::bs_fs_yx_bsv32_fsv16,
        format::bs_fs_yx_bsv32_fsv32
};

TEST(scatter_elements_update_gpu_fp16, d2411_axisF) {
    //  Dictionary : 2x4x1x1
    //  Indexes : 2x2x1x1
    //  Updates : 2x2x1x1
    //  Axis : 1
    //  Output : 2x4x1x1
    //  Input values in fp16
    //
    //  Input:
    //  3.f, 6.f, 5.f, 4.f,
    //  1.f, 7.f, 2.f, 9.f
    //
    //  Indexes:
    //  0.f, 1.f
    //  2.f, 3.f
    //
    //  Updates:
    //  10.f, 11.f,
    //  12.f, 13.f
    //
    //  Output:
    //  10.f, 11.f, 5.f, 4.f,
    //  1.f, 7.f, 12.f, 13.f

    auto& engine = get_test_engine();
    const auto data_type = data_types::f16;
    const auto plain_format = format::bfyx;

    for(const auto target_format : z_formats2D) {

        auto input1 = engine.allocate_memory({data_types::f16, format::bfyx, tensor{2, 4, 1, 1}}); // Dictionary
        auto input2 = engine.allocate_memory({data_types::f16, format::bfyx, tensor{2, 2, 1, 1}}); // Indexes
        auto input3 = engine.allocate_memory({data_types::f16, format::bfyx, tensor{2, 2, 1, 1}}); // Updates
        auto axis = 1;

        set_values(input1, {
                FLOAT16(3.0f), FLOAT16(6.0f), FLOAT16(5.0f), FLOAT16(4.0f),
                FLOAT16(1.0f), FLOAT16(7.0f), FLOAT16(2.0f), FLOAT16(9.0f)
        });

        set_values(input2, {
                FLOAT16(0.0f), FLOAT16(1.0f),
                FLOAT16(2.0f), FLOAT16(3.0f)
        });

        set_values(input3, {
                FLOAT16(10.0f), FLOAT16(11.0f),
                FLOAT16(12.0f), FLOAT16(13.0f)
        });

        topology topology;
        topology.add(input_layout("InputData", input1->get_layout()));
        topology.add(input_layout("InputIndices", input2->get_layout()));
        topology.add(input_layout("InputUpdates", input3->get_layout()));
        topology.add(reorder("InputData_Reordered", "InputData", target_format, data_type));
        topology.add(reorder("InputIndices_Reordered", "InputIndices", target_format, data_type));
        topology.add(reorder("InputUpdates_Reordered", "InputUpdates", target_format, data_type));

        topology.add(
                scatter_elements_update("scatter_elements_update", "InputData_Reordered", "InputIndices_Reordered",
                                        "InputUpdates_Reordered", axis)
        );

        topology.add(reorder("scatter_elements_update_plain", "scatter_elements_update", plain_format, data_type));

        network network(engine, topology);

        network.set_input_data("InputData", input1);
        network.set_input_data("InputIndices", input2);
        network.set_input_data("InputUpdates", input3);

        auto outputs = network.execute();

        auto output = outputs.at("scatter_elements_update_plain").get_memory();
        cldnn::mem_lock<uint16_t> output_ptr(output, get_test_stream());

        std::vector<float> expected_results = {
                10.f, 11.f, 5.f, 4.f,
                1.f, 7.f, 12.f, 13.f
        };

        for (size_t i = 0; i < expected_results.size(); ++i) {
            EXPECT_EQ(expected_results[i], float16_to_float32(output_ptr[i])) << "target_format=" << target_format << " i=" << i;
        }
    }
}



namespace {
template<typename T>
struct SEUParams {
    int64_t axis;
    tensor data_tensor;
    std::vector<T> data;
    tensor indices_tensor;
    std::vector<T> indices;
    std::vector<T> updates;
    std::vector<T> expected;
};

template<typename T>
using SEUParamsWithFormat = std::tuple<
    SEUParams<T>,
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
std::vector<SEUParams<T>> generateSEUParams2D() {
    const std::vector<SEUParams<T>> result = {
        {   1,
            // data
            tensor{2, 4, 1, 1},
            getValues<T>({ 0, 1, 2, 3, 4, 5, 6, 7 }),
            // indices
            tensor{2, 2, 1, 1},
            getValues<T>({ 0, 1, 2, 3 }),
            // updates
            getValues<T>({ -10, -11, -12, -13 }),
            // expected
            getValues<T>({ -10, -11, 2, 3, 4, 5, -12, -13 })
        },
        {   2,
            // data
            tensor{2, 1, 2, 2},
            getValues<T>({ 0, 1, 2, 3, 4, 5, 6, 7 }),
            // indices
            tensor{2, 1, 2, 1},
            getValues<T>({ 0, 1, 0, 1 }),
            // updates
            getValues<T>({ -10, -11, -12, -13 }),
            // expected
            getValues<T>({ -10, 1, 2, -11, -12, 5, 6, -13 })
        },
        {   3,
            // data
            tensor{2, 1, 2, 2},
            getValues<T>({ 0, 1, 2, 3, 4, 5, 6, 7 }),
            // indices
            tensor{2, 1, 1, 2},
            getValues<T>({ 0, 1, 0, 1 }),
            // updates
            getValues<T>({ -10, -11, -12, -13 }),
            // expected
            getValues<T>({ -10, 1, 2, -11, -12, 5, 6, -13 })
        },
    };

    return result;
}

template<typename T>
std::vector<SEUParams<T>> generateSEUParams3D() {
    const std::vector<SEUParams<T>> result = {
        {   1,
            // data
            tensor{2, 4, 1, 1, 3},
            getValues<T>({ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23 }),
            // indices
            tensor{2, 1, 1, 1, 2},
            getValues<T>({ 0, 3, 1, 2 }),
            // updates
            getValues<T>({ -100, -110, -120, -130 }),
            // expected
            getValues<T>({ -100, 1, 2, 3, 4, 5, 6, 7, 8, 9, -110, 11, 12, 13, 14, -120, 16, 17, 18, -130, 20, 21, 22, 23 })
        },
        {   4,
            // data
            tensor{2, 4, 1, 1, 3},
            getValues<T>({ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23 }),
            // indices
            tensor{2, 1, 1, 1, 2},
            getValues<T>({ 0, 1, 0, 1 }),
            // updates
            getValues<T>({ -100, -110, -120, -130 }),
            // expected
            getValues<T>({ -100, 1, -110, 3, 4, 5, 6, 7, 8, 9, 10, 11, -120, 13, -130, 15, 16, 17, 18, 19, 20, 21, 22, 23 })
        },
    };

    return result;
}

template<typename T>
std::vector<SEUParams<T>> generateSEUParams4D() {
    const std::vector<SEUParams<T>> result = {
        { 5,
          // data
          tensor{2, 4, 2, 1, 1, 3},
          getValues<T>({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                        24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47}),
          // indices
          tensor{2, 1, 1, 1, 1, 2},
          getValues<T>({2, 1, 1, 1, 2}),
          // updates
          getValues<T>({-100, -110, -120, -130}),
          // expected
          getValues<T>({0, 1, -100, -110, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                        24, -120, 26, -130, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47}),
        }
    };

    return result;
}

template<typename T>
float getError();

template<>
float getError<float>() {
    return 0.001;
}

template<>
float getError<half_t>() {
    return 0.2;
}

template<>
float getError<int32_t>() {
    return 0.0;
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
        default:
            return std::to_string(format);
    }
}

struct PrintToStringParamName {
    template<class T>
    std::string operator()(const testing::TestParamInfo<SEUParamsWithFormat<T> > &param) {
        std::stringstream buf;
        SEUParams<T> p;
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
        : public ::testing::TestWithParam<SEUParamsWithFormat<T> > {
public:
    void test() {
        const auto data_type = type_to_data_type<T>::value;
        SEUParams<T> params;
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
        topology.add(input_layout("InputData", data->get_layout()));
        topology.add(input_layout("InputIndices", indices->get_layout()));
        topology.add(input_layout("InputUpdates", updates->get_layout()));
        topology.add(reorder("InputData_Reordered", "InputData", target_format, data_type));
        topology.add(reorder("InputIndices_Reordered", "InputIndices", target_format, data_type));
        topology.add(reorder("InputUpdates_Reordered", "InputUpdates", target_format, data_type));
        topology.add(
                scatter_elements_update("ScatterEelementsUpdate", "InputData_Reordered", "InputIndices_Reordered",
                                        "InputUpdates_Reordered", params.axis)
        );
        topology.add(reorder("ScatterEelementsUpdate_Plain", "ScatterEelementsUpdate", plain_format, data_type));

        network network{engine, topology};

        network.set_input_data("InputData", data);
        network.set_input_data("InputIndices", indices);
        network.set_input_data("InputUpdates", updates);

        const auto outputs = network.execute();
        const auto output = outputs.at("ScatterEelementsUpdate_Plain").get_memory();
        const cldnn::mem_lock<T> output_ptr(output, get_test_stream());

        ASSERT_EQ(params.data.size(), output_ptr.size());
        ASSERT_EQ(params.expected.size(), output_ptr.size());
        for (uint32_t i = 0; i < output_ptr.size(); i++) {
            EXPECT_NEAR(output_ptr[i], params.expected[i], getError<T>())
                << "format=" << toString(target_format) << ", i=" << i;

//            std::cout << static_cast<int>(output_ptr[i]) << ", ";        getError<T>();
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
                                 ::testing::ValuesIn(generateSEUParams2D<float>()),
                                 ::testing::Values(format::bfyx),
                                 ::testing::ValuesIn(formats2D)
                         ),
                         PrintToStringParamName());

INSTANTIATE_TEST_SUITE_P(scatter_elements_update_gpu_formats_test_f16_2d,
                         scatter_elements_update_gpu_formats_test_f16,
                         ::testing::Combine(
                                 ::testing::ValuesIn(generateSEUParams2D<half_t>()),
                                 ::testing::Values(format::bfyx),
                                 ::testing::ValuesIn(formats2D)
                         ),
                         PrintToStringParamName());

INSTANTIATE_TEST_SUITE_P(scatter_elements_update_gpu_formats_test_i32_2d,
                         scatter_elements_update_gpu_formats_test_i32,
                         ::testing::Combine(
                                 ::testing::ValuesIn(generateSEUParams2D<int32_t>()),
                                 ::testing::Values(format::bfyx),
                                 ::testing::ValuesIn(formats2D)
                         ),
                         PrintToStringParamName());

INSTANTIATE_TEST_SUITE_P(scatter_elements_update_gpu_formats_test_f32_3d,
                         scatter_elements_update_gpu_formats_test_f32,
                         ::testing::Combine(
                                 ::testing::ValuesIn(generateSEUParams3D<float>()),
                                 ::testing::Values(format::bfzyx),
                                 ::testing::ValuesIn(formats3D)
                         ),
                         PrintToStringParamName());

INSTANTIATE_TEST_SUITE_P(scatter_elements_update_gpu_formats_test_f16_3d,
                         scatter_elements_update_gpu_formats_test_f16,
                         ::testing::Combine(
                                 ::testing::ValuesIn(generateSEUParams3D<half_t>()),
                                 ::testing::Values(format::bfzyx),
                                 ::testing::ValuesIn(formats3D)
                         ),
                         PrintToStringParamName());

INSTANTIATE_TEST_SUITE_P(scatter_elements_update_gpu_formats_test_i32_3d,
                         scatter_elements_update_gpu_formats_test_i32,
                         ::testing::Combine(
                                 ::testing::ValuesIn(generateSEUParams3D<int32_t>()),
                                 ::testing::Values(format::bfzyx),
                                 ::testing::ValuesIn(formats3D)
                         ),
                         PrintToStringParamName());

INSTANTIATE_TEST_SUITE_P(scatter_elements_update_gpu_formats_test_f32_4d,
                         scatter_elements_update_gpu_formats_test_f32,
                         ::testing::Combine(
                                 ::testing::ValuesIn(generateSEUParams4D<float>()),
                                 ::testing::Values(format::bfwzyx),
                                 ::testing::Values(format::bfwzyx)
                         ),
                         PrintToStringParamName());
