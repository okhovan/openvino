// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"
#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/activation.hpp>
#include <intel_gpu/primitives/reorg_yolo.hpp>

#include <cstddef>
#include <string>

using namespace cldnn;
using namespace ::tests;

namespace {
template<typename T>
struct ReorgYoloParams {
    tensor inputTensor;
    std::vector<T> input;
    uint32_t stride;
    std::vector<T> expected;
};

template<typename T>
using ReorgYoloParamsWithLayout = std::tuple<
    ReorgYoloParams<T>,
    format::type        // blocked layout
>;

const std::vector<format::type> dataFormats = {
    format::bfyx,
    format::b_fs_yx_fsv16,
    format::b_fs_yx_fsv32,
    format::bs_fs_yx_bsv16_fsv16,
    format::bs_fs_yx_bsv32_fsv16,
    format::bs_fs_yx_bsv32_fsv32
};

template<typename T>
std::vector<T> getValues(const std::vector<float> &values) {
    std::vector<T> result(values.begin(), values.end());
    return result;
}

template <typename T> float getError();

template<>
float getError<float>() {
    return 0.001;
}

template<>
float getError<half_t>() {
    return 0.2;
}

template<typename T>
std::vector<ReorgYoloParams<T>> generateParams() {
    static const std::vector<ReorgYoloParams<T>> result = {
        {
            tensor(1, 2, 7, 3),
            getValues<T>({0, 4, 1, 3, -2, -5, -2, -2, 1, -3, 1, -3, -4, 0, -2, 1, -1, -2, 3, -1, -3,
                          -1, -2, 3, 4, -3, -4, 1, 2, 0, -4, -5, -2, -2, -3, 2, 3, 1, -5, 2, -4, -2}),
            4,
            getValues<T>({0, 4, 1, 3, -2, -5, -2, -2, 1, -3, 1, -3, -4, 0, -2, 1, -1, -2, 3, -1, -3,
                          -1, -2, 3, 4, -3, -4, 1, 2, 0, -4, -5, -2, -2, -3, 2, 3, 1, -5, 2, -4, -2})
        },
        {
            tensor(1, 3, 10, 7),
            getValues<T>(
                    {-2, -3, -4, 3, -5, 4, 0, -4, -2, -4, -5, 0, -3, 0, -2, 0, 0, -5, -4, -1, 3, -1, 0, -1,
                     0, -2, 0, 4, 1, 4, 0, -1, -4, 2, -2, -5, -1, -1, -2, 1, 2, -2, -1, 2, 0, -1, 0, -5,
                     4, 4, 3, 0, -4, -4, -4, -2, 0, 1, -2, -1, 4, -2, -4, 1, -1, -3, -4, -1, 1, -4,

                     -2, -4, -5, 0, -4, 3, 4, -5, -4, -2, 0, 2, -4, -3, 3, -1, 1, -4, -5, 4, 2, -5, 2, -3,
                     0, 4, 3, 3, 1, 2, -1, -4, 1, -3, -3, -2, 3, 4, -2, -5, 1, 4, 4, -2, 2, 1, -5, -2,
                     -5, 1, 1, -2, -3, -3, -1, -5, 1, -3, -5, -3, -4, -1, 4, -3, 4, -1, 4, 3, 1, 4,

                     -2, -4, -4, 4, -3, 4, 2, -3, -2, 4, -3, 0, 1, -4, 4, 4, 0, 3, -1, 3, 3, -5, 0, 3,
                     -3, 1, -2, 4, -5, -5, 1, 0, -1, 0, -3, -2, 0, -3, 3, -2, -2, 0, -3, 4, -1, 2, -2, 2,
                     -3, -1, -4, -2, 0, 2, 0, 2, 0, -3, 4, 3, -5, -3, -5, 1, -5, -3, -5, 4, -3, 3}),
            6,
            getValues<T>(
                    {-2, -3, -4, 3, -5, 4, 0, -4, -2, -4, -5, 0, -3, 0, -2, 0, 0, -5, -4, -1, 3, -1, 0, -1,
                     0, -2, 0, 4, 1, 4, 0, -1, -4, 2, -2, -5, -1, -1, -2, 1, 2, -2, -1, 2, 0, -1, 0, -5,
                     4, 4, 3, 0, -4, -4, -4, -2, 0, 1, -2, -1, 4, -2, -4, 1, -1, -3, -4, -1, 1, -4,

                     -2, -4, -5, 0, -4, 3, 4, -5, -4, -2, 0, 2, -4, -3, 3, -1, 1, -4, -5, 4, 2, -5, 2, -3,
                     0, 4, 3, 3, 1, 2, -1, -4, 1, -3, -3, -2, 3, 4, -2, -5, 1, 4, 4, -2, 2, 1, -5, -2,
                     -5, 1, 1, -2, -3, -3, -1, -5, 1, -3, -5, -3, -4, -1, 4, -3, 4, -1, 4, 3, 1, 4,

                     -2, -4, -4, 4, -3, 4, 2, -3, -2, 4, -3, 0, 1, -4, 4, 4, 0, 3, -1, 3, 3, -5, 0, 3,
                     -3, 1, -2, 4, -5, -5, 1, 0, -1, 0, -3, -2, 0, -3, 3, -2, -2, 0, -3, 4, -1, 2, -2, 2,
                     -3, -1, -4, -2, 0, 2, 0, 2, 0, -3, 4, 3, -5, -3, -5, 1, -5, -3, -5, 4, -3, 3})
        }
    };
    return result;
}

template<typename T>
std::vector<ReorgYoloParams<T>> generateErrorParams() {
    static const std::vector<ReorgYoloParams<T>> result = {
        {
            tensor(1, 2, 7, 3),
            getValues<T>({0, 4, 1, 3, -2, -5, -2, -2, 1, -3, 1, -3, -4, 0, -2, 1, -1, -2, 3, -1, -3,
                          -1, -2, 3, 4, -3, -4, 1, 2, 0, -4, -5, -2, -2, -3, 2, 3, 1, -5, 2, -4, -2}),
            4,
            getValues<T>({0, 4, 1, 3, -2, -5, -2, -2, 1, -3, 1, -3, -4, 0, -2, 1, -1, -2, 3, -1, -3,
                          -1, -2, 3, 4, -3, -4, 1, 2, 0, -4, -5, -2, -2, -3, 2, 3, 1, -5, 2, -4, -2})
        },
        {
            tensor(1, 3, 10, 7),
            getValues<T>(
                    {-2, -3, -4, 3, -5, 4, 0, -4, -2, -4, -5, 0, -3, 0, -2, 0, 0, -5, -4, -1, 3, -1, 0, -1,
                     0, -2, 0, 4, 1, 4, 0, -1, -4, 2, -2, -5, -1, -1, -2, 1, 2, -2, -1, 2, 0, -1, 0, -5,
                     4, 4, 3, 0, -4, -4, -4, -2, 0, 1, -2, -1, 4, -2, -4, 1, -1, -3, -4, -1, 1, -4,

                     -2, -4, -5, 0, -4, 3, 4, -5, -4, -2, 0, 2, -4, -3, 3, -1, 1, -4, -5, 4, 2, -5, 2, -3,
                     0, 4, 3, 3, 1, 2, -1, -4, 1, -3, -3, -2, 3, 4, -2, -5, 1, 4, 4, -2, 2, 1, -5, -2,
                     -5, 1, 1, -2, -3, -3, -1, -5, 1, -3, -5, -3, -4, -1, 4, -3, 4, -1, 4, 3, 1, 4,

                     -2, -4, -4, 4, -3, 4, 2, -3, -2, 4, -3, 0, 1, -4, 4, 4, 0, 3, -1, 3, 3, -5, 0, 3,
                     -3, 1, -2, 4, -5, -5, 1, 0, -1, 0, -3, -2, 0, -3, 3, -2, -2, 0, -3, 4, -1, 2, -2, 2,
                     -3, -1, -4, -2, 0, 2, 0, 2, 0, -3, 4, 3, -5, -3, -5, 1, -5, -3, -5, 4, -3, 3}),
            6,
            getValues<T>(
                    {-2, -3, -4, 3, -5, 4, 0, -4, -2, -4, -5, 0, -3, 0, -2, 0, 0, -5, -4, -1, 3, -1, 0, -1,
                     0, -2, 0, 4, 1, 4, 0, -1, -4, 2, -2, -5, -1, -1, -2, 1, 2, -2, -1, 2, 0, -1, 0, -5,
                     4, 4, 3, 0, -4, -4, -4, -2, 0, 1, -2, -1, 4, -2, -4, 1, -1, -3, -4, -1, 1, -4,

                     -2, -4, -5, 0, -4, 3, 4, -5, -4, -2, 0, 2, -4, -3, 3, -1, 1, -4, -5, 4, 2, -5, 2, -3,
                     0, 4, 3, 3, 1, 2, -1, -4, 1, -3, -3, -2, 3, 4, -2, -5, 1, 4, 4, -2, 2, 1, -5, -2,
                     -5, 1, 1, -2, -3, -3, -1, -5, 1, -3, -5, -3, -4, -1, 4, -3, 4, -1, 4, 3, 1, 4,

                     -2, -4, -4, 4, -3, 4, 2, -3, -2, 4, -3, 0, 1, -4, 4, 4, 0, 3, -1, 3, 3, -5, 0, 3,
                     -3, 1, -2, 4, -5, -5, 1, 0, -1, 0, -3, -2, 0, -3, 3, -2, -2, 0, -3, 4, -1, 2, -2, 2,
                     -3, -1, -4, -2, 0, 2, 0, 2, 0, -3, 4, 3, -5, -3, -5, 1, -5, -3, -5, 4, -3, 3})
        }
    };
    return result;
}

struct PrintToStringParamName {
    template<class T>
    std::string operator()(const testing::TestParamInfo<ReorgYoloParamsWithLayout<T> > &param) {
        std::stringstream buf;
        ReorgYoloParams<T> p;
        format::type target_format;
        std::tie(p, target_format) = param.param;
        buf << "InputTensor=" << p.inputTensor.to_string()
            << ".stride=" << p.stride
            << ".TargetLayout=" << target_format;
        return buf.str();
    }
};
};  // namespace

template<typename T>
struct reorg_yolo_test
        : public ::testing::TestWithParam<ReorgYoloParamsWithLayout<T> > {
public:
    void test(bool expect_error = false) {
        const auto data_type = type_to_data_type<T>::value;
        ReorgYoloParams<T> params;
        const format::type plain_format = format::bfyx;
        format::type target_format;
        std::tie(params,  target_format) = this->GetParam();

        auto& engine = get_test_engine();

        auto input = engine.allocate_memory({data_type, plain_format, params.inputTensor});

        set_values(input, params.input);

        topology topology;
        topology.add(input_layout("input", input->get_layout()));
        topology.add(reorder("input_reordered", "input", target_format, data_type));
        topology.add(reorg_yolo("reorg_yolo", "input_reordered", params.stride));
        topology.add(reorder("reorg_yolo_reordered", "reorg_yolo", plain_format, data_type));

        network network(engine, topology);
        network.set_input_data("input", input);
        const auto result = network.execute();

        if (expect_error) {
            //do smth
        }


        auto out_mem = result.at("reorg_yolo_reordered").get_memory();
        cldnn::mem_lock<T> out_ptr(out_mem, get_test_stream());

        ASSERT_EQ(params.inputTensor.count(), out_ptr.size());
        for (size_t i = 0; i < params.expected.size(); ++i) {
            EXPECT_NEAR(params.expected[i], out_ptr[i], getError<T>())
                << "format=" << target_format << ", i= " << i;
        }
    }
};


using test_f32 = reorg_yolo_test<float>;
using test_f16 = reorg_yolo_test<half_t>;

TEST_P(test_f32, valid) {
    ASSERT_NO_FATAL_FAILURE(test());
}

TEST_P(test_f16, valid) {
    ASSERT_NO_FATAL_FAILURE(test());
}

TEST_P(test_f32, invalid) {
    EXPECT_THROW(test(true), std::invalid_argument);
}


INSTANTIATE_TEST_SUITE_P(reorg_yolo_f32,
                         test_f32,
                         ::testing::Combine(
                                 ::testing::ValuesIn(generateParams<float>()),
                                 ::testing::ValuesIn(dataFormats)),
                         PrintToStringParamName());

INSTANTIATE_TEST_SUITE_P(reorg_yolo_f16,
                         test_f16,
                         ::testing::Combine(
                                 ::testing::ValuesIn(generateParams<half_t>()),
                                 ::testing::ValuesIn(dataFormats)),
                         PrintToStringParamName());

INSTANTIATE_TEST_SUITE_P(reorg_yolo_invalid_input,
                         test_f32,
                         ::testing::Combine(
                                 ::testing::ValuesIn(generateErrorParams<float>()),
                                 ::testing::Values(format::bfyx)),
                         PrintToStringParamName());
