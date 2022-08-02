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

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"

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
    format::type,       // blocked layout
    bool                // should_fail
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
            tensor{1, 4, 2, 2},
            getValues<T>({
                0.0, 1.0,
                2.0, 3.0,

                4.0, 5.0,
                6.0, 7.0,

                8.0, 9.0,
                10.0, 11.0,

                12.0, 13.0,
                14.0, 15.0,
            }),
            2,
            getValues<T>({
                0.0f, 2.0f, 8.0f, 10.0f,
                1.0f, 3.0f, 9.0f, 11.0f,

                4.0f, 6.0f, 12.0f, 14.0f,
                5.0f, 7.0f, 13.0f, 15.0f,
            })
        },
        {
            tensor{2, 9, 3, 3},
            getValues<T>({
                0.0f, 1.0f, 2.0f,
                3.0f, 4.0f, 5.0f,
                6.0f, 7.0f, 8.0f,

                9.0f, 10.0f, 11.0f,
                12.0f, 13.0f, 14.0f,
                15.0f, 16.0f, 17.0f,

                18.0f, 19.0f, 20.0f,
                21.0f, 22.0f, 23.0f,
                24.0f, 25.0f, 26.0f,

                27.0f, 28.0f, 29.0f,
                30.0f, 31.0f, 32.0f,
                33.0f, 34.0f, 35.0f,

                36.0f, 37.0f, 38.0f, 39.0f,
                40.0f, 41.0f, 42.0f, 43.0f, 44.0f, 45.0f,
                46.0f, 47.0f, 48.0f, 49.0f, 50.0f, 51.0f, 52.0f, 53.0f, 54.0f,
                55.0f, 56.0f, 57.0f, 58.0f, 59.0f, 60.0f, 61.0f, 62.0f, 63.0f,
                64.0f, 65.0f, 66.0f, 67.0f, 68.0f, 69.0f, 70.0f, 71.0f, 72.0f,
                73.0f, 74.0f, 75.0f, 76.0f, 77.0f, 78.0f, 79.0f, 80.0f, 81.0f,
                82.0f, 83.0f, 84.0f, 85.0f, 86.0f, 87.0f, 88.0f, 89.0f, 90.0f,
                91.0f, 92.0f, 93.0f, 94.0f, 95.0f, 96.0f, 97.0f, 98.0f, 99.0f,
                100.0f, 101.0f, 102.0f, 103.0f, 104.0f, 105.0f, 106.0f, 107.0f, 108.0f,
                109.0f, 110.0f, 111.0f, 112.0f, 113.0f, 114.0f, 115.0f, 116.0f, 117.0f,
                118.0f, 119.0f, 120.0f, 121.0f, 122.0f, 123.0f, 124.0f, 125.0f, 126.0f,
                127.0f, 128.0f, 129.0f, 130.0f, 131.0f, 132.0f, 133.0f, 134.0f, 135.0f,
                136.0f, 137.0f, 138.0f, 139.0f, 140.0f, 141.0f, 142.0f, 143.0f, 144.0f,
                145.0f, 146.0f, 147.0f, 148.0f, 149.0f, 150.0f, 151.0f, 152.0f, 153.0f,
                154.0f, 155.0f, 156.0f, 157.0f, 158.0f, 159.0f, 160.0f, 161.0f
            }),
            3,
            getValues<T>({
                0.0f, 3.0f, 6.0f, 27.0f, 30.0f, 33.0f, 54.0f, 57.0f, 60.0f,
                1.0f, 4.0f, 7.0f, 28.0f, 31.0f, 34.0f, 55.0f, 58.0f, 61.0f,
                2.0f, 5.0f, 8.0f, 29.0f, 32.0f, 35.0f, 56.0f, 59.0f, 62.0f,
                9.0f, 12.0f, 15.0f, 36.0f, 39.0f, 42.0f, 63.0f, 66.0f, 69.0f,
                10.0f, 13.0f, 16.0f, 37.0f, 40.0f, 43.0f, 64.0f, 67.0f, 70.0f,
                11.0f, 14.0f, 17.0f, 38.0f, 41.0f, 44.0f, 65.0f, 68.0f, 71.0f,
                18.0f, 21.0f, 24.0f, 45.0f, 48.0f, 51.0f, 72.0f, 75.0f, 78.0f,
                19.0f, 22.0f, 25.0f, 46.0f, 49.0f, 52.0f, 73.0f, 76.0f, 79.0f,
                20.0f, 23.0f, 26.0f, 47.0f, 50.0f, 53.0f, 74.0f, 77.0f, 80.0f,
                81.0f, 84.0f, 87.0f, 108.0f, 111.0f, 114.0f, 135.0f, 138.0f, 141.0f,
                82.0f, 85.0f, 88.0f, 109.0f, 112.0f, 115.0f, 136.0f, 139.0f, 142.0f,
                83.0f, 86.0f, 89.0f, 110.0f, 113.0f, 116.0f, 137.0f, 140.0f, 143.0f,
                90.0f, 93.0f, 96.0f, 117.0f, 120.0f, 123.0f, 144.0f, 147.0f, 150.0f,
                91.0f, 94.0f, 97.0f, 118.0f, 121.0f, 124.0f, 145.0f, 148.0f, 151.0f,
                92.0f, 95.0f, 98.0f, 119.0f, 122.0f, 125.0f, 146.0f, 149.0f, 152.0f,
                99.0f, 102.0f, 105.0f, 126.0f, 129.0f, 132.0f, 153.0f, 156.0f, 159.0f,
                100.0f, 103.0f, 106.0f, 127.0f, 130.0f, 133.0f, 154.0f, 157.0f, 160.0f,
                101.0f, 104.0f, 107.0f, 128.0f, 131.0f, 134.0f, 155.0f, 158.0f, 161.0f,
            }),
        },
        {
            tensor{2, 5, 4, 4},
            getValues<T>({
                0.0f, 1.0f, 2.0f, 3.0f,
                4.0f, 5.0f, 6.0f, 7.0f,
                8.0f, 9.0f, 10.0f, 11.0f,
                12.0f, 13.0f, 14.0f, 15.0f,

                16.0f, 17.0f, 18.0f, 19.0f,
                20.0f, 21.0f, 22.0f, 23.0f,
                24.0f, 25.0f, 26.0f, 27.0f,
                28.0f, 29.0f, 30.0f, 31.0f,

                32.0f, 33.0f, 34.0f, 35.0f,
                36.0f, 37.0f, 38.0f, 39.0f,
                40.0f, 41.0f, 42.0f, 43.0f,
                44.0f, 45.0f, 46.0f, 47.0f,

                48.0f, 49.0f, 50.0f, 51.0f,
                52.0f, 53.0f, 54.0f, 55.0f,
                56.0f, 57.0f, 58.0f, 59.0f,
                60.0f, 61.0f, 62.0f, 63.0f,

                64.0f, 65.0f, 66.0f, 67.0f, 68.0f, 69.0f, 70.0f,
                71.0f, 72.0f, 73.0f, 74.0f, 75.0f, 76.0f, 77.0f, 78.0f, 79.0f, 80.0f,
                81.0f, 82.0f, 83.0f, 84.0f, 85.0f, 86.0f, 87.0f, 88.0f, 89.0f, 90.0f,
                91.0f, 92.0f, 93.0f, 94.0f, 95.0f, 96.0f, 97.0f, 98.0f, 99.0f, 100.0f,
                101.0f, 102.0f, 103.0f, 104.0f, 105.0f, 106.0f, 107.0f, 108.0f, 109.0f, 110.0f,
                111.0f, 112.0f, 113.0f, 114.0f, 115.0f, 116.0f, 117.0f, 118.0f, 119.0f, 120.0f,
                121.0f, 122.0f, 123.0f, 124.0f, 125.0f, 126.0f, 127.0f, 128.0f, 129.0f, 130.0f,
                131.0f, 132.0f, 133.0f, 134.0f, 135.0f, 136.0f, 137.0f, 138.0f, 139.0f, 140.0f,
                141.0f, 142.0f, 143.0f, 144.0f, 145.0f, 146.0f, 147.0f, 148.0f, 149.0f, 150.0f,
                151.0f, 152.0f, 153.0f, 154.0f, 155.0f, 156.0f, 157.0f, 158.0f, 159.0f,
            }),
            2,
            getValues<T>({
                0.0f, 2.0f,
                4.0f, 6.0f,

                16.0f, 18.0f,
                20.0f, 22.0f,

                32.0f, 34.0f,
                36.0f, 38.0f,

                48.0f, 50.0f,
                52.0f, 54.0f,

                1.0f, 3.0f,
                5.0f, 7.0f,

                17.0f, 19.0f,
                21.0f, 23.0f,

                33.0f, 35.0f,
                37.0f, 39.0f,

                49.0f, 51.0f,
                53.0f, 55.0f,

                8.0f, 10.0f, 12.0f, 14.0f, 24.0f, 26.0f, 28.0f, 30.0f,
                40.0f, 42.0f, 44.0f, 46.0f, 56.0f, 58.0f, 60.0f, 62.0f, 9.0f, 11.0f,
                13.0f, 15.0f, 25.0f, 27.0f, 29.0f, 31.0f, 41.0f, 43.0f, 45.0f, 47.0f,
                57.0f, 59.0f, 61.0f, 63.0f, 16.0f, 18.0f, 20.0f, 22.0f, 32.0f, 34.0f,
                36.0f, 38.0f, 48.0f, 50.0f, 52.0f, 54.0f, 64.0f, 66.0f, 68.0f, 70.0f,
                64.0f, 66.0f, 68.0f, 70.0f, 80.0f, 82.0f, 84.0f, 86.0f, 96.0f, 98.0f,
                100.0f, 102.0f, 112.0f, 114.0f, 116.0f, 118.0f, 65.0f, 67.0f, 69.0f, 71.0f,
                81.0f, 83.0f, 85.0f, 87.0f, 97.0f, 99.0f, 101.0f, 103.0f, 113.0f, 115.0f,
                117.0f, 119.0f, 72.0f, 74.0f, 76.0f, 78.0f, 88.0f, 90.0f, 92.0f, 94.0f,
                104.0f, 106.0f, 108.0f, 110.0f, 120.0f, 122.0f, 124.0f, 126.0f, 73.0f, 75.0f,
                77.0f, 79.0f, 89.0f, 91.0f, 93.0f, 95.0f, 105.0f, 107.0f, 109.0f, 111.0f,
                121.0f, 123.0f, 125.0f, 127.0f, 80.0f, 82.0f, 84.0f, 86.0f, 96.0f, 98.0f,
                100.0f, 102.0f, 112.0f, 114.0f, 116.0f, 118.0f, 128.0f, 130.0f, 132.0f, 134.0f,
            }),
        },
/*
        {
            tensor{2, 9, 6, 6},
            getValues<T>({
                0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f,
                12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f,
                18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f,
                24.0f, 25.0f, 26.0f, 27.0f, 28.0f, 29.0f,
                30.0f, 31.0f, 32.0f, 33.0f, 34.0f, 35.0f,

                36.0f, 37.0f, 38.0f, 39.0f, 40.0f, 41.0f,
                42.0f, 43.0f, 44.0f, 45.0f, 46.0f, 47.0f,
                48.0f, 49.0f, 50.0f, 51.0f, 52.0f, 53.0f,
                54.0f, 55.0f, 56.0f, 57.0f, 58.0f, 59.0f,
                60.0f, 61.0f, 62.0f, 63.0f, 64.0f, 65.0f,
                66.0f, 67.0f, 68.0f, 69.0f, 70.0f, 71.0f,

                72.0f, 73.0f, 74.0f, 75.0f, 76.0f, 77.0f,
                78.0f,
                79.0f, 80.0f, 81.0f, 82.0f, 83.0f, 84.0f,
                85.0f, 86.0f, 87.0f, 88.0f, 89.0f, 90.0f,
                91.0f, 92.0f, 93.0f, 94.0f, 95.0f, 96.0f,
                97.0f, 98.0f, 99.0f, 100.0f, 101.0f, 102.0f,
                103.0f, 104.0f, 105.0f, 106.0f, 107.0f, 108.0f,
                109.0f, 110.0f, 111.0f, 112.0f, 113.0f, 114.0f,
                115.0f, 116.0f, 117.0f, 118.0f, 119.0f, 120.0f,
                121.0f, 122.0f, 123.0f, 124.0f, 125.0f, 126.0f,
                127.0f, 128.0f, 129.0f, 130.0f, 131.0f, 132.0f,
                133.0f, 134.0f, 135.0f, 136.0f, 137.0f, 138.0f,
                139.0f, 140.0f, 141.0f, 142.0f, 143.0f, 144.0f,
                145.0f, 146.0f, 147.0f, 148.0f, 149.0f, 150.0f,
                151.0f, 152.0f, 153.0f, 154.0f, 155.0f, 156.0f,
                157.0f, 158.0f, 159.0f, 160.0f, 161.0f, 162.0f,
                163.0f, 164.0f, 165.0f, 166.0f, 167.0f, 168.0f,
                169.0f, 170.0f, 171.0f, 172.0f, 173.0f, 174.0f,
                175.0f, 176.0f, 177.0f, 178.0f, 179.0f, 180.0f,
                181.0f, 182.0f, 183.0f, 184.0f, 185.0f, 186.0f,
                187.0f, 188.0f, 189.0f, 190.0f, 191.0f, 192.0f,
                193.0f, 194.0f, 195.0f, 196.0f, 197.0f, 198.0f,
                199.0f, 200.0f, 201.0f, 202.0f, 203.0f, 204.0f,
                205.0f, 206.0f, 207.0f, 208.0f, 209.0f, 210.0f,
                211.0f, 212.0f, 213.0f, 214.0f, 215.0f, 216.0f,
                217.0f, 218.0f, 219.0f, 220.0f, 221.0f, 222.0f,
                223.0f, 224.0f, 225.0f, 226.0f, 227.0f, 228.0f,
                229.0f, 230.0f, 231.0f, 232.0f, 233.0f, 234.0f,
                235.0f, 236.0f, 237.0f, 238.0f, 239.0f, 240.0f,
                241.0f, 242.0f, 243.0f, 244.0f, 245.0f, 246.0f,
                247.0f, 248.0f, 249.0f, 250.0f, 251.0f, 252.0f,
                253.0f, 254.0f, 255.0f, 256.0f, 257.0f, 258.0f,
                259.0f, 260.0f, 261.0f, 262.0f, 263.0f, 264.0f,
                265.0f, 266.0f, 267.0f, 268.0f, 269.0f, 270.0f,
                271.0f, 272.0f, 273.0f, 274.0f, 275.0f, 276.0f,
                277.0f, 278.0f, 279.0f, 280.0f, 281.0f, 282.0f,
                283.0f, 284.0f, 285.0f, 286.0f, 287.0f, 288.0f,
                289.0f, 290.0f, 291.0f, 292.0f, 293.0f, 294.0f,
                295.0f, 296.0f, 297.0f, 298.0f, 299.0f, 300.0f,
                301.0f, 302.0f, 303.0f, 304.0f, 305.0f, 306.0f,
                307.0f, 308.0f, 309.0f, 310.0f, 311.0f, 312.0f,
                313.0f, 314.0f, 315.0f, 316.0f, 317.0f, 318.0f,
                319.0f, 320.0f, 321.0f, 322.0f, 323.0f, 324.0f,
                325.0f, 326.0f, 327.0f, 328.0f, 329.0f, 330.0f,
                331.0f, 332.0f, 333.0f, 334.0f, 335.0f, 336.0f,
                337.0f, 338.0f, 339.0f, 340.0f, 341.0f, 342.0f,
                343.0f, 344.0f, 345.0f, 346.0f, 347.0f, 348.0f,
                349.0f, 350.0f, 351.0f, 352.0f, 353.0f, 354.0f,
                355.0f, 356.0f, 357.0f, 358.0f, 359.0f, 360.0f,
                361.0f, 362.0f, 363.0f, 364.0f, 365.0f, 366.0f,
                367.0f, 368.0f, 369.0f, 370.0f, 371.0f, 372.0f,
                373.0f, 374.0f, 375.0f, 376.0f, 377.0f, 378.0f,
                379.0f, 380.0f, 381.0f, 382.0f, 383.0f, 384.0f,
                385.0f, 386.0f, 387.0f, 388.0f, 389.0f, 390.0f,
                391.0f, 392.0f, 393.0f, 394.0f, 395.0f, 396.0f,
                397.0f, 398.0f, 399.0f, 400.0f, 401.0f, 402.0f,
                403.0f, 404.0f, 405.0f, 406.0f, 407.0f, 408.0f,
                409.0f, 410.0f, 411.0f, 412.0f, 413.0f, 414.0f,
                415.0f, 416.0f, 417.0f, 418.0f, 419.0f, 420.0f,
                421.0f, 422.0f, 423.0f, 424.0f, 425.0f, 426.0f,
                427.0f, 428.0f, 429.0f, 430.0f, 431.0f, 432.0f,
                433.0f, 434.0f, 435.0f, 436.0f, 437.0f, 438.0f,
                439.0f, 440.0f, 441.0f, 442.0f, 443.0f, 444.0f,
                445.0f, 446.0f, 447.0f, 448.0f, 449.0f, 450.0f,
                451.0f, 452.0f, 453.0f, 454.0f, 455.0f, 456.0f,
                457.0f, 458.0f, 459.0f, 460.0f, 461.0f, 462.0f,
                463.0f, 464.0f, 465.0f, 466.0f, 467.0f, 468.0f,
                469.0f, 470.0f, 471.0f, 472.0f, 473.0f, 474.0f,
                475.0f, 476.0f, 477.0f, 478.0f, 479.0f, 480.0f,
                481.0f, 482.0f, 483.0f, 484.0f, 485.0f, 486.0f,
                487.0f, 488.0f, 489.0f, 490.0f, 491.0f, 492.0f,
                493.0f, 494.0f, 495.0f, 496.0f, 497.0f, 498.0f,
                499.0f, 500.0f, 501.0f, 502.0f, 503.0f, 504.0f,
                505.0f, 506.0f, 507.0f, 508.0f, 509.0f, 510.0f,
                511.0f, 512.0f, 513.0f, 514.0f, 515.0f, 516.0f,
                517.0f, 518.0f, 519.0f, 520.0f, 521.0f, 522.0f,
                523.0f, 524.0f, 525.0f, 526.0f, 527.0f, 528.0f,
                529.0f, 530.0f, 531.0f, 532.0f, 533.0f, 534.0f,
                535.0f, 536.0f, 537.0f, 538.0f, 539.0f, 540.0f,
                541.0f, 542.0f, 543.0f, 544.0f, 545.0f, 546.0f,
                547.0f, 548.0f, 549.0f, 550.0f, 551.0f, 552.0f,
                553.0f, 554.0f, 555.0f, 556.0f, 557.0f, 558.0f,
                559.0f, 560.0f, 561.0f, 562.0f, 563.0f, 564.0f,
                565.0f, 566.0f, 567.0f, 568.0f, 569.0f, 570.0f,
                571.0f, 572.0f, 573.0f, 574.0f, 575.0f, 576.0f,
                577.0f, 578.0f, 579.0f, 580.0f, 581.0f, 582.0f,
                583.0f, 584.0f, 585.0f, 586.0f, 587.0f, 588.0f,
                589.0f, 590.0f, 591.0f, 592.0f, 593.0f, 594.0f,
                595.0f, 596.0f, 597.0f, 598.0f, 599.0f, 600.0f,
                601.0f, 602.0f, 603.0f, 604.0f, 605.0f, 606.0f,
                607.0f, 608.0f, 609.0f, 610.0f, 611.0f, 612.0f,
                613.0f, 614.0f, 615.0f, 616.0f, 617.0f, 618.0f,
                619.0f, 620.0f, 621.0f, 622.0f, 623.0f, 624.0f,
                625.0f, 626.0f, 627.0f, 628.0f, 629.0f, 630.0f,
                631.0f, 632.0f, 633.0f, 634.0f, 635.0f, 636.0f,
                637.0f, 638.0f, 639.0f, 640.0f, 641.0f, 642.0f,
                643.0f, 644.0f, 645.0f, 646.0f, 647.0f,
            }),
            3,
            getValues<T>({
                0.0f, 3.0f, 6.0f, 9.0f,
                12.0f, 15.0f, 54.0f, 57.0f,
                60.0f, 63.0f, 66.0f, 69.0f,
                108.0f, 111.0f, 114.0f, 117.0f,
                120.0f, 123.0f, 162.0f, 165.0f, 168.0f, 171.0f, 174.0f, 177.0f, 216.0f, 219.0f, 222.0f, 225.0f, 228.0f, 231.0f, 270.0f, 273.0f, 276.0f, 279.0f, 282.0f, 285.0f, 1.0f, 4.0f, 7.0f, 10.0f, 13.0f, 16.0f, 55.0f, 58.0f, 61.0f, 64.0f, 67.0f, 70.0f, 109.0f, 112.0f, 115.0f, 118.0f, 121.0f, 124.0f, 163.0f, 166.0f, 169.0f, 172.0f, 175.0f, 178.0f, 217.0f, 220.0f, 223.0f, 226.0f, 229.0f, 232.0f, 271.0f, 274.0f, 277.0f, 280.0f, 283.0f, 286.0f, 2.0f, 5.0f, 8.0f, 11.0f, 14.0f, 17.0f, 56.0f, 59.0f, 62.0f, 65.0f, 68.0f, 71.0f, 110.0f, 113.0f, 116.0f, 119.0f, 122.0f, 125.0f, 164.0f, 167.0f, 170.0f, 173.0f, 176.0f, 179.0f, 218.0f, 221.0f, 224.0f, 227.0f, 230.0f, 233.0f, 272.0f, 275.0f, 278.0f, 281.0f, 284.0f, 287.0f, 18.0f, 21.0f, 24.0f, 27.0f, 30.0f, 33.0f, 72.0f, 75.0f, 78.0f, 81.0f, 84.0f, 87.0f, 126.0f, 129.0f, 132.0f, 135.0f, 138.0f, 141.0f, 180.0f, 183.0f, 186.0f, 189.0f, 192.0f, 195.0f, 234.0f, 237.0f, 240.0f, 243.0f, 246.0f, 249.0f, 288.0f, 291.0f, 294.0f, 297.0f, 300.0f, 303.0f, 19.0f, 22.0f, 25.0f, 28.0f, 31.0f, 34.0f, 73.0f, 76.0f, 79.0f, 82.0f, 85.0f, 88.0f, 127.0f, 130.0f, 133.0f, 136.0f, 139.0f, 142.0f, 181.0f, 184.0f, 187.0f, 190.0f, 193.0f, 196.0f, 235.0f, 238.0f, 241.0f, 244.0f, 247.0f, 250.0f, 289.0f, 292.0f, 295.0f, 298.0f, 301.0f, 304.0f, 20.0f, 23.0f, 26.0f, 29.0f, 32.0f, 35.0f, 74.0f, 77.0f, 80.0f, 83.0f, 86.0f, 89.0f, 128.0f, 131.0f, 134.0f, 137.0f, 140.0f, 143.0f, 182.0f, 185.0f, 188.0f, 191.0f, 194.0f, 197.0f, 236.0f, 239.0f, 242.0f, 245.0f, 248.0f, 251.0f, 290.0f, 293.0f, 296.0f, 299.0f, 302.0f, 305.0f, 36.0f, 39.0f, 42.0f, 45.0f, 48.0f, 51.0f, 90.0f, 93.0f, 96.0f, 99.0f, 102.0f, 105.0f, 144.0f, 147.0f, 150.0f, 153.0f, 156.0f, 159.0f, 198.0f, 201.0f, 204.0f, 207.0f, 210.0f, 213.0f, 252.0f, 255.0f, 258.0f, 261.0f, 264.0f, 267.0f, 306.0f, 309.0f, 312.0f, 315.0f, 318.0f, 321.0f, 37.0f, 40.0f, 43.0f, 46.0f, 49.0f, 52.0f, 91.0f, 94.0f, 97.0f, 100.0f, 103.0f, 106.0f, 145.0f, 148.0f, 151.0f, 154.0f, 157.0f, 160.0f, 199.0f, 202.0f, 205.0f, 208.0f, 211.0f, 214.0f, 253.0f, 256.0f, 259.0f, 262.0f, 265.0f, 268.0f, 307.0f, 310.0f, 313.0f, 316.0f, 319.0f, 322.0f, 38.0f, 41.0f, 44.0f, 47.0f, 50.0f, 53.0f, 92.0f, 95.0f, 98.0f, 101.0f, 104.0f, 107.0f, 146.0f, 149.0f, 152.0f, 155.0f, 158.0f, 161.0f, 200.0f, 203.0f, 206.0f, 209.0f, 212.0f, 215.0f, 254.0f, 257.0f, 260.0f, 263.0f, 266.0f, 269.0f, 308.0f, 311.0f, 314.0f, 317.0f, 320.0f, 323.0f, 324.0f, 327.0f, 330.0f, 333.0f, 336.0f, 339.0f, 378.0f, 381.0f, 384.0f, 387.0f, 390.0f, 393.0f, 432.0f, 435.0f, 438.0f, 441.0f, 444.0f, 447.0f, 486.0f, 489.0f, 492.0f, 495.0f, 498.0f, 501.0f, 540.0f, 543.0f, 546.0f, 549.0f, 552.0f, 555.0f, 594.0f, 597.0f, 600.0f, 603.0f, 606.0f, 609.0f, 325.0f, 328.0f, 331.0f, 334.0f, 337.0f, 340.0f, 379.0f, 382.0f, 385.0f, 388.0f, 391.0f, 394.0f, 433.0f, 436.0f, 439.0f, 442.0f, 445.0f, 448.0f, 487.0f, 490.0f, 493.0f, 496.0f, 499.0f, 502.0f, 541.0f, 544.0f, 547.0f, 550.0f, 553.0f, 556.0f, 595.0f, 598.0f, 601.0f, 604.0f, 607.0f, 610.0f, 326.0f, 329.0f, 332.0f, 335.0f, 338.0f, 341.0f, 380.0f, 383.0f, 386.0f, 389.0f, 392.0f, 395.0f, 434.0f, 437.0f, 440.0f, 443.0f, 446.0f, 449.0f, 488.0f, 491.0f, 494.0f, 497.0f, 500.0f, 503.0f, 542.0f, 545.0f, 548.0f, 551.0f, 554.0f, 557.0f, 596.0f, 599.0f, 602.0f, 605.0f, 608.0f, 611.0f, 342.0f, 345.0f, 348.0f, 351.0f, 354.0f, 357.0f, 396.0f, 399.0f, 402.0f, 405.0f, 408.0f, 411.0f, 450.0f, 453.0f, 456.0f, 459.0f, 462.0f, 465.0f, 504.0f, 507.0f, 510.0f, 513.0f, 516.0f, 519.0f, 558.0f, 561.0f, 564.0f, 567.0f, 570.0f, 573.0f, 612.0f, 615.0f, 618.0f, 621.0f, 624.0f, 627.0f, 343.0f, 346.0f, 349.0f, 352.0f, 355.0f, 358.0f, 397.0f, 400.0f, 403.0f, 406.0f, 409.0f, 412.0f, 451.0f, 454.0f, 457.0f, 460.0f, 463.0f, 466.0f, 505.0f, 508.0f, 511.0f, 514.0f, 517.0f, 520.0f, 559.0f, 562.0f, 565.0f, 568.0f, 571.0f, 574.0f, 613.0f, 616.0f, 619.0f, 622.0f, 625.0f, 628.0f, 344.0f, 347.0f, 350.0f, 353.0f, 356.0f, 359.0f, 398.0f, 401.0f, 404.0f, 407.0f, 410.0f, 413.0f, 452.0f, 455.0f, 458.0f, 461.0f, 464.0f, 467.0f, 506.0f, 509.0f, 512.0f, 515.0f, 518.0f, 521.0f, 560.0f, 563.0f, 566.0f, 569.0f, 572.0f, 575.0f, 614.0f, 617.0f, 620.0f, 623.0f, 626.0f, 629.0f, 360.0f, 363.0f, 366.0f, 369.0f, 372.0f, 375.0f, 414.0f, 417.0f, 420.0f, 423.0f, 426.0f, 429.0f, 468.0f, 471.0f, 474.0f, 477.0f, 480.0f, 483.0f, 522.0f, 525.0f, 528.0f, 531.0f, 534.0f, 537.0f, 576.0f, 579.0f, 582.0f, 585.0f, 588.0f, 591.0f, 630.0f, 633.0f, 636.0f, 639.0f, 642.0f, 645.0f, 361.0f, 364.0f, 367.0f, 370.0f, 373.0f, 376.0f, 415.0f, 418.0f, 421.0f, 424.0f, 427.0f, 430.0f, 469.0f, 472.0f, 475.0f, 478.0f, 481.0f, 484.0f, 523.0f, 526.0f, 529.0f, 532.0f, 535.0f, 538.0f, 577.0f, 580.0f, 583.0f, 586.0f, 589.0f, 592.0f, 631.0f, 634.0f, 637.0f, 640.0f, 643.0f, 646.0f, 362.0f, 365.0f, 368.0f, 371.0f, 374.0f, 377.0f, 416.0f, 419.0f, 422.0f, 425.0f, 428.0f, 431.0f, 470.0f, 473.0f, 476.0f, 479.0f, 482.0f, 485.0f, 524.0f, 527.0f, 530.0f, 533.0f, 536.0f, 539.0f, 578.0f, 581.0f, 584.0f, 587.0f, 590.0f, 593.0f, 632.0f, 635.0f, 638.0f, 641.0f, 644.0f, 647.0f,

            })
        },


*/
    };
    return result;
}

template<typename T>
std::vector<ReorgYoloParams<T>> generateInvalidParams() {
    static const std::vector<ReorgYoloParams<T>> result = {
        { // Feature < stride*stride
            tensor{1, 3, 4, 4},
            getValues<T>({
                0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f,
                11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f, 20.0f,
                21.0f, 22.0f, 23.0f, 24.0f, 25.0f, 26.0f, 27.0f, 28.0f, 29.0f, 30.0f,
                31.0f, 32.0f, 33.0f, 34.0f, 35.0f, 36.0f, 37.0f, 38.0f, 39.0f, 40.0f,
                41.0f, 42.0f, 43.0f, 44.0f, 45.0f, 46.0f, 47.0f,
            }),
            2,
            getValues<T>({}),
        },
        { // Height % stride != 0
            tensor{1, 4, 5, 4},
            getValues<T>({
                0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f,
                11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f, 20.0f,
                21.0f, 22.0f, 23.0f, 24.0f, 25.0f, 26.0f, 27.0f, 28.0f, 29.0f, 30.0f,
                31.0f, 32.0f, 33.0f, 34.0f, 35.0f, 36.0f, 37.0f, 38.0f, 39.0f, 40.0f,
                41.0f, 42.0f, 43.0f, 44.0f, 45.0f, 46.0f, 47.0f, 48.0f, 49.0f, 50.0f,
                51.0f, 52.0f, 53.0f, 54.0f, 55.0f, 56.0f, 57.0f, 58.0f, 59.0f, 60.0f,
                61.0f, 62.0f, 63.0f, 64.0f, 65.0f, 66.0f, 67.0f, 68.0f, 69.0f, 70.0f,
                71.0f, 72.0f, 73.0f, 74.0f, 75.0f, 76.0f, 77.0f, 78.0f, 79.0f,
            }),
            2,
            getValues<T>({}),
        },
        { // Width % stride != 0
            tensor{1, 4, 4, 5},
            getValues<T>({
                0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f,
                11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f, 20.0f,
                21.0f, 22.0f, 23.0f, 24.0f, 25.0f, 26.0f, 27.0f, 28.0f, 29.0f, 30.0f,
                31.0f, 32.0f, 33.0f, 34.0f, 35.0f, 36.0f, 37.0f, 38.0f, 39.0f, 40.0f,
                41.0f, 42.0f, 43.0f, 44.0f, 45.0f, 46.0f, 47.0f, 48.0f, 49.0f, 50.0f,
                51.0f, 52.0f, 53.0f, 54.0f, 55.0f, 56.0f, 57.0f, 58.0f, 59.0f, 60.0f,
                61.0f, 62.0f, 63.0f, 64.0f, 65.0f, 66.0f, 67.0f, 68.0f, 69.0f, 70.0f,
                71.0f, 72.0f, 73.0f, 74.0f, 75.0f, 76.0f, 77.0f, 78.0f, 79.0f,
            }),
            2,
            getValues<T>({}),
        },
    };
    return result;
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
    std::string operator()(const testing::TestParamInfo<ReorgYoloParamsWithLayout<T> > &param) {
        std::stringstream buf;
        ReorgYoloParams<T> p;
        format::type target_format;
        bool should_fail;
        std::tie(p, target_format, should_fail) = param.param;
        buf << "InputTensor=" << p.inputTensor.to_string()
            << ".stride=" << p.stride
            << ".TargetLayout=" << toString(target_format);
        return buf.str();
    }
};
};  // namespace

template<typename T>
struct reorg_yolo_test
        : public ::testing::TestWithParam<ReorgYoloParamsWithLayout<T> > {
public:
    void test() {
        ReorgYoloParams<T> params;
        format::type target_format;
        bool should_fail;
        std::tie(params, target_format, should_fail) = this->GetParam();

        if (should_fail) {
            ASSERT_THROW(run_test(params, target_format), std::invalid_argument);
        } else {
            ASSERT_NO_FATAL_FAILURE(run_test(params, target_format));
        }
    }

private:
    void run_test(const ReorgYoloParams<T> params, const format::type target_format) {
        const auto data_type = type_to_data_type<T>::value;
        const format::type plain_format = format::bfyx;

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

        auto out_mem = result.at("reorg_yolo_reordered").get_memory();
        cldnn::mem_lock<T> out_ptr(out_mem, get_test_stream());

        ASSERT_EQ(params.inputTensor.count(), out_ptr.size());
        ASSERT_EQ(params.expected.size(), out_ptr.size());
        std::cout << std::endl;
        for (size_t i = 0; i < params.expected.size(); ++i) {
            EXPECT_NEAR(params.expected[i], out_ptr[i], getError<T>()) << "format=" << target_format << ", i= " << i;
            //std::cout << (float)out_ptr[i] << ".0f, ";
        }
        std::cout << std::endl;
    }
};


using test_f32 = reorg_yolo_test<float>;
using test_f16 = reorg_yolo_test<half_t>;

TEST_P(test_f32, basic) {
    test();
}

TEST_P(test_f16, basic) {
    test();
}



INSTANTIATE_TEST_SUITE_P(reorg_yolo_f32,
                         test_f32,
                         ::testing::Combine(
                                 ::testing::ValuesIn(generateParams<float>()),
                                 ::testing::ValuesIn(dataFormats),
                                 ::testing::Values(false)),
                         PrintToStringParamName());

INSTANTIATE_TEST_SUITE_P(reorg_yolo_f16,
                         test_f16,
                         ::testing::Combine(
                                 ::testing::ValuesIn(generateParams<half_t>()),
                                 ::testing::ValuesIn(dataFormats),
                                 ::testing::Values(false)),
                         PrintToStringParamName());

INSTANTIATE_TEST_SUITE_P(reorg_yolo_invalid_input,
                         test_f32,
                         ::testing::Combine(
                                 ::testing::ValuesIn(generateInvalidParams<float>()),
                                 ::testing::Values(format::bfyx),
                                 ::testing::Values(true)),
                         PrintToStringParamName());
