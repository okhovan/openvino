// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/common.cl"
#include "include/batch_headers/data_types.cl"
#include "include/batch_headers/fetch_data.cl"

#if OUTPUT_LAYOUT_BFYX
    #define IW INPUT0_SIZES[0]
    #define IH INPUT0_SIZES[1]
    #define IC INPUT0_SIZES[2]
    #define B  INPUT0_SIZES[3]

#elif OUTPUT_LAYOUT_YXFB
    #define IW INPUT0_SIZES[3]
    #define IH INPUT0_SIZES[2]
    #define IC INPUT0_SIZES[1]
    #define B  INPUT0_SIZES[0]
#else
    #define IW INPUT0_SIZES[0]
    #define IH INPUT0_SIZES[1]
    #define IC INPUT0_SIZES[2]
    #define B  INPUT0_SIZES[3]
#endif

#define ic_off (IC / (STRIDE * STRIDE))
#define ih_off (IH * STRIDE)
#define iw_off (IW * STRIDE)

KERNEL (reorg_yolo_ref)(const __global UNIT_TYPE* input, __global UNIT_TYPE* output)
{
    int ic = get_global_id(2);
    int ih = get_global_id(1);
    int iw = get_global_id(0);

    const uint OC = IC * STRIDE * STRIDE;
    const uint OH = IH / STRIDE;
    const uint OW = IW / STRIDE;


    for (int b = 0; b < B; b++) {
        const uint dstPlanarIndex = b*IC*IH*IW + ic*IH*IW + ih*IW + iw;

        uint dstB = b;
        uint dstC;
        uint dstY;
        uint dstX;

        if(dstPlanarIndex < OW) {
            dstC = 0;
            dstY = 0;
            dstX = dstPlanarIndex;
        } else if(dstPlanarIndex < OH * OW) {
            dstC = 0;
            dstY = dstPlanarIndex / OW;
            dstX = dstPlanarIndex % OW;
        } else if(dstPlanarIndex < OC * OH * OW) {
            dstC = dstPlanarIndex / (OH * OW);
            uint dstXY = dstPlanarIndex % (OH * OW);
            dstY = dstXY / OW;
            dstX = dstXY % OW;
        } else if(dstPlanarIndex >= OC * OH * OW) {
            dstB = dstPlanarIndex / (OC * OH * OW);
            uint dstCXY = dstPlanarIndex % (OC * OH * OW);
            dstC = dstCXY / (OH * OW);
            uint dstXY = dstCXY % (OH * OW);
            dstY = dstXY / OW;
            dstX = dstXY % OW;
        }
        const uint dstIndex = OUTPUT_GET_INDEX(dstB, dstC, dstY, dstX);

        int oc = ic % ic_off; // 0, 0, 0, 0
        int offset = ic / ic_off; // 0, 1, 2, 3

        int rem = offset % STRIDE;  // 0, 1, 0, 1
        int div = offset / STRIDE;  // 0, 0, 1, 1

        int ow = iw * STRIDE + rem;
        int oh = ih * STRIDE + div;

        int srcIndex = b*ic_off*ih_off*iw_off + oc*ih_off*iw_off + oh*iw_off + ow;

        printf("src %d dstPlanar %d dst %d\n",
               srcIndex, dstPlanarIndex, dstIndex);

        output[/*dstPlanarIndex*/dstIndex] = input[srcIndex];
    }




/*
    const uint new_number_of_channels = IC * STRIDE * STRIDE;
    const uint new_heigth = IH / STRIDE;
    const uint new_width = IW / STRIDE;

    const uint b = 0;
uint cnt = 0;
    for(uint channel = 0; channel < IC; ++channel) {
        for(uint input_x = 0; input_x < IW; input_x += STRIDE) {
            for(uint input_y = 0; input_y < IH; input_y += STRIDE) {
                for(uint i = 0; i < STRIDE; ++i) {
                    for(uint j = 0; j < STRIDE; ++j) {
                        const uint input_idx =  INPUT0_GET_INDEX(b, channel, input_y + j, input_x + i);

                        const uint new_channel = channel*STRIDE*STRIDE + i*STRIDE + j;

                        const uint output_idx = OUTPUT_GET_INDEX(b, new_channel, input_y/STRIDE, input_x/STRIDE);

                        output[output_idx] = input[input_idx];
                        printf("%d: %d -> %d, %d %d %d %d - %d -> %d = %f\n",
                                 cnt,
                                 channel, new_channel,
                                 input_y, input_x, j, i,
                                 input_idx, output_idx, input[input_idx]);
                        ++cnt;
                    }
                }
            }
        }
    }
*/

//#if OUTPUT_LAYOUT_BFYX
//    int ic = get_global_id(2);
//    int ih = get_global_id(1);
//    int iw = get_global_id(0);
//        for (int b = 0; b < B; b++) {
//        int dstIndex = b*IC*IH*IW + ic*IH*IW + ih*IW + iw;
//
////ic_off = 1
////ih_off = 4
////iw_off 4
//
//        int oc = ic % ic_off; // 0, 0, 0, 0
//        int offset = ic / ic_off; // 0, 1, 2, 3
//
//        int rem = offset % STRIDE;  // 0, 1, 0, 1
//        int div = offset / STRIDE;  // 0, 0, 1, 1
//
//        int ow = iw * STRIDE + rem;
//        int oh = ih * STRIDE + div;
//
//        int srcIndex = b*ic_off*ih_off*iw_off + oc*ih_off*iw_off + oh*iw_off + ow;
//        //int newSrcIndex = GET_DATA_INDEX_RAW(OUTPUT, b, oc, oh, ow);
//
//        //printf("src %d -> dst %d \n", srcIndex, dstIndex);
//
//        output[dstIndex] = input[srcIndex];
//    }
//#elif OUTPUT_LAYOUT_YXFB
//    int ic = get_global_id(0) / B;
//    int ib = get_global_id(0) % B;
//    int ih = get_global_id(2);
//    int iw = get_global_id(1);
//    for (int b = 0; b < B; b++) {
//        int dstIndex = ib + ic*B + ih*IC*B + iw*IH*IC*B;
//
//        int oc = ic % ic_off;
//        int offset = ic / ic_off;
//
//        int ow = iw * STRIDE + offset % STRIDE;
//        int oh = ih * STRIDE + offset / STRIDE;
//
//        int srcIndex = b*ic_off*ih_off*iw_off + oc*ih_off*iw_off + oh*iw_off + ow;
//
//        output[dstIndex] = input[srcIndex];
//    }
//#endif
    

}
