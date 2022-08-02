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

inline void FUNC(planar_to_bfyx)(const uint dstPlanarIndex,
                                 const uint batch_num, const uint channel_num, const uint height, const uint width,
                                 uint* dstB, uint* dstC, uint* dstY, uint* dstX)
{
    if(dstPlanarIndex < width) {
        *dstB = 0;
        *dstC = 0;
        *dstY = 0;
        *dstX = dstPlanarIndex;
    } else if(dstPlanarIndex < height * width) {
        *dstB = 0;
        *dstC = 0;
        *dstY = dstPlanarIndex / width;
        *dstX = dstPlanarIndex % width;
    } else if(dstPlanarIndex < channel_num * height * width) {
        *dstB = 0;
        *dstC = dstPlanarIndex / (height * width);
        uint dstXY = dstPlanarIndex % (height * width);
        *dstY = dstXY / width;
        *dstX = dstXY % width;
    } else if(dstPlanarIndex >= channel_num * height * width) {
        *dstB = dstPlanarIndex / (channel_num * height * width);
        uint dstCXY = dstPlanarIndex % (channel_num * height * width);
        *dstC = dstCXY / (height * width);
        uint dstXY = dstCXY % (height * width);
        *dstY = dstXY / width;
        *dstX = dstXY % width;
    }
}

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
        uint dstB, dstC, dstY, dstX;
        FUNC_CALL(planar_to_bfyx)(dstPlanarIndex, B, OC, OH, OW, &dstB, &dstC, &dstY, &dstX);
        const uint dstIndex = OUTPUT_GET_INDEX(dstB, dstC, dstY, dstX);

        int oc = ic % ic_off; // 0, 0, 0, 0
        int offset = ic / ic_off; // 0, 1, 2, 3

        int rem = offset % STRIDE;  // 0, 1, 0, 1
        int div = offset / STRIDE;  // 0, 0, 1, 1

        int ow = iw * STRIDE + rem;
        int oh = ih * STRIDE + div;

        int srcPlanarIndex = b*ic_off*ih_off*iw_off + oc*ih_off*iw_off + oh*iw_off + ow;
        uint srcB, srcC, srcY, srcX;
        FUNC_CALL(planar_to_bfyx)(srcPlanarIndex, B, IC, IH, IW, &srcB, &srcC, &srcY, &srcX);
        const uint srcIndex = INPUT0_GET_INDEX(srcB, srcC, srcY, srcX);

        //printf("src %d dstPlanar %d dst %d\n", srcIndex, dstPlanarIndex, dstIndex);

        output[/*dstPlanarIndex*/dstIndex] = input[/*srcPlanarIndex*/srcIndex];
    }



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
