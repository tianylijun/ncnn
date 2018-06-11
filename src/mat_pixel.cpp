// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "mat.h"
#include <limits.h>
#include <algorithm>
#if __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON

namespace ncnn
{

static Mat from_rgb(const unsigned char* rgb, int w, int h)
{
    Mat m(w, h, 3);
    if (m.empty())
        return m;

    float* ptr0 = m.channel(0);
    float* ptr1 = m.channel(1);
    float* ptr2 = m.channel(2);

    int size = w * h;

#if __ARM_NEON
    int nn = size >> 3;
    int remain = size - (nn << 3);
#else
    int remain = size;
#endif // __ARM_NEON

#if __ARM_NEON
#if __aarch64__
    for (; nn>0; nn--)
    {
        uint8x8x3_t _rgb = vld3_u8(rgb);
        uint16x8_t _r16 = vmovl_u8(_rgb.val[0]);
        uint16x8_t _g16 = vmovl_u8(_rgb.val[1]);
        uint16x8_t _b16 = vmovl_u8(_rgb.val[2]);

        float32x4_t _rlow = vcvtq_f32_u32(vmovl_u16(vget_low_u16(_r16)));
        float32x4_t _rhigh = vcvtq_f32_u32(vmovl_u16(vget_high_u16(_r16)));
        float32x4_t _glow = vcvtq_f32_u32(vmovl_u16(vget_low_u16(_g16)));
        float32x4_t _ghigh = vcvtq_f32_u32(vmovl_u16(vget_high_u16(_g16)));
        float32x4_t _blow = vcvtq_f32_u32(vmovl_u16(vget_low_u16(_b16)));
        float32x4_t _bhigh = vcvtq_f32_u32(vmovl_u16(vget_high_u16(_b16)));

        vst1q_f32(ptr0, _rlow);
        vst1q_f32(ptr0+4, _rhigh);
        vst1q_f32(ptr1, _glow);
        vst1q_f32(ptr1+4, _ghigh);
        vst1q_f32(ptr2, _blow);
        vst1q_f32(ptr2+4, _bhigh);

        rgb += 3*8;
        ptr0 += 8;
        ptr1 += 8;
        ptr2 += 8;
    }
#else
    if (nn > 0)
    {
        asm volatile(
            "0:                             \n"
            "pld        [%1, #256]          \n"
            "vld3.u8    {d0-d2}, [%1]!      \n"
            "vmovl.u8   q8, d0              \n"
            "vmovl.u8   q9, d1              \n"
            "vmovl.u8   q10, d2             \n"
            "vmovl.u16  q0, d16             \n"
            "vmovl.u16  q1, d17             \n"
            "vmovl.u16  q2, d18             \n"
            "vmovl.u16  q3, d19             \n"
            "vmovl.u16  q8, d20             \n"
            "vmovl.u16  q9, d21             \n"
            "vcvt.f32.u32   q0, q0          \n"
            "vcvt.f32.u32   q1, q1          \n"
            "vcvt.f32.u32   q2, q2          \n"
            "vcvt.f32.u32   q3, q3          \n"
            "vcvt.f32.u32   q8, q8          \n"
            "subs       %0, #1              \n"
            "vst1.f32   {d0-d3}, [%2 :128]! \n"
            "vcvt.f32.u32   q9, q9          \n"
            "vst1.f32   {d4-d7}, [%3 :128]! \n"
            "vst1.f32   {d16-d19}, [%4 :128]!\n"
            "bne        0b                  \n"
            : "=r"(nn),     // %0
            "=r"(rgb),    // %1
            "=r"(ptr0),   // %2
            "=r"(ptr1),   // %3
            "=r"(ptr2)    // %4
            : "0"(nn),
            "1"(rgb),
            "2"(ptr0),
            "3"(ptr1),
            "4"(ptr2)
            : "cc", "memory", "q0", "q1", "q2", "q3", "q8", "q9", "q10"
        );
    }
#endif // __aarch64__
#endif // __ARM_NEON
    for (; remain>0; remain--)
    {
        *ptr0 = rgb[0];
        *ptr1 = rgb[1];
        *ptr2 = rgb[2];

        rgb += 3;
        ptr0++;
        ptr1++;
        ptr2++;
    }

    return m;
}

static void to_rgb(const Mat& m, unsigned char* rgb)
{
    const float* ptr0 = m.channel(0);
    const float* ptr1 = m.channel(1);
    const float* ptr2 = m.channel(2);

    int size = m.w * m.h;

#define SATURATE_CAST_UCHAR(X) (unsigned char)::std::min(::std::max((int)(X), 0), 255);

    int remain = size;

    for (; remain>0; remain--)
    {
        rgb[0] = SATURATE_CAST_UCHAR(*ptr0);
        rgb[1] = SATURATE_CAST_UCHAR(*ptr1);
        rgb[2] = SATURATE_CAST_UCHAR(*ptr2);

        rgb += 3;
        ptr0++;
        ptr1++;
        ptr2++;
    }

#undef SATURATE_CAST_UCHAR
}

static Mat from_gray(const unsigned char* gray, int w, int h)
{
    Mat m(w, h, 1);
    if (m.empty())
        return m;

    float* ptr = m;

    int size = w * h;

#if __ARM_NEON
    int nn = size >> 4;
    int remain = size - (nn << 4);
#else
    int remain = size;
#endif // __ARM_NEON

#if __ARM_NEON
#if __aarch64__
    for (; nn>0; nn--)
    {
        uint8x16_t _gray = vld1q_u8(gray);
        uint16x8_t _gray16_0 = vmovl_u8(vget_low_u8(_gray));
        uint16x8_t _gray16_1 = vmovl_u8(vget_high_u8(_gray));

        float32x4_t _graylow_0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(_gray16_0)));
        float32x4_t _grayhigh_0 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(_gray16_0)));
        float32x4_t _graylow_1 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(_gray16_1)));
        float32x4_t _grayhigh_1 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(_gray16_1)));

        vst1q_f32(ptr, _graylow_0);
        vst1q_f32(ptr+4, _grayhigh_0);
        vst1q_f32(ptr+8, _graylow_1);
        vst1q_f32(ptr+12, _grayhigh_1);

        gray += 16;
        ptr += 16;
    }
#else
    if (nn > 0)
    {
        asm volatile(
            "0:                             \n"
            "pld        [%1, #128]          \n"
            "vld1.u8    {d0,d1}, [%1]!      \n"
            "vmovl.u8   q8, d0              \n"
            "vmovl.u8   q9, d1              \n"
            "vmovl.u16  q0, d16             \n"
            "vmovl.u16  q1, d17             \n"
            "vmovl.u16  q2, d18             \n"
            "vmovl.u16  q3, d19             \n"
            "vcvt.f32.u32   q0, q0          \n"
            "vcvt.f32.u32   q1, q1          \n"
            "vcvt.f32.u32   q2, q2          \n"
            "vcvt.f32.u32   q3, q3          \n"
            "subs       %0, #1              \n"
            "vst1.f32   {d0-d3}, [%2 :128]! \n"
            "vst1.f32   {d4-d7}, [%2 :128]! \n"
            "bne        0b                  \n"
            : "=r"(nn),     // %0
            "=r"(gray),   // %1
            "=r"(ptr)     // %2
            : "0"(nn),
            "1"(gray),
            "2"(ptr)
            : "cc", "memory", "q0", "q1", "q2", "q3", "q8", "q9"
        );
    }
#endif // __aarch64__
#endif // __ARM_NEON
    for (; remain>0; remain--)
    {
        *ptr = *gray;

        gray++;
        ptr++;
    }

    return m;
}

static void to_gray(const Mat& m, unsigned char* gray)
{
    const float* ptr = m;

    int size = m.w * m.h;

#define SATURATE_CAST_UCHAR(X) (unsigned char)::std::min(::std::max((int)(X), 0), 255);

    int remain = size;

    for (; remain>0; remain--)
    {
        *gray = SATURATE_CAST_UCHAR(*ptr);

        gray++;
        ptr++;
    }

#undef SATURATE_CAST_UCHAR
}

static Mat from_rgba(const unsigned char* rgba, int w, int h)
{
    Mat m(w, h, 4);
    if (m.empty())
        return m;

    float* ptr0 = m.channel(0);
    float* ptr1 = m.channel(1);
    float* ptr2 = m.channel(2);
    float* ptr3 = m.channel(3);

    int size = w * h;

#if __ARM_NEON
    int nn = size >> 3;
    int remain = size - (nn << 3);
#else
    int remain = size;
#endif // __ARM_NEON

#if __ARM_NEON
#if __aarch64__
    for (; nn>0; nn--)
    {
        uint8x8x4_t _rgba = vld4_u8(rgba);
        int16x8_t _r16 = vreinterpretq_s16_u16(vmovl_u8(_rgba.val[0]));
        int16x8_t _g16 = vreinterpretq_s16_u16(vmovl_u8(_rgba.val[1]));
        int16x8_t _b16 = vreinterpretq_s16_u16(vmovl_u8(_rgba.val[2]));
        int16x8_t _a16 = vreinterpretq_s16_u16(vmovl_u8(_rgba.val[3]));

        float32x4_t _rlow = vcvtq_f32_s32(vmovl_s16(vget_low_s16(_r16)));
        float32x4_t _rhigh = vcvtq_f32_s32(vmovl_s16(vget_high_s16(_r16)));
        float32x4_t _glow = vcvtq_f32_s32(vmovl_s16(vget_low_s16(_g16)));
        float32x4_t _ghigh = vcvtq_f32_s32(vmovl_s16(vget_high_s16(_g16)));
        float32x4_t _blow = vcvtq_f32_s32(vmovl_s16(vget_low_s16(_b16)));
        float32x4_t _bhigh = vcvtq_f32_s32(vmovl_s16(vget_high_s16(_b16)));
        float32x4_t _alow = vcvtq_f32_s32(vmovl_s16(vget_low_s16(_a16)));
        float32x4_t _ahigh = vcvtq_f32_s32(vmovl_s16(vget_high_s16(_a16)));

        vst1q_f32(ptr0, _rlow);
        vst1q_f32(ptr0+4, _rhigh);
        vst1q_f32(ptr1, _glow);
        vst1q_f32(ptr1+4, _ghigh);
        vst1q_f32(ptr2, _blow);
        vst1q_f32(ptr2+4, _bhigh);
        vst1q_f32(ptr3, _alow);
        vst1q_f32(ptr3+4, _ahigh);

        rgba += 4*8;
        ptr0 += 8;
        ptr1 += 8;
        ptr2 += 8;
        ptr3 += 8;
    }
#else
    if (nn > 0)
    {
        asm volatile(
            "0:                             \n"
            "pld        [%1, #256]          \n"
            "vld4.u8    {d0-d3}, [%1]!      \n"
            "vmovl.u8   q8, d0              \n"
            "vmovl.u8   q9, d1              \n"
            "vmovl.u8   q10, d2             \n"
            "vmovl.u8   q11, d3             \n"
            "vmovl.u16  q0, d16             \n"
            "vmovl.u16  q1, d17             \n"
            "vmovl.u16  q2, d18             \n"
            "vmovl.u16  q3, d19             \n"
            "vmovl.u16  q8, d20             \n"
            "vmovl.u16  q9, d21             \n"
            "vmovl.u16  q10, d22            \n"
            "vmovl.u16  q11, d23            \n"
            "vcvt.f32.u32   q0, q0          \n"
            "vcvt.f32.u32   q1, q1          \n"
            "vcvt.f32.u32   q2, q2          \n"
            "vcvt.f32.u32   q3, q3          \n"
            "vcvt.f32.u32   q8, q8          \n"
            "vcvt.f32.u32   q9, q9          \n"
            "subs       %0, #1              \n"
            "vst1.f32   {d0-d3}, [%2 :128]! \n"
            "vcvt.f32.u32   q10, q10        \n"
            "vcvt.f32.u32   q11, q11        \n"
            "vst1.f32   {d4-d7}, [%3 :128]! \n"
            "vst1.f32   {d16-d19}, [%4 :128]!\n"
            "vst1.f32   {d20-d23}, [%5 :128]!\n"
            "bne        0b                  \n"
            : "=r"(nn),     // %0
            "=r"(rgba),   // %1
            "=r"(ptr0),   // %2
            "=r"(ptr1),   // %3
            "=r"(ptr2),   // %4
            "=r"(ptr3)    // %5
            : "0"(nn),
            "1"(rgba),
            "2"(ptr0),
            "3"(ptr1),
            "4"(ptr2),
            "5"(ptr3)
            : "cc", "memory", "q0", "q1", "q2", "q3", "q8", "q9", "q10", "q11"
        );
    }
#endif // __aarch64__
#endif // __ARM_NEON
    for (; remain>0; remain--)
    {
        *ptr0 = rgba[0];
        *ptr1 = rgba[1];
        *ptr2 = rgba[2];
        *ptr3 = rgba[3];

        rgba += 4;
        ptr0++;
        ptr1++;
        ptr2++;
        ptr3++;
    }

    return m;
}

static void to_rgba(const Mat& m, unsigned char* rgba)
{
    const float* ptr0 = m.channel(0);
    const float* ptr1 = m.channel(1);
    const float* ptr2 = m.channel(2);
    const float* ptr3 = m.channel(3);

    int size = m.w * m.h;

#define SATURATE_CAST_UCHAR(X) (unsigned char)::std::min(::std::max((int)(X), 0), 255);

    int remain = size;

    for (; remain>0; remain--)
    {
        rgba[0] = SATURATE_CAST_UCHAR(*ptr0);
        rgba[1] = SATURATE_CAST_UCHAR(*ptr1);
        rgba[2] = SATURATE_CAST_UCHAR(*ptr2);
        rgba[3] = SATURATE_CAST_UCHAR(*ptr3);

        rgba += 4;
        ptr0++;
        ptr1++;
        ptr2++;
        ptr3++;
    }

#undef SATURATE_CAST_UCHAR
}

/*
|R|   | 298    0     409 | | Y - 16  |
|G| = | 298  -100   -208 | | U - 128 |
|B|   | 298   516     0  | | V - 128 |

R = (298*(Y-16)+409*(V-128)+128)>>8
G = (298*(Y-16)-100*(U-128)-208*(V-128)+128)>>8
B = (298*(Y-16)+516*(U-128)+128)>>8

Y = (( 66 * R + 129 * G +  25 * B + 128) >> 8) +  16
U = ((-38 * R -  74 * G + 112 * B + 128) >> 8) + 128
V = ((112 * R -  94 * G -  18 * B + 128) >> 8) + 128
*/
void from_nv122rgb(const unsigned char* yuv, unsigned w, unsigned h, unsigned stride, unsigned roiX, unsigned roiY, unsigned roiW, unsigned roiH, unsigned char* pDst, unsigned bgrFlag)
{
    unsigned i = 0, j = 0;
    unsigned roiWDiv16, roiWHas8, roiWLeft;
    unsigned offsetH = 0, offsetW = 0;
    //printf("[%d %d %d   %d %d %d %d   %d]\n", w, h, stride, roiX, roiY, roiW, roiH, bgrFlag);
    const unsigned char * y  = yuv + roiX + roiY*stride;
    const unsigned char * uv = yuv + stride * h + (roiY>>1)*stride + ((roiX>>1)<<1);

    if (0 != (roiY&1)) offsetH = 1;
    if (0 != (roiX&1)) offsetW = 1;

    roiWDiv16 = (roiW - offsetW)>>4;
    roiWHas8  = (roiW - offsetW)&8;
    roiWLeft  = (roiW - offsetW)&7;

    //printf("[%d %d %d %d %d]\n", offsetW, offsetH, roiWDiv16, roiWHas8, roiWLeft);

    int16x8_t vsrc16x8_16  = vdupq_n_s16(16);
    int16x8_t vsrc16x8_128 = vdupq_n_s16(128);
    int32x4_t vsrc32x4_128 = vdupq_n_s32(128);
    int32x4_t vsrc32x4_0   = vdupq_n_s32(0);
    int32x4_t vsrc32x4_255 = vdupq_n_s32(255);

    //#pragma omp parallel for
    for( j = 0; j < roiH; j++)
    {
        const unsigned char *pCurY  = y + j*stride;
        const unsigned char *pCurUV = uv + ((j+offsetH)/2)*stride;
        unsigned char *pDstCur      = pDst + j*roiW*3;

        if (offsetW) //odd point process separate
        {
            int Y, U, V, R, G, B, Y298;
            Y = ((int32_t)*pCurY) - 16;
            U = ((int32_t)*pCurUV) - 128;
            V = ((int32_t)*(pCurUV+1)) - 128;

            Y298 = 298*Y;
            R = (Y298 + 409*(V) + 128)>>8;
            G = (Y298 - 100*(U) - 208*(V)+128)>>8;
            B = (Y298 + 516*(U) + 128)>>8;

            if (R < 0) R = 0;
            if (R > 255) R = 255;
            if (G < 0) G = 0;
            if (G > 255) G = 255;
            if (B < 0) B = 0;
            if (B > 255) B = 255;

            if (bgrFlag)
            {
                *pDstCur++ = (unsigned char)B;
                *pDstCur++ = (unsigned char)G;
                *pDstCur++ = (unsigned char)R;
            }
            else
            {
                *pDstCur++ = (unsigned char)R;
                *pDstCur++ = (unsigned char)G;
                *pDstCur++ = (unsigned char)B;
            }

            pCurY++;
            pCurUV += 2;
        }

        for( i = 0; i < roiWDiv16; i++)
        {
            int32x4x3_t vsrc32x4x3_0; // LOW  RGB
            int32x4x3_t vsrc32x4x3_1; // HIGH RGB

            vsrc32x4x3_0.val[0] = vsrc32x4_128; //R
            //vsrc32x4x3_0.val[1] = vsrc32x4_128; //G
            //vsrc32x4x3_0.val[2] = vsrc32x4_128; //B

            vsrc32x4x3_1.val[0] = vsrc32x4_128; //R
            //vsrc32x4x3_1.val[1] = vsrc32x4_128; //G
            //vsrc32x4x3_1.val[2] = vsrc32x4_128; //B

            uint8x8_t  vsrc8x8_y    = vld1_u8(pCurY); // [y0y1y2y3y4y5y6u7]
            uint16x8_t vsrc16x8_y_u = vmovl_u8(vsrc8x8_y);
            int16x8_t  vsrc16x8_y   = vreinterpretq_s16_u16(vsrc16x8_y_u);
            vsrc16x8_y = vsubq_s16(vsrc16x8_y, vsrc16x8_16);

            uint8x8x2_t vsrc8x8x2_uv = vld2_u8(pCurUV); // [u0u1u2u3u4u5u6u7] [v0v1v2v3v4v5v6v7]
            uint8x8x2_t vsrc8x8x2_u  = vzip_u8(vsrc8x8x2_uv.val[0], vsrc8x8x2_uv.val[0]); //[u0u0u1u1u2u2u3u3] [u4u4u5u5u6u6u7u7]
            uint8x8x2_t vsrc8x8x2_v  = vzip_u8(vsrc8x8x2_uv.val[1], vsrc8x8x2_uv.val[1]); //[v0v0v1v1v2v2v3v3] [v4v4v5v5v6v6v7v7]
            uint16x8_t  vsrc16x8_u_u = vmovl_u8(vsrc8x8x2_u.val[0]); //[u0u0u1u1u2u2u3u3]
            uint16x8_t  vsrc16x8_v_u = vmovl_u8(vsrc8x8x2_v.val[0]); //[v0v0v1v1v2v2v3v3]
            int16x8_t   vsrc16x8_u   = vreinterpretq_s16_u16(vsrc16x8_u_u);
            int16x8_t   vsrc16x8_v   = vreinterpretq_s16_u16(vsrc16x8_v_u);
            vsrc16x8_u = vsubq_s16(vsrc16x8_u, vsrc16x8_128);
            vsrc16x8_v = vsubq_s16(vsrc16x8_v, vsrc16x8_128);

            //R   R = (298*(Y-16)+409*(V-128)+128)>>8
            vsrc32x4x3_0.val[0] = vmlal_n_s16(vsrc32x4x3_0.val[0], vget_low_s16(vsrc16x8_y),  298);
            vsrc32x4x3_1.val[0] = vmlal_n_s16(vsrc32x4x3_1.val[0], vget_high_s16(vsrc16x8_y), 298);

            vsrc32x4x3_0.val[1] = vsrc32x4x3_0.val[0]; //vmlal_n_s16(vsrc32x4x3_0.val[1], vget_low_s16(vsrc16x8_y),  298);
            vsrc32x4x3_1.val[1] = vsrc32x4x3_1.val[0]; //vmlal_n_s16(vsrc32x4x3_1.val[1], vget_high_s16(vsrc16x8_y), 298);

            vsrc32x4x3_0.val[2] = vsrc32x4x3_0.val[0]; //vmlal_n_s16(vsrc32x4x3_0.val[2], vget_low_s16(vsrc16x8_y),  298);
            vsrc32x4x3_1.val[2] = vsrc32x4x3_1.val[0]; //vmlal_n_s16(vsrc32x4x3_1.val[2], vget_high_s16(vsrc16x8_y), 298);

            vsrc32x4x3_0.val[0] = vmlal_n_s16(vsrc32x4x3_0.val[0], vget_low_s16(vsrc16x8_v),  409);
            vsrc32x4x3_1.val[0] = vmlal_n_s16(vsrc32x4x3_1.val[0], vget_high_s16(vsrc16x8_v), 409);
            //G G = (298*(Y-16)-100*(U-128)-208*(V-128)+128)>>8
            //vsrc32x4x3_0.val[1] = vsrc32x4x3_0.val[0]; //vmlal_n_s16(vsrc32x4x3_0.val[1], vget_low_s16(vsrc16x8_y),  298);
            //vsrc32x4x3_1.val[1] = vsrc32x4x3_1.val[0]; //vmlal_n_s16(vsrc32x4x3_1.val[1], vget_high_s16(vsrc16x8_y), 298);

            vsrc32x4x3_0.val[1] = vmlal_n_s16(vsrc32x4x3_0.val[1], vget_low_s16(vsrc16x8_u),  -100);
            vsrc32x4x3_1.val[1] = vmlal_n_s16(vsrc32x4x3_1.val[1], vget_high_s16(vsrc16x8_u), -100);

            vsrc32x4x3_0.val[1] = vmlal_n_s16(vsrc32x4x3_0.val[1], vget_low_s16(vsrc16x8_v),  -208);
            vsrc32x4x3_1.val[1] = vmlal_n_s16(vsrc32x4x3_1.val[1], vget_high_s16(vsrc16x8_v), -208);
            //B B = (298*(Y-16)+516*(U-128)+128)>>8
            //vsrc32x4x3_0.val[2] = vsrc32x4x3_0.val[0]; //vmlal_n_s16(vsrc32x4x3_0.val[2], vget_low_s16(vsrc16x8_y),  298);
            //vsrc32x4x3_1.val[2] = vsrc32x4x3_1.val[0]; //vmlal_n_s16(vsrc32x4x3_1.val[2], vget_high_s16(vsrc16x8_y), 298);

            vsrc32x4x3_0.val[2] = vmlal_n_s16(vsrc32x4x3_0.val[2], vget_low_s16(vsrc16x8_u),  516);
            vsrc32x4x3_1.val[2] = vmlal_n_s16(vsrc32x4x3_1.val[2], vget_high_s16(vsrc16x8_u), 516);

            //shift right
            vsrc32x4x3_0.val[0] = vshrq_n_s32(vsrc32x4x3_0.val[0], 8);
            vsrc32x4x3_1.val[0] = vshrq_n_s32(vsrc32x4x3_1.val[0], 8);

            vsrc32x4x3_0.val[1] = vshrq_n_s32(vsrc32x4x3_0.val[1], 8);
            vsrc32x4x3_1.val[1] = vshrq_n_s32(vsrc32x4x3_1.val[1], 8);

            vsrc32x4x3_0.val[2] = vshrq_n_s32(vsrc32x4x3_0.val[2], 8);
            vsrc32x4x3_1.val[2] = vshrq_n_s32(vsrc32x4x3_1.val[2], 8);

            uint32x4_t mask;

            mask = vcltq_s32(vsrc32x4x3_0.val[0], vsrc32x4_255);
            vsrc32x4x3_0.val[0]  = vbslq_s32(mask, vsrc32x4x3_0.val[0], vsrc32x4_255);
            mask = vcltq_s32(vsrc32x4x3_1.val[0], vsrc32x4_255);
            vsrc32x4x3_1.val[0]  = vbslq_s32(mask, vsrc32x4x3_1.val[0], vsrc32x4_255);

            mask = vcltq_s32(vsrc32x4x3_0.val[1], vsrc32x4_255);
            vsrc32x4x3_0.val[1]  = vbslq_s32(mask, vsrc32x4x3_0.val[1], vsrc32x4_255);
            mask = vcltq_s32(vsrc32x4x3_1.val[1], vsrc32x4_255);
            vsrc32x4x3_1.val[1]  = vbslq_s32(mask, vsrc32x4x3_1.val[1], vsrc32x4_255);

            mask = vcltq_s32(vsrc32x4x3_0.val[2], vsrc32x4_255);
            vsrc32x4x3_0.val[2]  = vbslq_s32(mask, vsrc32x4x3_0.val[2], vsrc32x4_255);
            mask = vcltq_s32(vsrc32x4x3_1.val[2], vsrc32x4_255);
            vsrc32x4x3_1.val[2]  = vbslq_s32(mask, vsrc32x4x3_1.val[2], vsrc32x4_255);

            mask = vcgtq_s32(vsrc32x4x3_0.val[0], vsrc32x4_0);
            vsrc32x4x3_0.val[0]  = vbslq_s32(mask, vsrc32x4x3_0.val[0], vsrc32x4_0);
            mask = vcgtq_s32(vsrc32x4x3_1.val[0], vsrc32x4_0);
            vsrc32x4x3_1.val[0]  = vbslq_s32(mask, vsrc32x4x3_1.val[0], vsrc32x4_0);

            mask = vcgtq_s32(vsrc32x4x3_0.val[1], vsrc32x4_0);
            vsrc32x4x3_0.val[1]  = vbslq_s32(mask, vsrc32x4x3_0.val[1], vsrc32x4_0);
            mask = vcgtq_s32(vsrc32x4x3_1.val[1], vsrc32x4_0);
            vsrc32x4x3_1.val[1]  = vbslq_s32(mask, vsrc32x4x3_1.val[1], vsrc32x4_0);

            mask = vcgtq_s32(vsrc32x4x3_0.val[2], vsrc32x4_0);
            vsrc32x4x3_0.val[2]  = vbslq_s32(mask, vsrc32x4x3_0.val[2], vsrc32x4_0);
            mask = vcgtq_s32(vsrc32x4x3_1.val[2], vsrc32x4_0);
            vsrc32x4x3_1.val[2]  = vbslq_s32(mask, vsrc32x4x3_1.val[2], vsrc32x4_0);

            //narrow 32 to 8
            uint8x8x3_t vRet8x8x3;
            uint32x4x3_t vsrc32x4x3_u_0, vsrc32x4x3_u_1;
            vsrc32x4x3_u_0.val[0] = vreinterpretq_u32_s32(vsrc32x4x3_0.val[0]);
            vsrc32x4x3_u_0.val[1] = vreinterpretq_u32_s32(vsrc32x4x3_0.val[1]);
            vsrc32x4x3_u_0.val[2] = vreinterpretq_u32_s32(vsrc32x4x3_0.val[2]);
            vsrc32x4x3_u_1.val[0] = vreinterpretq_u32_s32(vsrc32x4x3_1.val[0]);
            vsrc32x4x3_u_1.val[1] = vreinterpretq_u32_s32(vsrc32x4x3_1.val[1]);
            vsrc32x4x3_u_1.val[2] = vreinterpretq_u32_s32(vsrc32x4x3_1.val[2]);

            //R
            uint16x4_t vR16x4_0 = vmovn_u32(vsrc32x4x3_u_0.val[0]);
            uint16x4_t vR16x4_1 = vmovn_u32(vsrc32x4x3_u_1.val[0]);
            uint16x8_t vR16x8   = vcombine_u16(vR16x4_0, vR16x4_1);
            if (bgrFlag)
                vRet8x8x3.val[2] = vmovn_u16(vR16x8);
            else
                vRet8x8x3.val[0] = vmovn_u16(vR16x8);

            //G
            uint16x4_t vG16x4_0 = vmovn_u32(vsrc32x4x3_u_0.val[1]);
            uint16x4_t vG16x4_1 = vmovn_u32(vsrc32x4x3_u_1.val[1]);
            uint16x8_t vG16x8   = vcombine_u16(vG16x4_0, vG16x4_1);
            vRet8x8x3.val[1]    = vmovn_u16(vG16x8);

            //B
            uint16x4_t vB16x4_0 = vmovn_u32(vsrc32x4x3_u_0.val[2]);
            uint16x4_t vB16x4_1 = vmovn_u32(vsrc32x4x3_u_1.val[2]);
            uint16x8_t vB16x8   = vcombine_u16(vB16x4_0, vB16x4_1);
            if (bgrFlag)
                vRet8x8x3.val[0] = vmovn_u16(vB16x8);
            else
                vRet8x8x3.val[2] = vmovn_u16(vB16x8);

            vst3_u8(pDstCur, vRet8x8x3);
            pDstCur += 24;
            pCurY += 8;

            //next 8 elements
            vsrc32x4x3_0.val[0] = vsrc32x4_128; //R
            //vsrc32x4x3_0.val[1] = vsrc32x4_128; //G
            //vsrc32x4x3_0.val[2] = vsrc32x4_128; //B

            vsrc32x4x3_1.val[0] = vsrc32x4_128; //R
            //vsrc32x4x3_1.val[1] = vsrc32x4_128; //G
            //vsrc32x4x3_1.val[2] = vsrc32x4_128; //B

            vsrc8x8_y    = vld1_u8(pCurY); // [y8y9y10y11y12y13y14u15]
            vsrc16x8_y_u = vmovl_u8(vsrc8x8_y);
            vsrc16x8_y   = vreinterpretq_s16_u16(vsrc16x8_y_u);
            vsrc16x8_y   = vsubq_s16(vsrc16x8_y, vsrc16x8_16);

            vsrc16x8_u_u = vmovl_u8(vsrc8x8x2_u.val[1]); //[u4u4u5u5u6u6u7u7]
            vsrc16x8_v_u = vmovl_u8(vsrc8x8x2_v.val[1]); //[v4v4v5v5v6v6v7v7]
            vsrc16x8_u   = vreinterpretq_s16_u16(vsrc16x8_u_u);
            vsrc16x8_v   = vreinterpretq_s16_u16(vsrc16x8_v_u);
            vsrc16x8_u   = vsubq_s16(vsrc16x8_u, vsrc16x8_128);
            vsrc16x8_v   = vsubq_s16(vsrc16x8_v, vsrc16x8_128);

            //R
            vsrc32x4x3_0.val[0] = vmlal_n_s16(vsrc32x4x3_0.val[0], vget_low_s16(vsrc16x8_y),  298);
            vsrc32x4x3_1.val[0] = vmlal_n_s16(vsrc32x4x3_1.val[0], vget_high_s16(vsrc16x8_y), 298);

            vsrc32x4x3_0.val[1] = vsrc32x4x3_0.val[0]; //vmlal_n_s16(vsrc32x4x3_0.val[1], vget_low_s16(vsrc16x8_y),  298);
            vsrc32x4x3_1.val[1] = vsrc32x4x3_1.val[0]; //vmlal_n_s16(vsrc32x4x3_1.val[1], vget_high_s16(vsrc16x8_y), 298);

            vsrc32x4x3_0.val[2] = vsrc32x4x3_0.val[0]; //vmlal_n_s16(vsrc32x4x3_0.val[2], vget_low_s16(vsrc16x8_y),  298);
            vsrc32x4x3_1.val[2] = vsrc32x4x3_1.val[0]; //vmlal_n_s16(vsrc32x4x3_1.val[2], vget_high_s16(vsrc16x8_y), 298);


            vsrc32x4x3_0.val[0] = vmlal_n_s16(vsrc32x4x3_0.val[0], vget_low_s16(vsrc16x8_v),  409);
            vsrc32x4x3_1.val[0] = vmlal_n_s16(vsrc32x4x3_1.val[0], vget_high_s16(vsrc16x8_v), 409);
            //G
            //vsrc32x4x3_0.val[1] = vsrc32x4x3_0.val[0]; //vmlal_n_s16(vsrc32x4x3_0.val[1], vget_low_s16(vsrc16x8_y),  298);
            //vsrc32x4x3_1.val[1] = vsrc32x4x3_1.val[0]; //vmlal_n_s16(vsrc32x4x3_1.val[1], vget_high_s16(vsrc16x8_y), 298);

            vsrc32x4x3_0.val[1] = vmlal_n_s16(vsrc32x4x3_0.val[1], vget_low_s16(vsrc16x8_u),  -100);
            vsrc32x4x3_1.val[1] = vmlal_n_s16(vsrc32x4x3_1.val[1], vget_high_s16(vsrc16x8_u), -100);

            vsrc32x4x3_0.val[1] = vmlal_n_s16(vsrc32x4x3_0.val[1], vget_low_s16(vsrc16x8_v),  -208);
            vsrc32x4x3_1.val[1] = vmlal_n_s16(vsrc32x4x3_1.val[1], vget_high_s16(vsrc16x8_v), -208);
            //B
            //vsrc32x4x3_0.val[2] = vsrc32x4x3_0.val[0]; //vmlal_n_s16(vsrc32x4x3_0.val[2], vget_low_s16(vsrc16x8_y),  298);
            //vsrc32x4x3_1.val[2] = vsrc32x4x3_1.val[0]; //vmlal_n_s16(vsrc32x4x3_1.val[2], vget_high_s16(vsrc16x8_y), 298);

            vsrc32x4x3_0.val[2] = vmlal_n_s16(vsrc32x4x3_0.val[2], vget_low_s16(vsrc16x8_u),  516);
            vsrc32x4x3_1.val[2] = vmlal_n_s16(vsrc32x4x3_1.val[2], vget_high_s16(vsrc16x8_u), 516);

            //shift right
            vsrc32x4x3_0.val[0] = vshrq_n_s32(vsrc32x4x3_0.val[0], 8);
            vsrc32x4x3_1.val[0] = vshrq_n_s32(vsrc32x4x3_1.val[0], 8);

            vsrc32x4x3_0.val[1] = vshrq_n_s32(vsrc32x4x3_0.val[1], 8);
            vsrc32x4x3_1.val[1] = vshrq_n_s32(vsrc32x4x3_1.val[1], 8);

            vsrc32x4x3_0.val[2] = vshrq_n_s32(vsrc32x4x3_0.val[2], 8);
            vsrc32x4x3_1.val[2] = vshrq_n_s32(vsrc32x4x3_1.val[2], 8);

            mask = vcltq_s32(vsrc32x4x3_0.val[0], vsrc32x4_255);
            vsrc32x4x3_0.val[0]  = vbslq_s32(mask, vsrc32x4x3_0.val[0], vsrc32x4_255);
            mask = vcltq_s32(vsrc32x4x3_1.val[0], vsrc32x4_255);
            vsrc32x4x3_1.val[0]  = vbslq_s32(mask, vsrc32x4x3_1.val[0], vsrc32x4_255);

            mask = vcltq_s32(vsrc32x4x3_0.val[1], vsrc32x4_255);
            vsrc32x4x3_0.val[1]  = vbslq_s32(mask, vsrc32x4x3_0.val[1], vsrc32x4_255);
            mask = vcltq_s32(vsrc32x4x3_1.val[1], vsrc32x4_255);
            vsrc32x4x3_1.val[1]  = vbslq_s32(mask, vsrc32x4x3_1.val[1], vsrc32x4_255);

            mask = vcltq_s32(vsrc32x4x3_0.val[2], vsrc32x4_255);
            vsrc32x4x3_0.val[2]  = vbslq_s32(mask, vsrc32x4x3_0.val[2], vsrc32x4_255);
            mask = vcltq_s32(vsrc32x4x3_1.val[2], vsrc32x4_255);
            vsrc32x4x3_1.val[2]  = vbslq_s32(mask, vsrc32x4x3_1.val[2], vsrc32x4_255);

            mask = vcgtq_s32(vsrc32x4x3_0.val[0], vsrc32x4_0);
            vsrc32x4x3_0.val[0]  = vbslq_s32(mask, vsrc32x4x3_0.val[0], vsrc32x4_0);
            mask = vcgtq_s32(vsrc32x4x3_1.val[0], vsrc32x4_0);
            vsrc32x4x3_1.val[0]  = vbslq_s32(mask, vsrc32x4x3_1.val[0], vsrc32x4_0);

            mask = vcgtq_s32(vsrc32x4x3_0.val[1], vsrc32x4_0);
            vsrc32x4x3_0.val[1]  = vbslq_s32(mask, vsrc32x4x3_0.val[1], vsrc32x4_0);
            mask = vcgtq_s32(vsrc32x4x3_1.val[1], vsrc32x4_0);
            vsrc32x4x3_1.val[1]  = vbslq_s32(mask, vsrc32x4x3_1.val[1], vsrc32x4_0);

            mask = vcgtq_s32(vsrc32x4x3_0.val[2], vsrc32x4_0);
            vsrc32x4x3_0.val[2]  = vbslq_s32(mask, vsrc32x4x3_0.val[2], vsrc32x4_0);
            mask = vcgtq_s32(vsrc32x4x3_1.val[2], vsrc32x4_0);
            vsrc32x4x3_1.val[2]  = vbslq_s32(mask, vsrc32x4x3_1.val[2], vsrc32x4_0);

            //narrow 32 to 8
            vsrc32x4x3_u_0.val[0] = vreinterpretq_u32_s32(vsrc32x4x3_0.val[0]);
            vsrc32x4x3_u_0.val[1] = vreinterpretq_u32_s32(vsrc32x4x3_0.val[1]);
            vsrc32x4x3_u_0.val[2] = vreinterpretq_u32_s32(vsrc32x4x3_0.val[2]);
            vsrc32x4x3_u_1.val[0] = vreinterpretq_u32_s32(vsrc32x4x3_1.val[0]);
            vsrc32x4x3_u_1.val[1] = vreinterpretq_u32_s32(vsrc32x4x3_1.val[1]);
            vsrc32x4x3_u_1.val[2] = vreinterpretq_u32_s32(vsrc32x4x3_1.val[2]);

            //R
            vR16x4_0 = vmovn_u32(vsrc32x4x3_u_0.val[0]);
            vR16x4_1 = vmovn_u32(vsrc32x4x3_u_1.val[0]);
            vR16x8   = vcombine_u16(vR16x4_0, vR16x4_1);
            if (bgrFlag)
                vRet8x8x3.val[2] = vmovn_u16(vR16x8);
            else
                vRet8x8x3.val[0] = vmovn_u16(vR16x8);

            //G
            vG16x4_0 = vmovn_u32(vsrc32x4x3_u_0.val[1]);
            vG16x4_1 = vmovn_u32(vsrc32x4x3_u_1.val[1]);
            vG16x8   = vcombine_u16(vG16x4_0, vG16x4_1);
            vRet8x8x3.val[1]    = vmovn_u16(vG16x8);

            //B
            vB16x4_0 = vmovn_u32(vsrc32x4x3_u_0.val[2]);
            vB16x4_1 = vmovn_u32(vsrc32x4x3_u_1.val[2]);
            vB16x8   = vcombine_u16(vB16x4_0, vB16x4_1);
            if (bgrFlag)
                vRet8x8x3.val[0] = vmovn_u16(vB16x8);
            else
                vRet8x8x3.val[2] = vmovn_u16(vB16x8);

            vst3_u8(pDstCur, vRet8x8x3);
            pDstCur += 24;
            pCurY += 8;
            pCurUV += 16;
        }

        if (roiWHas8)
        {
            int32x4x3_t vsrc32x4x3_0; // LOW  RGB
            int32x4x3_t vsrc32x4x3_1; // HIGH RGB

            vsrc32x4x3_0.val[0] = vsrc32x4_128; //R
            //vsrc32x4x3_0.val[1] = vsrc32x4_128; //G
            //vsrc32x4x3_0.val[2] = vsrc32x4_128; //B

            vsrc32x4x3_1.val[0] = vsrc32x4_128; //R
            //vsrc32x4x3_1.val[1] = vsrc32x4_128; //G
            //vsrc32x4x3_1.val[2] = vsrc32x4_128; //B

            uint8x8_t  vsrc8x8_y    = vld1_u8(pCurY); // [y0y1y2y3y4y5y6u7]
            uint16x8_t vsrc16x8_y_u = vmovl_u8(vsrc8x8_y);
            int16x8_t  vsrc16x8_y   = vreinterpretq_s16_u16(vsrc16x8_y_u);
            vsrc16x8_y = vsubq_s16(vsrc16x8_y, vsrc16x8_16);

            uint8x8x2_t vsrc8x8x2_uv = vld2_u8(pCurUV); // [u0u1u2u3u4u5u6u7] [v0v1v2v3v4v5v6v7]
            uint8x8x2_t vsrc8x8x2_u  = vzip_u8(vsrc8x8x2_uv.val[0], vsrc8x8x2_uv.val[0]); //[u0u0u1u1u2u2u3u3] [u4u4u5u5u6u6u7u7]
            uint8x8x2_t vsrc8x8x2_v  = vzip_u8(vsrc8x8x2_uv.val[1], vsrc8x8x2_uv.val[1]); //[v0v0v1v1v2v2v3v3] [v4v4v5v5v6v6v7v7]
            uint16x8_t  vsrc16x8_u_u = vmovl_u8(vsrc8x8x2_u.val[0]); //[u0u0u1u1u2u2u3u3]
            uint16x8_t  vsrc16x8_v_u = vmovl_u8(vsrc8x8x2_v.val[0]); //[v0v0v1v1v2v2v3v3]
            int16x8_t   vsrc16x8_u   = vreinterpretq_s16_u16(vsrc16x8_u_u);
            int16x8_t   vsrc16x8_v   = vreinterpretq_s16_u16(vsrc16x8_v_u);
            vsrc16x8_u = vsubq_s16(vsrc16x8_u, vsrc16x8_128);
            vsrc16x8_v = vsubq_s16(vsrc16x8_v, vsrc16x8_128);

            //R   R = (298*(Y-16)+409*(V-128)+128)>>8
            vsrc32x4x3_0.val[0] = vmlal_n_s16(vsrc32x4x3_0.val[0], vget_low_s16(vsrc16x8_y),  298);
            vsrc32x4x3_1.val[0] = vmlal_n_s16(vsrc32x4x3_1.val[0], vget_high_s16(vsrc16x8_y), 298);

            vsrc32x4x3_0.val[1] = vsrc32x4x3_0.val[0]; //vmlal_n_s16(vsrc32x4x3_0.val[1], vget_low_s16(vsrc16x8_y),  298);
            vsrc32x4x3_1.val[1] = vsrc32x4x3_1.val[0]; //vmlal_n_s16(vsrc32x4x3_1.val[1], vget_high_s16(vsrc16x8_y), 298);

            vsrc32x4x3_0.val[2] = vsrc32x4x3_0.val[0]; //vmlal_n_s16(vsrc32x4x3_0.val[2], vget_low_s16(vsrc16x8_y),  298);
            vsrc32x4x3_1.val[2] = vsrc32x4x3_1.val[0]; //vmlal_n_s16(vsrc32x4x3_1.val[2], vget_high_s16(vsrc16x8_y), 298);

            vsrc32x4x3_0.val[0] = vmlal_n_s16(vsrc32x4x3_0.val[0], vget_low_s16(vsrc16x8_v),  409);
            vsrc32x4x3_1.val[0] = vmlal_n_s16(vsrc32x4x3_1.val[0], vget_high_s16(vsrc16x8_v), 409);
            //G G = (298*(Y-16)-100*(U-128)-208*(V-128)+128)>>8
            //vsrc32x4x3_0.val[1] = vsrc32x4x3_0.val[0]; //vmlal_n_s16(vsrc32x4x3_0.val[1], vget_low_s16(vsrc16x8_y),  298);
            //vsrc32x4x3_1.val[1] = vsrc32x4x3_1.val[0]; //vmlal_n_s16(vsrc32x4x3_1.val[1], vget_high_s16(vsrc16x8_y), 298);

            vsrc32x4x3_0.val[1] = vmlal_n_s16(vsrc32x4x3_0.val[1], vget_low_s16(vsrc16x8_u),  -100);
            vsrc32x4x3_1.val[1] = vmlal_n_s16(vsrc32x4x3_1.val[1], vget_high_s16(vsrc16x8_u), -100);

            vsrc32x4x3_0.val[1] = vmlal_n_s16(vsrc32x4x3_0.val[1], vget_low_s16(vsrc16x8_v),  -208);
            vsrc32x4x3_1.val[1] = vmlal_n_s16(vsrc32x4x3_1.val[1], vget_high_s16(vsrc16x8_v), -208);
            //B B = (298*(Y-16)+516*(U-128)+128)>>8
            //vsrc32x4x3_0.val[2] = vsrc32x4x3_0.val[0]; //vmlal_n_s16(vsrc32x4x3_0.val[2], vget_low_s16(vsrc16x8_y),  298);
            //vsrc32x4x3_1.val[2] = vsrc32x4x3_1.val[0]; //vmlal_n_s16(vsrc32x4x3_1.val[2], vget_high_s16(vsrc16x8_y), 298);

            vsrc32x4x3_0.val[2] = vmlal_n_s16(vsrc32x4x3_0.val[2], vget_low_s16(vsrc16x8_u),  516);
            vsrc32x4x3_1.val[2] = vmlal_n_s16(vsrc32x4x3_1.val[2], vget_high_s16(vsrc16x8_u), 516);

            //shift right
            vsrc32x4x3_0.val[0] = vshrq_n_s32(vsrc32x4x3_0.val[0], 8);
            vsrc32x4x3_1.val[0] = vshrq_n_s32(vsrc32x4x3_1.val[0], 8);

            vsrc32x4x3_0.val[1] = vshrq_n_s32(vsrc32x4x3_0.val[1], 8);
            vsrc32x4x3_1.val[1] = vshrq_n_s32(vsrc32x4x3_1.val[1], 8);

            vsrc32x4x3_0.val[2] = vshrq_n_s32(vsrc32x4x3_0.val[2], 8);
            vsrc32x4x3_1.val[2] = vshrq_n_s32(vsrc32x4x3_1.val[2], 8);

            uint32x4_t mask;

            mask = vcltq_s32(vsrc32x4x3_0.val[0], vsrc32x4_255);
            vsrc32x4x3_0.val[0]  = vbslq_s32(mask, vsrc32x4x3_0.val[0], vsrc32x4_255);
            mask = vcltq_s32(vsrc32x4x3_1.val[0], vsrc32x4_255);
            vsrc32x4x3_1.val[0]  = vbslq_s32(mask, vsrc32x4x3_1.val[0], vsrc32x4_255);

            mask = vcltq_s32(vsrc32x4x3_0.val[1], vsrc32x4_255);
            vsrc32x4x3_0.val[1]  = vbslq_s32(mask, vsrc32x4x3_0.val[1], vsrc32x4_255);
            mask = vcltq_s32(vsrc32x4x3_1.val[1], vsrc32x4_255);
            vsrc32x4x3_1.val[1]  = vbslq_s32(mask, vsrc32x4x3_1.val[1], vsrc32x4_255);

            mask = vcltq_s32(vsrc32x4x3_0.val[2], vsrc32x4_255);
            vsrc32x4x3_0.val[2]  = vbslq_s32(mask, vsrc32x4x3_0.val[2], vsrc32x4_255);
            mask = vcltq_s32(vsrc32x4x3_1.val[2], vsrc32x4_255);
            vsrc32x4x3_1.val[2]  = vbslq_s32(mask, vsrc32x4x3_1.val[2], vsrc32x4_255);

            mask = vcgtq_s32(vsrc32x4x3_0.val[0], vsrc32x4_0);
            vsrc32x4x3_0.val[0]  = vbslq_s32(mask, vsrc32x4x3_0.val[0], vsrc32x4_0);
            mask = vcgtq_s32(vsrc32x4x3_1.val[0], vsrc32x4_0);
            vsrc32x4x3_1.val[0]  = vbslq_s32(mask, vsrc32x4x3_1.val[0], vsrc32x4_0);

            mask = vcgtq_s32(vsrc32x4x3_0.val[1], vsrc32x4_0);
            vsrc32x4x3_0.val[1]  = vbslq_s32(mask, vsrc32x4x3_0.val[1], vsrc32x4_0);
            mask = vcgtq_s32(vsrc32x4x3_1.val[1], vsrc32x4_0);
            vsrc32x4x3_1.val[1]  = vbslq_s32(mask, vsrc32x4x3_1.val[1], vsrc32x4_0);

            mask = vcgtq_s32(vsrc32x4x3_0.val[2], vsrc32x4_0);
            vsrc32x4x3_0.val[2]  = vbslq_s32(mask, vsrc32x4x3_0.val[2], vsrc32x4_0);
            mask = vcgtq_s32(vsrc32x4x3_1.val[2], vsrc32x4_0);
            vsrc32x4x3_1.val[2]  = vbslq_s32(mask, vsrc32x4x3_1.val[2], vsrc32x4_0);

            //narrow 32 to 8
            uint8x8x3_t vRet8x8x3;
            uint32x4x3_t vsrc32x4x3_u_0, vsrc32x4x3_u_1;
            vsrc32x4x3_u_0.val[0] = vreinterpretq_u32_s32(vsrc32x4x3_0.val[0]);
            vsrc32x4x3_u_0.val[1] = vreinterpretq_u32_s32(vsrc32x4x3_0.val[1]);
            vsrc32x4x3_u_0.val[2] = vreinterpretq_u32_s32(vsrc32x4x3_0.val[2]);
            vsrc32x4x3_u_1.val[0] = vreinterpretq_u32_s32(vsrc32x4x3_1.val[0]);
            vsrc32x4x3_u_1.val[1] = vreinterpretq_u32_s32(vsrc32x4x3_1.val[1]);
            vsrc32x4x3_u_1.val[2] = vreinterpretq_u32_s32(vsrc32x4x3_1.val[2]);

            //R
            uint16x4_t vR16x4_0 = vmovn_u32(vsrc32x4x3_u_0.val[0]);
            uint16x4_t vR16x4_1 = vmovn_u32(vsrc32x4x3_u_1.val[0]);
            uint16x8_t vR16x8   = vcombine_u16(vR16x4_0, vR16x4_1);
            if (bgrFlag)
                vRet8x8x3.val[2] = vmovn_u16(vR16x8);
            else
                vRet8x8x3.val[0] = vmovn_u16(vR16x8);

            //G
            uint16x4_t vG16x4_0 = vmovn_u32(vsrc32x4x3_u_0.val[1]);
            uint16x4_t vG16x4_1 = vmovn_u32(vsrc32x4x3_u_1.val[1]);
            uint16x8_t vG16x8   = vcombine_u16(vG16x4_0, vG16x4_1);
            vRet8x8x3.val[1]    = vmovn_u16(vG16x8);

            //B
            uint16x4_t vB16x4_0 = vmovn_u32(vsrc32x4x3_u_0.val[2]);
            uint16x4_t vB16x4_1 = vmovn_u32(vsrc32x4x3_u_1.val[2]);
            uint16x8_t vB16x8   = vcombine_u16(vB16x4_0, vB16x4_1);
            if (bgrFlag)
                vRet8x8x3.val[0] = vmovn_u16(vB16x8);
            else
                vRet8x8x3.val[2] = vmovn_u16(vB16x8);

            vst3_u8(pDstCur, vRet8x8x3);

            pDstCur += 24;
            pCurY += 8;
            pCurUV += 8;
        }

        for( w = 0; w < roiWLeft; w++)
        {
            int Y, U, V, R, G, B, Y298;
            Y = ((int32_t)*pCurY) - 16;
            U = ((int32_t)*pCurUV) - 128;
            V = ((int32_t)*(pCurUV+1)) - 128;

            Y298 = 298*(Y);
            R = (Y298 + 409*(V) + 128)>>8;
            G = (Y298 - 100*(U) - 208*(V) + 128)>>8;
            B = (Y298 + 516*(U) + 128)>>8;

            if (R < 0) R = 0;
            else if (R > 255) R = 255;

            if (G < 0) G = 0;
            else if (G > 255) G = 255;

            if (B < 0) B = 0;
            else if (B > 255) B = 255;

            if (bgrFlag)
            {
                *pDstCur++ = (unsigned char)B;
                *pDstCur++ = (unsigned char)G;
                *pDstCur++ = (unsigned char)R;
            }
            else
            {
                *pDstCur++ = (unsigned char)R;
                *pDstCur++ = (unsigned char)G;
                *pDstCur++ = (unsigned char)B;
            }

            pCurY++;
            if (w%2) pCurUV += 2;
        }
    }

    return;
}

static Mat from_yuv420sp(const unsigned char* yuv, int w, int h, int stride)
{
    unsigned int i,j;
    unsigned int dstw = w>>1;
    unsigned int dsth = h>>1;
    unsigned int left = dstw&0x0f;

    Mat m(dstw, dsth, 3, 1);
    if (m.empty())
        return m;

    unsigned char* pDst = (unsigned char*)m.data;
    const unsigned char * uv = yuv + stride * h;

    for( i = 0; i < dsth; i++)
    {
        const unsigned char *pCurY = yuv + (i<<1)*stride;
        const unsigned char *pCurUV = uv + i*2*dstw;
        unsigned char*pDstCur = pDst + i*3*dstw;

        for( j = 0; j < dstw; j += 16)
        {
            uint8x16x2_t vsrc8x16x2_y = vld2q_u8(pCurY);  //load 32 bytes
            uint8x16x2_t vsrc8x16x2_uv = vld2q_u8(pCurUV);//load 32 bytes
            uint8x16x3_t vsrc8x16x3;
            vsrc8x16x3.val[0] = vsrc8x16x2_y.val[0];
            vsrc8x16x3.val[1] = vsrc8x16x2_uv.val[0];
            vsrc8x16x3.val[2] = vsrc8x16x2_uv.val[1];
            vst3q_u8(pDstCur, vsrc8x16x3); //store 48 bytes

            pDstCur += 48;
            pCurY += 32;
            pCurUV += 32;
        }

        for( j = 0; j < left; j++)
        {
            *pDstCur++ = *pCurY;
            *pDstCur++ = *pCurUV;
            *pDstCur++ = *(pCurUV+1);

            pCurY  += 2;
            pCurUV += 2;
        }
    }

    return m;
}

static Mat from_rgb2bgr(const unsigned char* rgb, int w, int h)
{
    Mat m(w, h, 3);
    if (m.empty())
        return m;

    float* ptr0 = m.channel(0);
    float* ptr1 = m.channel(1);
    float* ptr2 = m.channel(2);

    int size = w * h;

#if __ARM_NEON
    int nn = size >> 3;
    int remain = size - (nn << 3);
#else
    int remain = size;
#endif // __ARM_NEON

#if __ARM_NEON
#if __aarch64__
    for (; nn>0; nn--)
    {
        uint8x8x3_t _rgb = vld3_u8(rgb);
        uint16x8_t _r16 = vmovl_u8(_rgb.val[0]);
        uint16x8_t _g16 = vmovl_u8(_rgb.val[1]);
        uint16x8_t _b16 = vmovl_u8(_rgb.val[2]);

        float32x4_t _rlow = vcvtq_f32_u32(vmovl_u16(vget_low_u16(_r16)));
        float32x4_t _rhigh = vcvtq_f32_u32(vmovl_u16(vget_high_u16(_r16)));
        float32x4_t _glow = vcvtq_f32_u32(vmovl_u16(vget_low_u16(_g16)));
        float32x4_t _ghigh = vcvtq_f32_u32(vmovl_u16(vget_high_u16(_g16)));
        float32x4_t _blow = vcvtq_f32_u32(vmovl_u16(vget_low_u16(_b16)));
        float32x4_t _bhigh = vcvtq_f32_u32(vmovl_u16(vget_high_u16(_b16)));

        vst1q_f32(ptr2, _rlow);
        vst1q_f32(ptr2+4, _rhigh);
        vst1q_f32(ptr1, _glow);
        vst1q_f32(ptr1+4, _ghigh);
        vst1q_f32(ptr0, _blow);
        vst1q_f32(ptr0+4, _bhigh);

        rgb += 3*8;
        ptr0 += 8;
        ptr1 += 8;
        ptr2 += 8;
    }
#else
    if (nn > 0)
    {
        asm volatile(
            "0:                             \n"
            "pld        [%1, #256]          \n"
            "vld3.u8    {d0-d2}, [%1]!      \n"
            "vmovl.u8   q8, d0              \n"
            "vmovl.u8   q9, d1              \n"
            "vmovl.u8   q10, d2             \n"
            "vmovl.u16  q0, d16             \n"
            "vmovl.u16  q1, d17             \n"
            "vmovl.u16  q2, d18             \n"
            "vmovl.u16  q3, d19             \n"
            "vmovl.u16  q8, d20             \n"
            "vmovl.u16  q9, d21             \n"
            "vcvt.f32.u32   q0, q0          \n"
            "vcvt.f32.u32   q1, q1          \n"
            "vcvt.f32.u32   q2, q2          \n"
            "vcvt.f32.u32   q3, q3          \n"
            "vcvt.f32.u32   q8, q8          \n"
            "subs       %0, #1              \n"
            "vst1.f32   {d0-d3}, [%4 :128]! \n"
            "vcvt.f32.u32   q9, q9          \n"
            "vst1.f32   {d4-d7}, [%3 :128]! \n"
            "vst1.f32   {d16-d19}, [%2 :128]!\n"
            "bne        0b                  \n"
            : "=r"(nn),     // %0
            "=r"(rgb),    // %1
            "=r"(ptr0),   // %2
            "=r"(ptr1),   // %3
            "=r"(ptr2)    // %4
            : "0"(nn),
            "1"(rgb),
            "2"(ptr0),
            "3"(ptr1),
            "4"(ptr2)
            : "cc", "memory", "q0", "q1", "q2", "q3", "q8", "q9", "q10"
        );
    }
#endif // __aarch64__
#endif // __ARM_NEON
    for (; remain>0; remain--)
    {
        *ptr0 = rgb[2];
        *ptr1 = rgb[1];
        *ptr2 = rgb[0];

        rgb += 3;
        ptr0++;
        ptr1++;
        ptr2++;
    }

    return m;
}

static void to_bgr2rgb(const Mat& m, unsigned char* rgb)
{
    const float* ptr0 = m.channel(0);
    const float* ptr1 = m.channel(1);
    const float* ptr2 = m.channel(2);

    int size = m.w * m.h;

#define SATURATE_CAST_UCHAR(X) (unsigned char)::std::min(::std::max((int)(X), 0), 255);

    int remain = size;

    for (; remain>0; remain--)
    {
        rgb[2] = SATURATE_CAST_UCHAR(*ptr0);
        rgb[1] = SATURATE_CAST_UCHAR(*ptr1);
        rgb[0] = SATURATE_CAST_UCHAR(*ptr2);

        rgb += 3;
        ptr0++;
        ptr1++;
        ptr2++;
    }

#undef SATURATE_CAST_UCHAR
}

static Mat from_rgb2gray(const unsigned char* rgb, int w, int h)
{
    // coeffs for r g b = 0.299f, 0.587f, 0.114f
    const unsigned char Y_shift = 8;//14
    const unsigned char R2Y = 77;
    const unsigned char G2Y = 150;
    const unsigned char B2Y = 29;

    Mat m(w, h, 1);
    if (m.empty())
        return m;

    float* ptr = m;

    int size = w * h;

#if __ARM_NEON
    int nn = size >> 3;
    int remain = size - (nn << 3);
#else
    int remain = size;
#endif // __ARM_NEON

#if __ARM_NEON
#if __aarch64__
    uint8x8_t _R2Y = vdup_n_u8(R2Y);
    uint8x8_t _G2Y = vdup_n_u8(G2Y);
    uint8x8_t _B2Y = vdup_n_u8(B2Y);
    for (; nn>0; nn--)
    {
        uint8x8x3_t _rgb = vld3_u8(rgb);

        uint16x8_t _y16 = vmull_u8(_rgb.val[0], _R2Y);
        _y16 = vmlal_u8(_y16, _rgb.val[1], _G2Y);
        _y16 = vmlal_u8(_y16, _rgb.val[2], _B2Y);
        _y16 = vshrq_n_u16(_y16, Y_shift);

        float32x4_t _ylow = vcvtq_f32_u32(vmovl_u16(vget_low_u16(_y16)));
        float32x4_t _yhigh = vcvtq_f32_u32(vmovl_u16(vget_high_u16(_y16)));

        vst1q_f32(ptr, _ylow);
        vst1q_f32(ptr+4, _yhigh);

        rgb += 3*8;
        ptr += 8;
    }
#else
    if (nn > 0)
    {
        asm volatile(
            "vdup.u8    d16, %6             \n"
            "vdup.u8    d17, %7             \n"
            "vdup.u8    d18, %8             \n"
            "0:                             \n"
            "pld        [%1, #256]          \n"
            "vld3.u8    {d0-d2}, [%1]!      \n"
            "vmull.u8   q2, d0, d16         \n"
            "vmlal.u8   q2, d1, d17         \n"
            "vmlal.u8   q2, d2, d18         \n"
            "vshr.u16   q2, q2, #8          \n" // Y_shift
            "vmovl.u16  q0, d4              \n"
            "vmovl.u16  q1, d5              \n"
            "vcvt.f32.u32   q0, q0          \n"
            "vcvt.f32.u32   q1, q1          \n"
            "subs       %0, #1              \n"
            "vst1.f32   {d0-d3}, [%2 :128]! \n"
            "bne        0b                  \n"
            : "=r"(nn),     // %0
            "=r"(rgb),    // %1
            "=r"(ptr)     // %2
            : "0"(nn),
            "1"(rgb),
            "2"(ptr),
            "r"(R2Y),     // %6
            "r"(G2Y),     // %7
            "r"(B2Y)      // %8
            : "cc", "memory", "q0", "q1", "q2", "q8", "q9"
        );
    }
#endif // __aarch64__
#endif // __ARM_NEON
    for (; remain>0; remain--)
    {
        *ptr = (rgb[0] * R2Y + rgb[1] * G2Y + rgb[2] * B2Y) >> Y_shift;

        rgb += 3;
        ptr++;
    }

    return m;
}

static Mat from_bgr2gray(const unsigned char* bgr, int w, int h)
{
    // coeffs for r g b = 0.299f, 0.587f, 0.114f
    const unsigned char Y_shift = 8;//14
    const unsigned char R2Y = 77;
    const unsigned char G2Y = 150;
    const unsigned char B2Y = 29;

    Mat m(w, h, 1);
    if (m.empty())
        return m;

    float* ptr = m;

    int size = w * h;

#if __ARM_NEON
    int nn = size >> 3;
    int remain = size - (nn << 3);
#else
    int remain = size;
#endif // __ARM_NEON

#if __ARM_NEON
#if __aarch64__
    uint8x8_t _R2Y = vdup_n_u8(R2Y);
    uint8x8_t _G2Y = vdup_n_u8(G2Y);
    uint8x8_t _B2Y = vdup_n_u8(B2Y);
    for (; nn>0; nn--)
    {
        uint8x8x3_t _rgb = vld3_u8(bgr);

        uint16x8_t _y16 = vmull_u8(_rgb.val[2], _R2Y);
        _y16 = vmlal_u8(_y16, _rgb.val[1], _G2Y);
        _y16 = vmlal_u8(_y16, _rgb.val[0], _B2Y);
        _y16 = vshrq_n_u16(_y16, Y_shift);

        float32x4_t _ylow = vcvtq_f32_u32(vmovl_u16(vget_low_u16(_y16)));
        float32x4_t _yhigh = vcvtq_f32_u32(vmovl_u16(vget_high_u16(_y16)));

        vst1q_f32(ptr, _ylow);
        vst1q_f32(ptr+4, _yhigh);

        bgr += 3*8;
        ptr += 8;
    }
#else
    if (nn > 0)
    {
        asm volatile(
            "vdup.u8    d16, %6             \n"
            "vdup.u8    d17, %7             \n"
            "vdup.u8    d18, %8             \n"
            "0:                             \n"
            "pld        [%1, #256]          \n"
            "vld3.u8    {d0-d2}, [%1]!      \n"
            "vmull.u8   q2, d2, d16         \n"
            "vmlal.u8   q2, d1, d17         \n"
            "vmlal.u8   q2, d0, d18         \n"
            "vshr.u16   q2, q2, #8          \n" // Y_shift
            "vmovl.u16  q0, d4              \n"
            "vmovl.u16  q1, d5              \n"
            "vcvt.f32.u32   q0, q0          \n"
            "vcvt.f32.u32   q1, q1          \n"
            "subs       %0, #1              \n"
            "vst1.f32   {d0-d3}, [%2 :128]! \n"
            "bne        0b                  \n"
            : "=r"(nn),     // %0
            "=r"(bgr),    // %1
            "=r"(ptr)     // %2
            : "0"(nn),
            "1"(bgr),
            "2"(ptr),
            "r"(R2Y),     // %6
            "r"(G2Y),     // %7
            "r"(B2Y)      // %8
            : "cc", "memory", "q0", "q1", "q2", "q8", "q9"
        );
    }
#endif // __aarch64__
#endif // __ARM_NEON
    for (; remain>0; remain--)
    {
        *ptr = (bgr[2] * R2Y + bgr[1] * G2Y + bgr[0] * B2Y) >> Y_shift;

        bgr += 3;
        ptr++;
    }

    return m;
}

static Mat from_gray2rgb(const unsigned char* gray, int w, int h)
{
    Mat m(w, h, 3);
    if (m.empty())
        return m;

    float* ptr0 = m.channel(0);
    float* ptr1 = m.channel(1);
    float* ptr2 = m.channel(2);

    int size = w * h;

#if __ARM_NEON
    int nn = size >> 4;
    int remain = size - (nn << 4);
#else
    int remain = size;
#endif // __ARM_NEON

#if __ARM_NEON
#if __aarch64__
    for (; nn>0; nn--)
    {
        uint8x16_t _gray = vld1q_u8(gray);
        uint16x8_t _gray16_0 = vmovl_u8(vget_low_u8(_gray));
        uint16x8_t _gray16_1 = vmovl_u8(vget_high_u8(_gray));

        float32x4_t _graylow_0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(_gray16_0)));
        float32x4_t _grayhigh_0 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(_gray16_0)));
        float32x4_t _graylow_1 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(_gray16_1)));
        float32x4_t _grayhigh_1 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(_gray16_1)));

        vst1q_f32(ptr0, _graylow_0);
        vst1q_f32(ptr0+4, _grayhigh_0);
        vst1q_f32(ptr0+8, _graylow_1);
        vst1q_f32(ptr0+12, _grayhigh_1);

        vst1q_f32(ptr1, _graylow_0);
        vst1q_f32(ptr1+4, _grayhigh_0);
        vst1q_f32(ptr1+8, _graylow_1);
        vst1q_f32(ptr1+12, _grayhigh_1);

        vst1q_f32(ptr2, _graylow_0);
        vst1q_f32(ptr2+4, _grayhigh_0);
        vst1q_f32(ptr2+8, _graylow_1);
        vst1q_f32(ptr2+12, _grayhigh_1);

        gray += 16;
        ptr0 += 16;
        ptr1 += 16;
        ptr2 += 16;
    }
#else
    if (nn > 0)
    {
        asm volatile(
            "0:                             \n"
            "pld        [%1, #128]          \n"
            "vld1.u8    {d0,d1}, [%1]!      \n"
            "vmovl.u8   q8, d0              \n"
            "vmovl.u8   q9, d1              \n"
            "vmovl.u16  q0, d16             \n"
            "vmovl.u16  q1, d17             \n"
            "vmovl.u16  q2, d18             \n"
            "vmovl.u16  q3, d19             \n"
            "vcvt.f32.u32   q0, q0          \n"
            "vcvt.f32.u32   q1, q1          \n"
            "vcvt.f32.u32   q2, q2          \n"
            "vcvt.f32.u32   q3, q3          \n"
            "subs       %0, #1              \n"
            "vst1.f32   {d0-d3}, [%2 :128]! \n"
            "vst1.f32   {d4-d7}, [%2 :128]! \n"
            "vst1.f32   {d0-d3}, [%3 :128]! \n"
            "vst1.f32   {d4-d7}, [%3 :128]! \n"
            "vst1.f32   {d0-d3}, [%4 :128]! \n"
            "vst1.f32   {d4-d7}, [%4 :128]! \n"
            "bne        0b                  \n"
            : "=r"(nn),     // %0
            "=r"(gray),   // %1
            "=r"(ptr0),   // %2
            "=r"(ptr1),   // %3
            "=r"(ptr2)    // %4
            : "0"(nn),
            "1"(gray),
            "2"(ptr0),
            "3"(ptr1),
            "4"(ptr2)
            : "cc", "memory", "q0", "q1", "q2", "q3", "q8", "q9"
        );
    }
#endif // __aarch64__
#endif // __ARM_NEON
    for (; remain>0; remain--)
    {
        *ptr0 = *gray;
        *ptr1 = *gray;
        *ptr2 = *gray;

        gray++;
        ptr0++;
        ptr1++;
        ptr2++;
    }

    return m;
}

static Mat from_rgba2rgb(const unsigned char* rgba, int w, int h)
{
    Mat m(w, h, 3);
    if (m.empty())
        return m;

    float* ptr0 = m.channel(0);
    float* ptr1 = m.channel(1);
    float* ptr2 = m.channel(2);

    int size = w * h;

#if __ARM_NEON
    int nn = size >> 3;
    int remain = size - (nn << 3);
#else
    int remain = size;
#endif // __ARM_NEON

#if __ARM_NEON
#if __aarch64__
    for (; nn>0; nn--)
    {
        uint8x8x4_t _rgba = vld4_u8(rgba);
        int16x8_t _r16 = vreinterpretq_s16_u16(vmovl_u8(_rgba.val[0]));
        int16x8_t _g16 = vreinterpretq_s16_u16(vmovl_u8(_rgba.val[1]));
        int16x8_t _b16 = vreinterpretq_s16_u16(vmovl_u8(_rgba.val[2]));

        float32x4_t _rlow = vcvtq_f32_s32(vmovl_s16(vget_low_s16(_r16)));
        float32x4_t _rhigh = vcvtq_f32_s32(vmovl_s16(vget_high_s16(_r16)));
        float32x4_t _glow = vcvtq_f32_s32(vmovl_s16(vget_low_s16(_g16)));
        float32x4_t _ghigh = vcvtq_f32_s32(vmovl_s16(vget_high_s16(_g16)));
        float32x4_t _blow = vcvtq_f32_s32(vmovl_s16(vget_low_s16(_b16)));
        float32x4_t _bhigh = vcvtq_f32_s32(vmovl_s16(vget_high_s16(_b16)));

        vst1q_f32(ptr0, _rlow);
        vst1q_f32(ptr0+4, _rhigh);
        vst1q_f32(ptr1, _glow);
        vst1q_f32(ptr1+4, _ghigh);
        vst1q_f32(ptr2, _blow);
        vst1q_f32(ptr2+4, _bhigh);

        rgba += 4*8;
        ptr0 += 8;
        ptr1 += 8;
        ptr2 += 8;
    }
#else
    if (nn > 0)
    {
        asm volatile(
            "0:                             \n"
            "pld        [%1, #256]          \n"
            "vld4.u8    {d0-d3}, [%1]!      \n"
            "vmovl.u8   q8, d0              \n"
            "vmovl.u8   q9, d1              \n"
            "vmovl.u8   q10, d2             \n"
            "vmovl.u16  q0, d16             \n"
            "vmovl.u16  q1, d17             \n"
            "vmovl.u16  q2, d18             \n"
            "vmovl.u16  q3, d19             \n"
            "vmovl.u16  q8, d20             \n"
            "vmovl.u16  q9, d21             \n"
            "vcvt.f32.u32   q0, q0          \n"
            "vcvt.f32.u32   q1, q1          \n"
            "vcvt.f32.u32   q2, q2          \n"
            "vcvt.f32.u32   q3, q3          \n"
            "vcvt.f32.u32   q8, q8          \n"
            "subs       %0, #1              \n"
            "vst1.f32   {d0-d3}, [%2 :128]! \n"
            "vcvt.f32.u32   q9, q9          \n"
            "vst1.f32   {d4-d7}, [%3 :128]! \n"
            "vst1.f32   {d16-d19}, [%4 :128]!\n"
            "bne        0b                  \n"
            : "=r"(nn),     // %0
            "=r"(rgba),   // %1
            "=r"(ptr0),   // %2
            "=r"(ptr1),   // %3
            "=r"(ptr2)    // %4
            : "0"(nn),
            "1"(rgba),
            "2"(ptr0),
            "3"(ptr1),
            "4"(ptr2)
            : "cc", "memory", "q0", "q1", "q2", "q3", "q8", "q9"
        );
    }
#endif // __aarch64__
#endif // __ARM_NEON
    for (; remain>0; remain--)
    {
        *ptr0 = rgba[0];
        *ptr1 = rgba[1];
        *ptr2 = rgba[2];

        rgba += 4;
        ptr0++;
        ptr1++;
        ptr2++;
    }

    return m;
}

static Mat from_rgba2bgr(const unsigned char* rgba, int w, int h)
{
    Mat m(w, h, 3);
    if (m.empty())
        return m;

    float* ptr0 = m.channel(0);
    float* ptr1 = m.channel(1);
    float* ptr2 = m.channel(2);

    int size = w * h;

#if __ARM_NEON
    int nn = size >> 3;
    int remain = size - (nn << 3);
#else
    int remain = size;
#endif // __ARM_NEON

#if __ARM_NEON
#if __aarch64__
    for (; nn>0; nn--)
    {
        uint8x8x4_t _rgba = vld4_u8(rgba);
        int16x8_t _r16 = vreinterpretq_s16_u16(vmovl_u8(_rgba.val[0]));
        int16x8_t _g16 = vreinterpretq_s16_u16(vmovl_u8(_rgba.val[1]));
        int16x8_t _b16 = vreinterpretq_s16_u16(vmovl_u8(_rgba.val[2]));

        float32x4_t _rlow = vcvtq_f32_s32(vmovl_s16(vget_low_s16(_r16)));
        float32x4_t _rhigh = vcvtq_f32_s32(vmovl_s16(vget_high_s16(_r16)));
        float32x4_t _glow = vcvtq_f32_s32(vmovl_s16(vget_low_s16(_g16)));
        float32x4_t _ghigh = vcvtq_f32_s32(vmovl_s16(vget_high_s16(_g16)));
        float32x4_t _blow = vcvtq_f32_s32(vmovl_s16(vget_low_s16(_b16)));
        float32x4_t _bhigh = vcvtq_f32_s32(vmovl_s16(vget_high_s16(_b16)));

        vst1q_f32(ptr2, _rlow);
        vst1q_f32(ptr2+4, _rhigh);
        vst1q_f32(ptr1, _glow);
        vst1q_f32(ptr1+4, _ghigh);
        vst1q_f32(ptr0, _blow);
        vst1q_f32(ptr0+4, _bhigh);

        rgba += 4*8;
        ptr0 += 8;
        ptr1 += 8;
        ptr2 += 8;
    }
#else
    if (nn > 0)
    {
        asm volatile(
            "0:                             \n"
            "pld        [%1, #256]          \n"
            "vld4.u8    {d0-d3}, [%1]!      \n"
            "vmovl.u8   q8, d0              \n"
            "vmovl.u8   q9, d1              \n"
            "vmovl.u8   q10, d2             \n"
            "vmovl.u16  q0, d16             \n"
            "vmovl.u16  q1, d17             \n"
            "vmovl.u16  q2, d18             \n"
            "vmovl.u16  q3, d19             \n"
            "vmovl.u16  q8, d20             \n"
            "vmovl.u16  q9, d21             \n"
            "vcvt.f32.u32   q0, q0          \n"
            "vcvt.f32.u32   q1, q1          \n"
            "vcvt.f32.u32   q2, q2          \n"
            "vcvt.f32.u32   q3, q3          \n"
            "vcvt.f32.u32   q8, q8          \n"
            "subs       %0, #1              \n"
            "vst1.f32   {d0-d3}, [%4 :128]! \n"
            "vcvt.f32.u32   q9, q9          \n"
            "vst1.f32   {d4-d7}, [%3 :128]! \n"
            "vst1.f32   {d16-d19}, [%2 :128]!\n"
            "bne        0b                  \n"
            : "=r"(nn),     // %0
            "=r"(rgba),   // %1
            "=r"(ptr0),   // %2
            "=r"(ptr1),   // %3
            "=r"(ptr2)    // %4
            : "0"(nn),
            "1"(rgba),
            "2"(ptr0),
            "3"(ptr1),
            "4"(ptr2)
            : "cc", "memory", "q0", "q1", "q2", "q3", "q8", "q9", "q10"
        );
    }
#endif // __aarch64__
#endif // __ARM_NEON
    for (; remain>0; remain--)
    {
        *ptr0 = rgba[2];
        *ptr1 = rgba[1];
        *ptr2 = rgba[0];

        rgba += 4;
        ptr0++;
        ptr1++;
        ptr2++;
    }

    return m;
}

static Mat from_rgba2gray(const unsigned char* rgba, int w, int h)
{
    // coeffs for r g b = 0.299f, 0.587f, 0.114f
    const unsigned char Y_shift = 8;//14
    const unsigned char R2Y = 77;
    const unsigned char G2Y = 150;
    const unsigned char B2Y = 29;

    Mat m(w, h, 1);
    if (m.empty())
        return m;

    float* ptr = m;

    int size = w * h;

#if __ARM_NEON
    int nn = size >> 3;
    int remain = size - (nn << 3);
#else
    int remain = size;
#endif // __ARM_NEON

#if __ARM_NEON
#if __aarch64__
    uint8x8_t _R2Y = vdup_n_u8(R2Y);
    uint8x8_t _G2Y = vdup_n_u8(G2Y);
    uint8x8_t _B2Y = vdup_n_u8(B2Y);
    for (; nn>0; nn--)
    {
        uint8x8x4_t _rgba = vld4_u8(rgba);

        uint16x8_t _y16 = vmull_u8(_rgba.val[0], _R2Y);
        _y16 = vmlal_u8(_y16, _rgba.val[1], _G2Y);
        _y16 = vmlal_u8(_y16, _rgba.val[2], _B2Y);
        _y16 = vshrq_n_u16(_y16, Y_shift);

        float32x4_t _ylow = vcvtq_f32_u32(vmovl_u16(vget_low_u16(_y16)));
        float32x4_t _yhigh = vcvtq_f32_u32(vmovl_u16(vget_high_u16(_y16)));

        vst1q_f32(ptr, _ylow);
        vst1q_f32(ptr+4, _yhigh);

        rgba += 4*8;
        ptr += 8;
    }
#else
    if (nn > 0)
    {
        asm volatile(
            "vdup.u8    d16, %6             \n"
            "vdup.u8    d17, %7             \n"
            "vdup.u8    d18, %8             \n"
            "0:                             \n"
            "pld        [%1, #256]          \n"
            "vld4.u8    {d0-d3}, [%1]!      \n"
            "vmull.u8   q2, d0, d16         \n"
            "vmlal.u8   q2, d1, d17         \n"
            "vmlal.u8   q2, d2, d18         \n"
            "vshr.u16   q2, q2, #8          \n" // Y_shift
            "vmovl.u16  q0, d4              \n"
            "vmovl.u16  q1, d5              \n"
            "vcvt.f32.u32   q0, q0          \n"
            "vcvt.f32.u32   q1, q1          \n"
            "subs       %0, #1              \n"
            "vst1.f32   {d0-d3}, [%2 :128]! \n"
            "bne        0b                  \n"
            : "=r"(nn),     // %0
            "=r"(rgba),   // %1
            "=r"(ptr)     // %2
            : "0"(nn),
            "1"(rgba),
            "2"(ptr),
            "r"(R2Y),     // %6
            "r"(G2Y),     // %7
            "r"(B2Y)      // %8
            : "cc", "memory", "q0", "q1", "q2", "q8", "q9"
        );
    }
#endif // __aarch64__
#endif // __ARM_NEON
    for (; remain>0; remain--)
    {
        *ptr = (rgba[0] * R2Y + rgba[1] * G2Y + rgba[2] * B2Y) >> Y_shift;

        rgba += 4;
        ptr++;
    }

    return m;
}

void resize_bilinear_c3(const unsigned char* src, int srcw, int srch, unsigned char* dst, int w, int h)
{
    const int INTER_RESIZE_COEF_BITS=11;
    const int INTER_RESIZE_COEF_SCALE=1 << INTER_RESIZE_COEF_BITS;
//     const int ONE=INTER_RESIZE_COEF_SCALE;

    double scale_x = (double)srcw / w;
    double scale_y = (double)srch / h;

    int* buf = new int[w + h + w + h];

    int* xofs = buf;//new int[w];
    int* yofs = buf + w;//new int[h];

    short* ialpha = (short*)(buf + w + h);//new short[w * 2];
    short* ibeta = (short*)(buf + w + h + w);//new short[h * 2];

    float fx;
    float fy;
    int sx;
    int sy;

#define SATURATE_CAST_SHORT(X) (short)::std::min(::std::max((int)(X + (X >= 0.f ? 0.5f : -0.5f)), SHRT_MIN), SHRT_MAX);

    for (int dx = 0; dx < w; dx++)
    {
        fx = (float)((dx + 0.5) * scale_x - 0.5);
        sx = fx;//cvFloor(fx);
        fx -= sx;

        if (sx >= srcw - 1)
        {
            sx = srcw - 2;
            fx = 1.f;
        }

        xofs[dx] = sx*3;

        float a0 = (1.f - fx) * INTER_RESIZE_COEF_SCALE;
        float a1 =        fx  * INTER_RESIZE_COEF_SCALE;

        ialpha[dx*2    ] = SATURATE_CAST_SHORT(a0);
        ialpha[dx*2 + 1] = SATURATE_CAST_SHORT(a1);
    }

    for (int dy = 0; dy < h; dy++)
    {
        fy = (float)((dy + 0.5) * scale_y - 0.5);
        sy = fy;//cvFloor(fy);
        fy -= sy;

        if (sy >= srch - 1)
        {
            sy = srch - 2;
            fy = 1.f;
        }

        yofs[dy] = sy*3;

        float b0 = (1.f - fy) * INTER_RESIZE_COEF_SCALE;
        float b1 =        fy  * INTER_RESIZE_COEF_SCALE;

        ibeta[dy*2    ] = SATURATE_CAST_SHORT(b0);
        ibeta[dy*2 + 1] = SATURATE_CAST_SHORT(b1);
    }

#undef SATURATE_CAST_SHORT

    // loop body
    Mat rowsbuf0((w*3 >> 1) + 3);
    Mat rowsbuf1((w*3 >> 1) + 3);
    short* rows0 = (short*)rowsbuf0.data;
    short* rows1 = (short*)rowsbuf1.data;

    int prev_sy1 = -1;

    for (int dy = 0; dy < h; dy++ )
    {
        int sy = yofs[dy];

        if (sy == prev_sy1)
        {
            // hresize one row
            short* rows0_old = rows0;
            rows0 = rows1;
            rows1 = rows0_old;
            const unsigned char *S1 = src + srcw * (sy+3);

            const short* ialphap = ialpha;
            short* rows1p = rows1;
            for ( int dx = 0; dx < w; dx++ )
            {
                int sx = xofs[dx];
                short a0 = ialphap[0];
                short a1 = ialphap[1];

                const unsigned char* S1p = S1 + sx;
#if __ARM_NEON
                int16x4_t _a0 = vdup_n_s16(a0);
                int16x4_t _a1 = vdup_n_s16(a1);
                uint8x8_t _S1 = vld1_u8(S1p);
                int16x8_t _S116 = vreinterpretq_s16_u16(vmovl_u8(_S1));
                int16x4_t _S1low = vget_low_s16(_S116);
                int16x4_t _S1high = vext_s16(_S1low, vget_high_s16(_S116), 3);
                int32x4_t _rows1 = vmull_s16(_S1low, _a0);
                _rows1 = vmlal_s16(_rows1, _S1high, _a1);
                int16x4_t _rows1_sr4 = vshrn_n_s32(_rows1, 4);
                vst1_s16(rows1p, _rows1_sr4);
#else
                rows1p[0] = (S1p[0]*a0 + S1p[3]*a1) >> 4;
                rows1p[1] = (S1p[1]*a0 + S1p[4]*a1) >> 4;
                rows1p[2] = (S1p[2]*a0 + S1p[5]*a1) >> 4;
#endif // __ARM_NEON

                ialphap += 2;
                rows1p += 3;
            }
        }
        else
        {
            // hresize two rows
            const unsigned char *S0 = src + srcw * (sy);
            const unsigned char *S1 = src + srcw * (sy+3);

            const short* ialphap = ialpha;
            short* rows0p = rows0;
            short* rows1p = rows1;
            for ( int dx = 0; dx < w; dx++ )
            {
                int sx = xofs[dx];
                short a0 = ialphap[0];
                short a1 = ialphap[1];

                const unsigned char* S0p = S0 + sx;
                const unsigned char* S1p = S1 + sx;
#if __ARM_NEON
                int16x4_t _a0 = vdup_n_s16(a0);
                int16x4_t _a1 = vdup_n_s16(a1);
                uint8x8_t _S0 = vld1_u8(S0p);
                uint8x8_t _S1 = vld1_u8(S1p);
                int16x8_t _S016 = vreinterpretq_s16_u16(vmovl_u8(_S0));
                int16x8_t _S116 = vreinterpretq_s16_u16(vmovl_u8(_S1));
                int16x4_t _S0low = vget_low_s16(_S016);
                int16x4_t _S1low = vget_low_s16(_S116);
                int16x4_t _S0high = vext_s16(_S0low, vget_high_s16(_S016), 3);
                int16x4_t _S1high = vext_s16(_S1low, vget_high_s16(_S116), 3);
                int32x4_t _rows0 = vmull_s16(_S0low, _a0);
                int32x4_t _rows1 = vmull_s16(_S1low, _a0);
                _rows0 = vmlal_s16(_rows0, _S0high, _a1);
                _rows1 = vmlal_s16(_rows1, _S1high, _a1);
                int16x4_t _rows0_sr4 = vshrn_n_s32(_rows0, 4);
                int16x4_t _rows1_sr4 = vshrn_n_s32(_rows1, 4);
                vst1_s16(rows0p, _rows0_sr4);
                vst1_s16(rows1p, _rows1_sr4);
#else
                rows0p[0] = (S0p[0]*a0 + S0p[3]*a1) >> 4;
                rows0p[1] = (S0p[1]*a0 + S0p[4]*a1) >> 4;
                rows0p[2] = (S0p[2]*a0 + S0p[5]*a1) >> 4;
                rows1p[0] = (S1p[0]*a0 + S1p[3]*a1) >> 4;
                rows1p[1] = (S1p[1]*a0 + S1p[4]*a1) >> 4;
                rows1p[2] = (S1p[2]*a0 + S1p[5]*a1) >> 4;
#endif // __ARM_NEON

                ialphap += 2;
                rows0p += 3;
                rows1p += 3;
            }
        }

        prev_sy1 = sy + 1;

        // vresize
        short b0 = ibeta[0];
        short b1 = ibeta[1];

        short* rows0p = rows0;
        short* rows1p = rows1;
        unsigned char* Dp = dst + w * 3 * (dy);

#if __ARM_NEON
        int nn = (w * 3) >> 3;
#else
        int nn = 0;
#endif
        int remain = (w * 3) - (nn << 3);

#if __ARM_NEON
#if __aarch64__
        int16x4_t _b0 = vdup_n_s16(b0);
        int16x4_t _b1 = vdup_n_s16(b1);
        int32x4_t _v2 = vdupq_n_s32(2);
        for (; nn>0; nn--)
        {
            int16x4_t _rows0p_sr4 = vld1_s16(rows0p);
            int16x4_t _rows1p_sr4 = vld1_s16(rows1p);
            int16x4_t _rows0p_1_sr4 = vld1_s16(rows0p+4);
            int16x4_t _rows1p_1_sr4 = vld1_s16(rows1p+4);

            int32x4_t _rows0p_sr4_mb0 = vmull_s16(_rows0p_sr4, _b0);
            int32x4_t _rows1p_sr4_mb1 = vmull_s16(_rows1p_sr4, _b1);
            int32x4_t _rows0p_1_sr4_mb0 = vmull_s16(_rows0p_1_sr4, _b0);
            int32x4_t _rows1p_1_sr4_mb1 = vmull_s16(_rows1p_1_sr4, _b1);

            int32x4_t _acc = _v2;
            _acc = vsraq_n_s32(_acc, _rows0p_sr4_mb0, 16);
            _acc = vsraq_n_s32(_acc, _rows1p_sr4_mb1, 16);

            int32x4_t _acc_1 = _v2;
            _acc_1 = vsraq_n_s32(_acc_1, _rows0p_1_sr4_mb0, 16);
            _acc_1 = vsraq_n_s32(_acc_1, _rows1p_1_sr4_mb1, 16);

            int16x4_t _acc16 = vshrn_n_s32(_acc, 2);
            int16x4_t _acc16_1 = vshrn_n_s32(_acc_1, 2);

            uint8x8_t _D = vqmovun_s16(vcombine_s16(_acc16, _acc16_1));

            vst1_u8(Dp, _D);

            Dp += 8;
            rows0p += 8;
            rows1p += 8;
        }
#else
        if (nn > 0)
        {
            asm volatile(
                "vdup.s16   d16, %8         \n"
                "mov        r4, #2          \n"
                "vdup.s16   d17, %9         \n"
                "vdup.s32   q12, r4         \n"
                "pld        [%0, #128]      \n"
                "vld1.s16   {d2-d3}, [%0 :128]!\n"
                "pld        [%1, #128]      \n"
                "vld1.s16   {d6-d7}, [%1 :128]!\n"
                "0:                         \n"
                "vmull.s16  q0, d2, d16     \n"
                "vmull.s16  q1, d3, d16     \n"
                "vorr.s32   q10, q12, q12   \n"
                "vorr.s32   q11, q12, q12   \n"
                "vmull.s16  q2, d6, d17     \n"
                "vmull.s16  q3, d7, d17     \n"
                "vsra.s32   q10, q0, #16    \n"
                "vsra.s32   q11, q1, #16    \n"
                "pld        [%0, #128]      \n"
                "vld1.s16   {d2-d3}, [%0 :128]!\n"
                "vsra.s32   q10, q2, #16    \n"
                "vsra.s32   q11, q3, #16    \n"
                "pld        [%1, #128]      \n"
                "vld1.s16   {d6-d7}, [%1 :128]!\n"
                "vshrn.s32  d20, q10, #2    \n"
                "vshrn.s32  d21, q11, #2    \n"
                "vqmovun.s16 d20, q10        \n"
                "vst1.8     {d20}, [%2]!    \n"
                "subs       %3, #1          \n"
                "bne        0b              \n"
                "sub        %0, #16         \n"
                "sub        %1, #16         \n"
                : "=r"(rows0p), // %0
                "=r"(rows1p), // %1
                "=r"(Dp),     // %2
                "=r"(nn)      // %3
                : "0"(rows0p),
                "1"(rows1p),
                "2"(Dp),
                "3"(nn),
                "r"(b0),      // %8
                "r"(b1)       // %9
                : "cc", "memory", "r4", "q0", "q1", "q2", "q3", "q8", "q9", "q10", "q11", "q12"
            );
        }
#endif // __aarch64__
#endif // __ARM_NEON
        for ( ; remain; --remain )
        {
//             D[x] = (rows0[x]*b0 + rows1[x]*b1) >> INTER_RESIZE_COEF_BITS;
            *Dp++ = (unsigned char)(( (short)((b0 * (short)(*rows0p++)) >> 16) + (short)((b1 * (short)(*rows1p++)) >> 16) + 2)>>2);
        }

        ibeta += 2;
    }

    delete[] buf;
}

void resize_bilinear_c1(const unsigned char* src, int srcw, int srch, unsigned char* dst, int w, int h)
{
    const int INTER_RESIZE_COEF_BITS=11;
    const int INTER_RESIZE_COEF_SCALE=1 << INTER_RESIZE_COEF_BITS;
//     const int ONE=INTER_RESIZE_COEF_SCALE;

    double scale_x = (double)srcw / w;
    double scale_y = (double)srch / h;

    int* buf = new int[w + h + w + h];

    int* xofs = buf;//new int[w];
    int* yofs = buf + w;//new int[h];

    short* ialpha = (short*)(buf + w + h);//new short[w * 2];
    short* ibeta = (short*)(buf + w + h + w);//new short[h * 2];

    float fx;
    float fy;
    int sx;
    int sy;

#define SATURATE_CAST_SHORT(X) (short)::std::min(::std::max((int)(X + (X >= 0.f ? 0.5f : -0.5f)), SHRT_MIN), SHRT_MAX);

    for (int dx = 0; dx < w; dx++)
    {
        fx = (float)((dx + 0.5) * scale_x - 0.5);
        sx = fx;//cvFloor(fx);
        fx -= sx;

        if (sx >= srcw - 1)
        {
            sx = srcw - 2;
            fx = 1.f;
        }

        xofs[dx] = sx;

        float a0 = (1.f - fx) * INTER_RESIZE_COEF_SCALE;
        float a1 =        fx  * INTER_RESIZE_COEF_SCALE;

        ialpha[dx*2    ] = SATURATE_CAST_SHORT(a0);
        ialpha[dx*2 + 1] = SATURATE_CAST_SHORT(a1);
    }

    for (int dy = 0; dy < h; dy++)
    {
        fy = (float)((dy + 0.5) * scale_y - 0.5);
        sy = fy;//cvFloor(fy);
        fy -= sy;

        if (sy >= srch - 1)
        {
            sy = srch - 2;
            fy = 1.f;
        }

        yofs[dy] = sy;

        float b0 = (1.f - fy) * INTER_RESIZE_COEF_SCALE;
        float b1 =        fy  * INTER_RESIZE_COEF_SCALE;

        ibeta[dy*2    ] = SATURATE_CAST_SHORT(b0);
        ibeta[dy*2 + 1] = SATURATE_CAST_SHORT(b1);
    }

#undef SATURATE_CAST_SHORT

    // loop body
    Mat rowsbuf0((w >> 1) + 1);
    Mat rowsbuf1((w >> 1) + 1);
    short* rows0 = (short*)rowsbuf0.data;
    short* rows1 = (short*)rowsbuf1.data;

    int prev_sy1 = -1;

    for (int dy = 0; dy < h; dy++ )
    {
        int sy = yofs[dy];

        if (sy == prev_sy1)
        {
            // hresize one row
            short* rows0_old = rows0;
            rows0 = rows1;
            rows1 = rows0_old;
            const unsigned char *S1 = src + srcw * (sy+1);

            const short* ialphap = ialpha;
            short* rows1p = rows1;
            for ( int dx = 0; dx < w; dx++ )
            {
                int sx = xofs[dx];
                short a0 = ialphap[0];
                short a1 = ialphap[1];

                const unsigned char* S1p = S1 + sx;
                rows1p[dx] = (S1p[0]*a0 + S1p[1]*a1) >> 4;

                ialphap += 2;
            }
        }
        else
        {
            // hresize two rows
            const unsigned char *S0 = src + srcw * (sy);
            const unsigned char *S1 = src + srcw * (sy+1);

            const short* ialphap = ialpha;
            short* rows0p = rows0;
            short* rows1p = rows1;
            for ( int dx = 0; dx < w; dx++ )
            {
                int sx = xofs[dx];
                short a0 = ialphap[0];
                short a1 = ialphap[1];

                const unsigned char* S0p = S0 + sx;
                const unsigned char* S1p = S1 + sx;
                rows0p[dx] = (S0p[0]*a0 + S0p[1]*a1) >> 4;
                rows1p[dx] = (S1p[0]*a0 + S1p[1]*a1) >> 4;

                ialphap += 2;
            }
        }

        prev_sy1 = sy + 1;

        // vresize
        short b0 = ibeta[0];
        short b1 = ibeta[1];

        short* rows0p = rows0;
        short* rows1p = rows1;
        unsigned char* Dp = dst + w * (dy);

#if __ARM_NEON
        int nn = w >> 3;
#else
        int nn = 0;
#endif
        int remain = w - (nn << 3);

#if __ARM_NEON
#if __aarch64__
        int16x4_t _b0 = vdup_n_s16(b0);
        int16x4_t _b1 = vdup_n_s16(b1);
        int32x4_t _v2 = vdupq_n_s32(2);
        for (; nn>0; nn--)
        {
            int16x4_t _rows0p_sr4 = vld1_s16(rows0p);
            int16x4_t _rows1p_sr4 = vld1_s16(rows1p);
            int16x4_t _rows0p_1_sr4 = vld1_s16(rows0p+4);
            int16x4_t _rows1p_1_sr4 = vld1_s16(rows1p+4);

            int32x4_t _rows0p_sr4_mb0 = vmull_s16(_rows0p_sr4, _b0);
            int32x4_t _rows1p_sr4_mb1 = vmull_s16(_rows1p_sr4, _b1);
            int32x4_t _rows0p_1_sr4_mb0 = vmull_s16(_rows0p_1_sr4, _b0);
            int32x4_t _rows1p_1_sr4_mb1 = vmull_s16(_rows1p_1_sr4, _b1);

            int32x4_t _acc = _v2;
            _acc = vsraq_n_s32(_acc, _rows0p_sr4_mb0, 16);
            _acc = vsraq_n_s32(_acc, _rows1p_sr4_mb1, 16);

            int32x4_t _acc_1 = _v2;
            _acc_1 = vsraq_n_s32(_acc_1, _rows0p_1_sr4_mb0, 16);
            _acc_1 = vsraq_n_s32(_acc_1, _rows1p_1_sr4_mb1, 16);

            int16x4_t _acc16 = vshrn_n_s32(_acc, 2);
            int16x4_t _acc16_1 = vshrn_n_s32(_acc_1, 2);

            uint8x8_t _D = vqmovun_s16(vcombine_s16(_acc16, _acc16_1));

            vst1_u8(Dp, _D);

            Dp += 8;
            rows0p += 8;
            rows1p += 8;
        }
#else
        if (nn > 0)
        {
            asm volatile(
                "vdup.s16   d16, %8         \n"
                "mov        r4, #2          \n"
                "vdup.s16   d17, %9         \n"
                "vdup.s32   q12, r4         \n"
                "pld        [%0, #128]      \n"
                "vld1.s16   {d2-d3}, [%0 :128]!\n"
                "pld        [%1, #128]      \n"
                "vld1.s16   {d6-d7}, [%1 :128]!\n"
                "0:                         \n"
                "vmull.s16  q0, d2, d16     \n"
                "vmull.s16  q1, d3, d16     \n"
                "vorr.s32   q10, q12, q12   \n"
                "vorr.s32   q11, q12, q12   \n"
                "vmull.s16  q2, d6, d17     \n"
                "vmull.s16  q3, d7, d17     \n"
                "vsra.s32   q10, q0, #16    \n"
                "vsra.s32   q11, q1, #16    \n"
                "pld        [%0, #128]      \n"
                "vld1.s32   {d2-d3}, [%0 :128]!\n"
                "vsra.s32   q10, q2, #16    \n"
                "vsra.s32   q11, q3, #16    \n"
                "pld        [%1, #128]      \n"
                "vld1.s32   {d6-d7}, [%1 :128]!\n"
                "vshrn.s32  d20, q10, #2    \n"
                "vshrn.s32  d21, q11, #2    \n"
                "vqmovun.s16 d20, q10        \n"
                "vst1.8     {d20}, [%2]!    \n"
                "subs       %3, #1          \n"
                "bne        0b              \n"
                "sub        %0, #16         \n"
                "sub        %1, #16         \n"
                : "=r"(rows0p), // %0
                "=r"(rows1p), // %1
                "=r"(Dp),     // %2
                "=r"(nn)      // %3
                : "0"(rows0p),
                "1"(rows1p),
                "2"(Dp),
                "3"(nn),
                "r"(b0),      // %8
                "r"(b1)       // %9
                : "cc", "memory", "r4", "q0", "q1", "q2", "q3", "q8", "q9", "q10", "q11", "q12"
            );
        }
#endif // __aarch64__
#endif // __ARM_NEON
        for ( ; remain; --remain )
        {
//             D[x] = (rows0[x]*b0 + rows1[x]*b1) >> INTER_RESIZE_COEF_BITS;
            *Dp++ = (unsigned char)(( (short)((b0 * (short)(*rows0p++)) >> 16) + (short)((b1 * (short)(*rows1p++)) >> 16) + 2)>>2);
        }

        ibeta += 2;
    }

    delete[] buf;
}

void resize_bilinear_c4(const unsigned char* src, int srcw, int srch, unsigned char* dst, int w, int h)
{
    const int INTER_RESIZE_COEF_BITS=11;
    const int INTER_RESIZE_COEF_SCALE=1 << INTER_RESIZE_COEF_BITS;
//     const int ONE=INTER_RESIZE_COEF_SCALE;

    double scale_x = (double)srcw / w;
    double scale_y = (double)srch / h;

    int* buf = new int[w + h + w + h];

    int* xofs = buf;//new int[w];
    int* yofs = buf + w;//new int[h];

    short* ialpha = (short*)(buf + w + h);//new short[w * 2];
    short* ibeta = (short*)(buf + w + h + w);//new short[h * 2];

    float fx;
    float fy;
    int sx;
    int sy;

#define SATURATE_CAST_SHORT(X) (short)::std::min(::std::max((int)(X + (X >= 0.f ? 0.5f : -0.5f)), SHRT_MIN), SHRT_MAX);

    for (int dx = 0; dx < w; dx++)
    {
        fx = (float)((dx + 0.5) * scale_x - 0.5);
        sx = fx;//cvFloor(fx);
        fx -= sx;

        if (sx >= srcw - 1)
        {
            sx = srcw - 2;
            fx = 1.f;
        }

        xofs[dx] = sx*4;

        float a0 = (1.f - fx) * INTER_RESIZE_COEF_SCALE;
        float a1 =        fx  * INTER_RESIZE_COEF_SCALE;

        ialpha[dx*2    ] = SATURATE_CAST_SHORT(a0);
        ialpha[dx*2 + 1] = SATURATE_CAST_SHORT(a1);
    }

    for (int dy = 0; dy < h; dy++)
    {
        fy = (float)((dy + 0.5) * scale_y - 0.5);
        sy = fy;//cvFloor(fy);
        fy -= sy;

        if (sy >= srch - 1)
        {
            sy = srch - 2;
            fy = 1.f;
        }

        yofs[dy] = sy*4;

        float b0 = (1.f - fy) * INTER_RESIZE_COEF_SCALE;
        float b1 =        fy  * INTER_RESIZE_COEF_SCALE;

        ibeta[dy*2    ] = SATURATE_CAST_SHORT(b0);
        ibeta[dy*2 + 1] = SATURATE_CAST_SHORT(b1);
    }

#undef SATURATE_CAST_SHORT

    // loop body
    Mat rowsbuf0((w*4 >> 1) + 4);
    Mat rowsbuf1((w*4 >> 1) + 4);
    short* rows0 = (short*)rowsbuf0.data;
    short* rows1 = (short*)rowsbuf1.data;

    int prev_sy1 = -1;

    for (int dy = 0; dy < h; dy++ )
    {
        int sy = yofs[dy];

        if (sy == prev_sy1)
        {
            // hresize one row
            short* rows0_old = rows0;
            rows0 = rows1;
            rows1 = rows0_old;
            const unsigned char *S1 = src + srcw * (sy+4);

            const short* ialphap = ialpha;
            short* rows1p = rows1;
            for ( int dx = 0; dx < w; dx++ )
            {
                int sx = xofs[dx];
                short a0 = ialphap[0];
                short a1 = ialphap[1];

                const unsigned char* S1p = S1 + sx;
#if __ARM_NEON
                int16x4_t _a0 = vdup_n_s16(a0);
                int16x4_t _a1 = vdup_n_s16(a1);
                uint8x8_t _S1 = vld1_u8(S1p);
                int16x8_t _S116 = vreinterpretq_s16_u16(vmovl_u8(_S1));
                int16x4_t _S1low = vget_low_s16(_S116);
                int16x4_t _S1high = vget_high_s16(_S116);
                int32x4_t _rows1 = vmull_s16(_S1low, _a0);
                _rows1 = vmlal_s16(_rows1, _S1high, _a1);
                int16x4_t _rows1_sr4 = vshrn_n_s32(_rows1, 4);
                vst1_s16(rows1p, _rows1_sr4);
#else
                rows1p[0] = (S1p[0]*a0 + S1p[4]*a1) >> 4;
                rows1p[1] = (S1p[1]*a0 + S1p[5]*a1) >> 4;
                rows1p[2] = (S1p[2]*a0 + S1p[6]*a1) >> 4;
                rows1p[3] = (S1p[3]*a0 + S1p[7]*a1) >> 4;
#endif // __ARM_NEON

                ialphap += 2;
                rows1p += 4;
            }
        }
        else
        {
            // hresize two rows
            const unsigned char *S0 = src + srcw * (sy);
            const unsigned char *S1 = src + srcw * (sy+4);

            const short* ialphap = ialpha;
            short* rows0p = rows0;
            short* rows1p = rows1;
            for ( int dx = 0; dx < w; dx++ )
            {
                int sx = xofs[dx];
                short a0 = ialphap[0];
                short a1 = ialphap[1];

                const unsigned char* S0p = S0 + sx;
                const unsigned char* S1p = S1 + sx;
#if __ARM_NEON
                int16x4_t _a0 = vdup_n_s16(a0);
                int16x4_t _a1 = vdup_n_s16(a1);
                uint8x8_t _S0 = vld1_u8(S0p);
                uint8x8_t _S1 = vld1_u8(S1p);
                int16x8_t _S016 = vreinterpretq_s16_u16(vmovl_u8(_S0));
                int16x8_t _S116 = vreinterpretq_s16_u16(vmovl_u8(_S1));
                int16x4_t _S0low = vget_low_s16(_S016);
                int16x4_t _S1low = vget_low_s16(_S116);
                int16x4_t _S0high = vget_high_s16(_S016);
                int16x4_t _S1high = vget_high_s16(_S116);
                int32x4_t _rows0 = vmull_s16(_S0low, _a0);
                int32x4_t _rows1 = vmull_s16(_S1low, _a0);
                _rows0 = vmlal_s16(_rows0, _S0high, _a1);
                _rows1 = vmlal_s16(_rows1, _S1high, _a1);
                int16x4_t _rows0_sr4 = vshrn_n_s32(_rows0, 4);
                int16x4_t _rows1_sr4 = vshrn_n_s32(_rows1, 4);
                vst1_s16(rows0p, _rows0_sr4);
                vst1_s16(rows1p, _rows1_sr4);
#else
                rows0p[0] = (S0p[0]*a0 + S0p[4]*a1) >> 4;
                rows0p[1] = (S0p[1]*a0 + S0p[5]*a1) >> 4;
                rows0p[2] = (S0p[2]*a0 + S0p[6]*a1) >> 4;
                rows0p[3] = (S0p[3]*a0 + S0p[7]*a1) >> 4;
                rows1p[0] = (S1p[0]*a0 + S1p[4]*a1) >> 4;
                rows1p[1] = (S1p[1]*a0 + S1p[5]*a1) >> 4;
                rows1p[2] = (S1p[2]*a0 + S1p[6]*a1) >> 4;
                rows1p[3] = (S1p[3]*a0 + S1p[7]*a1) >> 4;
#endif // __ARM_NEON

                ialphap += 2;
                rows0p += 4;
                rows1p += 4;
            }
        }

        prev_sy1 = sy + 1;

        // vresize
        short b0 = ibeta[0];
        short b1 = ibeta[1];

        short* rows0p = rows0;
        short* rows1p = rows1;
        unsigned char* Dp = dst + w * 4 * (dy);

#if __ARM_NEON
        int nn = (w * 4) >> 3;
#else
        int nn = 0;
#endif
        int remain = (w * 4) - (nn << 3);

#if __ARM_NEON
#if __aarch64__
        int16x4_t _b0 = vdup_n_s16(b0);
        int16x4_t _b1 = vdup_n_s16(b1);
        int32x4_t _v2 = vdupq_n_s32(2);
        for (; nn>0; nn--)
        {
            int16x4_t _rows0p_sr4 = vld1_s16(rows0p);
            int16x4_t _rows1p_sr4 = vld1_s16(rows1p);
            int16x4_t _rows0p_1_sr4 = vld1_s16(rows0p+4);
            int16x4_t _rows1p_1_sr4 = vld1_s16(rows1p+4);

            int32x4_t _rows0p_sr4_mb0 = vmull_s16(_rows0p_sr4, _b0);
            int32x4_t _rows1p_sr4_mb1 = vmull_s16(_rows1p_sr4, _b1);
            int32x4_t _rows0p_1_sr4_mb0 = vmull_s16(_rows0p_1_sr4, _b0);
            int32x4_t _rows1p_1_sr4_mb1 = vmull_s16(_rows1p_1_sr4, _b1);

            int32x4_t _acc = _v2;
            _acc = vsraq_n_s32(_acc, _rows0p_sr4_mb0, 16);
            _acc = vsraq_n_s32(_acc, _rows1p_sr4_mb1, 16);

            int32x4_t _acc_1 = _v2;
            _acc_1 = vsraq_n_s32(_acc_1, _rows0p_1_sr4_mb0, 16);
            _acc_1 = vsraq_n_s32(_acc_1, _rows1p_1_sr4_mb1, 16);

            int16x4_t _acc16 = vshrn_n_s32(_acc, 2);
            int16x4_t _acc16_1 = vshrn_n_s32(_acc_1, 2);

            uint8x8_t _D = vqmovun_s16(vcombine_s16(_acc16, _acc16_1));

            vst1_u8(Dp, _D);

            Dp += 8;
            rows0p += 8;
            rows1p += 8;
        }
#else
        if (nn > 0)
        {
            asm volatile(
                "vdup.s16   d16, %8         \n"
                "mov        r4, #2          \n"
                "vdup.s16   d17, %9         \n"
                "vdup.s32   q12, r4         \n"
                "pld        [%0, #128]      \n"
                "vld1.s16   {d2-d3}, [%0 :128]!\n"
                "pld        [%1, #128]      \n"
                "vld1.s16   {d6-d7}, [%1 :128]!\n"
                "0:                         \n"
                "vmull.s16  q0, d2, d16     \n"
                "vmull.s16  q1, d3, d16     \n"
                "vorr.s32   q10, q12, q12   \n"
                "vorr.s32   q11, q12, q12   \n"
                "vmull.s16  q2, d6, d17     \n"
                "vmull.s16  q3, d7, d17     \n"
                "vsra.s32   q10, q0, #16    \n"
                "vsra.s32   q11, q1, #16    \n"
                "pld        [%0, #128]      \n"
                "vld1.s32   {d2-d3}, [%0 :128]!\n"
                "vsra.s32   q10, q2, #16    \n"
                "vsra.s32   q11, q3, #16    \n"
                "pld        [%1, #128]      \n"
                "vld1.s32   {d6-d7}, [%1 :128]!\n"
                "vshrn.s32  d20, q10, #2    \n"
                "vshrn.s32  d21, q11, #2    \n"
                "vqmovun.s16 d20, q10        \n"
                "vst1.8     {d20}, [%2]!    \n"
                "subs       %3, #1          \n"
                "bne        0b              \n"
                "sub        %0, #16         \n"
                "sub        %1, #16         \n"
                : "=r"(rows0p), // %0
                "=r"(rows1p), // %1
                "=r"(Dp),     // %2
                "=r"(nn)      // %3
                : "0"(rows0p),
                "1"(rows1p),
                "2"(Dp),
                "3"(nn),
                "r"(b0),      // %8
                "r"(b1)       // %9
                : "cc", "memory", "r4", "q0", "q1", "q2", "q3", "q8", "q9", "q10", "q11", "q12"
            );
        }
#endif // __aarch64__
#endif // __ARM_NEON
        for ( ; remain; --remain )
        {
//             D[x] = (rows0[x]*b0 + rows1[x]*b1) >> INTER_RESIZE_COEF_BITS;
            *Dp++ = (unsigned char)(( (short)((b0 * (short)(*rows0p++)) >> 16) + (short)((b1 * (short)(*rows1p++)) >> 16) + 2)>>2);
        }

        ibeta += 2;
    }

    delete[] buf;
}

Mat Mat::from_pixels(const unsigned char* pixels, int type, int w, int h)
{
    if (type & PIXEL_CONVERT_MASK)
    {
        if (type == PIXEL_RGB2BGR || type == PIXEL_BGR2RGB)
            return from_rgb2bgr(pixels, w, h);

        if (type == PIXEL_RGB2GRAY)
            return from_rgb2gray(pixels, w, h);

        if (type == PIXEL_BGR2GRAY)
            return from_bgr2gray(pixels, w, h);

        if (type == PIXEL_GRAY2RGB || type == PIXEL_GRAY2BGR)
            return from_gray2rgb(pixels, w, h);

        if (type == PIXEL_RGBA2RGB)
            return from_rgba2rgb(pixels, w, h);

        if (type == PIXEL_RGBA2BGR)
            return from_rgba2bgr(pixels, w, h);

        if (type == PIXEL_RGBA2GRAY)
            return from_rgba2gray(pixels, w, h);
    }
    else
    {
        if (type == PIXEL_RGB || type == PIXEL_BGR)
            return from_rgb(pixels, w, h);

        if (type == PIXEL_GRAY)
            return from_gray(pixels, w, h);

        if (type == PIXEL_RGBA)
            return from_rgba(pixels, w, h);
    }

    return Mat();
}

Mat Mat::from_pixels_resize(const unsigned char* pixels, int type, int w, int h, int target_width, int target_height)
{
    if (w == target_width && h == target_height)
        return Mat::from_pixels(pixels, type, w, h);

    Mat m;

    int type_from = type & PIXEL_FORMAT_MASK;

    if (type_from == PIXEL_RGB || type_from == PIXEL_BGR)
    {
        unsigned char* dst = new unsigned char[target_width * target_height * 3];

        resize_bilinear_c3(pixels, w, h, dst, target_width, target_height);

        m = Mat::from_pixels(dst, type, target_width, target_height);

        delete[] dst;
    }
    else if (type_from == PIXEL_GRAY)
    {
        unsigned char* dst = new unsigned char[target_width * target_height];

        resize_bilinear_c1(pixels, w, h, dst, target_width, target_height);

        m = Mat::from_pixels(dst, type, target_width, target_height);

        delete[] dst;
    }
    else if (type_from == PIXEL_RGBA)
    {
        unsigned char* dst = new unsigned char[target_width * target_height * 4];

        resize_bilinear_c4(pixels, w, h, dst, target_width, target_height);

        m = Mat::from_pixels(dst, type, target_width, target_height);

        delete[] dst;
    }

    return m;
}

void Mat::to_pixels(unsigned char* pixels, int type)
{
    if (type & PIXEL_CONVERT_MASK)
    {
        if (type == PIXEL_RGB2BGR || type == PIXEL_BGR2RGB)
            return to_bgr2rgb(*this, pixels);
    }
    else
    {
        if (type == PIXEL_RGB || type == PIXEL_BGR)
            return to_rgb(*this, pixels);

        if (type == PIXEL_GRAY)
            return to_gray(*this, pixels);

        if (type == PIXEL_RGBA)
            return to_rgba(*this, pixels);
    }
}

void Mat::to_pixels_resize(unsigned char* pixels, int type, int target_width, int target_height)
{
    if (w == target_width && h == target_height)
        return to_pixels(pixels, type);

    int type_to = (type & PIXEL_CONVERT_MASK) ? (type >> PIXEL_CONVERT_SHIFT) : (type & PIXEL_FORMAT_MASK);

    if (type_to == PIXEL_RGB || type_to == PIXEL_BGR)
    {
        unsigned char* src = new unsigned char[w * h * 3];

        to_pixels(src, type);

        resize_bilinear_c3(src, w, h, pixels, target_width, target_height);

        delete[] src;
    }
    else if (type_to == PIXEL_GRAY)
    {
        unsigned char* src = new unsigned char[w * h];

        to_pixels(src, type);

        resize_bilinear_c1(src, w, h, pixels, target_width, target_height);

        delete[] src;
    }
    else if (type_to == PIXEL_RGBA)
    {
        unsigned char* src = new unsigned char[w * h * 4];

        to_pixels(src, type);

        resize_bilinear_c4(src, w, h, pixels, target_width, target_height);

        delete[] src;
    }
}

} // namespace ncnn
