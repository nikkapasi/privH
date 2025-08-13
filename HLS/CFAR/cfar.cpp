// ca_cfar_hls.cpp
#include "ca_cfar_hls.h"

static inline float getS(const float *S, int cols, int r, int c) {
    // Safe fetch from integral image with 0 for r<0 or c<0
    if (r < 0 || c < 0) return 0.0f;
    return S[r * cols + c];
}

static inline float rectSum(const float *S, int cols,
                            int r1, int c1, int r2, int c2) {
    // Inclusive rectangle sum on integral image S for [r1..r2, c1..c2]
    float A = getS(S, cols, r2,   c2);
    float B = getS(S, cols, r1-1, c2);
    float C = getS(S, cols, r2,   c1-1);
    float D = getS(S, cols, r1-1, c1-1);
    return A - B - C + D;
}

extern "C" {
void ca_cfar_2d(
    const float *rdm,
    int rows,
    int cols,
    int guard_r,
    int guard_d,
    int train_r,
    int train_d,
    float k,
    ap_uint<1> *detections,
    float *scratch)
{
#pragma HLS INTERFACE m_axi     port=rdm        offset=slave bundle=gmem0 depth=4096
#pragma HLS INTERFACE m_axi     port=detections offset=slave bundle=gmem1 depth=4096
#pragma HLS INTERFACE m_axi     port=scratch    offset=slave bundle=gmem2 depth=4096
#pragma HLS INTERFACE s_axilite port=rdm        bundle=control
#pragma HLS INTERFACE s_axilite port=detections bundle=control
#pragma HLS INTERFACE s_axilite port=scratch    bundle=control
#pragma HLS INTERFACE s_axilite port=rows       bundle=control
#pragma HLS INTERFACE s_axilite port=cols       bundle=control
#pragma HLS INTERFACE s_axilite port=guard_r    bundle=control
#pragma HLS INTERFACE s_axilite port=guard_d    bundle=control
#pragma HLS INTERFACE s_axilite port=train_r    bundle=control
#pragma HLS INTERFACE s_axilite port=train_d    bundle=control
#pragma HLS INTERFACE s_axilite port=k          bundle=control
#pragma HLS INTERFACE s_axilite port=return     bundle=control

    // -------- Pass 1: build integral image into 'scratch' ----------
    for (int i = 0; i < rows; i++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=16384
        float row_prefix = 0.0f;
        for (int j = 0; j < cols; j++) {
#pragma HLS PIPELINE II=1
            const int idx = i * cols + j;
            const float v = rdm[idx];
            row_prefix += v;
            const float above = (i > 0) ? scratch[(i-1)*cols + j] : 0.0f;
            scratch[idx] = row_prefix + above;
        }
    }

    // Geometry
    const int Wr = train_r + guard_r;
    const int Wd = train_d + guard_d;
    const int outer_h = 2*Wr + 1;
    const int outer_w = 2*Wd + 1;
    const int inner_h = 2*guard_r + 1;
    const int inner_w = 2*guard_d + 1;
    const int train_cells = outer_h * outer_w - inner_h * inner_w;

    // -------- Pass 2: CFAR on each CUT ----------
    for (int i = 0; i < rows; i++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=16384
        for (int j = 0; j < cols; j++) {
#pragma HLS PIPELINE II=1
            const int idx = i * cols + j;

            // Require full outer window to be inside the map
            const int r1 = i - Wr, r2 = i + Wr;
            const int c1 = j - Wd, c2 = j + Wd;
            const int gr1 = i - guard_r, gr2 = i + guard_r;
            const int gc1 = j - guard_d, gc2 = j + guard_d;

            ap_uint<1> det = 0;

            if (r1 >= 0 && c1 >= 0 && r2 < rows && c2 < cols && train_cells > 0) {
                const float sum_outer = rectSum(scratch, cols, r1,  c1,  r2,  c2);
                const float sum_inner = rectSum(scratch, cols, gr1, gc1, gr2, gc2);
                const float noise_sum = sum_outer - sum_inner;
                const float noise_mu  = noise_sum / (float)train_cells;
                const float thr       = k * noise_mu;
                const float cut       = rdm[idx];
                det = (cut > thr) ? ap_uint<1>(1) : ap_uint<1>(0);
            } else {
                det = 0;
            }

            detections[idx] = det;
        }
    }
}
}
