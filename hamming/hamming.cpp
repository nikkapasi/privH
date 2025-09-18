#include "hamming.h"
#include <hls_math.h>

template<int MAX_R, int MAX_D> // MAX_R = max range bins (rows), MAX_D = max Doppler bins (cols)
void rdm_hamming2d(hls::stream<data_t> &strm_in,
                   hls::stream<data_t> &strm_out,
                   int numRangeBins, int numVelocityBins) {
#pragma HLS INTERFACE axis      port=strm_in
#pragma HLS INTERFACE axis      port=strm_out
#pragma HLS INTERFACE s_axilite port=numRangeBins    bundle=CTRL
#pragma HLS INTERFACE s_axilite port=numVelocityBins bundle=CTRL
#pragma HLS INTERFACE s_axilite port=return          bundle=CTRL

    const data_t a0 = 0.54f, a1 = 0.46f;
    const data_t TWOPI = 6.28318530718f;

    // Precompute 1D windows
    static data_t w_r[MAX_D]; // along Doppler (columns)
    static data_t w_d[MAX_R]; // along Range  (rows)
#pragma HLS BIND_STORAGE variable=w_r type=ram_t2p impl=bram
#pragma HLS BIND_STORAGE variable=w_d type=ram_t2p impl=bram

    pre_cols:
    for (int r = 0; r < numVelocityBins; ++r) {
    #pragma HLS PIPELINE II=1
        data_t ang = (numVelocityBins > 1) ? (TWOPI * r / (numVelocityBins - 1)) : 0.f;
        w_r[r] = a0 - a1 * hls::cosf(ang);
    }
    pre_rows:
    for (int d = 0; d < numRangeBins; ++d) {
    #pragma HLS PIPELINE II=1
        data_t ang = (numRangeBins > 1) ? (TWOPI * d / (numRangeBins - 1)) : 0.f;
        w_d[d] = a0 - a1 * hls::cosf(ang);
    }

    // Apply separable 2D window: out = in * w_r[r] * w_d[d]
    rows:
    for (int d = 0; d < numRangeBins; ++d) {
        cols:
        for (int r = 0; r < numVelocityBins; ++r) {
        #pragma HLS PIPELINE II=1
            data_t x = strm_in.read();
            strm_out.write(x * w_r[r] * w_d[d]);
        }
    }
}
