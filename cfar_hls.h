#ifndef CFAR_HLS_H
#define CFAR_HLS_H

#include "hls_stream.h"
#include "ap_fixed.h"
#include "hls_math.h"

// Define the fixed-point type for the RDM data
// 16 bits total, 6 bits for the integer part
typedef ap_fixed<16, 6> data_t;

// HLS top-level function
void cfar_hls_detector(
    hls::stream<data_t>& rdm_stream_in,
    hls::stream<data_t>& cfar_stream_out,
    int range_bins,
    int doppler_bins,
    int Tr,
    int Td,
    int Gr,
    int Gd,
    data_t offset_db
);

#endif // CFAR_HLS_H
