#ifndef HAMMING_H
#define HAMMING_H

#include <hls_stream.h>

typedef float data_t;

template<int MAX_R, int MAX_D>
void rdm_hamming2d(hls::stream<data_t> &strm_in,
                   hls::stream<data_t> &strm_out,
                   int numRangeBins, int numVelocityBins);

#endif // HAMMING_H
