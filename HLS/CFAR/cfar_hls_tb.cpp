#include <iostream>
#include <vector>
#include "cfar_hls.h"

int main() {
    // Test parameters
    const int RANGE_BINS = 100;
    const int DOPPLER_BINS = 100;
    const int TR = 8;
    const int TD = 8;
    const int GR = 4;
    const int GD = 4;
    const data_t OFFSET_DB = 8.0;

    // Generate sample RDM data
    std::vector<std::vector<data_t>> rdm(RANGE_BINS, std::vector<data_t>(DOPPLER_BINS, 10.0));
    rdm[50][50] = 30.0; // Target

    // Prepare streams
    hls::stream<data_t> rdm_stream_in("rdm_stream_in");
    hls::stream<data_t> cfar_stream_out("cfar_stream_out");

    // Stream in the RDM data
    for (int r = 0; r < RANGE_BINS; ++r) {
        for (int d = 0; d < DOPPLER_BINS; ++d) {
            rdm_stream_in.write(rdm[r][d]);
        }
    }

    // Call the HLS DUT
    cfar_hls_detector(rdm_stream_in, cfar_stream_out, RANGE_BINS, DOPPLER_BINS, TR, TD, GR, GD, OFFSET_DB);

    // Check the output
    int errors = 0;
    for (int r = 0; r < RANGE_BINS; ++r) {
        for (int d = 0; d < DOPPLER_BINS; ++d) {
            data_t out_data = cfar_stream_out.read();
            if (r == 50 && d == 50) {
                if (out_data == 0) {
                    errors++;
                }
            } else {
                if (out_data != 0) {
                    errors++;
                }
            }
        }
    }

    // Print result
    if (errors == 0) {
        std::cout << "PASS: Test passed!" << std::endl;
    } else {
        std::cout << "FAIL: Test failed with " << errors << " errors." << std::endl;
    }

    return errors;
}
