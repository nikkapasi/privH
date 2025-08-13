#include "cfar_hls.h"

// Function to convert decibels to power
data_t db2pow_hls(data_t db) {
    return hls::pow(10.0, db / 10.0);
}

// Function to convert power to decibels
data_t pow2db_hls(data_t pow) {
    return 10.0 * hls::log10(pow);
}

void cfar_hls_detector(
    hls::stream<data_t>& rdm_stream_in,
    hls::stream<data_t>& cfar_stream_out,
    int range_bins,
    int doppler_bins,
    int Tr,
    int Td,
    int Gr,
    int Gd,
    data_t offset_db)
{
    // AXI-Lite interfaces for control signals
#pragma HLS INTERFACE s_axilite port=range_bins bundle=control
#pragma HLS INTERFACE s_axilite port=doppler_bins bundle=control
#pragma HLS INTERFACE s_axilite port=Tr bundle=control
#pragma HLS INTERFACE s_axilite port=Td bundle=control
#pragma HLS INTERFACE s_axilite port=Gr bundle=control
#pragma HLS INTERFACE s_axilite port=Gd bundle=control
#pragma HLS INTERFACE s_axilite port=offset_db bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    // AXI-Stream interfaces for data
#pragma HLS INTERFACE axis port=rdm_stream_in
#pragma HLS INTERFACE axis port=cfar_stream_out

    int radius_range = Tr + Gr;
    int radius_doppler = Td + Gd;

    // Line buffer to store rows of the RDM
    data_t line_buffer[2 * 8 + 1][100];
#pragma HLS ARRAY_PARTITION variable=line_buffer complete dim=1

    // Window buffer to store the current processing window
    data_t window_buffer[2 * 8 + 1][2 * 8 + 1];
#pragma HLS ARRAY_PARTITION variable=window_buffer complete dim=0

    // Loop over the RDM
    for (int r = 0; r < range_bins; ++r) {
        for (int d = 0; d < doppler_bins; ++d) {
#pragma HLS PIPELINE II=1

            // Read data from the input stream
            data_t in_data = rdm_stream_in.read();

            // Shift the line buffer
            for (int i = 0; i < 2 * radius_range; ++i) {
                line_buffer[i][d] = line_buffer[i + 1][d];
            }
            line_buffer[2 * radius_range][d] = in_data;

            // Shift the window buffer
            for (int i = 0; i < 2 * radius_range + 1; ++i) {
                for (int j = 0; j < 2 * radius_doppler; ++j) {
                    window_buffer[i][j] = window_buffer[i][j + 1];
                }
            }

            // Load new column into the window buffer
            for (int i = 0; i < 2 * radius_range + 1; ++i) {
                window_buffer[i][2 * radius_doppler] = line_buffer[i][d];
            }

            // Perform CFAR calculation
            if (r >= 2 * radius_range && d >= 2 * radius_doppler) {
                data_t noise_level = 0.0;
                int cell_count = 0;

                for (int wr = 0; wr < 2 * radius_range + 1; ++wr) {
                    for (int wd = 0; wd < 2 * radius_doppler + 1; ++wd) {
                        if (hls::abs(wr - radius_range) > Gr || hls::abs(wd - radius_doppler) > Gd) {
                            noise_level += db2pow_hls(window_buffer[wr][wd]);
                            cell_count++;
                        }
                    }
                }
                data_t average_noise = noise_level / cell_count;
                data_t threshold = pow2db_hls(average_noise) + offset_db;

                data_t cut = window_buffer[radius_range][radius_doppler];
                if (cut >= threshold) {
                    cfar_stream_out.write(cut);
                } else {
                    cfar_stream_out.write(0.0);
                }
            } else {
                cfar_stream_out.write(0.0);
            }
        }
    }
}
