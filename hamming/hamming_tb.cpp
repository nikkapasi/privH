// rdm_hamming2d_tb.cpp
#include "hamming.h"
#include <cmath>
#include <cstdio>
#include <vector>
#include <cstdlib>

// -----------------------------
// Simple golden model (CPU)
// -----------------------------
static void golden_hamming2d(std::vector<data_t> &out,
                             const std::vector<data_t> &in,
                             int rows, int cols)
{
    const data_t a0 = 0.54f, a1 = 0.46f;
    const data_t TWOPI = 6.28318530718f;

    std::vector<data_t> w_r(cols), w_d(rows);

    for (int c = 0; c < cols; ++c) {
        float ang = (cols > 1) ? (TWOPI * c / (cols - 1)) : 0.f;
        w_r[c] = a0 - a1 * std::cosf(ang);
    }
    for (int r = 0; r < rows; ++r) {
        float ang = (rows > 1) ? (TWOPI * r / (rows - 1)) : 0.f;
        w_d[r] = a0 - a1 * std::cosf(ang);
    }

    // Row-major: r in [0..rows-1], c in [0..cols-1]
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            int idx = r*cols + c;
            out[idx] = in[idx] * w_r[c] * w_d[r];
        }
    }
}

// -----------------------------
// Test bench
// -----------------------------
int main() {
    // You can change these; must be <= MAX_R/MAX_D
    const int rows = 32;   // numRangeBins
    const int cols = 48;   // numVelocityBins

    // Synthesis-time maxima for the DUT template
    const int MAX_R = 256;
    const int MAX_D = 256;

    // Generate deterministic input
    std::vector<data_t> host_in(rows*cols), host_golden(rows*cols), host_out(rows*cols);

    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            // Any pattern is fine; this one is easy to eyeball in logs
            host_in[r*cols + c] = 0.01f*r + 0.1f*c + 1.0f;
        }
    }

    // Build golden result on CPU
    golden_hamming2d(host_golden, host_in, rows, cols);

    // Drive DUT streams
    hls::stream<data_t> s_in, s_out;

    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c)
            s_in.write(host_in[r*cols + c]);

    // Call DUT
    rdm_hamming2d<MAX_R, MAX_D>(s_in, s_out, rows, cols);

    // Collect DUT output
    for (int i = 0; i < rows*cols; ++i)
        host_out[i] = s_out.read();

    // Compare
    const float EPS = 1e-5f; // loosen if you switch DUT to fixed-point
    int mismatches = 0;

    for (int i = 0; i < rows*cols; ++i) {
        float a = host_out[i], b = host_golden[i];
        float err = std::fabs(a - b);
        if (err > EPS && err > EPS * std::fabs(b)) {
            if (mismatches < 10) { // print first few
                int r = i / cols, c = i % cols;
                std::printf("Mismatch at (r=%d,c=%d): DUT=%g  GOLD=%g  |err|=%g\n", r, c, a, b, err);
            }
            ++mismatches;
        }
    }

    if (mismatches == 0) {
        std::puts("PASS: rdm_hamming2d output matches golden.");
        return 0;
    } else {
        std::printf("FAIL: %d mismatches out of %d elements.\n", mismatches, rows*cols);
        return 1;
    }
}
