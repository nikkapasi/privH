// ca_cfar_hls.h
#pragma once
#include <ap_int.h>

extern "C" {

/**
 * @brief 2-D Cell-Averaging CFAR over a Range–Doppler Map with runtime sizes.
 *
 * The kernel:
 *   1) builds an integral image of the input RDM (row-major floats),
 *   2) for each CUT (cell under test), averages the training cells
 *      in a (train + guard) window and applies threshold k * mean,
 *   3) writes a 1-bit detection mask (row-major) to memory.
 *
 * Edges where the full CFAR window does not fit are set to 0 (no detection).
 *
 * @param rdm          [rows*cols] input range–Doppler map (row-major, float)
 * @param rows         number of rows (range bins)
 * @param cols         number of cols (Doppler bins)
 * @param guard_r      guard half-width in rows (inner window half-height)
 * @param guard_d      guard half-width in cols (inner window half-width)
 * @param train_r      training half-width in rows (excluding guards)
 * @param train_d      training half-width in cols (excluding guards)
 * @param k            threshold scale (e.g., alpha from desired Pfa)
 * @param detections   [rows*cols] output 1-bit detections (row-major)
 * @param scratch      [rows*cols] workspace for integral image (float)
 */
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
    float *scratch
);

} // extern "C"
