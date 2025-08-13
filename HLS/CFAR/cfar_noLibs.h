// ca_cfar_hls.h
#pragma once
#include <cstdint>

extern "C" {

/**
 * 2-D Cell-Averaging CFAR over a Range–Doppler Map (runtime dimensions).
 *
 * Inputs:
 *   rdm         [rows*cols] float, row-major
 *   rows, cols  map size
 *   guard_r,d   guard half-widths
 *   train_r,d   training half-widths (exclude guards)
 *   k           threshold scale (alpha)
 *
 * Outputs:
 *   detections  [rows*cols] uint8_t (0/1) row-major mask
 *   scratch     [rows*cols] float workspace (integral image)
 */
void ca_cfar_2d(
    const float* rdm,
    int rows,
    int cols,
    int guard_r,
    int guard_d,
    int train_r,
    int train_d,
    float k,
    std::uint8_t* detections,
    float* scratch
);

} // extern "C"
