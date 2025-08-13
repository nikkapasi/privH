# cfar.py
# Allo-based CA-CFAR (Cell-Averaging CFAR) for 2-D range–Doppler maps.
# Author: ChatGPT (JULES guide)
#
# Requirements:
#   pip install allo  (see https://cornell-zhang.github.io/allo/)
#
# This module provides:
#   - Fixed-size kernels: prefix_rows, prefix_2d, cfar_from_prefix, cfar_top
#   - Dynamic-shape kernels (pre-allocated outputs): *_dyn
#   - Schedules for CPU (LLVM) and HLS (Vitis/Vivado)
#   - A small __main__ demo that runs the CPU version for sanity checking
#
# Notes:
#   - We implement CA-CFAR using a 2-D prefix sum (integral image) so
#     each rectangular sum is O(1).
#   - For hardware, scheduling primitives (pipeline, buffer_at, compose, etc.)
#     are defined at the end in build_* helpers.
#
#   API references:
#     - Schedule primitives: allo.customize(...), .pipeline(), .buffer_at(), .compose(), .build()
#     - Dynamic shapes: use float32[...] and pass sizes as int32 arguments
#
# SPDX-License-Identifier: MIT
from __future__ import annotations

import numpy as np

import allo
from allo.ir.types import float32, int32


# -------------------------
# Fixed-size kernels
# -------------------------

# Example default sizes; override by redefining with your constants when importing
M, N = 1024, 512  # rows (range) x cols (doppler)

# Training/guard half-sizes
Tr, Td = 8, 8
Gr, Gd = 2, 2


def prefix_rows(X: float32[M, N]) -> float32[M, N]:
    """Row-wise prefix sum: R[i, j] = sum_{k=0..j} X[i, k]."""
    R: float32[M, N] = 0.0
    for i in range(M):
        R[i, 0] = X[i, 0]
        for j in range(1, N):
            R[i, j] = R[i, j - 1] + X[i, j]
    return R


def prefix_2d(X: float32[M, N]) -> float32[M, N]:
    """2-D prefix sum S from X using a row prefix intermediate."""
    R = prefix_rows(X)
    S: float32[M, N] = 0.0
    # first row
    for j in range(N):
        S[0, j] = R[0, j]
    # remaining rows
    for i in range(1, M):
        for j in range(N):
            S[i, j] = S[i - 1, j] + R[i, j]
    return S


def cfar_from_prefix(
    X: float32[M, N],
    S: float32[M, N],
    alpha: float32,
) -> int32[M, N]:
    """CA-CFAR using 2-D prefix sums.
    Only computes detections on the interior where windows fit.
    """
    D: int32[M, N] = 0

    # Window geometry
    H_out = 2 * (Tr + Gr) + 1
    W_out = 2 * (Td + Gd) + 1
    H_g = 2 * Gr + 1
    W_g = 2 * Gd + 1
    Ntrain: int32 = H_out * W_out - H_g * W_g

    # Precondition: training+guard extents must be >= 1 in both dims
    # (Otherwise -1 indexing appears; CFAR without training data is ill-defined.)
    # If you need edge handling, consider a mode that clips windows or uses a sliding-sum impl.
    i_min = Tr + Gr
    i_max = M - (Tr + Gr)
    j_min = Td + Gd
    j_max = N - (Td + Gd)

    for i in range(i_min, i_max):
        for j in range(j_min, j_max):
            # Outer rectangle bounds
            i1 = i - (Tr + Gr)
            j1 = j - (Td + Gd)
            i2 = i + (Tr + Gr)
            j2 = j + (Td + Gd)

            # sum over [i1..i2] x [j1..j2] using inclusion-exclusion
            sum_outer = (
                S[i2, j2]
                - S[i1 - 1, j2]
                - S[i2, j1 - 1]
                + S[i1 - 1, j1 - 1]
            )

            # Guard (includes CUT)
            gi1 = i - Gr
            gj1 = j - Gd
            gi2 = i + Gr
            gj2 = j + Gd

            sum_guard = (
                S[gi2, gj2]
                - S[gi1 - 1, gj2]
                - S[gi2, gj1 - 1]
                + S[gi1 - 1, gj1 - 1]
            )

            training_sum = sum_outer - sum_guard
            noise_mean = training_sum / float32(Ntrain)
            threshold = noise_mean * alpha

            D[i, j] = 1 if X[i, j] > threshold else 0

    return D


def cfar_top(X: float32[M, N], alpha: float32) -> int32[M, N]:
    """Top-level: prefix_2d -> cfar_from_prefix."""
    S = prefix_2d(X)
    D = cfar_from_prefix(X, S, alpha)
    return D


# -------------------------
# Dynamic-shape kernels (pre-allocated outputs)
# -------------------------

def prefix_rows_dyn(
    X: float32[...],
    R: float32[...],
    Mv: int32,
    Nv: int32,
):
    """Row-wise prefix sum for dynamic shapes; R is pre-allocated by host."""
    for i in range(Mv):
        R[i, 0] = X[i, 0]
        for j in range(1, Nv):
            R[i, j] = R[i, j - 1] + X[i, j]


def prefix_2d_dyn(
    X: float32[...],
    S: float32[...],
    tmpR: float32[...],
    Mv: int32,
    Nv: int32,
):
    """2-D prefix with dynamic shapes; outputs S, temp buffer tmpR provided by caller."""
    prefix_rows_dyn(X, tmpR, Mv, Nv)
    # first row
    for j in range(Nv):
        S[0, j] = tmpR[0, j]
    # remaining rows
    for i in range(1, Mv):
        for j in range(Nv):
            S[i, j] = S[i - 1, j] + tmpR[i, j]


def cfar_from_prefix_dyn(
    X: float32[...],
    S: float32[...],
    D: int32[...],
    Mv: int32,
    Nv: int32,
    Trv: int32,
    Tdv: int32,
    Grv: int32,
    Gdv: int32,
    alpha: float32,
):
    """Dynamic-shape CFAR (writes into pre-allocated D)."""
    H_out = 2 * (Trv + Grv) + 1
    W_out = 2 * (Tdv + Gdv) + 1
    H_g = 2 * Grv + 1
    W_g = 2 * Gdv + 1
    Ntrain: int32 = H_out * W_out - H_g * W_g

    i_min = Trv + Grv
    i_max = Mv - (Trv + Grv)
    j_min = Tdv + Gdv
    j_max = Nv - (Tdv + Gdv)

    for i in range(i_min, i_max):
        for j in range(j_min, j_max):
            i1 = i - (Trv + Grv)
            j1 = j - (Tdv + Gdv)
            i2 = i + (Trv + Grv)
            j2 = j + (Tdv + Gdv)

            sum_outer = (
                S[i2, j2]
                - S[i1 - 1, j2]
                - S[i2, j1 - 1]
                + S[i1 - 1, j1 - 1]
            )

            gi1 = i - Grv
            gj1 = j - Gdv
            gi2 = i + Grv
            gj2 = j + Gdv

            sum_guard = (
                S[gi2, gj2]
                - S[gi1 - 1, gj2]
                - S[gi2, gj1 - 1]
                + S[gi1 - 1, gj1 - 1]
            )

            training_sum = sum_outer - sum_guard
            noise_mean = training_sum / float32(Ntrain)
            threshold = noise_mean * alpha
            D[i, j] = 1 if X[i, j] > threshold else 0


def cfar_top_dyn(
    X: float32[...],
    D: int32[...],
    tmpR: float32[...],
    S: float32[...],
    Mv: int32,
    Nv: int32,
    Trv: int32,
    Tdv: int32,
    Grv: int32,
    Gdv: int32,
    alpha: float32,
):
    """Dynamic top: prefix -> CFAR. All outputs pre-allocated by host (D, tmpR, S)."""
    prefix_2d_dyn(X, S, tmpR, Mv, Nv)
    cfar_from_prefix_dyn(X, S, D, Mv, Nv, Trv, Tdv, Grv, Gdv, alpha)


# -------------------------
# Schedules / Builders
# -------------------------

def schedule_fixed():
    """Create schedules for fixed-size kernels and compose into top."""
    s_rows = allo.customize(prefix_rows)
    s_rows.pipeline("j")

    s_ps = allo.customize(prefix_2d)
    # Buffer the output row when accumulating full 2-D prefix
    s_ps.buffer_at(s_ps.S, axis="i")
    s_ps.pipeline("j")

    s_cfar = allo.customize(cfar_from_prefix)
    s_cfar.pipeline("j")

    s_top = allo.customize(cfar_top)
    s_top.compose([s_ps, s_rows, s_cfar])
    return s_top, (s_rows, s_ps, s_cfar)


def schedule_dynamic():
    """Create schedules for dynamic kernels and compose into top."""
    s_rows = allo.customize(prefix_rows_dyn)
    s_rows.pipeline("j")

    s_ps = allo.customize(prefix_2d_dyn)
    s_ps.buffer_at(s_ps.S, axis="i")
    s_ps.pipeline("j")

    s_cfar = allo.customize(cfar_from_prefix_dyn)
    s_cfar.pipeline("j")

    s_top = allo.customize(cfar_top_dyn)
    s_top.compose([s_ps, s_rows, s_cfar])
    return s_top, (s_rows, s_ps, s_cfar)


def build_cpu_fixed():
    """Build the fixed-size top for CPU (LLVM)."""
    s_top, _ = schedule_fixed()
    return s_top.build(target="llvm")  # same as build()


def build_vhls_fixed():
    """Emit HLS C++ for the fixed-size top."""
    s_top, _ = schedule_fixed()
    return s_top.build(target="vhls")  # returns string of kernel.cpp


def build_vitis_hls_fixed(mode: str = "sw_emu", project: str = "cfar.prj"):
    """Create a Vitis HLS project (sw_emu | hw_emu | hw) for the fixed-size top.
    Returns an executable host handle similar to LLVM mod, but running via Vitis flows.
    """
    s_top, _ = schedule_fixed()
    return s_top.build(target="vitis_hls", mode=mode, project=project)


# -------------------------
# Host helpers
# -------------------------

def alpha_from_pfa(pfa: float, ntrain: int) -> np.float32:
    """For CA-CFAR with exponential noise: Pfa ≈ (1 + α/N)^(-N) → α = N * (Pfa^(-1/N) - 1)."""
    return np.float32(ntrain * (pfa ** (-1.0 / ntrain) - 1.0))


def ntrain_from_params(tr: int, td: int, gr: int, gd: int) -> int:
    h_out = 2 * (tr + gr) + 1
    w_out = 2 * (td + gd) + 1
    h_g = 2 * gr + 1
    w_g = 2 * gd + 1
    return h_out * w_out - h_g * w_g


# -------------------------
# Demo
# -------------------------

if __name__ == "__main__":
    # Small CPU sanity check with random data (won't run HLS here).
    m, n = 128, 64
    # Override globals for this quick run by re-binding symbols
    M, N = m, n  # type: ignore

    # Redefine kernels with new sizes by re-importing this module is typical,
    # but for a quick demonstration we just rebuild schedules using the same symbols.

    # Build CPU module
    s_top, _ = schedule_fixed()
    mod = s_top.build(target="llvm")

    # Inputs
    X = np.random.rand(m, n).astype(np.float32)

    # CFAR params
    tr, td, gr, gd = Tr, Td, Gr, Gd
    ntrain = ntrain_from_params(tr, td, gr, gd)
    pfa = 1e-6
    alpha = alpha_from_pfa(pfa, ntrain)

    # Run
    D = mod(X, np.float32(alpha))

    print("CFAR run complete.")
    print("Detections (sum of 1s):", int(D.sum()))
    # The output D is an int32 array with 0/1 values.
