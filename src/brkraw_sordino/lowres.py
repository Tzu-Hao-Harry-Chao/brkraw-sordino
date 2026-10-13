"""Reconstruction on a smaller matrix at the same FOV (option ``recon_matrix``, WI-0132).

A radial spoke reaches |k| = 0.5 (the Nyquist radius of the acquisition matrix ``N``,
omega = pi in ``serial.omega_columns``). A grid of ``N' = r N`` voxels over the same FOV
has voxels ``1 / r`` times larger and its Nyquist radius is ``r`` times the old one. So:

1. every trajectory point is divided by ``r`` (the new grid's units, |k'| = 0.5 at the
   new Nyquist radius), and
2. every spoke is cut after the last sample whose radius stays at or below ``0.5 r`` on
   all spokes (a sphere: a low-pass filter; the samples beyond are not read into the
   adjoint, so nothing outside the new grid's band folds back into the image).

The density weights are then those of the kept samples (``|k'|^2`` over the kept set, or
the golden Pipe weights estimated on the kept set and the small grid), and
``estimate_k0`` solves on the small grid with its virtual samples scaled the same way.
``output_scale`` makes the voxel values those of the full-resolution image's Fourier
low-pass (value-preserving), instead of a scale that depends on the grid size.

Only one ratio ``r`` for all axes: the cut of a radial acquisition is a sphere, and a
per-axis ratio (an ellipsoid cut, with |k|^2 weights of the unscaled positions) is not
implemented; such a request stops with a ``ValueError``.

The small grid keeps the full grid's shift (``recon.grid_shift``) in its own voxels, so the
FOV centre sits at ``N' // 2 + shift`` as ParaVision puts it for a scan of that matrix; the
affine is ``A T`` (``reduced_affine``).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

#: Fewest samples per spoke (after ``ignore_samples``) that the cut must keep.
MIN_KEPT_SAMPLES = 2
#: Rule name in the recon cache key; change it when the cut, the weights or the scale change.
RULE = "sphere-cut-v1"


def _item(value: Any) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"recon_matrix must be whole voxel counts, got {value!r}")
    if isinstance(value, str):
        text = value.strip()
        try:
            number = float(text)
        except ValueError:
            raise ValueError(f"recon_matrix must be whole voxel counts, got {value!r}") from None
    elif isinstance(value, (int, float, np.integer, np.floating)):
        number = float(value)
    else:
        raise ValueError(f"recon_matrix must be whole voxel counts, got {value!r}")
    if not np.isfinite(number) or number != int(number):
        raise ValueError(f"recon_matrix must be whole voxel counts, got {value!r}")
    if number < 2:
        raise ValueError(f"recon_matrix must be at least 2 voxels per axis, got {value!r}")
    return int(number)


def parse(value: Any) -> Optional[Tuple[int, int, int]]:
    """``recon_matrix`` as three voxel counts (read, phase, slice of ``PVM_Matrix``), or None.

    A number, a string (``"64"``, ``"64,64,64"``, ``"64 64 64"``, ``"[64, 64, 64]"``) and a
    list, tuple or array of one or three values; None (or an empty string) means the
    acquisition matrix. Only the form is checked here; ``resolve`` checks the values
    against the scan.
    """
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if text.lower() in ("", "null", "none"):
            return None
        if len(text) >= 2 and (text[0], text[-1]) in (("[", "]"), ("(", ")")):
            text = text[1:-1]
        items: list = text.replace(",", " ").split()
    elif isinstance(value, np.ndarray):
        items = value.reshape(-1).tolist()
    elif isinstance(value, (list, tuple)):
        items = list(value)
    else:
        items = [value]
    if len(items) == 1:
        n = _item(items[0])
        return (n, n, n)
    if len(items) == 3:
        return tuple(_item(v) for v in items)  # type: ignore[return-value]
    raise ValueError(f"recon_matrix takes one value or three (read, phase, slice), got {len(items)}: {value!r}")


@dataclass(frozen=True)
class Reduction:
    """A resolved ``recon_matrix``: the small matrix and its ratio to the acquisition matrix."""

    matrix: Tuple[int, int, int]     # N' (reconstruction order)
    base: Tuple[int, int, int]       # N, the acquisition matrix
    ratio: float                     # r = N' / N, the same on every axis (< 1)

    @property
    def scale(self) -> float:
        """``s = 1 / r``: trajectory and voxel-size factor."""
        return 1.0 / self.ratio

    def key(self) -> dict:
        return {"matrix": [int(v) for v in self.matrix], "rule": RULE}


def resolve(requested: Optional[Sequence[int]], base: Sequence[int]) -> Optional[Reduction]:
    """Check ``recon_matrix`` against the acquisition matrix; None when nothing is reduced.

    Raises ``ValueError`` (with the matrix in the message) when a value is larger than the
    acquisition matrix or the three values do not keep one ratio to it.
    """
    if requested is None:
        return None
    req = tuple(int(v) for v in requested)
    n = tuple(int(v) for v in base)
    if len(req) != 3 or len(n) != 3 or min(n) <= 0:
        raise ValueError(f"recon_matrix needs a 3-axis acquisition matrix, got {list(n)}")
    if req == n:
        return None
    bigger = [i for i in range(3) if req[i] > n[i]]
    if bigger:
        raise ValueError(
            f"recon_matrix {list(req)} is larger than the acquisition matrix {list(n)} on axis "
            f"{', '.join(str(i) for i in bigger)}; it can only lower the resolution (to enlarge the FOV "
            "at the same voxel size use ext_factors)")
    ratios = {Fraction(req[i], n[i]) for i in range(3)}
    if len(ratios) != 1:
        raise ValueError(
            f"recon_matrix {list(req)} does not keep one ratio to the acquisition matrix {list(n)} "
            f"(ratios {', '.join(str(Fraction(req[i], n[i])) for i in range(3))}); a radial scan is cut "
            "with a sphere, so every axis must shrink by the same factor (per-axis reduction is not supported)")
    r = float(next(iter(ratios)))
    return Reduction(matrix=req, base=n, ratio=r)  # type: ignore[arg-type]


class CutRows:
    """``TrajectoryRows`` of the kept samples in the small grid's units (``rows(lo, hi)``)."""

    def __init__(self, inner: Any, n_keep: int, scale: float):
        self._inner = inner
        self.n_pro = int(inner.n_pro) if hasattr(inner, "n_pro") else int(np.shape(inner)[0])
        self.n_keep = int(n_keep)
        self.n_samples = self.n_keep
        self.scale = float(scale)
        self._rows = inner.rows if hasattr(inner, "rows") else (lambda lo, hi: np.asarray(inner)[lo:hi])

    @property
    def shape(self) -> tuple:
        return (self.n_pro, self.n_keep, 3)

    def rows(self, lo: int, hi: int) -> np.ndarray:
        return self._rows(lo, hi)[:, :self.n_keep] * self.scale


@dataclass(frozen=True)
class Cut:
    """Where the spokes are cut and the numbers for the output scale."""

    n_keep: int                # samples per spoke kept, counted from sample 0 (before ignore_samples)
    n_samples: int             # samples per spoke of the full trajectory
    radius_kept: float         # largest |k| of a kept sample, full-grid units (<= 0.5 r)
    dmax_full: float           # largest |k|^2 of the full set (samples >= ignore_samples, and virtual ones)
    dmax_kept: float           # largest |k'|^2 of the kept set, small-grid units (and virtual ones)


def find_cut(rows: Any, n_pro: int, reduction: Reduction, ignore_samples: int,
             chunk_spokes: int = 4096, virtual_traj: Optional[np.ndarray] = None) -> Cut:
    """The spoke cut of ``reduction``: one pass over the trajectory rows, chunk by chunk.

    ``n_keep`` is the number of leading samples whose radius is at or below ``0.5 r`` on
    every spoke (radii grow along a spoke; the first sample past the limit on any spoke ends
    the kept part for all, so every kept point lies inside the small grid's band).
    """
    from . import serial

    get = rows.rows if hasattr(rows, "rows") else (lambda lo, hi: np.asarray(rows)[lo:hi])
    ign = int(ignore_samples)
    limit = 0.5 * reduction.ratio
    rmax = None
    for lo, hi in serial.spoke_ranges(int(n_pro), int(chunk_spokes)):
        rad = np.sqrt(np.square(get(lo, hi)).sum(-1)).max(axis=0)     # per sample, over these spokes
        rmax = rad if rmax is None else np.maximum(rmax, rad)
    assert rmax is not None
    n_samples = int(rmax.size)
    over = np.nonzero(rmax > limit)[0]          # no slack: every kept point within the band (wi-0132-choi-1)
    n_keep = int(over[0]) if over.size else n_samples
    # a later sample back inside the limit (a curved spoke) is cut too: the prefix is what is kept
    if n_keep - ign < MIN_KEPT_SAMPLES:
        raise ValueError(
            f"recon_matrix {list(reduction.matrix)} keeps {max(n_keep - ign, 0)} sample(s) per spoke "
            f"(radius {limit:.4f} of the acquisition's 0.5, ignore_samples {ign}); choose a larger matrix")
    full = rmax[ign:]
    dmax_full = float(np.max(full) ** 2)
    radius_kept = float(rmax[n_keep - 1])
    dmax_kept = float((np.max(rmax[ign:n_keep]) * reduction.scale) ** 2)
    if virtual_traj is not None:
        v2 = float(np.square(virtual_traj).sum(-1).max())
        dmax_full = max(dmax_full, v2)
        dmax_kept = max(dmax_kept, v2 * reduction.scale ** 2)
    logger.info("recon_matrix %s (ratio %.4g of %s): %s of %s samples per spoke kept (k-space radius %.4g of "
                "the acquisition's 0.5).", "x".join(str(v) for v in reduction.matrix), reduction.ratio,
                "x".join(str(v) for v in reduction.base), n_keep, n_samples, radius_kept)
    return Cut(n_keep=n_keep, n_samples=n_samples, radius_kept=radius_kept,
               dmax_full=dmax_full, dmax_kept=dmax_kept)


def output_scale(reduction: Reduction, cut: Cut) -> float:
    """Factor that gives the small grid the voxel values of the full image's low-pass.

    The adjoint is ``sum_m w_m y_m exp(i omega_m . n) / nf`` with ``w_m = |k|^2 / dmax`` and
    ``nf = sqrt(prod(grid) 8)``. On the small grid the phase ``omega'_m . n'`` of a voxel equals
    the full grid's at the same place, so with the full-resolution weights and ``nf`` the sum
    over the kept samples is the full image's low-pass at the new voxel centres. The small grid
    uses ``w'_m = |k'|^2 / dmax' = |k|^2 / (r^2 dmax')`` and ``nf' = r^1.5 nf``; the factor
    ``r^2 dmax' / dmax_full * r^1.5`` restores the full-resolution weights and ``nf``.
    """
    r = reduction.ratio
    return float(r ** 2 * cut.dmax_kept / cut.dmax_full * r ** 1.5)


def reduced_affine(affine: np.ndarray, scale: float, full_matrix: Sequence[int], small_matrix: Sequence[int],
                   shift: Optional[Sequence[float]] = None) -> np.ndarray:
    """``A T``: the small grid's affine, all three sequences in output axis order.

    The full grid holds the k-space phase origin (the FOV centre ParaVision records) at index
    ``p = N // 2 + shift`` (``recon.grid_shift_from_affine``; ``shift`` None = 0). The small grid
    keeps the same shift in its own voxels, ``p' = N' // 2 + shift``, and voxels ``scale`` times
    longer: ``T = [[s I, p - s p'], [0, 1]]``, so ``A T`` maps ``p'`` to the same world point as
    ``A`` maps ``p``. This is ParaVision's own rule for a scan of another matrix (the centre at
    ``RECO_rotate * N`` with ``RECO_rotate`` 0.5 in plane and ``0.5 - 0.5 / N`` on the slice axis,
    both ``N // 2`` plus a fixed voxel offset for even N): the phantom's native 60^3 scan 18 has
    exactly this affine for scan 13's 120^3 grid reduced to 60 (WI-0132). The FOV size is kept;
    its edges move by at most half a full-grid voxel on an axis whose shift is not -0.5.
    """
    a = np.asarray(affine, dtype=float)
    s = float(scale)
    sh = np.zeros(3) if shift is None else np.asarray(shift, dtype=float).reshape(3)
    p = np.array([int(v) // 2 for v in full_matrix], dtype=float) + sh
    q = np.array([int(v) // 2 for v in small_matrix], dtype=float) + sh
    t = np.eye(4)
    t[:3, :3] *= s
    t[:3, 3] = p - s * q
    return a @ t


__all__ = ["MIN_KEPT_SAMPLES", "RULE", "parse", "Reduction", "resolve", "CutRows", "Cut", "find_cut",
           "output_scale", "reduced_affine"]
