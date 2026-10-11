"""The image grid follows ParaVision's 3D grid (WI-0119).

ParaVision places the FOV centre (the k-space phase origin, set by the receiver
frequencies ``ACQ_O1_list``) at index ``RECO_rotate * N`` of each reconstruction
axis: ``N/2`` on read and phase and ``N/2 - 0.5`` on the slice axis of a 3D scan
(``RECO_rotate`` 0.5, 0.5, 0.5 - 0.5/N; the same in the method's reco and in a
Bruker 3D Cartesian scan), and ``VisuCorePosition``, hence the brkraw affine,
says the same. The adjoint NUFFT puts it at ``N // 2``. Before WI-0119 every
sordino image sat half a voxel too high on the slice axis (scan 13: 0.1 mm,
scan 18: 0.2 mm). The reconstruction now evaluates the image on ParaVision's
grid (a linear phase on the samples), so the affine is unchanged.
"""
import io
import os

import numpy as np
import pytest

from brkraw_sordino import hook, kcentre, recon
from brkraw_sordino.hook import _build_options
from brkraw_sordino.recon import phase_correction_factor, recon_dataobj
from brkraw_sordino.traj import get_trajectory

from test_serial_recon import N_POINTS, SHAPE, _info

def pv_grid(k, traj, shape):
    """Reference data on ParaVision's grid: the samples ``k`` (shape ``traj.shape[:-1]``) times the
    product's grid factor, for the whole-array references (``nufft_adjoint``, ``fill_centre``)."""
    g = recon.grid_shift_factor(traj, recon.grid_shift(shape))
    return k if g is None else np.asarray(k) * g


SIGMA = 1.2
#: blob centre tolerance in voxels (measured 0.015 at 120^3 and 60^3 on the phantom trajectories)
CENTRE_TOL = 0.03


@pytest.mark.parametrize("matrix, shift", [
    ([120, 120, 120], (0.0, 0.0, -0.5)),
    ([60, 60, 60], (0.0, 0.0, -0.5)),
    ([16, 16, 16], (0.0, 0.0, -0.5)),
    ([75, 75, 75], (0.5, 0.5, 0.0)),
    ([64, 75, 32], (0.0, 0.5, -0.5)),
])
def test_grid_shift_is_paravision_centre_minus_nufft_centre(matrix, shift):
    assert recon.grid_shift(matrix) == pytest.approx(shift)


def test_grid_shift_needs_three_axes():
    with pytest.raises(ValueError):
        recon.grid_shift([16, 16])


def _blob_kspace(traj, d):
    tr = np.asarray(traj, dtype=float)
    om = 2.0 * np.pi * tr
    return (np.exp(-2.0 * np.pi ** 2 * SIGMA ** 2 * (tr ** 2).sum(-1))
            * np.exp(-1j * (om @ np.asarray(d, dtype=float))))


def _raw(y, factor):
    if factor is not None:
        y = y / factor
    a = np.empty((2, y.shape[1], 1, y.shape[0]), dtype="<f8")
    a[0, :, 0, :] = y.real.T
    a[1, :, 0, :] = y.imag.T
    return a.tobytes(order="F")


def _centre(img, r=5, frac=0.05):
    m = np.abs(img)
    p = np.array(np.unravel_index(np.argmax(m), m.shape))
    sl = tuple(slice(max(0, c - r), min(n, c + r + 1)) for c, n in zip(p, m.shape))
    w = m[sl]
    w = np.where(w > frac * w.max(), w, 0.0)
    g = np.meshgrid(*[np.arange(s.start, s.stop) for s in sl], indexing="ij")
    return np.array([(gi * w).sum() / w.sum() for gi in g])


def _expected(d):
    n = np.asarray(SHAPE)
    return n // 2 + np.asarray(recon.grid_shift(SHAPE)) + np.asarray(d)


@pytest.mark.parametrize("k0", [False, True], ids=["plain", "estimate_k0"])
@pytest.mark.parametrize("d", [(0.0, 0.0, 0.0), (2.3, -1.6, 2.9), (-2.1, 1.4, -2.6)])
def test_recon_puts_a_blob_on_the_paravision_grid(tmp_path, k0, d):
    info = dict(_info(1, 1, True), FIDDataType=np.dtype("<f8"))
    options = _build_options({"cache_dir": str(tmp_path), "estimate_k0": k0})
    traj = get_trajectory(info, options)
    factor = phase_correction_factor(info, options, N_POINTS)
    raw = _raw(_blob_kspace(traj, d), factor)
    vt = kcentre.leading_points(info, options.ignore_samples or 1) if k0 else None
    out = io.BytesIO()
    recon_dataobj(io.BytesIO(raw), traj, info, out, options, phase_factor=factor, virtual_traj=vt)
    img = np.frombuffer(out.getvalue(), dtype=np.complex128).reshape(SHAPE[::-1]).T
    assert np.abs(_centre(img) - _expected(d)).max() < CENTRE_TOL


def test_frames_put_a_blob_on_the_paravision_grid(tmp_path):
    from brkraw_sordino import frames
    from brkraw_sordino.recon import phase_correction_rows
    from brkraw_sordino.traj import trajectory_rows

    from test_golden_recon import _golden_info

    d = (1.7, -2.2, 2.4)
    info = dict(_golden_info(1), FIDDataType=np.dtype("<f8"))
    options = _build_options({"cache_dir": str(tmp_path), "estimate_k0": False,
                              "frame_spokes": int(info["NPro"])})
    traj = get_trajectory(info, options)
    factor = phase_correction_rows(info, options, N_POINTS)[:]
    raw = _raw(_blob_kspace(traj, d), factor)
    plan = frames.make_plan(info, options)
    out = io.BytesIO()
    frames.recon_frames(io.BytesIO(raw), trajectory_rows(info, options), info, out, options, plan,
                        phase_factor=phase_correction_rows(info, options, N_POINTS))
    per = int(np.prod(SHAPE))
    img = np.frombuffer(out.getvalue(), dtype=np.complex128)[:per].reshape(SHAPE[::-1]).T
    assert np.abs(_centre(img) - _expected(d)).max() < CENTRE_TOL


def test_the_grid_rule_is_in_the_recon_cache_key(tmp_path):
    from test_hook_read import _Fid, _Scan, _recon_info

    info = _recon_info(1)
    options = hook._build_options(dict(cache_dir=str(tmp_path)))
    params = hook._build_cache_params(_Scan(), None, _Fid(), options, info)
    assert params["grid"] == {"rule": recon.GRID_RULE, "shift": list(recon.grid_shift(info["Matrix"]))}


ZIP = os.environ.get("BRKRAW_SORDINO_GOLDEN_ZIP")


@pytest.mark.skipif(not ZIP or not os.path.exists(ZIP or ""), reason="phantom zip not available")
@pytest.mark.parametrize("sid", [13, 18])
def test_the_rule_is_where_paravision_puts_the_fov_centre(sid):
    """FOV centre from the scan offsets (subject frame), through VisuCorePosition and
    VisuCoreOrientation to a voxel index: equals N // 2 + grid_shift and RECO_rotate * N."""
    import brkraw

    scan = brkraw.load(ZIP).get_scan(sid)
    reco = scan.get_reco(1)
    acqp, vp = scan.acqp, reco.visu_pars
    g = np.asarray(acqp.get("ACQ_grad_matrix"), dtype=float).reshape(-1, 3, 3)[0]
    off = np.array([float(np.ravel(acqp.get(k))[0])
                    for k in ("ACQ_read_offset", "ACQ_phase1_offset", "ACQ_slice_offset")])
    centre_subject = g.T @ off
    orient = np.asarray(vp.get("VisuCoreOrientation"), dtype=float).reshape(3, 3)
    pos = np.asarray(vp.get("VisuCorePosition"), dtype=float).reshape(3)
    size = np.asarray(vp.get("VisuCoreSize"), dtype=float)
    res = np.asarray(vp.get("VisuCoreExtent"), dtype=float) / size
    index = orient @ (centre_subject - pos) / res
    matrix = [int(v) for v in size]
    rule = np.asarray(matrix) // 2 + np.asarray(recon.grid_shift(matrix))
    rotate = np.ravel(np.asarray(reco.reco.get("RECO_rotate"), dtype=float))[:3] * np.asarray(matrix)
    assert index == pytest.approx(rule, abs=1e-6)
    assert rotate == pytest.approx(rule, abs=1e-6)
