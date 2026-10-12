"""The image grid follows the affine (WI-0119, D-0207).

The k-space phase origin of a sordino scan is the FOV centre: the sequence moves it to
the slice-pack position with the receiver frequencies (``ACQ_O1_list``,
``SetBeforeAcquisition``). The adjoint NUFFT puts that point at index ``N // 2`` of each
reconstruction axis. The affine brkraw returns for the scan says where that point is:
``A^-1 c`` (``c`` the FOV centre in the affine's world frame). The reconstruction moves
its grid by ``shift = A^-1 c - N // 2`` (per output axis, then put on the reconstruction
axis that ``orientation.correct`` sends there), so the image lands where the affine says.
``c`` is read from ParaVision's own record of the FOV centre on its image grid
(``RECO_rotate * N`` through ``VisuCorePosition`` / ``VisuCoreOrientation``); on the
phantom it equals the slice-pack position the sequence frequency-shifts to.

Run 1 hard-coded ParaVision's 3D rule ``(N/2, N/2, N/2 - 0.5)``; run 2 derives the same
numbers for the phantom from the affine, and follows the affine where the rule was a
guess: odd sizes, other planes, flipped affine axes (``ACQ_scaling_phase < 0``).

The check on the phantom dataset (real data) is a manual script,
``scripts/validate/check_grid_centre_phantom.py`` (WI-0123).
"""
import io

import numpy as np
import pytest

from brkraw_sordino import hook, kcentre, recon
from brkraw_sordino.hook import _build_options
from brkraw_sordino.recon import phase_correction_factor, recon_dataobj
from brkraw_sordino.traj import get_trajectory

from test_serial_recon import N_POINTS, SHAPE, _info


# --- the derivation on synthetic affines ------------------------------------------------

def _rot(a, b, c):
    """A proper rotation from three angles (radians)."""
    ca, sa, cb, sb, cc, sc = np.cos(a), np.sin(a), np.cos(b), np.sin(b), np.cos(c), np.sin(c)
    rx = np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]])
    ry = np.array([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]])
    rz = np.array([[cc, -sc, 0], [sc, cc, 0], [0, 0, 1]])
    return rz @ ry @ rx


def _affine(R, res, centre, centre_index):
    """Affine (output index -> world) with direction columns ``R``, voxel sizes ``res``, and the
    world point ``centre`` at output index ``centre_index``."""
    M = np.asarray(R, float) @ np.diag(res)
    A = np.eye(4)
    A[:3, :3] = M
    A[:3, 3] = np.asarray(centre, float) - M @ np.asarray(centre_index, float)
    return A


@pytest.mark.parametrize("matrix, centre_index, shift", [
    ([120, 120, 120], (60.0, 60.0, 59.5), (0.0, 0.0, -0.5)),      # scan 13 (measured, p8)
    ([60, 60, 60], (30.0, 30.0, 29.5), (0.0, 0.0, -0.5)),         # scan 18
    ([75, 75, 75], (37.5, 37.5, 37.0), (0.5, 0.5, 0.0)),          # odd: RECO_rotate 0.5, 0.5 - 0.5/N
    ([64, 75, 33], (32.0, 37.5, 16.0), (0.0, 0.5, 0.0)),
    ([16, 16, 16], (8.0, 8.0, 8.0), (0.0, 0.0, 0.0)),             # an affine with the FFT centre
])
def test_shift_is_affine_centre_minus_nufft_centre(matrix, centre_index, shift):
    c = (-0.766, -1.392, 3.896)
    A = _affine(np.eye(3), (0.2, 0.25, 0.3), c, centre_index)
    got = recon.grid_shift_from_affine(A, c, (0, 1, 2), matrix)
    assert got == pytest.approx(shift, abs=1e-9)


@pytest.mark.parametrize("plane, order", [("axial", (0, 1, 2)), ("coronal", (0, 2, 1)),
                                          ("sagittal", (1, 2, 0))])
def test_shift_goes_to_the_reconstruction_axis_of_each_output_axis(plane, order):
    """Output axis j holds reconstruction axis order[j]; the shift found on output axis j is
    the shift of reconstruction axis order[j]."""
    from brkraw_sordino import orientation

    info = {"GradientOrientation": np.eye(3), "SliceOrientation": plane}
    assert orientation.axis_order(info) == order
    matrix = [64, 75, 33]                                   # read, phase, slice
    n_out = [matrix[order[j]] for j in range(3)]
    # the affine centre on each output axis: geometric centre (N-1)/2 on the output axis that
    # holds the slice axis, N/2 elsewhere
    centre_index = [(n - 1) / 2.0 if order[j] == 2 else n / 2.0 for j, n in enumerate(n_out)]
    c = (1.1, -2.3, 0.7)
    A = _affine(_rot(0.3, -0.2, 0.9), (0.2, 0.25, 0.3), c, centre_index)
    got = recon.grid_shift_from_affine(A, c, orientation.axis_order(info), matrix)
    want = [0.0, 0.0, 0.0]
    for j in range(3):
        want[order[j]] = centre_index[j] - matrix[order[j]] // 2
    assert got == pytest.approx(tuple(want), abs=1e-9)
    assert got == pytest.approx((0.0, 0.5, 0.0), abs=1e-9)  # 64: N/2 = N//2; 75: 37.5 - 37; 33: 16 - 16


def test_shift_does_not_depend_on_the_world_frame():
    """A world-side transform (subject pose wrap, RAS flip, world rotation) moves the affine and
    the centre together: the shift is the same (brkraw's spaces differ only by such transforms)."""
    c = np.array([0.4, -1.2, 2.5])
    A = _affine(_rot(0.1, 0.5, -0.4), (0.2, 0.2, 0.4), c, (60.0, 60.0, 59.5))
    T = np.eye(4)
    T[:3, :3] = np.diag([-1.0, 1.0, -1.0]) @ _rot(np.pi / 2, 0.0, np.pi)
    T[:3, 3] = (3.0, -7.0, 11.0)
    c2 = (T @ np.r_[c, 1.0])[:3]
    base = recon.grid_shift_from_affine(A, c, (0, 1, 2), [120, 120, 120])
    assert recon.grid_shift_from_affine(T @ A, c2, (0, 1, 2), [120, 120, 120]) == pytest.approx(base, abs=1e-9)


@pytest.mark.parametrize("n, centre, want", [
    (120, 60.0, -1.0),      # even: N/2 -> N - 1 - N/2 = N/2 - 1, one voxel below N // 2
    (120, 59.5, -0.5),      # the geometric centre (N-1)/2 does not move under a flip
    (75, 37.5, -0.5),       # odd: N/2 -> N/2 - 1 = 36.5, N // 2 = 37
])
def test_a_flipped_affine_axis_moves_the_shift(n, centre, want):
    """brkraw flips voxel axis 1 of the affine for ACQ_scaling_phase < 0 (and axis 2 for coronal):
    index i -> n - 1 - i. The derivation follows: the centre moves to n - 1 - centre."""
    from brkraw.resolver.affine import flip_voxel_axis_affine

    matrix = [n, n, n]
    c = (0.3, 0.9, -1.7)
    A = _affine(np.eye(3), (0.2, 0.2, 0.2), c, (n / 2.0, centre, (n - 1) / 2.0))
    Af = flip_voxel_axis_affine(A, axis=1, shape=tuple(matrix))
    before = recon.grid_shift_from_affine(A, c, (0, 1, 2), matrix)
    after = recon.grid_shift_from_affine(Af, c, (0, 1, 2), matrix)
    assert before[1] == pytest.approx(centre - n // 2, abs=1e-9)
    assert after[1] == pytest.approx(want, abs=1e-9)
    assert after[0] == pytest.approx(before[0], abs=1e-9)
    assert after[2] == pytest.approx(before[2], abs=1e-9)


def test_shift_needs_three_axes_and_an_invertible_affine():
    with pytest.raises(ValueError):
        recon.grid_shift_from_affine(np.eye(4), (0, 0, 0), (0, 1), [16, 16])
    bad = np.eye(4)
    bad[2, 2] = 0.0
    with pytest.raises(ValueError):
        recon.grid_shift_from_affine(bad, (0, 0, 0), (0, 1, 2), [16, 16, 16])


# --- ParaVision's FOV centre from its parameters ------------------------------------------

def _visu(pos, orient_rows, extent, size):
    return {"VisuCoreDim": 3, "VisuCorePosition": [list(pos)], "VisuCoreOrientation": [list(np.ravel(orient_rows))],
            "VisuCoreExtent": list(extent), "VisuCoreSize": list(size)}


@pytest.mark.parametrize("size, rotate", [([120, 120, 120], [0.5, 0.5, 0.4958333333333333]),
                                          ([75, 75, 75], [0.5, 0.5, 0.4933333333333333]),
                                          ([64, 75, 33], [0.5, 0.5, 0.4848484848484848])])
def test_fov_centre_is_rotate_times_size_through_visu(size, rotate):
    R = _rot(0.2, 0.4, -0.3)
    res = np.array([0.2, 0.25, 0.3])
    pos = np.array([-12.7, 3.1, -8.0])
    visu = _visu(pos, R, res * np.asarray(size), size)
    reco = {"RECO_rotate": [[v] for v in rotate], "RECO_transposition": [0], "RECO_size": list(size)}
    c = recon.paravision_fov_centre(visu, reco)
    want = pos + R.T @ (res * np.asarray(rotate) * np.asarray(size))
    assert c == pytest.approx(want, abs=1e-9)


def test_fov_centre_transposition_one_swaps_the_first_two_axes():
    """RECO_rotate is in acquisition order; RECO_transposition 1 swaps the first two image axes
    (Bruker FieldMap scan 5, sagittal: (0.5, 0.5, 0.4921875) and Visu index (32, 32, 31.5))."""
    size_acq = [64, 75, 33]
    rotate = [0.5, 0.6, 0.4848484848484848]
    size_img = [75, 64, 33]
    res = np.array([0.25, 0.2, 0.3])
    visu = _visu((0.0, 0.0, 0.0), np.eye(3), res * np.asarray(size_img), size_img)
    reco = {"RECO_rotate": [[v] for v in rotate], "RECO_transposition": [1], "RECO_size": size_acq}
    c = recon.paravision_fov_centre(visu, reco)
    idx_img = np.array([0.6 * 75, 0.5 * 64, rotate[2] * 33])
    assert c == pytest.approx(res * idx_img, abs=1e-9)


@pytest.mark.parametrize("change", ["transposition2", "size", "dim2", "missing"])
def test_fov_centre_is_none_when_it_cannot_be_read(change):
    size = [16, 16, 16]
    visu = _visu((0.0, 0.0, 0.0), np.eye(3), [3.2, 3.2, 3.2], size)
    reco = {"RECO_rotate": [[0.5], [0.5], [0.46875]], "RECO_transposition": [0], "RECO_size": size}
    if change == "transposition2":
        reco["RECO_transposition"] = [2]
    elif change == "size":
        visu["VisuCoreSize"] = [16, 16, 8]
    elif change == "dim2":
        visu["VisuCoreDim"] = 2
    else:
        reco.pop("RECO_rotate")
    assert recon.paravision_fov_centre(visu, reco) is None


# --- the reconstruction puts the image where the shift says ---------------------------------

SIGMA = 1.2
#: blob centre tolerance in voxels (measured 0.015 at 120^3 and 60^3 on the phantom trajectories)
CENTRE_TOL = 0.03


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


SHIFTS = {"none": None, "pv": [0.0, 0.0, -0.5], "odd": [0.5, -1.0, 0.25]}


@pytest.mark.parametrize("k0", [False, True], ids=["plain", "estimate_k0"])
@pytest.mark.parametrize("shift", list(SHIFTS))
@pytest.mark.parametrize("d", [(0.0, 0.0, 0.0), (2.3, -1.6, 2.9)])
def test_recon_puts_a_blob_where_the_grid_shift_says(tmp_path, k0, shift, d):
    info = dict(_info(1, 1, True), FIDDataType=np.dtype("<f8"))
    if SHIFTS[shift] is not None:
        info["GridShift"] = SHIFTS[shift]
    options = _build_options({"cache_dir": str(tmp_path), "estimate_k0": k0})
    traj = get_trajectory(info, options)
    factor = phase_correction_factor(info, options, N_POINTS)
    raw = _raw(_blob_kspace(traj, d), factor)
    vt = kcentre.leading_points(info, options.ignore_samples or 1) if k0 else None
    out = io.BytesIO()
    recon_dataobj(io.BytesIO(raw), traj, info, out, options, phase_factor=factor, virtual_traj=vt)
    img = np.frombuffer(out.getvalue(), dtype=np.complex128).reshape(SHAPE[::-1]).T
    want = np.asarray(SHAPE) // 2 + np.asarray(SHIFTS[shift] or [0.0] * 3) + np.asarray(d)
    assert np.abs(_centre(img) - want).max() < CENTRE_TOL


def test_frames_put_a_blob_where_the_grid_shift_says(tmp_path):
    from brkraw_sordino import frames
    from brkraw_sordino.recon import phase_correction_rows
    from brkraw_sordino.traj import trajectory_rows

    from test_golden_recon import _golden_info

    d = (1.7, -2.2, 2.4)
    info = dict(_golden_info(1), FIDDataType=np.dtype("<f8"), GridShift=[0.5, 0.0, -0.5])
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
    want = np.asarray(SHAPE) // 2 + np.array([0.5, 0.0, -0.5]) + np.asarray(d)
    assert np.abs(_centre(img) - want).max() < CENTRE_TOL


def test_the_grid_shift_is_in_the_recon_cache_key(tmp_path):
    from test_hook_read import _Fid, _Scan, _recon_info

    options = hook._build_options(dict(cache_dir=str(tmp_path)))
    info = _recon_info(1)
    plain = hook._build_cache_params(_Scan(), None, _Fid(), options, info)
    assert plain["grid"] == {"rule": recon.GRID_RULE, "shift": [0.0, 0.0, 0.0]}
    shifted = hook._build_cache_params(_Scan(), None, _Fid(), options, dict(info, GridShift=[0.0, 0.0, -0.5]))
    assert shifted["grid"] == {"rule": recon.GRID_RULE, "shift": [0.0, 0.0, -0.5]}
    from brkraw_sordino.recon import build_recon_cache_path
    assert build_recon_cache_path(tmp_path, plain).name != build_recon_cache_path(tmp_path, shifted).name
