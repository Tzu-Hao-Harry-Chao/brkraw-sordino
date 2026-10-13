"""``recon_matrix``: a smaller reconstruction matrix at the same FOV (WI-0132, D-0230).

Part 1: the option's forms and errors. Part 2: the affine of the small grid (voxels
``N / N'`` larger, the FOV box kept) and the grid shift that puts the image there. Part 3:
the reconstruction equals the full-resolution image's Fourier low-pass (same sphere, value
preserving) on a synthetic object; frames and estimate_k0 run on the small grid. Part 4:
the hook (shape, cache key, size estimate, affine).
"""
import io

import numpy as np
import pytest

from brkraw_sordino import hook, lowres
from brkraw_sordino.recon import (grid_shift_from_affine, parse_volume_shape, recon_dataobj)
from brkraw_sordino.traj import get_trajectory, trajectory_rows

from test_serial_recon import N_PRO, SHAPE, _fid_frames, _info


# ---------------------------------------------------------------- part 1: the option
@pytest.mark.parametrize("value,expected", [
    (None, None), ("", None), ("null", None), (64, (64, 64, 64)), (64.0, (64, 64, 64)), ("64", (64, 64, 64)),
    ([64], (64, 64, 64)), ([64, 32, 16], (64, 32, 16)), ((8, 8, 8), (8, 8, 8)), ("64,64,32", (64, 64, 32)),
    ("64 64 32", (64, 64, 32)), ("[64, 64, 32]", (64, 64, 32)), (np.array([8, 8, 8]), (8, 8, 8)),
])
def test_forms(value, expected):
    assert lowres.parse(value) == expected


@pytest.mark.parametrize("value,match", [
    (0, "at least 2"), (-8, "at least 2"), (1, "at least 2"), (64.5, "whole voxel"), ("abc", "whole voxel"),
    (True, "whole voxel"), ([64, 64], "one value or three"), ("64,64", "one value or three"),
    ([8, 8, 8, 8], "one value or three"), (float("nan"), "whole voxel"), ({"a": 1}, "whole voxel"),
])
def test_bad_forms(value, match):
    with pytest.raises(ValueError, match=match):
        lowres.parse(value)


def test_resolve():
    assert lowres.resolve(None, (16, 16, 16)) is None
    assert lowres.resolve((16, 16, 16), (16, 16, 16)) is None          # the full grid: the usual path
    red = lowres.resolve((8, 8, 8), (16, 16, 16))
    assert red.ratio == 0.5 and red.scale == 2.0 and red.matrix == (8, 8, 8)
    assert lowres.resolve((32, 32, 16), (64, 64, 32)).ratio == 0.5   # one ratio, anisotropic matrix
    with pytest.raises(ValueError, match="larger than the acquisition matrix"):
        lowres.resolve((32, 32, 32), (16, 16, 16))
    with pytest.raises(ValueError, match="one ratio"):
        lowres.resolve((8, 8, 4), (16, 16, 16))


# ---------------------------------------------------------------- part 2: affine and grid
def _affine(rng):
    q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    a = np.eye(4)
    a[:3, :3] = q @ np.diag([0.2, 0.25, 0.3])
    a[:3, 3] = rng.uniform(-10, 10, 3)
    return a


def _paravision_affine(direction, delta, n, centre, rotate):
    """A ParaVision-style affine: voxel ``delta`` along the unit ``direction`` columns, the FOV centre
    ``centre`` at index ``rotate * n`` per axis (``RECO_rotate``; ``recon.paravision_fov_centre``)."""
    a = np.eye(4)
    a[:3, :3] = direction * delta
    a[:3, 3] = centre - a[:3, :3] @ (np.asarray(rotate) * n)
    return a


@pytest.mark.parametrize("n,m", [(16, 8), (120, 60), (120, 40), (96, 64), (160, 80)])
@pytest.mark.parametrize("order", [(0, 1, 2), (1, 0, 2), (2, 0, 1)])
def test_reduced_affine_is_paravisions_affine_of_the_small_matrix(n, m, order):
    """ParaVision puts the FOV centre at RECO_rotate * N, 0.5 in plane and 0.5 - 0.5 / N on the slice
    axis. Reducing the N grid gives exactly the affine ParaVision writes for an N' scan of the same
    FOV and centre (scan 18 vs scan 13 on the phantom), and the grid shift stays in voxels."""
    rng = np.random.default_rng(n + m)
    q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    centre = rng.uniform(-10, 10, 3)
    fov = 24.0
    slice_axis = order.index(2)                          # output axis that holds reconstruction axis 2
    rot = lambda size: [0.5 - (0.5 / size if j == slice_axis else 0.0) for j in range(3)]  # noqa: E731
    a_full = _paravision_affine(q, fov / n, n, centre, rot(n))
    a_small = _paravision_affine(q, fov / m, m, centre, rot(m))
    shift_recon = grid_shift_from_affine(a_full, centre, order, [n] * 3)
    assert np.allclose(grid_shift_from_affine(a_small, centre, order, [m] * 3), shift_recon)
    shift_out = [shift_recon[order[j]] for j in range(3)]
    got = lowres.reduced_affine(a_full, n / m, [n] * 3, [m] * 3, shift_out)
    assert np.allclose(got, a_small, atol=1e-9)
    # FOV size kept, directions kept
    assert np.allclose(np.linalg.norm(got[:3, :3], axis=0) * m, fov)
    assert np.allclose(got[:3, :3] / np.linalg.norm(got[:3, :3], axis=0), q)


def test_reduced_affine_without_a_shift_maps_the_nufft_centre():
    a = _affine(np.random.default_rng(1))
    got = lowres.reduced_affine(a, 2.0, [16] * 3, [8] * 3, None)
    assert np.allclose(got @ [4, 4, 4, 1.0], a @ [8, 8, 8, 1.0])          # N'//2 <-> N//2
    assert np.allclose(np.linalg.norm(got[:3, :3], axis=0), 2 * np.linalg.norm(a[:3, :3], axis=0))


# ---------------------------------------------------------------- part 3: reconstruction
def _opts(tmp_path, **kw):
    return hook._build_options(dict(cache_dir=str(tmp_path), estimate_k0=False, **kw))


def _read(buf, shape):
    return np.frombuffer(buf.getvalue(), dtype=np.complex128).reshape(tuple(shape)[::-1]).T


def _setup(tmp_path, n_frames=1):
    info = _info(n_rx=1, n_frames=n_frames, phase=False)
    options = _opts(tmp_path)
    traj = get_trajectory(info, options)
    fid = b"".join(_fid_frames(info, traj, n_frames, 1, None))
    return info, options, fid


def _small(info, tmp_path, m, virtual=None, **kw):
    options = _opts(tmp_path, recon_matrix=m, **kw)
    red = lowres.resolve(options.recon_matrix, info["Matrix"])
    rows = trajectory_rows(info, options)
    cut = lowres.find_cut(rows, int(info["NPro"]), red, options.ignore_samples, virtual_traj=virtual)
    small_info = dict(info)                         # no grid shift: both grids centred at N // 2
    return options, red, cut, lowres.CutRows(rows, cut.n_keep, red.scale), small_info


def _lowpass_reference(img, red, radius):
    """Full image -> its DFT (value preserving) -> sphere |k| <= radius (grid units) -> the small grid's
    voxel centres (full-grid index N//2 + s (i' - N'//2), lowres.reduced_affine), evaluated directly."""
    n = img.shape[0]
    f = np.fft.fftn(img, norm="forward")
    k = np.fft.fftfreq(n) * n
    kk = np.meshgrid(k, k, k, indexing="ij")
    f = np.where(np.sqrt(sum(c ** 2 for c in kk)) <= radius + 1e-9, f, 0.0)
    m = red.matrix[0]
    x = n // 2 + red.scale * (np.arange(m) - m // 2)
    e = np.exp(2j * np.pi * np.outer(x, k) / n)                       # (m, n)
    return np.einsum("ia,jb,kc,abc->ijk", e, e, e, f)


def _compare(a, b):
    a, b = a.reshape(-1), b.reshape(-1)
    corr = abs(np.vdot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b))
    nrmse = np.linalg.norm(a - b) / np.linalg.norm(b)
    return float(corr), float(nrmse), float(np.abs(a).mean() / np.abs(b).mean())


def test_small_grid_is_the_full_image_lowpass(tmp_path):
    info, options, fid = _setup(tmp_path)
    full = io.BytesIO()
    recon_dataobj(io.BytesIO(fid), trajectory_rows(info, options), info, full, options)
    img = _read(full, SHAPE)
    opt_s, red, cut, rows, small_info = _small(info, tmp_path, 8)
    out = io.BytesIO()
    recon_dataobj(io.BytesIO(fid), rows, small_info, out, opt_s, out_scale=lowres.output_scale(red, cut))
    got = _read(out, [8, 8, 8])
    ref = _lowpass_reference(img, red, cut.radius_kept * SHAPE[0])
    corr, nrmse, ratio = _compare(got, ref)
    # measured on this object (2026-10-13, finufft 2.5.1): corr 0.99971, NRMSE 0.024, mean ratio 0.9956
    # (N' = 12: 0.99996, 0.0089, 0.9995; N' = 4 keeps 12 of 64 samples: 0.973, 0.23, 0.857)
    assert corr > 0.998 and nrmse < 0.06 and abs(ratio - 1.0) < 0.02, (corr, nrmse, ratio)
    assert cut.n_keep < cut.n_samples and cut.radius_kept <= 0.25 + 1e-12


def test_no_kept_point_leaves_the_small_grid_band(tmp_path):
    info, _, _ = _setup(tmp_path)
    for m in (4, 6, 8, 12):
        _, red, cut, rows, _ = _small(info, tmp_path, m)
        r = np.sqrt(np.square(rows.rows(0, N_PRO)).sum(-1))
        assert r.max() <= 0.5 + 1e-12
        assert rows.rows(0, N_PRO).shape == (N_PRO, cut.n_keep, 3)


def test_too_small_a_matrix_is_refused(tmp_path):
    info, _, _ = _setup(tmp_path)
    with pytest.raises(ValueError, match="sample"):
        _small(info, tmp_path, 2, ignore_samples=8)


def test_default_path_is_untouched(tmp_path):
    """recon_matrix equal to the matrix (or absent): the usual path. finufft's threaded spreading
    changes the last bits from run to run (measured: the same options twice differ bitwise), so
    the values are held to 1e-12 of the largest; the cache key is checked in test_recon_cache_key."""
    info, options, fid = _setup(tmp_path)
    a, b = io.BytesIO(), io.BytesIO()
    recon_dataobj(io.BytesIO(fid), trajectory_rows(info, options), info, a, options)
    opts_eq = _opts(tmp_path, recon_matrix=list(SHAPE))
    assert parse_volume_shape(info, opts_eq) == list(SHAPE)
    recon_dataobj(io.BytesIO(fid), trajectory_rows(info, opts_eq), info, b, opts_eq)
    c = io.BytesIO()
    recon_dataobj(io.BytesIO(fid), trajectory_rows(info, options), info, c, options)
    va, vb, vc = (np.frombuffer(x.getvalue(), dtype=np.complex128) for x in (a, b, c))
    tol = 1e-12 * np.abs(va).max()
    assert np.allclose(va, vc, rtol=0, atol=tol) and np.allclose(va, vb, rtol=0, atol=tol)


def test_frames_on_the_small_grid(tmp_path):
    """frame mode (one repetition per frame) gives what recon_dataobj gives on the small grid."""
    from brkraw_sordino import frames as f

    info, _, fid = _setup(tmp_path, n_frames=2)
    opt_s, red, cut, rows, small_info = _small(info, tmp_path, 8, frame_spokes=N_PRO)
    scale = lowres.output_scale(red, cut)
    plan = f.make_plan(small_info, opt_s)
    assert plan is not None and plan.n_frames == 2
    a, b = io.BytesIO(), io.BytesIO()
    f.recon_frames(io.BytesIO(fid), rows, small_info, a, opt_s, plan, out_scale=scale)
    recon_dataobj(io.BytesIO(fid), rows, small_info, b, opt_s, out_scale=scale)
    va = np.frombuffer(a.getvalue(), dtype=np.complex128)
    vb = np.frombuffer(b.getvalue(), dtype=np.complex128)
    assert va.shape == vb.shape == (2 * 8 ** 3,)
    assert np.allclose(va, vb, rtol=0, atol=1e-9 * np.abs(vb).max())


@pytest.mark.parametrize("method", ["toeplitz", "samples"])
def test_estimate_k0_on_the_small_grid(tmp_path, method):
    from brkraw_sordino import kcentre

    info, _, fid = _setup(tmp_path)
    virtual = kcentre.leading_points(info, 1)
    opt_s, red, cut, rows, small_info = _small(info, tmp_path, 8, virtual=virtual)
    scale = lowres.output_scale(red, cut)
    plain, filled = io.BytesIO(), io.BytesIO()
    recon_dataobj(io.BytesIO(fid), rows, small_info, plain, opt_s, out_scale=scale)
    k0 = []
    recon_dataobj(io.BytesIO(fid), rows, small_info, filled, opt_s, out_scale=scale,
                  virtual_traj=virtual * red.scale, k0_out=k0, k0_method=method)
    p, q = _read(plain, [8, 8, 8]), _read(filled, [8, 8, 8])
    assert np.isfinite(q).all() and len(k0) == 1 and np.isfinite(k0[0][0])
    corr, _, _ = _compare(q, p)
    assert corr > 0.95                    # the centre fill changes the image, not its shape
    assert not np.allclose(p, q)


# ---------------------------------------------------------------- part 4: the hook
class _Scan:
    scan_id = 3


class _Fid:
    name = "fid"


def _hook_info():
    info = _info(n_rx=1, n_frames=1, phase=False)
    info["GradientOrientation"] = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
    info["SliceOrientation"] = "coronal"
    return info


@pytest.fixture
def hooked(monkeypatch):
    info = _hook_info()
    monkeypatch.setattr(hook, "_parse_recon_info", lambda scan: dict(info))
    monkeypatch.setattr(hook, "_get_fid_entry", lambda scan: _Fid())
    monkeypatch.setattr(hook, "_grid_shift", lambda scan, rid, ri: None)
    return info


def test_plan_shape_key_and_estimate(tmp_path, hooked):
    full = hook._plan(_Scan(), None, {"cache_dir": str(tmp_path), "estimate_k0": False})
    small = hook._plan(_Scan(), None, {"cache_dir": str(tmp_path), "estimate_k0": False, "recon_matrix": "8"})
    same = hook._plan(_Scan(), None, {"cache_dir": str(tmp_path), "estimate_k0": False, "recon_matrix": 16})
    assert small["info"]["shape"][:3] == [8, 8, 8] and full["info"]["shape"][:3] == [16, 16, 16]
    assert small["img_cache_path"] != full["img_cache_path"]
    assert same["img_cache_path"] == full["img_cache_path"] and same["options"].recon_matrix is None
    assert small["info"]["cache_nbytes"] * 8 == full["info"]["cache_nbytes"]
    assert small["info"]["peak_nbytes"] < full["info"]["peak_nbytes"]
    assert "GridShift" not in small["recon_info"] and "GridShift" not in full["recon_info"]


@pytest.mark.parametrize("value,match", [(32, "larger"), ([8, 8, 4], "one ratio"), ([8, 8], "one value or three")])
def test_hook_errors(tmp_path, hooked, value, match):
    with pytest.raises(ValueError, match=match):
        hook._plan(_Scan(), None, {"cache_dir": str(tmp_path), "recon_matrix": value})


@pytest.mark.parametrize("ext", [None, 1.5])
def test_get_affine(tmp_path, hooked, monkeypatch, ext):
    base = np.array([[0.2, 0, 0, -1.6], [0, 0.2, 0, -1.4], [0, 0, 0.25, 2.0], [0, 0, 0, 1.0]])
    monkeypatch.setattr(hook, "get_affine_helper", lambda scan, rid, decimals=None, **kw: base.copy())
    kw = {"cache_dir": str(tmp_path), "recon_matrix": 8}
    if ext is not None:
        kw["ext_factors"] = ext
    scan = _Scan()
    got = hook.get_affine(scan, None, **kw)
    expect = lowres.reduced_affine(base, 2.0, [16] * 3, [8] * 3, None)     # no grid shift here (fixture)
    if ext is not None:
        m = int(8 * ext)
        expect = hook._apply_ext_factor_affine(expect, tuple([-(m // 2 - 4)] * 3))
    assert np.allclose(got, expect)
    assert np.allclose(np.diag(got)[:3], [0.4, 0.4, 0.5])
    # without the option: the brkraw affine itself
    assert np.array_equal(hook.get_affine(_Scan(), None, cache_dir=str(tmp_path)), base)
