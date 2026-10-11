"""Density-weight choice for golden scans (WI-0113 run 6, D-0199 1).

The capped Pipe-Menon rule of run 5 stays the default; the user may choose the |k|^2
weights (``k2``) or the uncapped Pipe-Menon weights (``pipe``) with the hook option
``golden_weights``. Default scans keep |k|^2 whatever the option says (a warning when a
non-default value is given), and their recon cache keys do not change. The default
golden key is the key of ``da72d83`` (caches made there stay valid).
"""
import logging

import numpy as np
import pytest

from brkraw_sordino import dcf, hook, recon
from brkraw_sordino.hook import _build_options
from brkraw_sordino.recon import phase_correction_rows
from brkraw_sordino.traj import get_trajectory

from test_golden_recon import _Entry, _Scan, _golden_info, _hook_image
from test_golden_density import _pipe_reference, _weighted_adjoint
from test_serial_recon import N_POINTS, SHAPE, _fid_frames, _info as _serial_info, _rel

#: dcf.RULE at da72d83 (the merged run-5 rule): the default golden key must not change.
RULE_DA72D83 = {"method": "pipe-menon", "iterations": 10, "upsampfac": 2.0,
                "scale": "sum of |k|^2/max", "virtual_samples": "included with estimate_k0",
                "cap": "min(Pipe, |k|^2/max), then the same sum"}


def _opts(tmp_path, **kw):
    return _build_options({"cache_dir": str(tmp_path), **kw})


# ------------------------------------------------------------------ parsing
def test_default_is_capped(tmp_path):
    assert _opts(tmp_path).golden_weights == "capped"
    assert dcf.CHOICES == ("capped", "pipe", "k2") and dcf.DEFAULT == "capped"


@pytest.mark.parametrize("text, value", [("capped", "capped"), ("Pipe", "pipe"), (" K2 ", "k2"),
                                         ("k2", "k2"), ("PIPE", "pipe")])
def test_choice_is_read_case_insensitively(tmp_path, text, value):
    assert _opts(tmp_path, golden_weights=text).golden_weights == value


@pytest.mark.parametrize("bad", ["voronoi", "", "k^2", 2, True, None])
def test_other_values_are_refused_by_name(tmp_path, bad):
    with pytest.raises(ValueError, match="golden_weights") as err:
        _opts(tmp_path, golden_weights=bad)
    assert "capped" in str(err.value) and "pipe" in str(err.value) and "k2" in str(err.value)


def test_option_is_known(tmp_path, caplog):
    with caplog.at_level(logging.WARNING, logger="brkraw_sordino.hook"):
        _opts(tmp_path, golden_weights="pipe")
    assert not [r for r in caplog.records if "unknown" in r.getMessage()]


# ------------------------------------------------------------------ cache keys
def test_default_golden_key_is_the_merged_key(tmp_path):
    assert dcf.RULE == RULE_DA72D83
    g = hook._build_cache_params(_Scan(), None, _Entry(b""), _opts(tmp_path), _golden_info())
    assert g["density"] == RULE_DA72D83
    assert "golden_weights" not in g["options"]


def test_each_choice_has_its_own_golden_key(tmp_path):
    keys = {}
    for choice in dcf.CHOICES:
        p = hook._build_cache_params(_Scan(), None, _Entry(b""), _opts(tmp_path, golden_weights=choice),
                                     _golden_info())
        keys[choice] = p["density"]
        assert p["density"] == dcf.rule(choice)
    assert len({repr(v) for v in keys.values()}) == 3
    assert keys["pipe"]["cap"] == "none"
    assert keys["k2"]["method"] == "|k|^2/max"


def test_default_scan_keys_do_not_depend_on_the_choice(tmp_path):
    info = _serial_info(1, 1)
    base = hook._build_cache_params(_Scan(), None, _Entry(b""), _opts(tmp_path), info)
    assert "density" not in base and "golden_weights" not in base["options"]
    for choice in dcf.CHOICES:
        p = hook._build_cache_params(_Scan(), None, _Entry(b""), _opts(tmp_path, golden_weights=choice), info)
        assert p == base


def test_uses_sample_weights_follows_the_choice():
    g = _golden_info(1, "GoldenGridSampling")
    assert dcf.uses_sample_weights(g) and dcf.uses_sample_weights(g, "capped")
    assert dcf.uses_sample_weights(g, "pipe")
    assert not dcf.uses_sample_weights(g, "k2")
    for choice in dcf.CHOICES:
        assert not dcf.uses_sample_weights(_serial_info(1, 1), choice)


# ------------------------------------------------------------------ images
def _case(tmp_path, mode):
    info = _golden_info(1, mode)
    options = _opts(tmp_path / "t", estimate_k0=False)
    traj = get_trajectory(info, options)
    factor = phase_correction_rows(info, options, N_POINTS)[:]
    (fid,) = _fid_frames(info, traj, 1, 1, factor)
    vol = np.frombuffer(fid, dtype="<i4").reshape((2, N_POINTS, 1, info["NPro"]), order="F")
    k = ((vol[0] + 1j * vol[1]).T[:, 0, :]) * factor
    return info, traj, fid, k


def _uncapped_reference(traj, shape):
    from mrinufft.operators.interfaces.finufft import MRIfinufft

    omega = np.ascontiguousarray(traj.reshape(-1, 3) / 0.5 * np.pi, dtype=np.float64)
    w = np.abs(np.asarray(MRIfinufft.pipe(omega, tuple(shape), max_iter=10, osf=2, normalize=False)))
    k2 = np.square(traj).sum(-1).reshape(-1)
    return w * ((k2.sum() / k2.max()) / w.sum())


@pytest.mark.parametrize("mode", ["GoldenSampling", "GoldenGridSampling"])
def test_each_choice_reaches_the_image(tmp_path, monkeypatch, mode):
    from brkraw_sordino.orientation import correct as correct_orientation

    info, traj, fid, k = _case(tmp_path, mode)
    from test_grid_centre import pv_grid
    k = pv_grid(k, traj, SHAPE)                       # ParaVision's grid (WI-0119)
    tr = traj[:, 1:, :]
    refs = {
        "capped": correct_orientation(_weighted_adjoint(k[:, 1:], tr, SHAPE, _pipe_reference(tr, SHAPE)[0]), info),
        "pipe": correct_orientation(_weighted_adjoint(k[:, 1:], tr, SHAPE, _uncapped_reference(tr, SHAPE)), info),
        "k2": correct_orientation(recon.nufft_adjoint(k[:, 1:], tr, SHAPE, 1), info),
    }
    for choice, ref in refs.items():
        img = _hook_image(tmp_path / choice, monkeypatch, info, fid, golden_weights=choice,
                          estimate_k0=False)[..., 0]
        assert _rel(img, ref) < 1e-8, choice
    assert _rel(refs["pipe"], refs["capped"]) > 1e-3 and _rel(refs["k2"], refs["capped"]) > 1e-3


def test_default_scan_ignores_the_choice_with_a_warning(tmp_path, monkeypatch, caplog):
    from test_serial_recon import _info

    info = _info(1, 1)
    options = _opts(tmp_path / "t", estimate_k0=False)
    traj = get_trajectory(info, options)
    factor = phase_correction_rows(info, options, N_POINTS)
    (fid,) = _fid_frames(info, traj, 1, 1, None if factor is None else factor[:])
    plain = _hook_image(tmp_path / "a", monkeypatch, info, fid, estimate_k0=False, frame_spokes=None)
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="brkraw_sordino.hook"):
        other = _hook_image(tmp_path / "b", monkeypatch, info, fid, estimate_k0=False, frame_spokes=None,
                            golden_weights="pipe")
    assert _rel(other, plain) < 1e-12          # two finufft runs differ by ~1e-16 (threads)
    assert [r for r in caplog.records if "golden_weights" in r.getMessage()
            and "golden" in r.getMessage().lower()]


def test_metadata_names_the_weights(tmp_path, monkeypatch):
    info, traj, fid, k = _case(tmp_path, "GoldenSampling")
    scan = _Scan()
    monkeypatch.setattr(hook, "_parse_recon_info", lambda s: dict(info))
    monkeypatch.setattr(hook, "_get_fid_entry", lambda s: _Entry(fid))
    hook.get_dataobj(scan, None, cache_dir=str(tmp_path / "m"), estimate_k0=False, golden_weights="pipe",
                     frame_spokes="repetition")
    assert scan._sordino_recon_meta["density_weights"] == "pipe"
    dinfo = _serial_info(1, 1)
    monkeypatch.setattr(hook, "_parse_recon_info", lambda s: dict(dinfo))
    monkeypatch.setattr(hook, "_get_fid_entry", lambda s: _Entry(b"\0" * (8 * N_POINTS * dinfo["NPro"])))
    scan2 = _Scan()
    hook.get_dataobj(scan2, None, cache_dir=str(tmp_path / "d"), estimate_k0=False, golden_weights="pipe")
    assert scan2._sordino_recon_meta["density_weights"] == "k2"


# ------------------------------------------------------------------ memory
def test_k2_choice_holds_no_weights(tmp_path, monkeypatch):
    from brkraw_sordino import memguard

    info = _golden_info(1, "GoldenGridSampling")
    monkeypatch.setattr(hook, "_parse_recon_info", lambda scan: dict(info))
    monkeypatch.setattr(hook, "_get_fid_entry", lambda scan: _Entry(b"\0" * (8 * N_POINTS * info["NPro"])))
    for choice, expect in (("k2", 0), ("pipe", memguard.dcf_nbytes(info["NPro"], N_POINTS - 1, SHAPE)),
                           ("capped", memguard.dcf_nbytes(info["NPro"], N_POINTS - 1, SHAPE))):
        gi = hook.get_dataobj_info(_Scan(), None, cache_dir=str(tmp_path / choice), frame_spokes="repetition",
                                   estimate_k0=False, golden_weights=choice)
        assert gi["recon_dcf_nbytes"] == expect, choice


def test_pipe_weights_cap_argument_is_the_choice(tmp_path):
    info = _golden_info(1, "GoldenGridSampling")
    options = _opts(tmp_path, estimate_k0=False)
    traj = get_trajectory(info, options)
    tr = traj[:, 1:, :]
    capped = dcf.pipe_weights(traj, info["NPro"], 1, SHAPE, 50, cap=True).w.reshape(-1)
    uncapped = dcf.pipe_weights(traj, info["NPro"], 1, SHAPE, 50, cap=False).w.reshape(-1)
    assert _rel(capped, _pipe_reference(tr, SHAPE)[0]) < 1e-9
    assert _rel(uncapped, _uncapped_reference(tr, SHAPE)) < 1e-9
