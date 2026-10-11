"""``estimate_k0`` is on by default (WI-0113 run 6, D-0200 2).

Rules:

* not given and ``correct_ramptime`` on: on;
* not given and ``correct_ramptime`` off: off, with one info line (no error: the user did not
  ask for it);
* given as true with ``correct_ramptime`` off: the error as before;
* not given and the scan has no sequence timing (the solve cannot run): off with one info line;
  given as true: the error as before;
* a general ZTE: off with the info line as before;
* ``brkraw hook preset sordino`` (``asdict(_build_options({}))``) shows ``estimate_k0: true``.
"""
import logging
from dataclasses import asdict

import pytest

from brkraw_sordino import hook
from brkraw_sordino.hook import _build_options, _resolve_k0


def _opts(tmp_path, **kw):
    return _build_options({"cache_dir": str(tmp_path), **kw})


def _info(version):
    from test_ramp_model import _info as info
    return info(version)


def test_on_by_default(tmp_path):
    assert _opts(tmp_path).estimate_k0 is True


def test_preset_shows_it_on(tmp_path, monkeypatch):
    monkeypatch.setattr(hook, "_get_cache_dir", lambda path: tmp_path)
    preset = asdict(_build_options({}))
    assert preset["estimate_k0"] is True
    assert preset["golden_weights"] == "capped"


def test_off_without_error_when_ramp_correction_is_off(tmp_path, caplog):
    with caplog.at_level(logging.INFO, logger="brkraw_sordino.hook"):
        o = _opts(tmp_path, correct_ramptime=False)
    assert (o.estimate_k0, o.correct_ramptime) == (False, False)
    lines = [r for r in caplog.records if "estimate_k0" in r.getMessage()]
    assert len(lines) == 1 and lines[0].levelno == logging.INFO
    assert "correct_ramptime" in lines[0].getMessage()


def test_explicit_true_with_ramp_correction_off_is_still_an_error(tmp_path):
    with pytest.raises(ValueError, match="estimate_k0"):
        _opts(tmp_path, estimate_k0=True, correct_ramptime=False)


def test_explicit_false_stays_off(tmp_path):
    assert _opts(tmp_path, estimate_k0=False).estimate_k0 is False
    assert _opts(tmp_path, estimate_k0="false").estimate_k0 is False


def test_general_zte_still_switches_it_off(tmp_path):
    assert _resolve_k0(_opts(tmp_path), _info("zte")).estimate_k0 is False
    assert _resolve_k0(_opts(tmp_path), _info("v3")).estimate_k0 is True


def test_missing_timing_raises_a_k0_timing_error(tmp_path):
    with pytest.raises(hook.K0TimingError, match="estimate_k0"):
        _resolve_k0(_opts(tmp_path), {})
    assert issubclass(hook.K0TimingError, ValueError)


class _Scan:
    scan_id = 3


def test_default_without_timing_is_off_with_an_info_line(tmp_path, monkeypatch, caplog):
    from test_serial_recon import _info as serial_info

    info = serial_info(1, 1)
    for key in ("SordinoVersion", "RampTime", "AcqDelayTotal", "ReadDelay"):
        info.pop(key, None)
    monkeypatch.setattr(hook, "_parse_recon_info", lambda scan: dict(info))
    monkeypatch.setattr(hook, "_get_fid_entry", lambda scan: None)
    monkeypatch.setattr(hook, "_resolve_k0", _raise_timing)
    monkeypatch.setattr(hook, "_check_fid_size", lambda *a, **k: None)
    monkeypatch.setattr(hook, "_fid_nbytes", lambda entry: None)
    monkeypatch.setattr(hook, "_get_fid_identity", lambda entry: "fid")
    with caplog.at_level(logging.INFO, logger="brkraw_sordino.hook"):
        plan = hook._plan(_Scan(), None, {"cache_dir": str(tmp_path)})
    assert plan["options"].estimate_k0 is False
    assert [r for r in caplog.records if "estimate_k0" in r.getMessage() and r.levelno == logging.INFO]
    with pytest.raises(hook.K0TimingError):
        hook._plan(_Scan(), None, {"cache_dir": str(tmp_path), "estimate_k0": True})


def _raise_timing(options, recon_info):
    if options.estimate_k0:
        raise hook.K0TimingError("estimate_k0 needs the sequence timing parameters: missing")
    return options


def test_default_changes_the_recon_cache_key_against_explicit_false(tmp_path):
    from brkraw_sordino.recon import build_recon_cache_path

    a, b = _opts(tmp_path), _opts(tmp_path, estimate_k0=False)
    assert build_recon_cache_path(tmp_path, {"options": asdict(a)}) != \
        build_recon_cache_path(tmp_path, {"options": asdict(b)})
