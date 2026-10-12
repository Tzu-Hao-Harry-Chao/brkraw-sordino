"""Sequence generation labels 0.1 / 0.2 / 0.3 / 1.x / zte (WI-0125, D-0217).

The labels are display only. The internal values "v1" | "v2" | "v3" | "zte" stay in the
timing model and in the cache keys; the pinned keys below were taken on the commit before the
labels existed (d6bc1d3) and must never move because of a label change.
"""
import re
from pathlib import Path

import pytest

from brkraw_sordino import hook, timing, traj as traj_mod
from brkraw_sordino.hook import _build_options

from test_ramp_model import _info

ROOT = Path(__file__).resolve().parents[1]

#: trajectory_cache_key of test_ramp_model._info(v) with default options, on d6bc1d3 (before WI-0125)
PINNED_KEYS = {
    "v1": "cf54b1ad0ba5a69650096eca387f50c3219aea833559646e20770cf6ab93a8ce",
    "v2": "f4b9a5fa6dcfd8f435ab18f5fade7fe731dc162c9cbf08f68a0b5d8f9e922d44",
    "v3": "f4b9a5fa6dcfd8f435ab18f5fade7fe731dc162c9cbf08f68a0b5d8f9e922d44",
    "zte": "ea6839e443f8d531450c5bd4b80243bb8d2020e645ecfc2b6b48dbde4ce146ab",
}


@pytest.mark.parametrize("version", sorted(PINNED_KEYS))
def test_trajectory_cache_key_is_unchanged(tmp_path, version):
    opts = _build_options({"cache_dir": str(tmp_path)})
    inp = traj_mod._trajectory_inputs(_info(version), opts)
    key = traj_mod.trajectory_cache_key(inp["grad_params"], inp["n_samples"], inp["model"])
    assert key == PINNED_KEYS[version]


def test_internal_values_are_the_cache_key_values():
    for version in ("v1", "v2", "v3", "zte"):
        assert timing.detect_version(_info(version)) == version
        assert timing.read_timing(_info(version)).version == version
        assert timing.describe(timing.read_timing(_info(version)),
                               timing.tuning_for(version))["timing"]["version"] == version
    assert set(timing.TIMING_TUNING) == {"v1", "v2", "v3", "zte"}
    assert set(timing.GENERATION_LABELS) == set(timing.TIMING_TUNING)


@pytest.mark.parametrize("version,method,label", [
    ("v1", None, "0.1"),
    ("v1", "sordino", "0.1"),                       # the method name matters for v3 only
    ("v2", None, "0.2"),
    ("v3", "sordino_260801", "0.3"),
    ("v3", "sordino", "1.x"),
    ("v3", None, "0.3 or 1.x"),
    ("v3", "something_else", "0.3 or 1.x"),
    ("zte", None, "zte"),
])
def test_generation_label(version, method, label):
    assert timing.generation_label(version, method) == label


class _Method:
    def __init__(self, name):
        self._name = name

    def get(self, key):
        return self._name if key == "Method" else None


class _Scan:
    scan_id = 3

    def __init__(self, name):
        self.method = None if name is False else _Method(name)


@pytest.mark.parametrize("version,name,label,method", [
    ("v3", "<User:sordino_260801>", "0.3", "sordino_260801"),
    ("v3", "<User:sordino>", "1.x", "sordino"),
    ("v3", None, "0.3 or 1.x", None),
    ("v3", False, "0.3 or 1.x", None),              # no method object at all
    ("v2", "<User:sordino_260122_trig>", "0.2", "sordino_260122_trig"),
    ("v1", "<User:mjm_zte_231005>", "0.1", "mjm_zte_231005"),
    ("zte", "<User:zte>", "zte", "zte"),
])
def test_info_items_come_from_the_method_name(version, name, label, method):
    assert hook._sequence_generation(_Scan(name), _info(version)) == (label, method)


def test_unrecognised_sequence_gives_none():
    assert hook._sequence_generation(_Scan("<User:x>"), dict(_info("v2"), TrigSegmentMode=None,
                                                             RampDelay_ms=None))[0] is None


def test_get_dataobj_info_carries_the_generation(tmp_path, monkeypatch):
    from test_serial_recon import _info as serial_info

    info = dict(serial_info(1, 1), TrigSegmentMode=None, RampDelay_ms=None,
                MaximizeRampTime="Yes", RFWait_ms=0.01)          # internal v3
    monkeypatch.setattr(hook, "_parse_recon_info", lambda scan: dict(info))
    monkeypatch.setattr(hook, "_get_fid_entry", lambda scan: None)
    monkeypatch.setattr(hook, "_check_fid_size", lambda *a, **k: None)
    monkeypatch.setattr(hook, "_fid_nbytes", lambda entry: None)
    monkeypatch.setattr(hook, "_get_fid_identity", lambda entry: "fid")
    for name, label in (("<User:sordino_260801>", "0.3"), ("<User:sordino>", "1.x"), (None, "0.3 or 1.x")):
        plan = hook._plan(_Scan(name), None, {"cache_dir": str(tmp_path)})
        assert plan["info"]["sequence_generation"] == label
        assert plan["info"]["sequence_method"] == (None if name is None else name[6:-1])


def test_no_person_name_and_no_old_labels_in_public_text():
    """Public code and docs use the generation labels and no person's name."""
    public = [ROOT / "README.md", ROOT / "RELEASE_NOTES.md", ROOT / "docs" / "technical-note.md",
              ROOT / "src" / "brkraw_sordino" / "docs.md"]
    public += sorted((ROOT / "src" / "brkraw_sordino").glob("*.py"))
    for path in public:
        text = path.read_text(encoding="utf-8")
        assert "sordino_martin" not in text and "sordino_proto" not in text, path.name
    note = (ROOT / "docs" / "technical-note.md").read_text(encoding="utf-8")
    table = note.split("depends on the sequence generation", 1)[1].split("$T_{\\mathrm{RF}}$ is", 1)[0]
    assert not re.search(r"^\| v[123] \|", table, re.M)
    for label in ("0.1", "0.2", "0.3", "1.x", "zte"):
        assert f"| {label} |" in table
