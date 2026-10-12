"""The derived grid shift on the phantom gives run 1's numbers (WI-0119, D-0207; manual check, real data).

Manual validation script, not a unit test (WI-0123): it reads the phantom dataset, so it
lives outside ``tests/`` and is neither collected by a plain ``pytest`` nor run by CI. Run it by
naming the file, with ``BRKRAW_SORDINO_GOLDEN_ZIP`` set to the phantom zip that holds the
scans 13 (Default, matrix 120) and 18 (Default, matrix 60)::

    BRKRAW_SORDINO_GOLDEN_ZIP=/path/to/phantom.zip pytest -v scripts/validate/check_grid_centre_phantom.py

Without the variable it stops with a message. The derivation on synthetic affines stays in
``tests/test_grid_centre.py``.
"""
import os

import numpy as np
import pytest

from brkraw_sordino import hook, recon

ZIP = os.environ.get("BRKRAW_SORDINO_GOLDEN_ZIP")
if not (ZIP and os.path.isfile(ZIP)):
    pytest.exit("set BRKRAW_SORDINO_GOLDEN_ZIP to the phantom dataset zip (see the docstring)", returncode=2)


def _slicegeo_position(geo):
    if isinstance(geo, (list, tuple)):
        if (len(geo) >= 2 and isinstance(geo[0], (list, tuple)) and len(geo[0]) == 9
                and isinstance(geo[1], (list, tuple)) and len(geo[1]) == 3):
            return np.array([float(v) for v in geo[1]])
        for g in geo:
            r = _slicegeo_position(g)
            if r is not None:
                return r
    return None


@pytest.mark.parametrize("sid, n", [(13, 120), (18, 60)])
def test_phantom_shift_from_the_affine_is_run_1s(sid, n):
    """Scans 13 and 18 (axial, even): the derived shift is exactly (0, 0, -0.5), and the FOV centre
    ParaVision records is the slice-pack position the sequence frequency-shifts to."""
    import brkraw

    scan = brkraw.load(ZIP).get_scan(sid)
    info = hook._parse_recon_info(scan)
    assert hook._grid_shift(scan, None, info) == (0.0, 0.0, -0.5)
    reco = scan.get_reco(1)
    c = recon.paravision_fov_centre(reco.visu_pars, reco.reco)
    assert c == pytest.approx(_slicegeo_position(scan.method.get("PVM_SliceGeo")), abs=1e-6)
    assert np.asarray(info["Matrix"]).tolist() == [n, n, n]


def test_phantom_plan_carries_the_shift_into_recon_info_and_key(tmp_path):
    import brkraw

    scan = brkraw.load(ZIP).get_scan(18)
    plan = hook._plan(scan, None, {"cache_dir": str(tmp_path)})
    assert plan["recon_info"]["GridShift"] == [0.0, 0.0, -0.5]
    assert plan["cache_params"]["grid"] == {"rule": recon.GRID_RULE, "shift": [0.0, 0.0, -0.5]}
