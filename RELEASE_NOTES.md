# Unreleased (branch golden-weights-k0, on brkraw-060)

Documentation (WI-0138): `brkraw hook docs sordino` (`src/brkraw_sordino/docs.md`) is now a CLI usage page of
37 lines, 80 columns: three usage lines and one line per option with its default. The option tables, cautions,
developer checks and the size-estimate table it held moved unchanged to Appendix A of `docs/technical-note.md`.

Behaviour change, `estimate_k0` on every trajectory (WI-0137, D-0234): `estimate_k0` and `correct_ramptime` are now
independent. With `correct_ramptime: false` the k-space centre is estimated on the straight trajectory (the virtual
samples continue each spoke's line back to the RF centre); a general ZTE (no `RampTime`) is no longer reconstructed
without it. The default stays `estimate_k0: true`, so **results with `correct_ramptime: false` and general ZTE results
change** (the centre is filled); their recon caches get a new key, and `estimate_k0: false` gives the previous result
and finds the old caches. An explicit `estimate_k0: true` with `correct_ramptime: false` is no longer an error. Every
other combination keeps its results and cache keys. The least-squares solve of the centre uses 10 iterations up to an
unsampled centre of 1.3 k-grid units (every SORDINO scan: unchanged), 40 up to 1.8 and 160 above (synthetic sweep:
with 10 the fill degrades above 1.3); the count enters the cache key only when it is not 10, and a centre wider than
2.0 k-grid units gives a warning (the reconstruction still runs). ZTE-mode scans of `sordino_260801` (`SequenceMode = ZTE`) are not detected from the header: set
`correct_ramptime: false` for them.

New option `recon_matrix` (WI-0132, D-0230): reconstruct on a smaller matrix at the same FOV (`-H sordino:recon_matrix=60`
or `[60, 60, 60]`; one ratio on every axis, not above `PVM_Matrix`). Each spoke is cut at the new Nyquist radius (a
sphere, no taper), the density weights and `estimate_k0` work on the kept samples, and the values follow the full image's
low-pass. The affine is the one ParaVision writes for that matrix (voxels `Matrix / recon_matrix` larger, the FOV centre
kept). Leaving it out changes nothing (same results and cache keys). On the phantom (scan 13, 120^3 to 60^3): NRMSE 0.033
against the full image's Fourier low-pass, scale within 0.2 %, affine equal to the native 60^3 scan 18. Golden scans
with `capped`/`pipe` weights: same shape, global scale 0.74 of the full image (weights re-estimated on the kept samples).

Documentation (WI-0122, no behaviour change): `README.md` is now the short entry (description, install, hook install,
minimal use). `brkraw hook docs sordino` (`src/brkraw_sordino/docs.md`) shows the option tables, defaults and cautions.
The algorithms, with their equations, are in `docs/technical-note.md`. `tests/test_docs.py` keeps the option tables in
step with the code.

Sequence generations (WI-0125, no change to results or cache keys): the sequences are now named 0.1 (`mjm_zte_231005`),
0.2 (`sordino_260122_trig`), 0.3 (`sordino_260801`, golden sample extension) and 1.x (`sordino`, the stable release);
zte stays. The earlier "v1/v2/v3" wording is gone from the docs and log messages. 0.3 and 1.x share one ramp model and
are told apart by the method name only (`"0.3 or 1.x"` when it says neither). `get_dataobj_info` has two new items,
`sequence_generation` and `sequence_method`, and the recon cache `.json` keeps `sequence_generation`; no existing item
changed. Internally the model values `v1`/`v2`/`v3`/`zte` stay, so trajectory caches are not rebuilt.

Behaviour changes for every scan:

- Image grid (WI-0119): images are now placed where the scan's affine says
  (grid shift = affine index of the FOV centre minus the NUFFT centre `N // 2`).
  For the axial phantom scans this moves them half a voxel down the slice axis
  (they sat 0.1 mm too high at 0.2 mm voxels). Default and golden trajectories,
  frames and `estimate_k0` all follow; the affine does not change. Recon caches
  written before this change are not read (reconstructed once).

- `estimate_k0` is now on by default. The dead-time centre of k-space is
  estimated and filled for SORDINO 0.1 to 1.x data; general ZTE data and scans
  without the sequence timing values are reconstructed without it (info line).
  With `correct_ramptime: false` and no explicit `estimate_k0` it is off (info
  line); `estimate_k0: true` with `correct_ramptime: false` is still an error.
  Images differ from earlier versions, reconstruction takes longer and needs
  more memory (on small computers the memory check may now stop and offer a
  higher limit). Set `estimate_k0: false` for the previous result. Recon caches
  made with the old default are not reused under the new default (they are
  found again with `estimate_k0: false`).

Golden trajectories (`TrajectoryMode` GoldenSampling / GoldenGridSampling of
`sordino_260801`; merged into brkraw-060 at da72d83):

- New option `golden_weights`: `capped` (default; Pipe-Menon weights capped at
  the |k|^2 weight, then one common factor), `pipe` (uncapped Pipe-Menon) or
  `k2` (|k|^2). Default scans are unchanged. The default keeps the recon caches
  made at da72d83.
- Already on brkraw-060 (da72d83): golden trajectory lists, frames (subset,
  sliding window, accumulation; `frame_spokes`, `frame_step`,
  `frame_accumulate`), the Golden Grid head rule for `sordino_260801`, and the
  capped sample-based density weights. See "Golden trajectories and frames" in
  the README.

# Release v0.2.4

Date: 2026-01-31
Changes since 0.2.2

- chore: prepare release v0.2.4 (edca175)
- Refactor: standardize logging and enhance off-resonance options (b4f7aa1)
- fix: update release_pr ensure_pr call (222d7a8)
- docs: release notes for v0.2.3 (6c842b5)
- chore: prepare release v0.2.3 (9a6b87d)
- fix: apply off-resonance correction during recon (f473ca6)
- ci: patch method to retreive PR (9b4d011)
