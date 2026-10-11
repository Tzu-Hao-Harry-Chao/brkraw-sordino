# Unreleased (branch golden-weights-k0, on brkraw-060)

Behaviour changes for every scan:

- Image grid (WI-0119): images are now on ParaVision's 3D grid, which the
  affine describes. Earlier images sat half a voxel too high on the slice axis
  (0.1 mm at 0.2 mm voxels). Default and golden trajectories, frames and
  `estimate_k0` all change by that half voxel; the affine does not change.
  Recon caches written before this change are not read (reconstructed once).

- `estimate_k0` is now on by default. The dead-time centre of k-space is
  estimated and filled for SORDINO v1-v3 data; general ZTE data and scans
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
