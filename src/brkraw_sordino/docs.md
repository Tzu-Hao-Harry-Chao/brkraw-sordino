# brkraw-sordino: options and practical notes

SORDINO-ZTE reconstruction hook for BrkRaw. This page is what `brkraw hook docs sordino`
shows: how to pass options, what each option does, and what to watch for. The algorithms
behind them are in `docs/technical-note.md` of the repository.

## Passing options

```bash
brkraw convert /path/to/study -s 3 -r 1 \
  --hook-arg sordino:ext_factors=1.2 --hook-arg sordino:offset=2 --hook-arg sordino:split_ch=false
```

Or from YAML. `brkraw hook preset sordino -o hook_args.yaml` writes a template with every
reconstruction option at its default; edit it and pass it (repeatable):

```bash
brkraw convert /path/to/study -s 3 -r 1 --hook-args-yaml hook_args.yaml
```

```yaml
hooks:
  sordino:
    ext_factors: 1.2
    offset: 2
    split_ch: false
    # as_complex: true  # optional, return (real, imag)
    # cache_dir: ~/.brkraw/cache/sordino  # optional (add manually if needed)
    # frame_spokes: 3200      # golden scan: 20 subsets of 160 spokes, 2 s at a TR of 0.625 ms
    # frame_accumulate: true  # frames of 3,200, 6,400, 9,600 ... spokes
```

- `--hook-arg` values override YAML. YAML takes `{hooks: {sordino: {...}}}` or `{sordino: {...}}`.
- `BRKRAW_CONVERT_HOOK_ARGS_YAML` holds comma-separated YAML paths.
- Booleans are case-insensitive: `true`, `True`, `TRUE`, `false`, `False` (also `1`, `0`,
  `yes`, `no`, `on`, `off`); anything else is an error.
- A key the hook does not know is ignored with a one-line warning. `ramp_model` and
  `correct_phase` were removed and have no alias.

## Reconstruction options

| Option | Default | Meaning |
| --- | --- | --- |
| `ext_factors` | `1.0` | Scalar or 3 values. Enlarges the grid (so the FOV) at the same voxel size: axis `i` gets `int(Matrix[i] * ext_factors[i])` voxels (16 x 1.5 = 24, 16 x 1.1 = 17). Order: read, phase, slice of `PVM_Matrix`. |
| `ignore_samples` | `1` | Samples dropped at the start of every spoke (0 is read as 1). |
| `offset` | `0` | First repetition to reconstruct. |
| `num_frames` | `null` | Repetitions to reconstruct from `offset` (`null`: all). |
| `correct_ramptime` | `true` | Ramp-time correction: places each sample on the ramping gradient and removes the phase the FOV-offset frequency (`ACQ_O1_list`) leaves while it ramps. Off: every spoke keeps one fixed gradient vector, no phase correction. Timing values live in `timing.py`. |
| `estimate_k0` | `true` | Fills the k-space centre that the dead time leaves unmeasured (SORDINO v1-v3): the dark centre of the adjoint image is filled. Changes the image and costs time and memory (see Cautions). `false` gives the plain adjoint. |
| `correct_spoketiming` | `false` | Aligns every spoke to the centre of its repetition's scan time (needs several repetitions; not with frames). |
| `offreso_freqs` | `null` | Off-resonance in Hz, one per receive channel: a number, a list or `"120,-80"`; all forms read the same, a non-finite value is refused. |
| `golden_weights` | `capped` | Density weights of golden scans (case-insensitive): `capped` Pipe-Menon weights capped at the \|k\|^2 weight; `pipe` the same without the cap (less streaking, more noise in thinly covered k-space); `k2` the \|k\|^2 weights of uniform straight spokes (no estimate, no extra memory or time; streaks). Each choice has its own cache; `capped` keeps the caches made before the option existed. A Default scan always uses `k2` (a warning when the option is not `capped`). |
| `split_ch` | `false` | `false` merges channels (magnitude: root sum of squares; complex: coherent sum). |
| `as_complex` | `false` | Return `(real, imag)` instead of magnitude. |
| `mem_limit` | `0.5` | GB for the FID segments of `correct_spoketiming` (divided by a safety factor 3.4). Not the reconstruction memory limit. |
| `clear_cache` | `true` | Removes leftover temporary files after conversion; finished caches stay. |
| `cache_dir` | `~/.brkraw/cache/sordino` | Cache folder (or `BRKRAW_CONFIG_HOME`). |

Frames (golden scans; a Default scan only when `frame_spokes` is a spoke count). The spoke
directions come from the method parameter `TrajectoryMode` (`Default`, `GoldenSampling`,
`GoldenGridSampling`; older data without it are `Default`); the hook reads it, there is no option.

| Option | Default | Meaning |
| --- | --- | --- |
| `frame_spokes` | `null` | Spokes per frame. `null` or `subset`: the method subset, the smallest count that covers the sphere (`NGoldenSpokesPerSubset`; one grid frame). `repetition`: one repetition per frame. Integer: a spoke count. Default scans keep one repetition per frame (`subset` is refused). |
| `frame_step` | `null` | Spokes between frame starts (`null`: `frame_spokes`). Smaller is a sliding window, larger leaves spokes out. |
| `frame_accumulate` | `false` | Every frame starts at the first spoke and holds `frame_spokes` + k x `frame_step` spokes; the last one holds every spoke read. |

Read-time options (they choose what is returned from the cache and are not part of its key):

| Option | Default | Meaning |
| --- | --- | --- |
| `frames` | `null` | Frames to return, brkraw rules: an int picks one and removes the frame axis, a list keeps the axis in that order, `"start:stop[:step]"` is a slice. Frame 0 is frame `offset`. Only these frames are read from a cache; with no cache the whole scan (or `num_frames`) is reconstructed. |
| `axis` | `null` | The frame axis is data axis 3 (`3`, `-1`, `"cycle"` or `"repetition"`). `axis` without `frames` is an error. |
| `max_memory_gb` | half of physical memory | Limit of the size check (4 GB when memory cannot be read). Over it, the hook stops before reconstructing or reading with `SordinoResourceError` (a `MemoryError`) whose `retry_kwargs` holds the smallest passing limit (0.1 GB steps), for example `{"max_memory_gb": 1.8}`; the brkraw CLI uses it to ask "Proceed anyway?". The cache must also fit on the disk. Ask for fewer frames or raise the limit. |
| `allow_short_fid` | `true` | A scan stopped before its last repetition leaves a short FID. `true`: the hook measures the FID first and, when short, logs one warning (bytes found, needed, missing; complete frames of planned) and reconstructs only the complete frames; the bytes of the incomplete last frame are not used, the NIfTI frame count is the number of frames returned (voxel size, affine and TR unchanged), `offset` and `num_frames` count complete frames; a `ValueError` if none is complete or `offset` is at or after the last. `false`: a short FID stops with a `ValueError` before anything is reconstructed. A FID of the planned size or longer is read as usual. |

Frames from the CLI (sliding window of 3,200 spokes every 1,600):

```bash
brkraw convert /path/to/study -s 11 -r 1 \
  --hook-arg sordino:frame_spokes=3200 --hook-arg sordino:frame_step=1600
```

## Cautions

- **`ext_factors` axes.** The three values follow the acquisition order (read, phase,
  slice), not anatomical axes and not the output array axes: the hook reorders axes after
  reconstruction, so which anatomical direction an index widens depends on the scan. Check
  your protocol once (for example `[1.5, 1, 1]` and see which side grew). A scalar or three
  equal values widen every axis, so the order does not matter for them. "RPS" is also an orientation code (Right-Posterior-Superior); here it is never
  one. Example (two approved coronal scans, readout `H_F`, quadruped, `Head_Supine`,
  `subject_ras`): index 0 widens posterior-anterior, 1 left-right, 2 inferior-superior; the
  output array holds index 1 on axis 0 and index 0 on axis 1.
- **`estimate_k0` is on by default** and costs time and memory (a 160^3 scan with NPro 80,876,
  NPoints 640, OverSampling 8 and 2 receivers, synthetic FID, macOS, default limit: 3.0 GiB measured without
  it, 4.9 GiB with it). Memory is a fixed part plus one chunk of
  spokes and does not grow with spokes or frames. On an 8 GB computer (limit 4 GiB) such a scan
  stops with an offer of a limit of 5.3 GB; `estimate_k0: false` fits. Set to `true`
  explicitly it stops with an error when `correct_ramptime` is false or timing values are
  missing; left at the default it is switched off with an info line then, and on a general
  ZTE. Caches made before it became the default are found again only with `estimate_k0: false`.
- **Golden frames need disk.** The cache holds one complex128 volume per frame and channel.
  The default frames of a 288,000-spoke, 160-spoke-subset scan at 120^3 are 1,800 frames:
  49.8 GB of cache (46.4 GiB), 24.9 GB as float64 magnitude in memory, 6.2 GB as uint16 NIfTI. The size
  check runs first and stops with the sizes; use larger `frame_spokes`, fewer repetitions
  (`num_frames`) or `frame_spokes: repetition`. A sliding window also keeps one running-sum
  copy per open frame start (`frame_spokes` / `frame_step` copies).
- **NIfTI is uint16 with slope and intercept.** Converted files are stored as 16-bit
  integers; the true value is `inter + slope * stored`.
- **Frames:** `correct_spoketiming` cannot be combined with frames (use
  `frame_spokes: repetition`). `estimate_k0` is estimated once from all spokes read. NIfTI
  `pixdim[4]` is `frame_step` x the spoke TR in the unit asked for. Frame spokes, centre times
  and interval are in `scan._sordino_recon_meta["frames"]`.
- **Grid shift and cache regeneration.** The image is placed where the brkraw affine says; the
  reconstruction grid is moved by a derived shift (technical note, section 8). For the phantom
  scans (axial) it is (0, 0, -0.5) voxel: earlier images sat half a voxel too high on the slice
  axis. Recon caches written before this change are not read; they are reconstructed once.
  ParaVision's own reconstruction (pdata/1) sits about half a voxel low on the slice axis; this
  is not corrected here.
- **Cache files** (`cache_dir`): `recon_<hash>.bin` (complex128, all frames) with its `.json`
  (also `kspace_gap`, `k0`, `short_fid`). They stay for reuse until `brkraw cache clear`;
  `clear_cache` removes only temporary files. The key covers scan, reco, FID and every option that
  changes the values; `as_complex`, `split_ch`, `clear_cache`, `cache_dir` and the read-time
  options are not in it, so changing them reads the same cache.
- **Result metadata** (`scan._sordino_recon_meta`, also in the cache `.json`): `kspace_gap` (radius of
  the unmeasured centre, first acquired and first kept sample), `k0` (`[real, imag]` per channel and
  frame; with frames one estimate for all), `short_fid` (`fid_nbytes`, `expected_nbytes`,
  `frame_nbytes`, `frames_planned`, `frames_complete`; null for a complete FID; a short FID has its
  own cache), `frames`, `density_weights`. A Golden Grid head, if applied, logs a warning; no option.

## For developers: tests and manual checks

`tests/` holds only tests on synthetic data; `pytest tests` needs no dataset and the CI never reads one.
Checks that read a real dataset are scripts in `scripts/validate/` (named `check_*.py`, so a plain `pytest` does not
collect them) and are run by hand, when wanted: `BRKRAW_SORDINO_GOLDEN_ZIP=/path/to/phantom.zip pytest -v
scripts/validate/check_golden_phantom.py scripts/validate/check_grid_centre_phantom.py`. Without the variable they stop
with a message. They compare the golden trajectories with the scanner's `traj` file and the derived grid shift with
the phantom's recorded geometry.

## Size estimate without reading data

`brkraw_sordino.get_dataobj_info(scan, reco_id, **options)` takes the options of `get_dataobj`
and returns a dict:

| Key | Content |
| --- | --- |
| `shape`, `dtype`, `count`, `nbytes` | each returned array, how many (channels with `split_ch`, x 2 with `as_complex`), total bytes |
| `frames`, `frames_reconstructed` | frames returned; frames in the reconstruction and its cache |
| `cached`, `cache_path`, `cache_dir`, `cache_dtype`, `cache_nbytes` | a valid cache exists and its file (`cache_dtype` is complex128 until one exists) |
| `peak_nbytes` | the estimate compared with the limit: returned arrays, three cache frames, and without a cache `recon_nbytes` |
| `recon_nbytes`, `recon_chunk_spokes`, `recon_chunks` | reconstruction step estimate, spokes per chunk, chunks (0 and null when cached) |
| `recon_frames_open`, `recon_dcf_nbytes` | running-sum copies held at once with frames (null without); bytes of the golden density weights and their grid (0 otherwise) |
| `recon_k0_method`, `recon_k0_reason` | `"toeplitz"` or `"samples"` (with `estimate_k0`, no cache); `"rule"` or `"limit"` |
| `disk_nbytes`, `disk_free_nbytes` | disk a new cache needs (0 when cached), free space (null when cached or unreadable) |
| `limit_nbytes`, `limit_source` | the limit `get_dataobj` applies; `max_memory_gb option`, `half of physical memory` or `fallback 4 GB` |
| `frames_planned`, `fid_short_nbytes` | frames in `PVM_NRepetitions`; bytes the FID is short (0 complete, null unknown); with a short FID `frames_reconstructed` is the complete frames |
