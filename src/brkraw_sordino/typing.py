from dataclasses import dataclass
from typing import Tuple, Optional, Union
from pathlib import Path


@dataclass
class Options:
    ext_factors: Tuple[float, float, float]
    ignore_samples: int
    offset: int
    num_frames: Optional[int]
    correct_spoketiming: bool
    correct_ramptime: bool
    offreso_freqs: Tuple[Optional[Union[float, int]], ...]
    mem_limit: float
    clear_cache: bool
    split_ch: bool
    cache_dir: Path
    as_complex: bool
    # BRK-0066: algebraic estimate of the k-space centre (kcentre.py), SORDINO v1-v3 only.
    # The hook's default is on since WI-0113 run 6 (D-0200 2; hook._build_options); an Options
    # built directly keeps False so that internal callers state it.
    estimate_k0: bool = False
    # WI-0113 CP3 (frames.py): golden frames; keyed through the frame plan, not as raw options.
    frame_spokes: Optional[Union[str, int]] = None
    frame_step: Optional[int] = None
    frame_accumulate: bool = False
    # WI-0113 run 6 (D-0199 1): density weights of golden scans (dcf.CHOICES); keyed through
    # the weight rule of golden scans only (hook._build_cache_params), not as a raw option.
    golden_weights: str = "capped"


__all__ = [
    'Options'
]
