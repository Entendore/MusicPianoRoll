"""
Core data structures, constants, feature flags, and utility functions.

ALGORITHM IMPROVEMENTS:
1. NoteIndex: Added active_events_map_at() for O(log n + k) event lookup,
   avoiding O(n) linear scans in hot rendering paths.
   Added bucketed time-indexing for faster visible_in() range queries.
2. KeyDetector: Vectorized Pearson correlation with NumPy — computes all
   24 key correlations in a single matrix operation instead of a Python loop.
3. BeatDetector: Histogram-based BPM detection with log-spaced bins for
   robustness against tempo doubling/halving ambiguity.
4. Added lru_cache-friendly helpers and precomputed lookup tables.
"""
import math
import colorsys
import bisect
import json
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Union, Tuple
from PySide6.QtGui import QColor

# ═══════════════════════════════════════════════════════════════
#  OPTIONAL LIBRARY DETECTION
# ═══════════════════════════════════════════════════════════════
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    np = None
    HAS_NUMPY = False

try:
    import mido
    from mido import MidiFile
    HAS_MIDO = True
except ImportError:
    mido = None
    MidiFile = None
    HAS_MIDO = False

try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    njit = None
    HAS_NUMBA = False

try:
    from pydub import AudioSegment
    HAS_PYDUB = True
except ImportError:
    AudioSegment = None
    HAS_PYDUB = False

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    cp = None
    HAS_CUPY = False

try:
    from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
    HAS_QTMULTIMEDIA = True
except ImportError:
    QMediaPlayer = None
    QAudioOutput = None
    HAS_QTMULTIMEDIA = False

# ═══════════════════════════════════════════════════════════════
#  DATA CLASSES
# ═══════════════════════════════════════════════════════════════
@dataclass
class NoteEvent:
    note: int
    start_time: float
    duration: float
    velocity: int
    channel: int
    track: int

@dataclass
class AudioData:
    samples: list = field(default_factory=list)
    sample_rate: int = 44100
    duration: float = 0.0
    channels: int = 1
    waveform_peaks: list = field(default_factory=list)
    spectrum_data: list = field(default_factory=list)
    path: str = ""

@dataclass
class PlaylistEntry:
    midi_path: Optional[str] = None
    audio_path: Optional[str] = None
    name: str = ""
    notes: List[NoteEvent] = field(default_factory=list)
    audio_data: Optional[AudioData] = None
    duration: float = 0.0
    track_info: Dict[int, str] = field(default_factory=dict)
    tempo_map: list = field(default_factory=list)
    ticks_per_beat: int = 480
    note_count: int = 0
    file_size: int = 0
    file_type: str = "midi"
    beats: list = field(default_factory=list)
    bpm: float = 0.0
    key: str = ""

# ═══════════════════════════════════════════════════════════════
#  EXPORT PRESETS & EXTENSIONS
# ═══════════════════════════════════════════════════════════════
@dataclass
class ExportPreset:
    name: str; width: int; height: int; fps: int
    codec: str; crf: int; preset: str; description: str

YOUTUBE_PRESETS = [
    ExportPreset("YouTube 1080p", 1920, 1080, 60, "libx264", 18, "medium", "Full HD 60fps"),
    ExportPreset("Shorts 1080×1920", 1080, 1920, 60, "libx264", 18, "medium", "Vertical 9:16"),
    ExportPreset("Custom", 1920, 1080, 60, "libx264", 18, "medium", "Custom"),
]

MIDI_EXTS = {'.mid', '.midi'}
AUDIO_EXTS = {'.wav', '.mp3'}
ALL_EXTS = MIDI_EXTS | AUDIO_EXTS
FILE_FILTER = "Music (*.mid *.midi *.wav *.mp3);;MIDI (*.mid *.midi);;Audio (*.wav *.mp3);;All (*)"

# ═══════════════════════════════════════════════════════════════
#  SPATIAL INDEX & ANALYSIS
# ═══════════════════════════════════════════════════════════════
class NoteIndex:
    """
    Spatial index for fast note lookup by time range.
    
    IMPROVEMENTS over original:
    - Bucketed time index: partitions notes into fixed-duration buckets
      so visible_in() only scans relevant buckets instead of all notes
      with start_time <= t_end.  O(B + k) where B = overlapping buckets,
      vs O(n) for long pieces.
    - active_events_map_at(): returns Dict[int, NoteEvent] for O(1)
      per-note lookup by pitch, eliminating the O(n) linear scan in
      _get_active_key_color and the tick() effects loop.
    - Precomputed end_times array for efficient backward scanning.
    """
    BUCKET_SECONDS = 4.0  # Each bucket covers 4 seconds

    def __init__(self, notes: List[NoteEvent]):
        self.notes = sorted(notes, key=lambda n: n.start_time)
        self.start_times = [n.start_time for n in self.notes]
        self.end_times = [n.start_time + n.duration for n in self.notes]

        # ── Build time buckets for accelerated range queries ──
        self._buckets: Dict[int, List[int]] = {}
        for i, n in enumerate(self.notes):
            bi = int(n.start_time / self.BUCKET_SECONDS)
            self._buckets.setdefault(bi, []).append(i)

        # Max end_time per bucket — allows skipping entire buckets
        self._bucket_max_end: Dict[int, float] = {}
        for bi, indices in self._buckets.items():
            self._bucket_max_end[bi] = max(self.end_times[i] for i in indices)

    def visible_in(self, t_start: float, t_end: float) -> List[NoteEvent]:
        """
        Return notes visible in the time window [t_start, t_end].
        Uses bucket pruning: only scans buckets that could contain
        overlapping notes, skipping far-away buckets entirely.
        """
        bi_start = max(0, int(t_start / self.BUCKET_SECONDS) - 1)
        bi_end = int(t_end / self.BUCKET_SECONDS)
        result = []
        for bi in range(bi_start, bi_end + 1):
            if bi not in self._buckets:
                continue
            if self._bucket_max_end[bi] < t_start:
                continue
            for i in self._buckets[bi]:
                if self.end_times[i] >= t_start and self.start_times[i] <= t_end:
                    result.append(self.notes[i])
        return result

    def active_at(self, t: float) -> set:
        """Return set of note numbers (pitches) active at time t."""
        idx = bisect.bisect_right(self.start_times, t)
        active = set()
        for i in range(idx - 1, -1, -1):
            if self.end_times[i] < t:
                break
            active.add(self.notes[i].note)
        return active

    def active_events_map_at(self, t: float) -> Dict[int, 'NoteEvent']:
        """
        Return {pitch: NoteEvent} for all notes active at time t.
        O(log n + k) where k = active notes at time t.
        """
        idx = bisect.bisect_right(self.start_times, t)
        result: Dict[int, NoteEvent] = {}
        for i in range(idx - 1, -1, -1):
            if self.end_times[i] < t:
                break
            n = self.notes[i]
            if n.note not in result:
                result[n.note] = n
        return result


class BeatDetector:
    """
    IMPROVEMENT: Uses histogram binning in log-space instead of simple median.
    """
    @staticmethod
    def analyze(notes, duration) -> tuple:
        if len(notes) < 4:
            return [], 0.0
        intervals = []
        sorted_notes = sorted(notes, key=lambda n: n.start_time)
        for i in range(1, len(sorted_notes)):
            diff = sorted_notes[i].start_time - sorted_notes[i-1].start_time
            if 0.2 < diff < 2.0:
                intervals.append(diff)
        if not intervals:
            return [], 0.0

        if HAS_NUMPY:
            arr = np.array(intervals)
            log_arr = np.log2(arr)
            hist, bin_edges = np.histogram(log_arr, bins=60)
            best_bin = np.argmax(hist)
            median_interval = float(2 ** ((bin_edges[best_bin] + bin_edges[best_bin + 1]) / 2))
        else:
            median_interval = sorted(intervals)[len(intervals) // 2]

        bpm = 60.0 / median_interval if median_interval > 0 else 0
        while bpm > 200:
            bpm /= 2.0
        while bpm < 40:
            bpm *= 2.0
        beats = list(np.arange(0, duration, median_interval)) if HAS_NUMPY and median_interval > 0 else []
        return beats, round(bpm, 1)


# ═══════════════════════════════════════════════════════════════
#  KEY DETECTOR — Module-level profile constants
#  (Must be at module scope so list comprehensions in the class
#   body can see them — Python class scope doesn't leak into
#   comprehensions the way function scope does.)
# ═══════════════════════════════════════════════════════════════
_KD_PROFILE_MAJ = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
_KD_PROFILE_MIN = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
_KD_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

if HAS_NUMPY:
    _KD_ALL_PROFILES = np.vstack([
        np.roll(np.array(_KD_PROFILE_MAJ, dtype=np.float64), r) for r in range(12)
    ] + [
        np.roll(np.array(_KD_PROFILE_MIN, dtype=np.float64), r) for r in range(12)
    ])  # shape (24, 12)
    _KD_PROF_MEANS = _KD_ALL_PROFILES.mean(axis=1, keepdims=True)
    _KD_PROF_CENTERED = _KD_ALL_PROFILES - _KD_PROF_MEANS
    _KD_PROF_NORMS = np.sqrt(np.sum(_KD_PROF_CENTERED ** 2, axis=1))  # shape (24,)


class KeyDetector:
    """
    IMPROVEMENT: Vectorized Pearson correlation using NumPy matrix operations.
    Computes all 24 key profiles in a single matrix multiplication.
    """
    PROFILE_MAJ = _KD_PROFILE_MAJ
    PROFILE_MIN = _KD_PROFILE_MIN
    NAMES = _KD_NAMES

    @staticmethod
    def detect(notes) -> str:
        if not notes:
            return "C"
        pc = [0.0] * 12
        for n in notes:
            pc[n.note % 12] += n.duration * n.velocity
        total = sum(pc)
        if total == 0:
            return "C"
        pc_norm = [v / total for v in pc]

        if HAS_NUMPY:
            pc_arr = np.array(pc_norm, dtype=np.float64)
            pc_mean = pc_arr.mean()
            pc_centered = pc_arr - pc_mean
            pc_norm_val = np.sqrt(np.dot(pc_centered, pc_centered))

            if pc_norm_val < 1e-10:
                return "C"

            # Single matrix multiply computes all 24 correlations at once
            correlations = _KD_PROF_CENTERED @ pc_centered
            correlations /= (_KD_PROF_NORMS * pc_norm_val + 1e-10)

            best_idx = int(np.argmax(correlations))
            root = best_idx % 12
            mode = "major" if best_idx < 12 else "minor"
            return f"{_KD_NAMES[root]} {mode}"
        else:
            # Pure Python fallback
            best_key, best_corr = "C", -1.0
            for root in range(12):
                maj_rot = _KD_PROFILE_MAJ[root:] + _KD_PROFILE_MAJ[:root]
                min_rot = _KD_PROFILE_MIN[root:] + _KD_PROFILE_MIN[:root]
                for mode, profile in [("major", maj_rot), ("minor", min_rot)]:
                    corr = sum((a - sum(pc_norm)/12) * (b - sum(profile)/12)
                               for a, b in zip(pc_norm, profile))
                    norm = math.sqrt(
                        sum((a - sum(pc_norm)/12)**2 for a in pc_norm) *
                        sum((b - sum(profile)/12)**2 for b in profile)
                    )
                    corr = corr / norm if norm > 0 else 0
                    if corr > best_corr:
                        best_corr = corr
                        best_key = f"{_KD_NAMES[root]} {mode}"
            return best_key

# ═══════════════════════════════════════════════════════════════
#  SMALL HELPERS
# ═══════════════════════════════════════════════════════════════

_BLACK_NOTES = frozenset({1, 3, 6, 8, 10})
def is_black(note: int) -> bool: return note % 12 in _BLACK_NOTES

_NAMES_HELPER = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "A", "A#", "B"]
def note_name(n: int) -> str: return _NAMES_HELPER[n % 12] + str(n // 12 - 1)

def fmt_time(s: float) -> str:
    s = max(0, s); h = int(s) // 3600; m = (int(s) % 3600) // 60; sec = int(s) % 60
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"

def color_c(r, g, b, a=255) -> QColor: return QColor(r, g, b, a)
def color_hsv(h, s=1.0, v=1.0, a=255) -> QColor:
    r, g, b = colorsys.hsv_to_rgb(h % 1, min(1, max(0, s)), min(1, max(0, v)))
    return QColor(int(r * 255), int(g * 255), int(b * 255), int(a))

# Color cache for frequently-requested rainbow colors
_COLOR_CACHE: Dict[Tuple, QColor] = {}
def cached_color_hsv(h, s=1.0, v=1.0, a=255) -> QColor:
    key = (int(h * 1000), int(s * 100), int(v * 100), a)
    c = _COLOR_CACHE.get(key)
    if c is None:
        c = color_hsv(h, s, v, a)
        _COLOR_CACHE[key] = c
        if len(_COLOR_CACHE) > 4096:
            _COLOR_CACHE.clear()
    return c