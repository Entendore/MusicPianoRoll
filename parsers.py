"""
MIDI parsing, audio file parsing, directory scanning,
and miscellaneous helper functions.

ALGORITHM IMPROVEMENTS:
1. t2s (tick-to-seconds): Uses bisect + precomputed cumulative time array
   for O(log T) lookup instead of O(T) linear scan through tempo map.
2. _compute_peaks: Vectorized with numpy reshape/axis operations.
3. FFT computation: Vectorized batch processing using stride tricks
   and compute_binned_spectra_batch for entire audio at once.
"""

import os
import bisect
import subprocess
import logging

from data import (
    mido, MidiFile, HAS_MIDO,
    HAS_NUMPY, HAS_PYDUB, HAS_CUPY,
    AudioSegment, cp,
    NoteEvent, AudioData, MIDI_EXTS, AUDIO_EXTS, ALL_EXTS,
)
if HAS_NUMPY:
    from data import np

from kernels import compute_binned_spectrum
if HAS_NUMPY:
    from kernels import compute_binned_spectra_batch

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
#  MIDI PARSER
# ═══════════════════════════════════════════════════════════════
def parse_midi(filepath):
    if not HAS_MIDO:
        raise RuntimeError("mido is required for MIDI parsing. Install with: pip install mido")

    logger.info("Parsing MIDI file: %s", filepath)
    mid = MidiFile(filepath)
    tpb = mid.ticks_per_beat
    tmap = []

    # Extract tempo map
    for tr in mid.tracks:
        at = 0
        for msg in tr:
            at += msg.time
            if msg.type == "set_tempo":
                tmap.append((at, msg.tempo))
    tmap.sort()
    if not tmap:
        tmap = [(0, 500000)]

    # Precompute cumulative seconds for O(log T) tick→second
    tempo_ticks = [t[0] for t in tmap]
    tempo_values = [t[1] for t in tmap]
    cumulative_secs = [0.0] * len(tmap)
    for i in range(1, len(tmap)):
        cumulative_secs[i] = cumulative_secs[i - 1] + mido.tick2second(
            tempo_ticks[i] - tempo_ticks[i - 1], tpb, tempo_values[i - 1]
        )

    def t2s(tick):
        """Convert absolute tick to seconds. O(log T) via bisect."""
        idx = bisect.bisect_right(tempo_ticks, tick) - 1
        if idx < 0:
            idx = 0
        return cumulative_secs[idx] + mido.tick2second(tick - tempo_ticks[idx], tpb, tempo_values[idx])

    notes = []
    for ti, tr in enumerate(mid.tracks):
        at = 0
        active = {}
        for msg in tr:
            at += msg.time
            if msg.type == "note_on" and msg.velocity > 0:
                active[(msg.note, msg.channel)] = (at, msg.velocity)
            elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                k = (msg.note, msg.channel)
                if k in active:
                    st, vel = active.pop(k)
                    s, e = t2s(st), t2s(at)
                    dur = max(e - s, 0.02)
                    notes.append(NoteEvent(msg.note, s, dur, vel, msg.channel, ti))

    total = max((n.start_time + n.duration for n in notes), default=0) + 2.0

    info = {}
    for i, tr in enumerate(mid.tracks):
        nm = f"Track {i}"
        for msg in tr:
            if msg.type == "track_name" and msg.name:
                nm = msg.name
                break
        info[i] = nm

    logger.info("MIDI parsed successfully: %d notes, %.2fs duration, %d tracks", len(notes), total, len(info))
    return notes, total, info, tmap, tpb


# ═══════════════════════════════════════════════════════════════
#  AUDIO FILE PARSER
# ═══════════════════════════════════════════════════════════════
def parse_audio_file(filepath) -> AudioData:
    logger.info("Parsing audio file: %s", filepath)
    data = AudioData(path=filepath)
    ext = os.path.splitext(filepath)[1].lower()

    if ext == '.wav' and HAS_NUMPY:
        try:
            import wave as wm
            wf = wm.open(filepath, 'rb')
            data.channels = wf.getnchannels()
            data.sample_rate = wf.getframerate()
            nf = wf.getnframes()
            data.duration = nf / data.sample_rate
            samp_width = wf.getsampwidth()
            raw = wf.readframes(nf)
            wf.close()

            if samp_width == 2:
                s = np.frombuffer(raw, dtype=np.int16).astype(np.float64)
            elif samp_width == 4:
                s = np.frombuffer(raw, dtype=np.int32).astype(np.float64)
            elif samp_width == 1:
                s = np.frombuffer(raw, dtype=np.uint8).astype(np.float64) - 128.0
            else:
                s = np.frombuffer(raw, dtype=np.int16).astype(np.float64)

            if data.channels > 1:
                s = s.reshape(-1, data.channels).mean(axis=1)

            max_val = 2.0 ** (samp_width * 8 - 1) if samp_width >= 2 else 128.0
            sn = s / max_val
            data.samples = sn.tolist()

            _compute_peaks(data, sn)
            logger.info("WAV parsed successfully: %.2fs, %dHz", data.duration, data.sample_rate)

        except Exception as e:
            logger.error("WAV parsing error: %s", e, exc_info=True)

    elif HAS_PYDUB:
        try:
            seg = AudioSegment.from_file(filepath)
            data.channels = seg.channels
            data.sample_rate = seg.frame_rate
            data.duration = seg.duration_seconds

            s = np.array(seg.get_array_of_samples(), dtype=np.float64)
            if data.channels > 1:
                s = s.reshape(-1, data.channels).mean(axis=1)

            sn = s / (2 ** 15)
            data.samples = sn.tolist()

            _compute_peaks(data, sn)
            logger.info("Audio parsed via pydub: %.2fs, %dHz", data.duration, data.sample_rate)

        except Exception as e:
            logger.error("Audio parsing error (pydub): %s", e, exc_info=True)

    else:
        if ext in AUDIO_EXTS:
            logger.warning("Cannot parse %s: install numpy (for WAV) or pydub (for MP3/other) to enable audio loading", filepath)
        else:
            logger.warning("Cannot parse %s: no suitable decoder available", filepath)

    return data


def _compute_peaks(data: AudioData, samples, pps=30):
    """
    Vectorized peak computation using numpy reshape.
    """
    if not HAS_NUMPY or len(samples) == 0:
        return

    # Waveform peaks (vectorized)
    tp = max(1, int(data.duration * pps))
    cs = max(1, len(samples) // tp)
    n_full = len(samples) // cs

    if n_full > 0 and cs > 0:
        reshaped = samples[:n_full * cs].reshape(n_full, cs)
        max_vals = reshaped.max(axis=1)
        min_vals = reshaped.min(axis=1)
        data.waveform_peaks = list(zip(max_vals.tolist(), min_vals.tolist()))
        remainder_start = n_full * cs
        if remainder_start < len(samples):
            data.waveform_peaks.append(
                (float(samples[remainder_start:].max()), float(samples[remainder_start:].min()))
            )
        data.waveform_peaks = data.waveform_peaks[:tp]
    else:
        data.waveform_peaks = [(0.0, 0.0)] * tp

    # Spectrum (FFT) computation
    fft_size = 2048
    hop = fft_size // 2
    nb = 64

    # GPU Acceleration (CuPy)
    if HAS_CUPY:
        try:
            cps = cp.asarray(samples)
            n_frames = max(0, (len(samples) - fft_size) // hop + 1)
            if n_frames > 0:
                indices = cp.arange(fft_size)[cp.newaxis, :] + cp.arange(n_frames)[:, cp.newaxis] * hop
                frames = cps[indices]
                window = cp.hanning(fft_size)
                frames = frames * window
                spectra = cp.abs(cp.fft.rfft(frames, axis=1))
                # Convert back to numpy for batch binning
                sd = compute_binned_spectra_batch(cp.asnumpy(spectra), nb)
                data.spectrum_data = sd.tolist()
                return
        except Exception:
            pass

    # CPU (NumPy) — Vectorized batch processing
    if HAS_NUMPY:
        n_frames = max(0, (len(samples) - fft_size) // hop + 1)
        if n_frames > 0:
            indices = np.arange(fft_size)[np.newaxis, :] + np.arange(n_frames)[:, np.newaxis] * hop
            frames = samples[indices]
            window = np.hanning(fft_size)
            frames = frames * window
            spectra = np.abs(np.fft.rfft(frames, axis=1))

            binned = compute_binned_spectra_batch(spectra, nb)
            data.spectrum_data = binned.tolist()
            return

    # Fallback: per-frame Python loop
    sd = []
    pos = 0
    while pos + fft_size <= len(samples):
        ck = samples[pos:pos + fft_size]
        w = ck * np.hanning(fft_size)
        sp = np.abs(np.fft.rfft(w))
        sd.append(compute_binned_spectrum(sp, nb))
        pos += hop
    data.spectrum_data = sd


# ═══════════════════════════════════════════════════════════════
#  DIRECTORY SCANNER
# ═══════════════════════════════════════════════════════════════
def scan_directory(d, rec=True):
    f = []
    try:
        if rec:
            for r, _, fn in os.walk(d):
                for n in fn:
                    if os.path.splitext(n)[1].lower() in ALL_EXTS:
                        f.append(os.path.join(r, n))
        else:
            for n in os.listdir(d):
                fp = os.path.join(d, n)
                if os.path.isfile(fp) and os.path.splitext(n)[1].lower() in ALL_EXTS:
                    f.append(fp)
    except Exception:
        pass
    return sorted(f)


# ═══════════════════════════════════════════════════════════════
#  FFMPEG & CODEC HELPERS
# ═══════════════════════════════════════════════════════════════
def check_ffmpeg():
    try:
        return subprocess.run(
            ['ffmpeg', '-version'], capture_output=True, timeout=5
        ).returncode == 0
    except Exception:
        return False


def audio_codec_for(ext):
    if ext in ('.mp4', '.mov', '.m4v'):
        return 'aac'
    if ext in ('.webm', '.mkv'):
        return 'libvorbis'
    return 'aac'