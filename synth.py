import io, os, struct, tempfile, logging
from data import HAS_NUMPY
if HAS_NUMPY: from data import np
from kernels import synth_kernel
logger = logging.getLogger(__name__)
_temp_files = []

def cleanup_temps():
    for p in _temp_files:
        try: os.unlink(p)
        except: pass
    _temp_files.clear()

def synthesize_midi_audio(notes, duration):
    if not HAS_NUMPY: return None
    try:
        sr, total = 44100, int(44100 * duration) + 44100 * 2
        samples = np.zeros(total, dtype=np.float64)
        nn = np.array([n.note for n in notes], dtype=np.float64)
        st = np.array([n.start_time for n in notes], dtype=np.float64)
        du = np.array([n.duration for n in notes], dtype=np.float64)
        ve = np.array([n.velocity for n in notes], dtype=np.float64)
        synth_kernel(samples, sr, nn, st, du, ve)
        pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
        bio = io.BytesIO(); data = pcm.tobytes(); bits = 16
        bio.write(b'RIFF'); bio.write(struct.pack('<I', 36 + len(data))); bio.write(b'WAVEfmt ')
        bio.write(struct.pack('<IHHIIHH', 16, 1, 1, sr, sr * (bits//8), 1, bits))
        bio.write(b'data'); bio.write(struct.pack('<I', len(data))); bio.write(data)
        tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False); name = tmp.name
        tmp.write(bio.getvalue()); tmp.close(); _temp_files.append(name)
        return name
    except Exception as e:
        logger.error(f"Synth failed: {e}"); return None