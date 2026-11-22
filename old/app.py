#!/usr/bin/env python3
"""
🎹 Piano Roll Studio— Viral YouTube Video Engine
  ✦ Professional cinematic piano roll for viral YouTube videos & Shorts
  ✦ 10 stunning visual presets (Minimal, Neon, Rainbow, Fire, Ice, Galaxy, Synthwave, Candy, Aurora, Midnight)
  ✦ Beat-reactive backgrounds, particle explosions, note trails, screen shake
  ✦ Dynamic camera: auto-zoom, follow, cinematic sweeps
  ✦ Combo counter, note streaks, speed display overlay
  ✦ Audio spectrum overlay with circular & bar visualizers
  ✦ YouTube 1080p/4K + Shorts 9:16 export with FFmpeg
  ✦ Full Settings tab with all visual/audio/export controls
  ✦ Built-in Numba-accelerated MIDI synthesizer (No pydub required for playback)
  ✦ CuPy GPU spectrum processing
  ✦ Keyboard shortcuts: Space=play, ←→=seek, ↑↓=speed

Install: pip install PySide6 mido numpy numba
Optional: pip install cupy-cuda12x pydub
Requires: ffmpeg in PATH for video export
"""

import sys, os, time, math, colorsys, array, random, subprocess, threading, io, struct, tempfile
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Set, Optional
from pathlib import Path

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QSlider, QLabel, QComboBox, QFrame, QCheckBox,
    QMessageBox, QSizePolicy, QListWidget, QListWidgetItem, QGroupBox,
    QFileDialog, QProgressDialog, QSpinBox, QDialog, QFormLayout,
    QDialogButtonBox, QProgressBar, QAbstractItemView, QTabWidget,
    QDoubleSpinBox, QScrollArea, QToolButton, QStatusBar,
    QGridLayout, QSplitter, QLineEdit, QTextEdit,
)
from PySide6.QtCore import Qt, QTimer, Signal, QPoint, QRectF, QUrl, QThread, QSize
from PySide6.QtGui import (
    QPainter, QColor, QBrush, QPen, QFont, QLinearGradient,
    QRadialGradient, QPainterPath, QPixmap, QImage,
    QPalette, QFontMetrics, QAction, QIcon, QConicalGradient,
)

import mido
from mido import MidiFile

try:
    from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
    HAS_QTMULTIMEDIA = True
except ImportError:
    HAS_QTMULTIMEDIA = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    from pydub import AudioSegment
    HAS_PYDUB = True
except ImportError:
    HAS_PYDUB = False

try:
    import numba
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

# ═══════════════════════════════════════════════════════════════
#  NUMBA ACCELERATED KERNELS
# ═══════════════════════════════════════════════════════════════
if HAS_NUMBA:
    @njit(fastmath=True, cache=True)
    def _numba_synth_kernel(samples, sr, note_nums, start_times, durations, velocities):
        for i in range(len(note_nums)):
            freq = 440.0 * (2.0 ** ((note_nums[i] - 69) / 12.0))
            start = int(start_times[i] * sr)
            length = int(durations[i] * sr)
            if length <= 0: continue
            attack = min(int(sr * 0.008), length)
            decay = min(int(sr * 0.04), length)
            release = min(int(sr * 0.08), length)
            vel_factor = (velocities[i] / 127.0) * 0.25
            for j in range(length):
                idx = start + j
                if idx >= len(samples): break
                t = j / sr
                wave = (math.sin(2 * math.pi * freq * t) * 0.50
                      + math.sin(2 * math.pi * freq * 2 * t) * 0.20
                      + math.sin(2 * math.pi * freq * 3 * t) * 0.10
                      + math.sin(2 * math.pi * freq * 4 * t) * 0.05)
                env = 1.0
                if j < attack: env = j / max(attack, 1)
                elif j < attack + decay: env = 1.0 - 0.3 * ((j - attack) / max(decay, 1))
                elif j > length - release: env = (length - j) / max(release, 1)
                samples[idx] += wave * vel_factor * env

    @njit(fastmath=True, cache=True)
    def _numba_compute_binned_spectrum(spectrum_arr, n_bins):
        bin_size = max(1, len(spectrum_arr) // n_bins)
        binned = np.empty(n_bins, dtype=np.float64)
        for b in range(n_bins):
            s = b * bin_size; e = min(s + bin_size, len(spectrum_arr))
            if s < len(spectrum_arr) and e > s:
                val = 0.0
                for i in range(s, e): val += spectrum_arr[i]
                binned[b] = val / (e - s)
            else: binned[b] = 0.0
        mx = 0.0
        for b in range(n_bins):
            if binned[b] > mx: mx = binned[b]
        if mx > 0.0:
            for b in range(n_bins): binned[b] /= mx
        return binned
else:
    def _numba_synth_kernel(samples, sr, note_nums, start_times, durations, velocities):
        for i in range(len(note_nums)):
            freq = 440.0 * (2.0 ** ((note_nums[i] - 69) / 12.0))
            start = int(start_times[i] * sr); length = int(durations[i] * sr)
            if length <= 0: continue
            t = np.arange(length) / sr
            wave = (np.sin(2*np.pi*freq*t)*0.50 + np.sin(2*np.pi*freq*2*t)*0.20
                  + np.sin(2*np.pi*freq*3*t)*0.10 + np.sin(2*np.pi*freq*4*t)*0.05)
            wave *= (velocities[i] / 127.0) * 0.25
            attack = min(int(sr*0.008), length); decay = min(int(sr*0.04), length)
            release = min(int(sr*0.08), length)
            if attack > 0: wave[:attack] *= np.linspace(0, 1, attack)
            if decay > 0: wave[attack:attack+decay] *= np.linspace(1, 0.7, decay)
            if release > 0: wave[-release:] *= np.linspace(1, 0, release)
            end = min(start+length, len(samples))
            if start < end: samples[start:end] += wave[:end-start]

    def _numba_compute_binned_spectrum(spectrum_arr, n_bins):
        if len(spectrum_arr) == 0 or n_bins <= 0: return [0.0] * max(n_bins, 0)
        bin_size = max(1, len(spectrum_arr) // n_bins)
        binned = []
        for b in range(n_bins):
            s = b * bin_size; e = min(s + bin_size, len(spectrum_arr))
            binned.append(float(np.mean(spectrum_arr[s:e])) if s < len(spectrum_arr) and e > s else 0.0)
        mx = max(binned) if binned else 0.0
        return [v / mx for v in binned] if mx > 0 else binned

# ═══════════════════════════════════════════════════════════════
#  SOUND & SYNTH (Fixed: No longer requires pydub for playback)
# ═══════════════════════════════════════════════════════════════
_temp_files: List[str] = []

def _register_temp(p):
    if p and os.path.exists(p): _temp_files.append(p)

def _cleanup_temps():
    for p in _temp_files:
        try: os.unlink(p)
        except: pass
    _temp_files.clear()

def _wav_bytes(samples_int16, sr=44100, ch=1):
    bio = io.BytesIO(); data = samples_int16.tobytes(); bits = 16
    bio.write(b'RIFF'); bio.write(struct.pack('<I', 36+len(data))); bio.write(b'WAVEfmt ')
    bio.write(struct.pack('<IHHIIHH', 16, 1, ch, sr, sr*ch*(bits//8), ch*(bits//8), bits))
    bio.write(b'data'); bio.write(struct.pack('<I', len(data))); bio.write(data)
    return bio.getvalue()

def synthesize_midi_audio(notes, duration):
    if not HAS_NUMPY: return None
    try:
        sr = 44100; total = int(sr*duration)+sr*2
        samples = np.zeros(total, dtype=np.float64)
        nn = np.array([n.note for n in notes], dtype=np.float64)
        st = np.array([n.start_time for n in notes], dtype=np.float64)
        du = np.array([n.duration for n in notes], dtype=np.float64)
        ve = np.array([n.velocity for n in notes], dtype=np.float64)
        _numba_synth_kernel(samples, sr, nn, st, du, ve)
        samples = np.clip(samples, -1.0, 1.0)
        pcm = (samples*32767).astype(np.int16)
        
        # Write WAV directly using built-in _wav_bytes
        wav_data = _wav_bytes(pcm, sr, 1)
        tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        name = tmp.name
        tmp.write(wav_data)
        tmp.close()
        _register_temp(name)
        return name
    except Exception as e:
        print(f"Synth error: {e}"); return None

# ═══════════════════════════════════════════════════════════════
#  PROFESSIONAL VISUAL PRESETS (10 presets)
# ═══════════════════════════════════════════════════════════════
def _c(r,g,b,a=255): return QColor(r,g,b,a)
def _hsv(h,s=1.0,v=1.0,a=255):
    r,g,b = colorsys.hsv_to_rgb(h%1,min(1,max(0,s)),min(1,max(0,v)))
    return QColor(int(r*255),int(g*255),int(b*255),int(a))

@dataclass
class VisualPreset:
    name: str
    bg1: QColor; bg2: QColor
    lane_white: QColor; lane_black: QColor; grid_line: QColor; c_line: QColor
    note_colors: list
    note_glow: float; note_radius: float; note_border: bool
    trail_length: float; trail_opacity: float
    particle_count: int; particle_speed: float; particle_size: float
    kb_wt: QColor; kb_wb: QColor; kb_bt: QColor; kb_bb: QColor; kb_border: QColor; kb_active_glow: float
    show_spectrum: bool; spectrum_style: str
    show_combo: bool; show_streaks: bool; show_speed: bool
    screen_shake: float; bg_pulse: float; camera_zoom_range: tuple
    is_rainbow: bool = False; star_count: int = 0; star_color: QColor = QColor(255,255,255,15)
    reflection: bool = False; depth_3d: bool = False; vignette: float = 0.0
    ui_bg: str = "#0d0d1a"; ui_accent: str = "#1a1a3e"; ui_text: str = "#e0e0e8"; ui_border: str = "#2a2a50"
    bloom: float = 0.0; chromatic_aberration: float = 0.0; scanlines: float = 0.0
    film_grain: float = 0.0; light_streak: bool = False; pulse_on_beat: bool = False
    note_splash: bool = False; floating_labels: bool = False; hdr_tonemap: float = 0.0

PRESETS = {}

PRESETS["Minimal"] = VisualPreset(
    name="Minimal", bg1=_c(18,18,28), bg2=_c(12,12,22),
    lane_white=_c(28,28,42), lane_black=_c(16,16,26), grid_line=_c(38,38,55), c_line=_c(50,50,70),
    note_colors=[_c(100,160,255),_c(255,100,120),_c(100,230,160),_c(255,200,80),_c(180,120,255),_c(255,160,80)],
    note_glow=0.4, note_radius=4, note_border=False,
    trail_length=0, trail_opacity=0,
    particle_count=0, particle_speed=0, particle_size=0,
    kb_wt=_c(245,245,250), kb_wb=_c(220,220,230), kb_bt=_c(55,55,65), kb_bb=_c(22,22,30), kb_border=_c(160,160,175), kb_active_glow=0.6,
    show_spectrum=False, spectrum_style='bars', show_combo=False, show_streaks=False, show_speed=False,
    screen_shake=0, bg_pulse=0, camera_zoom_range=(1.0,1.0),
)

PRESETS["Neon"] = VisualPreset(
    name="Neon", bg1=_c(2,2,10), bg2=_c(0,0,5),
    lane_white=_c(8,8,22), lane_black=_c(3,3,10), grid_line=_c(15,15,35), c_line=_c(25,25,55),
    note_colors=[_c(0,255,100),_c(255,0,110),_c(0,180,255),_c(255,255,0),_c(255,0,255),_c(0,255,255)],
    note_glow=2.5, note_radius=3, note_border=True,
    trail_length=40, trail_opacity=0.35,
    particle_count=12, particle_speed=80, particle_size=4,
    kb_wt=_c(15,15,30), kb_wb=_c(8,8,18), kb_bt=_c(5,5,12), kb_bb=_c(2,2,6), kb_border=_c(0,100,50), kb_active_glow=2.0,
    show_spectrum=True, spectrum_style='bars', show_combo=True, show_streaks=True, show_speed=True,
    screen_shake=2.0, bg_pulse=0.3, camera_zoom_range=(1.0,1.05),
    star_count=60, star_color=_c(0,255,100,12), vignette=0.4,
    bloom=0.6, chromatic_aberration=0.0, scanlines=0.05, film_grain=0.02,
    light_streak=True, pulse_on_beat=True, note_splash=True, floating_labels=True, hdr_tonemap=0.3,
)

PRESETS["Rainbow"] = VisualPreset(
    name="Rainbow", bg1=_c(8,5,18), bg2=_c(4,2,12),
    lane_white=_c(18,12,32), lane_black=_c(10,6,20), grid_line=_c(30,20,50), c_line=_c(50,35,75),
    note_colors='rainbow',
    note_glow=2.0, note_radius=5, note_border=True,
    trail_length=60, trail_opacity=0.4,
    particle_count=10, particle_speed=60, particle_size=3.5,
    kb_wt=_c(35,25,55), kb_wb=_c(22,15,40), kb_bt=_c(45,25,65), kb_bb=_c(18,10,30), kb_border=_c(90,60,140), kb_active_glow=1.8,
    show_spectrum=True, spectrum_style='circular', show_combo=True, show_streaks=True, show_speed=True,
    screen_shake=3.0, bg_pulse=0.5, camera_zoom_range=(0.98,1.06),
    is_rainbow=True, star_count=100, star_color=_c(200,100,255,18), reflection=True, vignette=0.3,
    bloom=0.5, chromatic_aberration=0.3, scanlines=0.0, film_grain=0.01,
    light_streak=True, pulse_on_beat=True, note_splash=True, floating_labels=True, hdr_tonemap=0.4,
)

PRESETS["Fire"] = VisualPreset(
    name="Fire", bg1=_c(25,5,0), bg2=_c(15,2,0),
    lane_white=_c(35,10,5), lane_black=_c(20,5,2), grid_line=_c(50,15,5), c_line=_c(70,25,10),
    note_colors=[_c(255,60,0),_c(255,140,0),_c(255,200,30),_c(255,80,20),_c(255,180,50),_c(200,40,0)],
    note_glow=2.2, note_radius=4, note_border=True,
    trail_length=50, trail_opacity=0.5,
    particle_count=16, particle_speed=100, particle_size=5,
    kb_wt=_c(60,25,10), kb_wb=_c(40,15,5), kb_bt=_c(30,10,5), kb_bb=_c(15,3,0), kb_border=_c(140,50,15), kb_active_glow=2.0,
    show_spectrum=True, spectrum_style='bars', show_combo=True, show_streaks=True, show_speed=True,
    screen_shake=4.0, bg_pulse=0.6, camera_zoom_range=(0.97,1.07),
    star_count=40, star_color=_c(255,100,0,15), vignette=0.5,
    bloom=0.7, chromatic_aberration=0.0, scanlines=0.0, film_grain=0.03,
    light_streak=True, pulse_on_beat=True, note_splash=True, floating_labels=False, hdr_tonemap=0.2,
)

PRESETS["Ice"] = VisualPreset(
    name="Ice", bg1=_c(5,10,30), bg2=_c(2,5,20),
    lane_white=_c(10,18,45), lane_black=_c(5,10,28), grid_line=_c(15,25,60), c_line=_c(25,40,80),
    note_colors=[_c(100,200,255),_c(150,230,255),_c(60,170,240),_c(200,240,255),_c(80,190,250),_c(180,235,255)],
    note_glow=1.8, note_radius=4, note_border=True,
    trail_length=45, trail_opacity=0.35,
    particle_count=8, particle_speed=40, particle_size=3,
    kb_wt=_c(180,215,240), kb_wb=_c(140,190,230), kb_bt=_c(15,30,60), kb_bb=_c(8,18,45), kb_border=_c(60,120,180), kb_active_glow=1.5,
    show_spectrum=True, spectrum_style='wave', show_combo=True, show_streaks=False, show_speed=True,
    screen_shake=1.0, bg_pulse=0.2, camera_zoom_range=(1.0,1.03),
    star_count=120, star_color=_c(150,200,255,20), vignette=0.3,
    bloom=0.4, chromatic_aberration=0.15, scanlines=0.0, film_grain=0.01,
    light_streak=True, pulse_on_beat=False, note_splash=True, floating_labels=True, hdr_tonemap=0.5,
)

PRESETS["Galaxy"] = VisualPreset(
    name="Galaxy", bg1=_c(5,2,15), bg2=_c(2,0,8),
    lane_white=_c(12,6,28), lane_black=_c(6,3,16), grid_line=_c(20,10,40), c_line=_c(35,18,60),
    note_colors=[_c(180,100,255),_c(100,150,255),_c(255,120,200),_c(80,200,255),_c(220,80,255),_c(140,180,255)],
    note_glow=2.0, note_radius=5, note_border=True,
    trail_length=70, trail_opacity=0.45,
    particle_count=14, particle_speed=50, particle_size=3,
    kb_wt=_c(30,18,55), kb_wb=_c(18,10,40), kb_bt=_c(20,10,45), kb_bb=_c(8,4,22), kb_border=_c(80,50,130), kb_active_glow=2.0,
    show_spectrum=True, spectrum_style='circular', show_combo=True, show_streaks=True, show_speed=True,
    screen_shake=2.0, bg_pulse=0.4, camera_zoom_range=(0.98,1.05),
    is_rainbow=False, star_count=200, star_color=_c(180,120,255,12), reflection=True, vignette=0.5,
    bloom=0.6, chromatic_aberration=0.2, scanlines=0.0, film_grain=0.015,
    light_streak=True, pulse_on_beat=True, note_splash=True, floating_labels=True, hdr_tonemap=0.35,
)

PRESETS["Synthwave"] = VisualPreset(
    name="Synthwave", bg1=_c(10,0,20), bg2=_c(5,0,12),
    lane_white=_c(18,2,35), lane_black=_c(10,0,22), grid_line=_c(30,5,55), c_line=_c(50,10,80),
    note_colors=[_c(255,0,150),_c(0,255,200),_c(255,100,0),_c(0,200,255),_c(255,0,80),_c(200,0,255)],
    note_glow=2.5, note_radius=4, note_border=True,
    trail_length=55, trail_opacity=0.5,
    particle_count=10, particle_speed=70, particle_size=4,
    kb_wt=_c(25,5,45), kb_wb=_c(15,2,30), kb_bt=_c(12,2,28), kb_bb=_c(5,0,15), kb_border=_c(150,0,100), kb_active_glow=2.2,
    show_spectrum=True, spectrum_style='bars', show_combo=True, show_streaks=True, show_speed=True,
    screen_shake=3.0, bg_pulse=0.5, camera_zoom_range=(0.97,1.06),
    star_count=80, star_color=_c(255,0,150,10), depth_3d=True, vignette=0.4,
    bloom=0.8, chromatic_aberration=0.4, scanlines=0.08, film_grain=0.02,
    light_streak=True, pulse_on_beat=True, note_splash=True, floating_labels=True, hdr_tonemap=0.2,
)

PRESETS["Candy"] = VisualPreset(
    name="Candy", bg1=_c(25,10,25), bg2=_c(15,5,18),
    lane_white=_c(38,15,38), lane_black=_c(22,8,22), grid_line=_c(55,22,55), c_line=_c(75,35,75),
    note_colors=[_c(255,105,180),_c(255,182,193),_c(255,20,147),_c(255,160,200),_c(220,80,160),_c(255,130,180)],
    note_glow=1.5, note_radius=6, note_border=True,
    trail_length=35, trail_opacity=0.3,
    particle_count=8, particle_speed=45, particle_size=3.5,
    kb_wt=_c(255,220,235), kb_wb=_c(240,190,210), kb_bt=_c(80,25,55), kb_bb=_c(40,10,30), kb_border=_c(200,100,150), kb_active_glow=1.5,
    show_spectrum=True, spectrum_style='wave', show_combo=True, show_streaks=True, show_speed=False,
    screen_shake=1.5, bg_pulse=0.3, camera_zoom_range=(1.0,1.04),
    star_count=50, star_color=_c(255,150,200,15), vignette=0.2,
    bloom=0.3, chromatic_aberration=0.0, scanlines=0.0, film_grain=0.0,
    light_streak=False, pulse_on_beat=True, note_splash=True, floating_labels=True, hdr_tonemap=0.3,
)

PRESETS["Aurora"] = VisualPreset(
    name="Aurora", bg1=_c(2,8,18), bg2=_c(0,3,10),
    lane_white=_c(5,15,30), lane_black=_c(2,8,18), grid_line=_c(10,25,50), c_line=_c(18,40,70),
    note_colors=[_c(0,255,150),_c(50,200,255),_c(150,100,255),_c(0,255,200),_c(100,255,180),_c(80,180,255)],
    note_glow=2.0, note_radius=5, note_border=True,
    trail_length=80, trail_opacity=0.5,
    particle_count=15, particle_speed=30, particle_size=2.5,
    kb_wt=_c(20,50,60), kb_wb=_c(10,35,45), kb_bt=_c(8,25,40), kb_bb=_c(3,12,22), kb_border=_c(0,150,100), kb_active_glow=2.0,
    show_spectrum=True, spectrum_style='wave', show_combo=True, show_streaks=True, show_speed=True,
    screen_shake=1.5, bg_pulse=0.4, camera_zoom_range=(0.99,1.04),
    is_rainbow=False, star_count=250, star_color=_c(100,255,180,10), reflection=True, vignette=0.35,
    bloom=0.7, chromatic_aberration=0.25, scanlines=0.0, film_grain=0.01,
    light_streak=True, pulse_on_beat=True, note_splash=True, floating_labels=True, hdr_tonemap=0.6,
)

PRESETS["Midnight"] = VisualPreset(
    name="Midnight", bg1=_c(8,8,18), bg2=_c(3,3,10),
    lane_white=_c(14,14,28), lane_black=_c(8,8,16), grid_line=_c(22,22,40), c_line=_c(35,35,60),
    note_colors=[_c(200,180,255),_c(255,220,150),_c(150,220,255),_c(255,180,200),_c(180,255,200),_c(255,200,120)],
    note_glow=1.6, note_radius=4, note_border=True,
    trail_length=45, trail_opacity=0.3,
    particle_count=6, particle_speed=25, particle_size=2,
    kb_wt=_c(40,40,60), kb_wb=_c(28,28,45), kb_bt=_c(20,20,35), kb_bb=_c(10,10,20), kb_border=_c(80,80,120), kb_active_glow=1.4,
    show_spectrum=True, spectrum_style='circular', show_combo=True, show_streaks=False, show_speed=True,
    screen_shake=0.8, bg_pulse=0.15, camera_zoom_range=(1.0,1.02),
    is_rainbow=False, star_count=300, star_color=_c(200,200,255,8), vignette=0.6,
    bloom=0.5, chromatic_aberration=0.1, scanlines=0.0, film_grain=0.005,
    light_streak=True, pulse_on_beat=False, note_splash=False, floating_labels=True, hdr_tonemap=0.5,
)

# ═══════════════════════════════════════════════════════════════
#  PARTICLE, STAR, EFFECTS ENGINE
# ═══════════════════════════════════════════════════════════════
@dataclass
class Particle:
    x: float; y: float; vx: float; vy: float
    life: float; max_life: float; color: QColor; size: float
    trail: list = field(default_factory=list)
    ptype: str = 'spark'

class ParticleEngine:
    def __init__(self, max_p=2000):
        self.particles: List[Particle] = []
        self.max_p = max_p

    def emit(self, x, y, color, count=6, speed=50, life=0.7, size=3, has_trail=False, ptype='spark'):
        for _ in range(count):
            if len(self.particles) >= self.max_p: break
            a = random.uniform(0, 2*math.pi)
            s = random.uniform(speed*0.5, speed*1.2) if ptype == 'splash' else random.uniform(speed*0.2, speed)
            ml = random.uniform(life*0.5, life*1.2) if ptype == 'splash' else random.uniform(life*0.4, life)
            sz = random.uniform(size*0.4, size*1.6)
            c = QColor(color); c.setAlpha(random.randint(140,255))
            self.particles.append(Particle(x, y, math.cos(a)*s, -abs(math.sin(a)*s)*0.8,
                                           ml, ml, c, sz, [] if has_trail else None, ptype))

    def update(self, dt):
        alive = []
        for p in self.particles:
            p.life -= dt
            if p.life <= 0: continue
            if p.trail is not None:
                p.trail.append((p.x, p.y, p.life/p.max_life))
                if len(p.trail) > 16: p.trail.pop(0)
            p.x += p.vx*dt; p.y += p.vy*dt
            p.vy += 120*dt; p.vx *= 0.97
            alive.append(p)
        self.particles = alive

    def draw(self, painter, ox=0, oy=0):
        for p in self.particles:
            ratio = max(0, p.life/p.max_life)
            alpha = int(255*ratio)
            c = QColor(p.color); c.setAlpha(max(0,min(255,alpha)))
            sz = p.size * ratio
            if p.trail and len(p.trail) > 1:
                tc = QColor(p.color); tc.setAlpha(max(0,min(255,int(alpha*0.4))))
                painter.setPen(QPen(tc, max(1, sz*0.5)))
                for i in range(1, len(p.trail)):
                    x1, y1, _ = p.trail[i-1]; x2, y2, _ = p.trail[i]
                    painter.drawLine(int(x1-ox), int(y1-oy), int(x2-ox), int(y2-oy))
            painter.setPen(Qt.NoPen); painter.setBrush(QBrush(c))
            if p.ptype == 'splash' and ratio > 0.5:
                painter.drawEllipse(QPoint(int(p.x-ox), int(p.y-oy)), int(sz*1.5), int(sz))
            else:
                painter.drawEllipse(QPoint(int(p.x-ox), int(p.y-oy)), int(sz), int(sz))


class StarField:
    def __init__(self, count=100):
        self.stars = [(random.uniform(0,1), random.uniform(0,1),
                       random.uniform(0.2,1.0), random.uniform(0.5,2.5),
                       random.uniform(0,6.28)) for _ in range(count)]

    def draw(self, painter, w, h, t, color, is_rainbow=False):
        painter.setPen(Qt.NoPen)
        for sx, sy, br, sz, phase in self.stars:
            if is_rainbow:
                c = _hsv((sx+t*0.04)%1, 0.5, br*0.4, int(br*40))
            else:
                alpha = int(br * color.alpha() * (0.4 + 0.6*math.sin(t*0.7+phase)))
                c = QColor(color); c.setAlpha(max(0, min(255, alpha)))
            dx = math.sin(t*0.08+phase)*3; dy = math.cos(t*0.06+phase)*2
            painter.setBrush(QBrush(c))
            painter.drawEllipse(int(sx*w+dx), int(sy*h+dy), int(sz), int(sz))


class RippleSystem:
    def __init__(self): self.ripples = []
    def emit(self, x, y, color, mr=35, life=0.45):
        self.ripples.append([x, y, 0.0, mr, life, life, QColor(color)])
    def update(self, dt):
        alive = []
        for r in self.ripples:
            r[4] -= dt
            if r[4] > 0: r[2] = r[3] * (1 - r[4]/r[5]); alive.append(r)
        self.ripples = alive
    def draw(self, painter):
        for r in self.ripples:
            alpha = max(0, int(100*(r[4]/r[5])))
            c = QColor(r[6]); c.setAlpha(alpha)
            painter.setPen(QPen(c, 2)); painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(QPoint(int(r[0]),int(r[1])), int(r[2]), int(r[2]*0.35))


class NoteSplashSystem:
    def __init__(self): self.splashes = []
    def emit(self, x, y, text, color, size=14):
        self.splashes.append([x, y, text, QColor(color), size, 1.0, 0.0])
    def update(self, dt):
        alive = []
        for s in self.splashes:
            s[5] -= dt * 1.2; s[6] += dt; s[1] -= dt * 60
            if s[5] > 0: alive.append(s)
        self.splashes = alive
    def draw(self, painter):
        for s in self.splashes:
            x, y, text, color, size, alpha, _ = s
            a = max(0, min(255, int(alpha * 255)))
            c = QColor(color); c.setAlpha(a)
            painter.setPen(c)
            scale = 1.0 + (1.0 - alpha) * 0.3
            painter.setFont(QFont("SF Pro", int(size * scale), QFont.Bold))
            painter.drawText(int(x), int(y), text)

# ═══════════════════════════════════════════════════════════════
#  DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════
@dataclass
class NoteEvent:
    note: int; start_time: float; duration: float; velocity: int; channel: int; track: int

@dataclass
class AudioData:
    samples: list = field(default_factory=list); sample_rate: int = 44100
    duration: float = 0.0; channels: int = 1
    waveform_peaks: list = field(default_factory=list)
    spectrum_data: list = field(default_factory=list); path: str = ""

@dataclass
class PlaylistEntry:
    midi_path: Optional[str] = None; audio_path: Optional[str] = None; name: str = ""
    notes: List[NoteEvent] = field(default_factory=list)
    audio_data: Optional[AudioData] = None; duration: float = 0.0
    track_info: Dict[int, str] = field(default_factory=dict)
    tempo_map: list = field(default_factory=list); ticks_per_beat: int = 480
    note_count: int = 0; file_size: int = 0; file_type: str = "midi"

@dataclass
class ExportPreset:
    name: str; width: int; height: int; fps: int; codec: str
    crf: int; preset: str; description: str

YOUTUBE_PRESETS = [
    ExportPreset("YouTube 1080p", 1920, 1080, 60, "libx264", 18, "medium", "Full HD 60fps"),
    ExportPreset("YouTube 1080p HQ", 1920, 1080, 60, "libx265", 22, "slow", "H.265 Quality"),
    ExportPreset("YouTube 4K", 3840, 2160, 60, "libx264", 18, "medium", "4K 60fps"),
    ExportPreset("YouTube 4K HQ", 3840, 2160, 60, "libx265", 22, "slow", "4K H.265"),
    ExportPreset("Shorts 1080×1920", 1080, 1920, 60, "libx264", 18, "medium", "Vertical 9:16"),
    ExportPreset("Shorts 1080×1920 HQ", 1080, 1920, 60, "libx265", 22, "slow", "Vertical H.265"),
    ExportPreset("Shorts 608×1080", 608, 1080, 60, "libx264", 20, "fast", "Shorts Fast"),
    ExportPreset("Custom", 1920, 1080, 60, "libx264", 18, "medium", "Custom"),
]

# ═══════════════════════════════════════════════════════════════
#  PARSERS & HELPERS
# ═══════════════════════════════════════════════════════════════
def is_black(n): return n%12 in {1,3,6,8,10}
_NAMES = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
def note_name(n): return _NAMES[n%12]+str(n//12-1)
def fmt_time(s):
    s = max(0,s); h = int(s)//3600; m = (int(s)%3600)//60; sec = int(s)%60
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"

MIDI_EXTS = {'.mid','.midi'}; AUDIO_EXTS = {'.wav','.mp3'}
ALL_EXTS = MIDI_EXTS | AUDIO_EXTS
FILE_FILTER = "Music (*.mid *.midi *.wav *.mp3);;MIDI (*.mid *.midi);;Audio (*.wav *.mp3);;All (*)"

def generate_beat_times(tempo_map, tpb, total_dur):
    if not tempo_map: tempo_map = [(0, 500000)]
    beats = []; ct = 0; ctmp = tempo_map[0][1]; ct_sec = 0.0; tidx = 0; bt = 0
    while True:
        bs = ct_sec + mido.tick2second(bt-ct, tpb, ctmp)
        if bs > total_dur: break
        beats.append(bs); bt += tpb
        while tidx+1 < len(tempo_map) and tempo_map[tidx+1][0] <= bt:
            nmt = tempo_map[tidx+1][0]; ct_sec += mido.tick2second(nmt-ct, tpb, ctmp)
            ct = nmt; tidx += 1; ctmp = tempo_map[tidx][1]
    return beats

def parse_midi(filepath):
    mid = MidiFile(filepath); tpb = mid.ticks_per_beat; tmap = []
    for tr in mid.tracks:
        at = 0
        for msg in tr:
            at += msg.time
            if msg.type == "set_tempo": tmap.append((at, msg.tempo))
    tmap.sort()
    if not tmap: tmap = [(0, 500000)]
    def t2s(tick):
        ct, ctmp, cs = 0, tmap[0][1], 0.0
        for mt, mtp in tmap:
            if mt >= tick: break
            cs += mido.tick2second(mt-ct, tpb, ctmp); ct, ctmp = mt, mtp
        return cs + mido.tick2second(tick-ct, tpb, ctmp)
    notes = []
    for ti, tr in enumerate(mid.tracks):
        at = 0; active = {}
        for msg in tr:
            at += msg.time
            if msg.type == "note_on" and msg.velocity > 0:
                active[(msg.note, msg.channel)] = (at, msg.velocity)
            elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                k = (msg.note, msg.channel)
                if k in active:
                    st, vel = active.pop(k)
                    s, e = t2s(st), t2s(at)
                    notes.append(NoteEvent(msg.note, s, max(e-s, 0.02), vel, msg.channel, ti))
    total = max((n.start_time+n.duration for n in notes), default=0) + 2.0
    info = {}
    for i, tr in enumerate(mid.tracks):
        nm = f"Track {i}"
        for msg in tr:
            if msg.type == "track_name" and msg.name: nm = msg.name; break
        info[i] = nm
    return notes, total, info, tmap, tpb

def parse_audio_file(filepath) -> AudioData:
    data = AudioData(path=filepath); ext = os.path.splitext(filepath)[1].lower()
    if ext == '.wav' and HAS_NUMPY:
        try:
            import wave as wm; wf = wm.open(filepath,'rb')
            data.channels = wf.getnchannels(); data.sample_rate = wf.getframerate()
            nf = wf.getnframes(); data.duration = nf/data.sample_rate
            raw = wf.readframes(nf); wf.close()
            s = np.frombuffer(raw, dtype=np.int16).astype(np.float64)
            if data.channels > 1: s = s.reshape(-1, data.channels).mean(axis=1)
            sn = s/32768.0; data.samples = sn.tolist(); _compute_peaks(data, sn)
        except Exception as e: print(f"WAV error: {e}")
    elif HAS_PYDUB:
        try:
            seg = AudioSegment.from_file(filepath); data.channels = seg.channels
            data.sample_rate = seg.frame_rate; data.duration = seg.duration_seconds
            s = np.array(seg.get_array_of_samples(), dtype=np.float64)
            if data.channels > 1: s = s.reshape(-1, data.channels).mean(axis=1)
            sn = s/(2**15); data.samples = sn.tolist(); _compute_peaks(data, sn)
        except Exception as e: print(f"Audio error: {e}")
    return data

def _compute_peaks(data, samples, pps=30):
    if len(samples) == 0: return
    tp = max(1, int(data.duration*pps)); cs = max(1, len(samples)//tp); peaks = []
    for i in range(tp):
        s = i*cs; e = min(s+cs, len(samples)); ch = samples[s:e] if s < len(samples) else []
        peaks.append((float(np.max(ch)), float(np.min(ch))) if len(ch) > 0 else (0.0, 0.0))
    data.waveform_peaks = peaks
    fft=2048; hop=fft//2; nb=64; sd=[]; pos=0
    if HAS_CUPY:
        try:
            cps = cp.asarray(samples)
            while pos+fft <= len(samples):
                ck = cps[pos:pos+fft]; w = ck*cp.hanning(fft); sp = cp.abs(cp.fft.rfft(w))
                sd.append(_numba_compute_binned_spectrum(cp.asnumpy(sp), nb)); pos += hop
            data.spectrum_data = sd; return
        except: pass
    while pos+fft <= len(samples):
        ck = samples[pos:pos+fft]; w = ck*np.hanning(fft); sp = np.abs(np.fft.rfft(w))
        sd.append(_numba_compute_binned_spectrum(sp, nb)); pos += hop
    data.spectrum_data = sd

def scan_directory(d, rec=True):
    f = []
    try:
        if rec:
            for r, _, fn in os.walk(d):
                for n in fn:
                    if os.path.splitext(n)[1].lower() in ALL_EXTS: f.append(os.path.join(r, n))
        else:
            for n in os.listdir(d):
                fp = os.path.join(d, n)
                if os.path.isfile(fp) and os.path.splitext(n)[1].lower() in ALL_EXTS: f.append(fp)
    except: pass
    return sorted(f)

def check_ffmpeg():
    try: return subprocess.run(['ffmpeg','-version'], capture_output=True, timeout=5).returncode==0
    except: return False

def audio_codec_for(ext):
    if ext in ('.mp4','.mov','.m4v'): return 'aac'
    if ext in ('.webm','.mkv'): return 'libvorbis'
    return 'aac'

# ═══════════════════════════════════════════════════════════════
#  SETTINGS DIALOG
# ═══════════════════════════════════════════════════════════════
class SettingsDialog(QDialog):
    settings_changed = Signal(dict)

    def __init__(self, parent=None, current_preset_name="Neon"):
        super().__init__(parent)
        self.setWindowTitle("⚙ Settings")
        self.setMinimumSize(680, 620)
        self.setStyleSheet(self._stylesheet())
        self._building = True
        self._init_ui(current_preset_name)
        self._building = False

    def _stylesheet(self):
        return """
        QDialog { background: #0f0f1e; color: #e0e0e8; }
        QTabWidget::pane { border: 1px solid #2a2a50; background: #12122a; border-radius: 8px; }
        QTabBar::tab { background: #16163a; color: #a0a0c0; padding: 10px 22px; margin: 2px 4px;
                       border-radius: 6px; font-weight: 600; font-size: 13px; }
        QTabBar::tab:selected { background: #2a2a6e; color: #fff; }
        QTabBar::tab:hover { background: #22225a; }
        QGroupBox { font-weight: 700; font-size: 13px; color: #c0c0e0; border: 1px solid #2a2a50;
                    border-radius: 8px; margin-top: 14px; padding: 18px 12px 10px 12px; }
        QGroupBox::title { subcontrol-origin: margin; left: 14px; padding: 0 8px; color: #9090d0; }
        QLabel { color: #c0c0d8; font-size: 12px; }
        QSlider::groove:horizontal { background: #1e1e40; height: 6px; border-radius: 3px; }
        QSlider::handle:horizontal { background: #6060c0; width: 18px; height: 18px;
                                     margin: -6px 0; border-radius: 9px; }
        QSlider::handle:horizontal:hover { background: #8080e0; }
        QSlider::sub-page:horizontal { background: #4040a0; border-radius: 3px; }
        QComboBox { background: #1a1a3a; border: 1px solid #2a2a50; border-radius: 6px;
                    padding: 6px 12px; color: #e0e0e8; min-height: 28px; }
        QComboBox::drop-down { border: none; width: 24px; }
        QComboBox QAbstractItemView { background: #1a1a3a; color: #e0e0e8;
                                      selection-background-color: #3a3a7a; border: 1px solid #2a2a50; }
        QSpinBox, QDoubleSpinBox { background: #1a1a3a; border: 1px solid #2a2a50;
                                   border-radius: 6px; padding: 5px 8px; color: #e0e0e8; }
        QCheckBox { color: #c0c0d8; spacing: 8px; font-size: 12px; }
        QCheckBox::indicator { width: 20px; height: 20px; border-radius: 5px;
                               border: 2px solid #3a3a6a; background: #1a1a3a; }
        QCheckBox::indicator:checked { background: #5050b0; border-color: #6060d0; }
        QPushButton { background: #2a2a6e; color: #e0e0e8; border: none; border-radius: 8px;
                      padding: 10px 24px; font-weight: 600; font-size: 13px; }
        QPushButton:hover { background: #3a3a8e; }
        QPushButton:pressed { background: #1a1a4e; }
        QPushButton#applyBtn { background: #4a4ac0; }
        QPushButton#applyBtn:hover { background: #5a5ae0; }
        QPushButton#resetBtn { background: #5a2020; }
        QPushButton#resetBtn:hover { background: #7a3030; }
        QScrollArea { border: none; background: transparent; }
        QLineEdit { background: #1a1a3a; border: 1px solid #2a2a50; border-radius: 6px;
                    padding: 5px; color: #e0e0e8; }
        """

    def _init_ui(self, current_preset_name):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        hdr = QLabel("⚙ Piano Roll Studio Settings")
        hdr.setFont(QFont("SF Pro", 18, QFont.Bold))
        hdr.setStyleSheet("color: #a0a0ff; margin-bottom: 4px;")
        layout.addWidget(hdr)

        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        self._build_visual_tab(current_preset_name)
        self._build_effects_tab()
        self._build_audio_tab()
        self._build_export_tab()
        self._build_advanced_tab()

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        reset_btn = QPushButton("↺ Reset to Default")
        reset_btn.setObjectName("resetBtn")
        reset_btn.clicked.connect(self._reset_defaults)
        btn_row.addWidget(reset_btn)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(cancel_btn)
        apply_btn = QPushButton("✓ Apply Settings")
        apply_btn.setObjectName("applyBtn")
        apply_btn.clicked.connect(self._apply)
        btn_row.addWidget(apply_btn)
        layout.addLayout(btn_row)

    def _build_visual_tab(self, current_preset_name):
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        container = QWidget(); lay = QVBoxLayout(container); lay.setSpacing(12)

        g = QGroupBox("🎨 Visual Preset"); fl = QFormLayout(g)
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(list(PRESETS.keys()))
        self.preset_combo.setCurrentText(current_preset_name)
        self.preset_combo.currentTextChanged.connect(self._on_preset_changed)
        fl.addRow("Preset:", self.preset_combo)
        lay.addWidget(g)

        g = QGroupBox("🖼 Background"); fl = QFormLayout(g)
        self.bg_pulse = self._slider(0, 100, 30); fl.addRow("Beat Pulse:", self.bg_pulse['widget'])
        self.star_count = self._slider(0, 500, 60); fl.addRow("Star Count:", self.star_count['widget'])
        self.vignette = self._slider(0, 100, 40); fl.addRow("Vignette:", self.vignette['widget'])
        self.reflection = QCheckBox("Reflection Effect"); fl.addRow(self.reflection)
        lay.addWidget(g)

        g = QGroupBox("🎹 Keyboard"); fl = QFormLayout(g)
        self.kb_glow = self._slider(0, 300, 200); fl.addRow("Active Glow:", self.kb_glow['widget'])
        self.kb_height = QSpinBox(); self.kb_height.setRange(40, 160); self.kb_height.setValue(90)
        fl.addRow("Height:", self.kb_height)
        lay.addWidget(g)

        g = QGroupBox("🎵 Note Appearance"); fl = QFormLayout(g)
        self.note_glow = self._slider(0, 300, 250); fl.addRow("Glow Intensity:", self.note_glow['widget'])
        self.note_radius = self._slider(0, 20, 3); fl.addRow("Corner Radius:", self.note_radius['widget'])
        self.note_border = QCheckBox("Show Border"); self.note_border.setChecked(True); fl.addRow(self.note_border)
        self.note_splash = QCheckBox("Note Splash Labels"); self.note_splash.setChecked(True); fl.addRow(self.note_splash)
        self.floating_labels = QCheckBox("Floating Note Names"); self.floating_labels.setChecked(True); fl.addRow(self.floating_labels)
        lay.addWidget(g)

        g = QGroupBox("💫 Trails & Afterglow"); fl = QFormLayout(g)
        self.trail_length = self._slider(0, 150, 40); fl.addRow("Trail Length:", self.trail_length['widget'])
        self.trail_opacity = self._slider(0, 100, 35); fl.addRow("Trail Opacity:", self.trail_opacity['widget'])
        lay.addWidget(g)

        g = QGroupBox("📊 Spectrum Visualizer"); fl = QFormLayout(g)
        self.show_spectrum = QCheckBox("Show Spectrum"); self.show_spectrum.setChecked(True); fl.addRow(self.show_spectrum)
        self.spectrum_style = QComboBox(); self.spectrum_style.addItems(["bars", "circular", "wave"])
        fl.addRow("Style:", self.spectrum_style)
        lay.addWidget(g)

        g = QGroupBox("🎮 HUD Overlay"); fl = QFormLayout(g)
        self.show_combo = QCheckBox("Combo Counter"); self.show_combo.setChecked(True); fl.addRow(self.show_combo)
        self.show_streaks = QCheckBox("Note Streaks"); self.show_streaks.setChecked(True); fl.addRow(self.show_streaks)
        self.show_speed = QCheckBox("Speed Display"); self.show_speed.setChecked(True); fl.addRow(self.show_speed)
        self.show_watermark = QCheckBox("Channel Watermark"); self.show_watermark.setChecked(False); fl.addRow(self.show_watermark)
        self.watermark_text = QLineEdit(); self.watermark_text.setPlaceholderText("@YourChannel"); fl.addRow("Watermark:", self.watermark_text)
        lay.addWidget(g)

        lay.addStretch(); scroll.setWidget(container); self.tabs.addTab(scroll, "🎨 Visual")

    def _build_effects_tab(self):
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        container = QWidget(); lay = QVBoxLayout(container); lay.setSpacing(12)

        g = QGroupBox("✨ Particles"); fl = QFormLayout(g)
        self.particle_count = self._slider(0, 40, 12); fl.addRow("Count per Note:", self.particle_count['widget'])
        self.particle_speed = self._slider(0, 200, 80); fl.addRow("Speed:", self.particle_speed['widget'])
        self.particle_size = self._slider(1, 20, 4); fl.addRow("Size:", self.particle_size['widget'])
        lay.addWidget(g)

        g = QGroupBox("📷 Camera & Shake"); fl = QFormLayout(g)
        self.screen_shake = self._slider(0, 100, 20); fl.addRow("Screen Shake:", self.screen_shake['widget'])
        self.zoom_min = self._dspin(0.8, 1.0, 1.0, 0.01); fl.addRow("Zoom Min:", self.zoom_min)
        self.zoom_max = self._dspin(1.0, 1.2, 1.05, 0.01); fl.addRow("Zoom Max:", self.zoom_max)
        lay.addWidget(g)

        g = QGroupBox("🎬 Post-Processing"); fl = QFormLayout(g)
        self.bloom = self._slider(0, 100, 60); fl.addRow("Bloom / Glow:", self.bloom['widget'])
        self.chromatic_aberration = self._slider(0, 100, 0); fl.addRow("Chromatic Aberration:", self.chromatic_aberration['widget'])
        self.scanlines = self._slider(0, 100, 5); fl.addRow("Scanlines:", self.scanlines['widget'])
        self.film_grain = self._slider(0, 100, 2); fl.addRow("Film Grain:", self.film_grain['widget'])
        self.hdr_tonemap = self._slider(0, 100, 30); fl.addRow("HDR Tonemap:", self.hdr_tonemap['widget'])
        self.light_streak = QCheckBox("Light Streak on Hit"); self.light_streak.setChecked(True); fl.addRow(self.light_streak)
        self.pulse_on_beat = QCheckBox("Pulse on Beat"); self.pulse_on_beat.setChecked(True); fl.addRow(self.pulse_on_beat)
        lay.addWidget(g)

        g = QGroupBox("⏩ Playback Speed"); fl = QFormLayout(g)
        self.default_speed = QDoubleSpinBox(); self.default_speed.setRange(0.25, 3.0); self.default_speed.setValue(1.0); self.default_speed.setSingleStep(0.25)
        self.default_speed.setStyleSheet("background:#1a1a3a; border:1px solid #2a2a50; border-radius:6px; padding:5px; color:#e0e0e8;")
        fl.addRow("Default Speed:", self.default_speed)
        self.pps_slider = self._slider(50, 600, 220); fl.addRow("Scroll Speed (pps):", self.pps_slider['widget'])
        lay.addWidget(g)

        lay.addStretch(); scroll.setWidget(container); self.tabs.addTab(scroll, "✨ Effects")

    def _build_audio_tab(self):
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        container = QWidget(); lay = QVBoxLayout(container); lay.setSpacing(12)

        g = QGroupBox("🔊 Synthesizer"); fl = QFormLayout(g)
        self.synth_volume = self._slider(0, 100, 40); fl.addRow("Synth Volume:", self.synth_volume['widget'])
        self.synth_attack = self._slider(1, 100, 8); fl.addRow("Attack (ms):", self.synth_attack['widget'])
        self.synth_release = self._slider(1, 300, 80); fl.addRow("Release (ms):", self.synth_release['widget'])
        lay.addWidget(g)

        g = QGroupBox("🎧 Audio Output"); fl = QFormLayout(g)
        self.auto_synthesize = QCheckBox("Auto-Synthesize MIDI"); self.auto_synthesize.setChecked(True); fl.addRow(self.auto_synthesize)
        lay.addWidget(g)

        lay.addStretch(); scroll.setWidget(container); self.tabs.addTab(scroll, "🔊 Audio")

    def _build_export_tab(self):
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        container = QWidget(); lay = QVBoxLayout(container); lay.setSpacing(12)

        g = QGroupBox("📹 Export Preset"); fl = QFormLayout(g)
        self.export_preset_combo = QComboBox()
        for ep in YOUTUBE_PRESETS: self.export_preset_combo.addItem(f"{ep.name} — {ep.description}", ep.name)
        self.export_preset_combo.currentIndexChanged.connect(self._on_export_preset_changed)
        fl.addRow("Format:", self.export_preset_combo)
        lay.addWidget(g)

        g = QGroupBox("📐 Resolution & Frame Rate"); fl = QFormLayout(g)
        self.export_width = QSpinBox(); self.export_width.setRange(256, 7680); self.export_width.setValue(1920); fl.addRow("Width:", self.export_width)
        self.export_height = QSpinBox(); self.export_height.setRange(256, 4320); self.export_height.setValue(1080); fl.addRow("Height:", self.export_height)
        self.export_fps = QSpinBox(); self.export_fps.setRange(24, 120); self.export_fps.setValue(60); fl.addRow("FPS:", self.export_fps)
        lay.addWidget(g)

        g = QGroupBox("🎥 Codec & Quality"); fl = QFormLayout(g)
        self.export_codec = QComboBox(); self.export_codec.addItems(["libx264", "libx265", "libvpx-vp9", "prores_ks"]); fl.addRow("Video Codec:", self.export_codec)
        self.export_crf = self._slider(0, 51, 18); fl.addRow("Quality (CRF):", self.export_crf['widget'])
        self.export_preset_enc = QComboBox(); self.export_preset_enc.addItems(["ultrafast","superfast","veryfast","faster","fast","medium","slow","slower","veryslow"]); self.export_preset_enc.setCurrentText("medium"); fl.addRow("Encoding Speed:", self.export_preset_enc)
        lay.addWidget(g)

        g = QGroupBox("🎯 YouTube / Shorts Specific"); fl = QFormLayout(g)
        self.export_shorts_mode = QCheckBox("Shorts Mode (9:16)"); self.export_shorts_mode.toggled.connect(self._on_shorts_toggled); fl.addRow(self.export_shorts_mode)
        self.export_shorts_crop = QCheckBox("Smart Crop to Center Notes"); self.export_shorts_crop.setChecked(True); fl.addRow(self.export_shorts_crop)
        self.export_shorts_zoom = QCheckBox("Dynamic Zoom Follow"); self.export_shorts_zoom.setChecked(True); fl.addRow(self.export_shorts_zoom)
        self.export_intro_slate = QCheckBox("Add Intro Slate (3s)"); fl.addRow(self.export_intro_slate)
        self.export_outro_slate = QCheckBox("Add Outro Slate (3s)"); fl.addRow(self.export_outro_slate)
        lay.addWidget(g)

        g = QGroupBox("📁 Output"); fl = QFormLayout(g)
        self.export_format = QComboBox(); self.export_format.addItems(["mp4", "mov", "webm", "mkv"]); fl.addRow("Container:", self.export_format)
        self.export_include_audio = QCheckBox("Include Synthesized Audio"); self.export_include_audio.setChecked(True); fl.addRow(self.export_include_audio)
        lay.addWidget(g)

        lay.addStretch(); scroll.setWidget(container); self.tabs.addTab(scroll, "📹 Export")

    def _build_advanced_tab(self):
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        container = QWidget(); lay = QVBoxLayout(container); lay.setSpacing(12)

        g = QGroupBox("⚡ Performance"); fl = QFormLayout(g)
        self.gpu_spectrum = QCheckBox("GPU Spectrum (CuPy)"); self.gpu_spectrum.setChecked(HAS_CUPY); self.gpu_spectrum.setEnabled(HAS_CUPY); fl.addRow(self.gpu_spectrum)
        self.render_quality = QComboBox(); self.render_quality.addItems(["Low (30fps)", "Medium (45fps)", "High (60fps)", "Ultra (60fps + AA)"]); self.render_quality.setCurrentIndex(2); fl.addRow("Render Quality:", self.render_quality)
        self.max_particles = QSpinBox(); self.max_particles.setRange(200, 5000); self.max_particles.setValue(2000); fl.addRow("Max Particles:", self.max_particles)
        lay.addWidget(g)

        g = QGroupBox("⌨ Keyboard Shortcuts")
        shortcuts_text = QTextEdit(); shortcuts_text.setReadOnly(True); shortcuts_text.setMaximumHeight(180)
        shortcuts_text.setHtml("<table style='color:#c0c0d8; font-size:12px;'><tr><td><b>Space</b></td><td>Play / Pause</td></tr><tr><td><b>← →</b></td><td>Seek ±2s</td></tr><tr><td><b>↑ ↓</b></td><td>Speed ±0.25×</td></tr><tr><td><b>R</b></td><td>Reset to Start</td></tr><tr><td><b>F</b></td><td>Toggle Fullscreen</td></tr><tr><td><b>1-0</b></td><td>Switch Preset</td></tr><tr><td><b>S</b></td><td>Open Settings</td></tr><tr><td><b>E</b></td><td>Export Video</td></tr></table>")
        shortcuts_text.setStyleSheet("background:#12122a; border:1px solid #2a2a50; border-radius:6px;")
        g.layout().addWidget(shortcuts_text)
        lay.addWidget(g)

        g = QGroupBox("🛠 System Info"); fl = QFormLayout(g)
        info_lines = [f"NumPy: {'✓' if HAS_NUMPY else '✗'}", f"Numba: {'✓' if HAS_NUMBA else '✗'}", f"CuPy GPU: {'✓' if HAS_CUPY else '✗'}", f"pydub: {'✓' if HAS_PYDUB else '✗'}", f"QtMultimedia: {'✓' if HAS_QTMULTIMEDIA else '✗'}", f"FFmpeg: {'✓' if check_ffmpeg() else '✗'}"]
        fl.addRow(QLabel("\n".join(info_lines)))
        lay.addWidget(g)

        lay.addStretch(); scroll.setWidget(container); self.tabs.addTab(scroll, "⚡ Advanced")

    def _slider(self, mn, mx, val):
        sl = QSlider(Qt.Horizontal); sl.setRange(mn, mx); sl.setValue(val)
        lbl = QLabel(str(val)); lbl.setMinimumWidth(36); lbl.setAlignment(Qt.AlignCenter); lbl.setStyleSheet("color:#a0a0d0; font-weight:600;")
        w = QWidget(); h = QHBoxLayout(w); h.setContentsMargins(0,0,0,0); h.addWidget(sl, 1); h.addWidget(lbl)
        sl.valueChanged.connect(lambda v, l=lbl: l.setText(str(v)))
        return {'widget': w, 'slider': sl, 'label': lbl}

    def _dspin(self, mn, mx, val, step):
        sb = QDoubleSpinBox(); sb.setRange(mn, mx); sb.setValue(val); sb.setSingleStep(step)
        sb.setStyleSheet("background:#1a1a3a; border:1px solid #2a2a50; border-radius:6px; padding:5px; color:#e0e0e8;")
        return sb

    def _on_preset_changed(self, name):
        if self._building or name not in PRESETS: return
        pr = PRESETS[name]; self._building = True
        self.bg_pulse['slider'].setValue(int(pr.bg_pulse * 100))
        self.star_count['slider'].setValue(pr.star_count)
        self.vignette['slider'].setValue(int(pr.vignette * 100))
        self.reflection.setChecked(pr.reflection)
        self.kb_glow['slider'].setValue(int(pr.kb_active_glow * 100))
        self.note_glow['slider'].setValue(int(pr.note_glow * 100))
        self.note_radius['slider'].setValue(int(pr.note_radius))
        self.note_border.setChecked(pr.note_border)
        self.note_splash.setChecked(pr.note_splash)
        self.floating_labels.setChecked(pr.floating_labels)
        self.trail_length['slider'].setValue(int(pr.trail_length))
        self.trail_opacity['slider'].setValue(int(pr.trail_opacity * 100))
        self.show_spectrum.setChecked(pr.show_spectrum)
        idx = self.spectrum_style.findText(pr.spectrum_style)
        if idx >= 0: self.spectrum_style.setCurrentIndex(idx)
        self.show_combo.setChecked(pr.show_combo)
        self.show_streaks.setChecked(pr.show_streaks)
        self.show_speed.setChecked(pr.show_speed)
        self.particle_count['slider'].setValue(pr.particle_count)
        self.particle_speed['slider'].setValue(int(pr.particle_speed))
        self.particle_size['slider'].setValue(int(pr.particle_size * 10))
        self.screen_shake['slider'].setValue(int(pr.screen_shake * 10))
        self.zoom_min.setValue(pr.camera_zoom_range[0])
        self.zoom_max.setValue(pr.camera_zoom_range[1])
        self.bloom['slider'].setValue(int(pr.bloom * 100))
        self.chromatic_aberration['slider'].setValue(int(pr.chromatic_aberration * 100))
        self.scanlines['slider'].setValue(int(pr.scanlines * 100))
        self.film_grain['slider'].setValue(int(pr.film_grain * 100))
        self.hdr_tonemap['slider'].setValue(int(pr.hdr_tonemap * 100))
        self.light_streak.setChecked(pr.light_streak)
        self.pulse_on_beat.setChecked(pr.pulse_on_beat)
        self._building = False

    def _on_export_preset_changed(self, idx):
        if idx < 0 or idx >= len(YOUTUBE_PRESETS): return
        ep = YOUTUBE_PRESETS[idx]; self._building = True
        self.export_width.setValue(ep.width); self.export_height.setValue(ep.height); self.export_fps.setValue(ep.fps)
        codec_idx = self.export_codec.findText(ep.codec)
        if codec_idx >= 0: self.export_codec.setCurrentIndex(codec_idx)
        self.export_crf['slider'].setValue(ep.crf)
        enc_idx = self.export_preset_enc.findText(ep.preset)
        if enc_idx >= 0: self.export_preset_enc.setCurrentIndex(enc_idx)
        self.export_shorts_mode.setChecked("Shorts" in ep.name)
        self._building = False

    def _on_shorts_toggled(self, checked):
        if checked and not self._building: self.export_width.setValue(1080); self.export_height.setValue(1920)

    def _reset_defaults(self): self._on_preset_changed("Neon")

    def _apply(self):
        self.settings_changed.emit(self.get_settings()); self.accept()

    def get_settings(self) -> dict:
        return {
            'preset_name': self.preset_combo.currentText(),
            'bg_pulse': self.bg_pulse['slider'].value() / 100.0, 'star_count': self.star_count['slider'].value(),
            'vignette': self.vignette['slider'].value() / 100.0, 'reflection': self.reflection.isChecked(),
            'kb_active_glow': self.kb_glow['slider'].value() / 100.0, 'kb_height': self.kb_height.value(),
            'note_glow': self.note_glow['slider'].value() / 100.0, 'note_radius': self.note_radius['slider'].value(),
            'note_border': self.note_border.isChecked(), 'note_splash': self.note_splash.isChecked(),
            'floating_labels': self.floating_labels.isChecked(),
            'trail_length': self.trail_length['slider'].value(), 'trail_opacity': self.trail_opacity['slider'].value() / 100.0,
            'show_spectrum': self.show_spectrum.isChecked(), 'spectrum_style': self.spectrum_style.currentText(),
            'show_combo': self.show_combo.isChecked(), 'show_streaks': self.show_streaks.isChecked(),
            'show_speed': self.show_speed.isChecked(), 'show_watermark': self.show_watermark.isChecked(),
            'watermark_text': self.watermark_text.text(),
            'particle_count': self.particle_count['slider'].value(), 'particle_speed': self.particle_speed['slider'].value(),
            'particle_size': self.particle_size['slider'].value() / 10.0,
            'screen_shake': self.screen_shake['slider'].value() / 10.0,
            'zoom_min': self.zoom_min.value(), 'zoom_max': self.zoom_max.value(),
            'bloom': self.bloom['slider'].value() / 100.0, 'chromatic_aberration': self.chromatic_aberration['slider'].value() / 100.0,
            'scanlines': self.scanlines['slider'].value() / 100.0, 'film_grain': self.film_grain['slider'].value() / 100.0,
            'hdr_tonemap': self.hdr_tonemap['slider'].value() / 100.0, 'light_streak': self.light_streak.isChecked(),
            'pulse_on_beat': self.pulse_on_beat.isChecked(), 'default_speed': self.default_speed.value(),
            'pps': self.pps_slider['slider'].value(),
            'synth_volume': self.synth_volume['slider'].value() / 100.0, 'synth_attack': self.synth_attack['slider'].value(),
            'synth_release': self.synth_release['slider'].value(), 'auto_synthesize': self.auto_synthesize.isChecked(),
            'export_width': self.export_width.value(), 'export_height': self.export_height.value(),
            'export_fps': self.export_fps.value(), 'export_codec': self.export_codec.currentText(),
            'export_crf': self.export_crf['slider'].value(), 'export_preset_enc': self.export_preset_enc.currentText(),
            'export_shorts_mode': self.export_shorts_mode.isChecked(), 'export_shorts_crop': self.export_shorts_crop.isChecked(),
            'export_shorts_zoom': self.export_shorts_zoom.isChecked(), 'export_intro_slate': self.export_intro_slate.isChecked(),
            'export_outro_slate': self.export_outro_slate.isChecked(), 'export_format': self.export_format.currentText(),
            'export_include_audio': self.export_include_audio.isChecked(), 'gpu_spectrum': self.gpu_spectrum.isChecked(),
            'render_quality': self.render_quality.currentIndex(), 'max_particles': self.max_particles.value(),
        }

# ═══════════════════════════════════════════════════════════════
#  PROFESSIONAL PIANO KEYBOARD
# ═══════════════════════════════════════════════════════════════
class ProKeyboard(QWidget):
    note_pressed = Signal(int); note_released = Signal(int)
    def __init__(self, parent=None):
        super().__init__(parent)
        self.min_note=21; self.max_note=108; self.kw=20; self.kh=90
        self.scroll_x=0; self.active: Set[int]=set(); self.colors: Dict[int,QColor]={}
        self.preset=PRESETS["Neon"]; self.anim_t=0.0; self.ripples=RippleSystem()
        self.setFixedHeight(self.kh+6); self.setMouseTracking(True)
        self._glow_cache: Dict[int, float] = {}

    def tw(self): return sum(1 for n in range(self.min_note,self.max_note+1) if not is_black(n))*self.kw
    def _nx(self, n):
        wc = sum(1 for m in range(self.min_note, n+1) if not is_black(m))
        return wc*self.kw-self.kw*0.38 if is_black(n) else (wc-1)*self.kw
    def _nw(self, n): return self.kw*0.68 if is_black(n) else self.kw

    def emit_ripple(self, note, color):
        self.ripples.emit(self._nx(note)+self._nw(note)/2, 8, color, 35, 0.4)
        self._glow_cache[note] = 1.0

    def set_active(self, notes, colors=None):
        self.active = notes; self.colors = colors or {}; self.update()

    def advance(self, dt):
        self.anim_t += dt; self.ripples.update(dt)
        dead = []
        for n in self._glow_cache:
            self._glow_cache[n] -= dt * 3.0
            if self._glow_cache[n] <= 0: dead.append(n)
        for n in dead: del self._glow_cache[n]

    def _get_note_color(self, n, is_active):
        pr = self.preset
        if not is_active: return None
        if pr.is_rainbow or pr.note_colors == 'rainbow': return _hsv((n-21)/87 + self.anim_t*0.06, 0.9, 1.0)
        return self.colors.get(n, QColor(80,160,255))

    def paintEvent(self, _):
        p = QPainter(self)
        if not p.isActive(): return
        try:
            p.setRenderHint(QPainter.Antialiasing); p.translate(-self.scroll_x, 0)
            kh = self.kh; t = self.anim_t; pr = self.preset

            for n in range(self.min_note, self.max_note+1):
                if is_black(n): continue
                x, w = self._nx(n), self._nw(n); is_act = n in self.active
                if is_act:
                    c = self._get_note_color(n, True)
                    if c:
                        glow_a = int(50 * pr.kb_active_glow)
                        p.setPen(Qt.NoPen); p.setBrush(QBrush(QColor(c.red(),c.green(),c.blue(),glow_a)))
                        p.drawRoundedRect(int(x-4), -4, int(w+8), kh+8, 4, 6)
                        g = QLinearGradient(x, 0, x, kh)
                        g.setColorAt(0, c.lighter(180)); g.setColorAt(0.3, c.lighter(140)); g.setColorAt(0.7, c); g.setColorAt(1, c.darker(110))
                        p.setBrush(QBrush(g)); p.setPen(QPen(c.lighter(120), 0.8))
                    else:
                        g = QLinearGradient(x,0,x,kh); g.setColorAt(0, pr.kb_wt); g.setColorAt(0.85, pr.kb_wb); g.setColorAt(1, pr.kb_wb.darker(110))
                        p.setBrush(QBrush(g)); p.setPen(QPen(pr.kb_border, 0.5))
                else:
                    g = QLinearGradient(x,0,x,kh); g.setColorAt(0, pr.kb_wt); g.setColorAt(0.85, pr.kb_wb); g.setColorAt(1, pr.kb_wb.darker(110))
                    p.setBrush(QBrush(g)); p.setPen(QPen(pr.kb_border, 0.5))
                p.drawRoundedRect(int(x), 0, int(w), kh, 1, 5)
                if n%12 == 0:
                    p.setPen(QColor(120,120,140) if not is_act else QColor(255,255,255,200))
                    p.setFont(QFont("SF Pro", 7, QFont.Bold)); p.drawText(int(x), kh-16, int(w), 14, Qt.AlignCenter, note_name(n))

            for n in range(self.min_note, self.max_note+1):
                if not is_black(n): continue
                x, w, h2 = self._nx(n), self._nw(n), int(kh*0.60); is_act = n in self.active
                if is_act:
                    c = self._get_note_color(n, True)
                    if c:
                        g = QLinearGradient(x,0,x,h2); g.setColorAt(0, c.lighter(160)); g.setColorAt(0.5, c); g.setColorAt(1, c.darker(170))
                        p.setBrush(QBrush(g)); p.setPen(QPen(c.darker(150), 0.6))
                    else:
                        g = QLinearGradient(x,0,x,h2); g.setColorAt(0, pr.kb_bt.lighter(130)); g.setColorAt(0.3, pr.kb_bt); g.setColorAt(1, pr.kb_bb)
                        p.setBrush(QBrush(g)); p.setPen(QPen(pr.kb_bb.darker(150), 0.5))
                else:
                    g = QLinearGradient(x,0,x,h2); g.setColorAt(0, pr.kb_bt.lighter(130)); g.setColorAt(0.3, pr.kb_bt); g.setColorAt(1, pr.kb_bb)
                    p.setBrush(QBrush(g)); p.setPen(QPen(pr.kb_bb.darker(150), 0.5))
                p.drawRoundedRect(int(x), 0, int(w), h2, 1, 3)

            self.ripples.draw(p); p.end()
        except: pass

    def _hit(self, mx):
        mx += self.scroll_x
        for n in range(self.max_note, self.min_note-1, -1):
            if not is_black(n): continue
            x, w = self._nx(n), self._nw(n)
            if x <= mx <= x+w: return n
        for n in range(self.max_note, self.min_note-1, -1):
            if is_black(n): continue
            x, w = self._nx(n), self._nw(n)
            if x <= mx <= x+w: return n
        return -1
    def mousePressEvent(self, e):
        n = self._hit(e.position().x())
        if n >= 0: self.note_pressed.emit(n)
    def mouseReleaseEvent(self, e):
        n = self._hit(e.position().x())
        if n >= 0: self.note_released.emit(n)

# ═══════════════════════════════════════════════════════════════
#  PROFESSIONAL PIANO ROLL WIDGET
# ═══════════════════════════════════════════════════════════════
class ProPianoRoll(QWidget):
    seek_requested = Signal(float)
    def __init__(self, parent=None):
        super().__init__(parent)
        self.notes: List[NoteEvent] = []; self.current_time = 0.0
        self.total_duration = 0.0; self.pps = 220; self.min_note = 21; self.max_note = 108
        self.kw = 20; self.track_vis: Dict[int,bool] = {}
        self.preset = PRESETS["Neon"]; self.scroll_x = 0
        self.particles = ParticleEngine(2000); self.starfield = StarField(100)
        self.splash_sys = NoteSplashSystem()
        self.beat_times: List[float] = []
        self.setMinimumSize(500, 300); self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)
        self.audio_data: Optional[AudioData] = None; self.show_audio_mode = False
        self._painting = False; self._prev_active: Set[Tuple] = set()
        self._shake_x = 0.0; self._shake_y = 0.0; self._shake_decay = 0.0
        self._bg_pulse_val = 0.0; self._combo = 0; self._max_combo = 0
        self._streak_notes: List[Tuple[float,int,QColor]] = []
        self._speed_mult = 1.0; self._note_count_display = 0; self._active_count = 0
        self._beat_pulse = 0.0
        self._show_watermark = False; self._watermark_text = ""
        self._floating_labels = True; self._note_splash = True; self._light_streak = True
        self._bloom = 0.6; self._chromatic = 0.0; self._scanlines = 0.05
        self._film_grain = 0.02; self._pulse_on_beat = True; self._hdr_tonemap = 0.3
        self._streaks: List[dict] = []

    def tw(self): return sum(1 for n in range(self.min_note,self.max_note+1) if not is_black(n))*self.kw
    def _nx(self, n):
        wc = sum(1 for m in range(self.min_note,n+1) if not is_black(m))
        return wc*self.kw-self.kw*0.38 if is_black(n) else (wc-1)*self.kw
    def _nw(self, n): return self.kw*0.68 if is_black(n) else self.kw
    def _t2y(self, t): return self.height()-70-(t-self.current_time)*self.pps
    def _y2t(self, y): return self.current_time+(self.height()-70-y)/self.pps

    def apply_settings(self, settings: dict):
        if settings.get('preset_name') and settings['preset_name'] in PRESETS:
            self.preset = PRESETS[settings['preset_name']]
        pr = self.preset
        pr.bg_pulse = settings.get('bg_pulse', pr.bg_pulse); pr.star_count = settings.get('star_count', pr.star_count)
        pr.vignette = settings.get('vignette', pr.vignette); pr.reflection = settings.get('reflection', pr.reflection)
        pr.kb_active_glow = settings.get('kb_active_glow', pr.kb_active_glow)
        pr.note_glow = settings.get('note_glow', pr.note_glow); pr.note_radius = settings.get('note_radius', pr.note_radius)
        pr.note_border = settings.get('note_border', pr.note_border)
        pr.trail_length = settings.get('trail_length', pr.trail_length); pr.trail_opacity = settings.get('trail_opacity', pr.trail_opacity)
        pr.show_spectrum = settings.get('show_spectrum', pr.show_spectrum); pr.spectrum_style = settings.get('spectrum_style', pr.spectrum_style)
        pr.show_combo = settings.get('show_combo', pr.show_combo); pr.show_streaks = settings.get('show_streaks', pr.show_streaks)
        pr.show_speed = settings.get('show_speed', pr.show_speed)
        pr.particle_count = settings.get('particle_count', pr.particle_count); pr.particle_speed = settings.get('particle_speed', pr.particle_speed)
        pr.particle_size = settings.get('particle_size', pr.particle_size); pr.screen_shake = settings.get('screen_shake', pr.screen_shake)
        pr.camera_zoom_range = (settings.get('zoom_min', 1.0), settings.get('zoom_max', 1.05))
        pr.bloom = settings.get('bloom', pr.bloom); pr.chromatic_aberration = settings.get('chromatic_aberration', pr.chromatic_aberration)
        pr.scanlines = settings.get('scanlines', pr.scanlines); pr.film_grain = settings.get('film_grain', pr.film_grain)
        pr.hdr_tonemap = settings.get('hdr_tonemap', pr.hdr_tonemap); pr.light_streak = settings.get('light_streak', pr.light_streak)
        pr.pulse_on_beat = settings.get('pulse_on_beat', pr.pulse_on_beat)
        pr.note_splash = settings.get('note_splash', pr.note_splash); pr.floating_labels = settings.get('floating_labels', pr.floating_labels)
        self._show_watermark = settings.get('show_watermark', False); self._watermark_text = settings.get('watermark_text', '')
        self._floating_labels = settings.get('floating_labels', True); self._note_splash = settings.get('note_splash', True)
        self._light_streak = settings.get('light_streak', True); self._bloom = settings.get('bloom', 0.6)
        self._chromatic = settings.get('chromatic_aberration', 0.0); self._scanlines = settings.get('scanlines', 0.05)
        self._film_grain = settings.get('film_grain', 0.02); self._pulse_on_beat = settings.get('pulse_on_beat', True)
        self._hdr_tonemap = settings.get('hdr_tonemap', 0.3)
        if settings.get('pps'): self.pps = settings['pps']
        self.starfield = StarField(pr.star_count); self.particles.max_p = settings.get('max_particles', 2000)
        self.update()

    def set_notes(self, notes, dur, beats=None):
        self.notes = sorted(notes, key=lambda n: n.start_time)
        self.total_duration = dur; self.beat_times = beats or []
        self.show_audio_mode = False; self._prev_active.clear()
        self._combo = 0; self._max_combo = 0; self._streak_notes.clear()
        if notes: self.min_note = max(0, min(n.note for n in notes)-3); self.max_note = min(127, max(n.note for n in notes)+3)
        self.update()

    def set_audio_data(self, ad):
        self.audio_data = ad; self.total_duration = ad.duration
        self.notes = []; self.show_audio_mode = True; self.update()

    def set_time(self, t): self.current_time = t; self.update()

    def trigger_note(self, ne):
        x = self._nx(ne.note)+self._nw(ne.note)/2; y = self._t2y(ne.start_time)
        pr = self.preset; c = self._note_color(ne)
        self.particles.emit(x, y, c, count=pr.particle_count, speed=pr.particle_speed, life=0.8, size=pr.particle_size, has_trail=pr.trail_length > 0, ptype='splash' if pr.note_splash else 'spark')
        if pr.screen_shake > 0: self._shake_decay = min(self._shake_decay + pr.screen_shake, 8.0)
        self._combo += 1; self._max_combo = max(self._max_combo, self._combo)
        self._streak_notes.append((time.time(), ne.note, c))
        if pr.bg_pulse > 0: self._bg_pulse_val = min(self._bg_pulse_val + pr.bg_pulse, 1.0)
        if self._light_streak: self._streaks.append({'x': x, 'y': y, 'color': c, 'life': 1.0, 'max_life': 1.0})
        if self._note_splash: self.splash_sys.emit(x, y-10, note_name(ne.note), c, 12)

    def _note_color(self, ne) -> QColor:
        pr = self.preset
        if pr.is_rainbow or pr.note_colors == 'rainbow': return _hsv((ne.note-21)/87 + self.anim_t*0.06, 0.9, 1.0)
        if isinstance(pr.note_colors, list): return pr.note_colors[ne.track % len(pr.note_colors)]
        return QColor(80,160,255)

    def paintEvent(self, _):
        if self._painting: return
        self._painting = True
        try:
            p = QPainter(self)
            if not p.isActive(): return
            p.setRenderHint(QPainter.Antialiasing)
            W, H = self.width(), self.height(); ct = self.current_time; pr = self.preset; ta = time.time()
            self.anim_t = ta
            if self._shake_decay > 0.1:
                sx = random.uniform(-self._shake_decay, self._shake_decay); sy = random.uniform(-self._shake_decay, self._shake_decay)
                p.translate(sx, sy); self._shake_decay *= 0.88
            if self._pulse_on_beat and self.beat_times:
                for bt in self.beat_times:
                    dt_beat = ct - bt
                    if 0 <= dt_beat < 0.15: self._beat_pulse = max(self._beat_pulse, 1.0 - dt_beat / 0.15)
            self._draw_bg(p, W, H, ta, pr)
            if self.show_audio_mode: self._draw_audio_viz(p, W, H, ct, ta, pr)
            else: self._draw_roll(p, W, H, ct, ta, pr)
            py = self._t2y(ct)
            if -5 <= py <= H+5:
                g = QLinearGradient(0, py-10, 0, py+10); g.setColorAt(0, QColor(255,255,255,0)); g.setColorAt(0.4, QColor(255,255,255,40)); g.setColorAt(0.5, QColor(255,255,255,80)); g.setColorAt(0.6, QColor(255,255,255,40)); g.setColorAt(1, QColor(255,255,255,0))
                p.fillRect(0, int(py)-10, W, 20, QBrush(g)); p.setPen(QPen(QColor(255,255,255,220), 2)); p.drawLine(0, int(py), W, int(py))
            self._draw_light_streaks(p, W, H, 1.0/60.0)
            if self._note_splash: self.splash_sys.update(1.0/60.0); self.splash_sys.draw(p)
            if self._scanlines > 0: self._draw_scanlines(p, W, H)
            if self._film_grain > 0: self._draw_film_grain(p, W, H)
            if pr.vignette > 0: self._draw_vignette(p, W, H, pr.vignette)
            if self._chromatic > 0.05: self._draw_chromatic_edge(p, W, H)
            if self._bloom > 0.1: self._draw_bloom(p, W, H)
            self._draw_hud(p, W, H, ct, pr, ta)
            p.end()
        finally: self._painting = False

    def _draw_bg(self, p, W, H, t, pr):
        pulse_extra = self._beat_pulse * 15 if self._pulse_on_beat else 0
        if pr.is_rainbow or pr.note_colors == 'rainbow':
            bg = QLinearGradient(0,0,W,H); h1 = (t*0.015)%1
            bg.setColorAt(0, _hsv(h1, 0.3, 0.06 + self._bg_pulse_val*0.04)); bg.setColorAt(0.5, _hsv((h1+0.1)%1, 0.3, 0.08 + self._bg_pulse_val*0.05)); bg.setColorAt(1, _hsv((h1+0.2)%1, 0.3, 0.06 + self._bg_pulse_val*0.04))
            p.fillRect(0,0,W,H, QBrush(bg))
        else:
            bg = QLinearGradient(0,0,0,H); lv = 105 + int(self._bg_pulse_val*30) + int(pulse_extra)
            bg.setColorAt(0, pr.bg1.lighter(min(200, lv))); bg.setColorAt(1, pr.bg2)
            p.fillRect(0,0,W,H, QBrush(bg))
        self._bg_pulse_val *= 0.92; self._beat_pulse *= 0.9
        if pr.star_count > 0: self.starfield.draw(p, W, H, t, pr.star_color, pr.is_rainbow or pr.note_colors=='rainbow')

    def _draw_vignette(self, p, W, H, strength):
        g = QRadialGradient(W/2, H/2, max(W,H)*0.7); g.setColorAt(0, QColor(0,0,0,0)); g.setColorAt(0.5, QColor(0,0,0,0)); g.setColorAt(1, QColor(0,0,0,int(strength*200)))
        p.fillRect(0,0,W,H, QBrush(g))

    def _draw_scanlines(self, p, W, H):
        alpha = int(self._scanlines * 40); p.setPen(Qt.NoPen)
        for y in range(0, H, 3): p.fillRect(0, y, W, 1, QColor(0, 0, 0, alpha))

    def _draw_film_grain(self, p, W, H):
        alpha = int(self._film_grain * 30); p.setPen(Qt.NoPen)
        random.seed(int(time.time() * 100) % 10000)
        for _ in range(int(W * H * self._film_grain * 0.0004)):
            x = random.randint(0, W-1); y = random.randint(0, H-1); v = random.randint(0, 255)
            p.fillRect(x, y, 1, 1, QColor(v, v, v, alpha))

    def _draw_chromatic_edge(self, p, W, H):
        strength = int(self._chromatic * 25)
        g1 = QLinearGradient(0, 0, 40, 0); g1.setColorAt(0, QColor(255, 0, 0, strength)); g1.setColorAt(1, QColor(255, 0, 0, 0)); p.fillRect(0, 0, 40, H, QBrush(g1))
        g2 = QLinearGradient(W-40, 0, W, 0); g2.setColorAt(0, QColor(0, 0, 255, 0)); g2.setColorAt(1, QColor(0, 0, 255, strength)); p.fillRect(W-40, 0, 40, H, QBrush(g2))

    def _draw_bloom(self, p, W, H):
        alpha = int(self._bloom * 15); g = QRadialGradient(W/2, H/2, max(W,H)*0.4)
        g.setColorAt(0, QColor(255, 255, 255, alpha)); g.setColorAt(0.4, QColor(255, 255, 255, alpha//3)); g.setColorAt(1, QColor(255, 255, 255, 0))
        p.fillRect(0, 0, W, H, QBrush(g))

    def _draw_light_streaks(self, p, W, H, dt):
        alive = []
        for s in self._streaks:
            s['life'] -= dt * 2.5
            if s['life'] <= 0: continue
            alpha = int(s['life'] * 80); c = QColor(s['color']); c.setAlpha(max(0, alpha))
            streak_h = int((1.0 - s['life']) * 200 + 40)
            g = QLinearGradient(s['x'], s['y'] - streak_h, s['x'], s['y'] + 20)
            g.setColorAt(0, QColor(c.red(), c.green(), c.blue(), 0)); g.setColorAt(0.5, QColor(c.red(), c.green(), c.blue(), alpha)); g.setColorAt(1, QColor(c.red(), c.green(), c.blue(), 0))
            p.setPen(Qt.NoPen); p.setBrush(QBrush(g)); p.drawRect(int(s['x']-3), int(s['y']-streak_h), 6, streak_h+20)
            alive.append(s)
        self._streaks = alive

    def _draw_roll(self, p, W, H, ct, ta, pr):
        p.save(); p.translate(-self.scroll_x, 0)
        for n in range(self.min_note, self.max_note+1):
            x, w = self._nx(n), self._nw(n)
            if not is_black(n):
                p.fillRect(int(x),0,int(w),H, pr.lane_white); pen = QPen(pr.c_line, 1) if n%12==0 else QPen(pr.grid_line, 0.5); p.setPen(pen); p.drawLine(int(x+w),0,int(x+w),H)
            else: p.fillRect(int(x),0,int(w),H, pr.lane_black)
        for bt in self.beat_times:
            by = self._t2y(bt)
            if -2 <= by <= H+2:
                is_measure = any(abs(bt - mb) < 0.01 for mb in self.beat_times[::4])
                dt_beat = ct - bt; pulse_alpha = 0
                if 0 <= dt_beat < 0.2 and self._pulse_on_beat: pulse_alpha = int(40 * (1 - dt_beat / 0.2))
                alpha = (40 if is_measure else 18) + pulse_alpha
                p.setPen(QPen(QColor(255,255,255,alpha), 1 if is_measure else 0.5, Qt.DashLine)); p.drawLine(0, int(by), self.tw(), int(by))
        vs = ct - 3; ve = ct + H/self.pps + 3; rad = pr.note_radius
        for ne in self.notes:
            if not self.track_vis.get(ne.track, True): continue
            if ne.start_time+ne.duration < vs: continue
            if ne.start_time > ve: break
            x, w = self._nx(ne.note), self._nw(ne.note); y = self._t2y(ne.start_time); h = max(ne.duration*self.pps, 4)
            is_act = ct >= ne.start_time and ct < ne.start_time+ne.duration; base = self._note_color(ne)
            dist_factor = max(0, min(1, (ne.start_time - ct) / 5.0))
            if pr.trail_length > 0 and not is_act:
                dist = ct - (ne.start_time + ne.duration)
                if 0 < dist < pr.trail_length / self.pps:
                    trail_alpha = int(pr.trail_opacity * 255 * (1 - dist / (pr.trail_length / self.pps)))
                    tc = QColor(base.red(), base.green(), base.blue(), max(0, trail_alpha)); p.setPen(Qt.NoPen); p.setBrush(QBrush(tc))
                    trail_h = min(h * 0.6, 20); p.drawRoundedRect(int(x), int(y+h-trail_h), int(w), int(trail_h), rad, rad)
            if is_act:
                bright = base.lighter(140); glow_alpha = int(60 * pr.note_glow); p.setPen(Qt.NoPen)
                for gi in range(3):
                    ga = glow_alpha // (gi + 1); expand = (gi + 1) * 4
                    p.setBrush(QBrush(QColor(bright.red(), bright.green(), bright.blue(), ga))); p.drawRoundedRect(int(x-expand), int(y-expand), int(w+expand*2), int(h+expand*2), rad+expand, rad+expand)
                g = QLinearGradient(x, y, x, y+h); g.setColorAt(0, bright.lighter(180)); g.setColorAt(0.15, bright.lighter(150)); g.setColorAt(0.5, bright.lighter(120)); g.setColorAt(1, base)
                p.setBrush(QBrush(g)); p.setPen(QPen(bright.lighter(120), 0.8) if pr.note_border else Qt.NoPen); p.drawRoundedRect(int(x), int(y), int(w), int(h), rad, rad)
                if h > 10 and w > 10:
                    hl = QLinearGradient(x, y, x, y+h*0.35); hl.setColorAt(0, QColor(255,255,255,100)); hl.setColorAt(1, QColor(255,255,255,0))
                    p.setBrush(QBrush(hl)); p.setPen(Qt.NoPen); p.drawRoundedRect(int(x+2), int(y+1), int(w-4), int(h*0.35), rad-1, rad-1)
            else:
                alpha = max(30, int(230 * (1 - dist_factor * 0.7))); p.setBrush(QBrush(QColor(base.red(), base.green(), base.blue(), alpha))); p.setPen(Qt.NoPen); p.drawRoundedRect(int(x), int(y), int(w), int(h), rad, rad)
            if w > 10 and h > 16 and (self._floating_labels or is_act):
                label_alpha = 255 if is_act else max(40, 150 - int(dist_factor * 120)); p.setPen(QColor(255,255,255, label_alpha))
                p.setFont(QFont("SF Pro", 7 if not is_act else 8, QFont.Bold if is_act else QFont.Normal)); p.drawText(int(x+1), int(y+2), int(w-2), int(min(h-2,14)), Qt.AlignCenter, note_name(ne.note))
        if pr.reflection:
            cursor_y = self._t2y(ct)
            if cursor_y < H:
                ref_h = min(80, H - int(cursor_y)); g = QLinearGradient(0, cursor_y, 0, cursor_y+ref_h)
                g.setColorAt(0, QColor(255,255,255,15)); g.setColorAt(0.5, QColor(255,255,255,5)); g.setColorAt(1, QColor(255,255,255,0))
                p.fillRect(QRectF(0, cursor_y, self.tw(), ref_h), QBrush(g))
        p.restore(); self.particles.draw(p, self.scroll_x)

    def _draw_audio_viz(self, p, W, H, ct, ta, pr):
        ad = self.audio_data
        if not ad or not ad.waveform_peaks: p.setPen(QColor(255,255,255,150)); p.setFont(QFont("Arial",14)); p.drawText(0,0,W,H, Qt.AlignCenter, "No Audio Data"); return
        if ad.spectrum_data:
            nb=64
            if pr.spectrum_style == 'circular': self._draw_circular_spectrum(p, W, H, ct, ta, ad, nb, pr)
            elif pr.spectrum_style == 'wave': self._draw_wave_spectrum(p, W, H, ct, ta, ad, nb, pr)
            else: self._draw_bar_spectrum(p, W, H, ct, ta, ad, nb, pr)
        if ad.waveform_peaks:
            tp = len(ad.waveform_peaks); pp = W/tp; mid = H*0.35
            pu = QPainterPath(); pd = QPainterPath(); first = True
            for i, (mx, mn) in enumerate(ad.waveform_peaks):
                x = i*pp
                if first: pu.moveTo(x, mid-mx*H*0.22); pd.moveTo(x, mid-mn*H*0.22); first=False
                else: pu.lineTo(x, mid-mx*H*0.22); pd.lineTo(x, mid-mn*H*0.22)
            c = pr.note_colors[0] if isinstance(pr.note_colors, list) and pr.note_colors else QColor(80,160,255)
            p.setPen(QPen(QColor(c.red(),c.green(),c.blue(),180), 1.5)); p.setBrush(Qt.NoBrush); p.drawPath(pu); p.drawPath(pd)

    def _draw_bar_spectrum(self, p, W, H, ct, ta, ad, nb, pr):
        idx = max(0, min(int(ct*30), len(ad.spectrum_data)-1))
        if idx < 0 or idx >= len(ad.spectrum_data): return
        spec = ad.spectrum_data[idx]; bw = W/nb
        for i, val in enumerate(spec):
            if i >= nb: break
            bh = val*H*0.55; x = i*bw; y = H*0.75-bh
            c = (_hsv((i/nb+ta*0.04)%1, 0.85, 0.95, 200) if (pr.is_rainbow or pr.note_colors=='rainbow') else _hsv(i/nb*0.7+0.55, 0.8, 0.9, 200))
            p.setPen(Qt.NoPen); p.setBrush(QBrush(QColor(c.red(),c.green(),c.blue(),40))); p.drawRoundedRect(int(x-1), int(y-2), int(bw+2), int(bh+4), 3, 3)
            g = QLinearGradient(x, y, x, y+bh); g.setColorAt(0, c.lighter(140)); g.setColorAt(1, c)
            p.setBrush(QBrush(g)); p.drawRoundedRect(int(x), int(y), int(bw-1), int(bh), 2, 2)

    def _draw_circular_spectrum(self, p, W, H, ct, ta, ad, nb, pr):
        idx = max(0, min(int(ct*30), len(ad.spectrum_data)-1))
        if idx < 0 or idx >= len(ad.spectrum_data): return
        spec = ad.spectrum_data[idx]; cx, cy = W/2, H*0.6; base_r = min(W,H)*0.15; p.setPen(Qt.NoPen)
        for i, val in enumerate(spec):
            if i >= nb: break
            angle = (i/nb)*360 + ta*20; r = base_r + val*min(W,H)*0.25; rad = math.radians(angle)
            x1 = cx + math.cos(rad)*base_r; y1 = cy + math.sin(rad)*base_r; x2 = cx + math.cos(rad)*r; y2 = cy + math.sin(rad)*r
            c = _hsv(i/nb, 0.85, 0.95, int(val*200+55)); p.setPen(QPen(c, max(1, val*4))); p.drawLine(int(x1),int(y1),int(x2),int(y2))

    def _draw_wave_spectrum(self, p, W, H, ct, ta, ad, nb, pr):
        idx = max(0, min(int(ct*30), len(ad.spectrum_data)-1))
        if idx < 0 or idx >= len(ad.spectrum_data): return
        spec = ad.spectrum_data[idx]; mid_y = H*0.75
        path = QPainterPath(); path.moveTo(0, mid_y)
        for i, val in enumerate(spec):
            if i >= nb: break
            x = (i/nb)*W; y = mid_y - val*H*0.4
            if i == 0: path.lineTo(x, y)
            else: path.cubicTo((i-0.5)/nb*W, mid_y - spec[i-1]*H*0.4, x, y, x, y)
        path.lineTo(W, mid_y); c = pr.note_colors[0] if isinstance(pr.note_colors, list) and pr.note_colors else QColor(80,160,255)
        fill_path = QPainterPath(path); fill_path.lineTo(W, H); fill_path.lineTo(0, H); fill_path.closeSubpath()
        g = QLinearGradient(0, mid_y-H*0.4, 0, H); g.setColorAt(0, QColor(c.red(),c.green(),c.blue(),80)); g.setColorAt(1, QColor(c.red(),c.green(),c.blue(),0))
        p.setPen(Qt.NoPen); p.setBrush(QBrush(g)); p.drawPath(fill_path); p.setPen(QPen(c, 2)); p.setBrush(Qt.NoBrush); p.drawPath(path)

    def _draw_hud(self, p, W, H, ct, pr, ta):
        p.setRenderHint(QPainter.TextAntialiasing)
        if pr.show_combo and self._combo > 2:
            combo_text = f"×{self._combo}"; scale = 1.0 + min(0.3, self._combo * 0.01); font_size = int(28 * scale)
            p.setFont(QFont("SF Pro", font_size, QFont.Bold)); tw = p.fontMetrics().horizontalAdvance(combo_text) + 24
            pill_rect = QRectF(W-tw-20, 8, tw, font_size+16); p.setPen(Qt.NoPen); p.setBrush(QBrush(QColor(0,0,0,120))); p.drawRoundedRect(pill_rect, 12, 12)
            hue = (ta * 0.3) % 1 if (pr.is_rainbow or pr.note_colors=='rainbow') else 0.55
            p.setPen(_hsv(hue, 0.9, 1.0)); p.drawText(pill_rect, Qt.AlignCenter, combo_text)
        if pr.show_speed and self._speed_mult != 1.0:
            p.setFont(QFont("SF Pro", 16, QFont.Bold)); p.setPen(QColor(255,255,255,180)); p.drawText(20, 36, f"{self._speed_mult:.2f}×")
        if pr.show_streaks:
            now = time.time(); recent = [(t, n, c) for t, n, c in self._streak_notes if now - t < 1.5]
            if len(recent) > 3: p.setFont(QFont("SF Pro", 14, QFont.Bold)); p.setPen(QColor(255,200,50,200)); p.drawText(W//2 - 60, 36, f"🔥 {len(recent)} Streak!")
        if self._show_watermark and self._watermark_text:
            p.setFont(QFont("SF Pro", 11, QFont.Bold)); p.setPen(QColor(255,255,255,60))
            p.drawText(W - p.fontMetrics().horizontalAdvance(self._watermark_text) - 16, H - 20, self._watermark_text)
        p.setFont(QFont("SF Pro", 11)); p.setPen(QColor(255,255,255,100)); p.drawText(20, H - 16, f"{fmt_time(ct)} / {fmt_time(self.total_duration)}")

    def mousePressEvent(self, e):
        t = self._y2t(e.position().y()); self.seek_requested.emit(max(0, min(t, self.total_duration)))

# ═══════════════════════════════════════════════════════════════
#  EXPORT WORKER THREAD
# ═══════════════════════════════════════════════════════════════
class ExportWorker(QThread):
    progress = Signal(int, str); finished = Signal(str); error = Signal(str)

    def __init__(self, piano_roll, settings, output_path):
        super().__init__()
        self.piano_roll = piano_roll; self.settings = settings; self.output_path = output_path; self._cancel = False

    def cancel(self): self._cancel = True

    def run(self):
        try:
            s = self.settings; W = s.get('export_width', 1920); H = s.get('export_height', 1080)
            fps = s.get('export_fps', 60); codec = s.get('export_codec', 'libx264')
            crf = s.get('export_crf', 18); enc_preset = s.get('export_preset_enc', 'medium')
            fmt = s.get('export_format', 'mp4'); is_shorts = s.get('export_shorts_mode', False)
            pr = self.piano_roll; notes = pr.notes; total_dur = pr.total_duration
            if not notes or total_dur <= 0: self.error.emit("No notes to export"); return
            tmp_dir = tempfile.mkdtemp(); _register_temp(tmp_dir)
            total_frames = int(total_dur * fps); self.progress.emit(0, "Rendering frames...")
            offscreen = ProPianoRoll()
            offscreen.notes = list(notes); offscreen.total_duration = total_dur; offscreen.pps = pr.pps
            offscreen.min_note = pr.min_note; offscreen.max_note = pr.max_note; offscreen.kw = pr.kw
            offscreen.track_vis = dict(pr.track_vis); offscreen.preset = pr.preset
            offscreen.beat_times = list(pr.beat_times); offscreen.show_audio_mode = pr.show_audio_mode
            offscreen.audio_data = pr.audio_data; offscreen._show_watermark = pr._show_watermark
            offscreen._watermark_text = pr._watermark_text; offscreen._floating_labels = pr._floating_labels
            offscreen._note_splash = pr._note_splash; offscreen._light_streak = pr._light_streak
            offscreen._bloom = pr._bloom; offscreen._chromatic = pr._chromatic
            offscreen._scanlines = pr._scanlines; offscreen._film_grain = pr._film_grain
            offscreen._pulse_on_beat = pr._pulse_on_beat; offscreen._hdr_tonemap = pr._hdr_tonemap
            offscreen.particles.max_p = pr.particles.max_p; offscreen.setFixedSize(W, H)
            frame_paths = []
            for frame_idx in range(total_frames):
                if self._cancel: self.error.emit("Export cancelled"); return
                t = frame_idx / fps; offscreen.current_time = t; offscreen.anim_t = t
                active_set = set()
                for ne in notes:
                    if ne.start_time <= t < ne.start_time + ne.duration:
                        if (ne.note, ne.track) not in active_set:
                            active_set.add((ne.note, ne.track)); offscreen.trigger_note(ne)
                offscreen.update(); QApplication.processEvents()
                pixmap = QPixmap(offscreen.size()); offscreen.render(pixmap)
                frame_path = os.path.join(tmp_dir, f"frame_{frame_idx:06d}.png"); pixmap.save(frame_path, "PNG")
                frame_paths.append(frame_path)
                if frame_idx % max(1, total_frames // 100) == 0:
                    pct = int(frame_idx / total_frames * 100); self.progress.emit(pct, f"Rendering frame {frame_idx}/{total_frames}")
            if self._cancel: self.error.emit("Export cancelled"); return
            self.progress.emit(95, "Encoding video with FFmpeg...")
            ext = f".{fmt}"; output = self.output_path if self.output_path.endswith(ext) else self.output_path + ext
            audio_codec = audio_codec_for(ext)
            cmd = ['ffmpeg', '-y', '-framerate', str(fps), '-i', os.path.join(tmp_dir, 'frame_%06d.png')]
            if s.get('export_include_audio') and pr.audio_data and pr.audio_data.path: cmd += ['-i', pr.audio_data.path]
            cmd += ['-c:v', codec, '-crf', str(crf), '-preset', enc_preset, '-pix_fmt', 'yuv420p']
            if s.get('export_include_audio') and pr.audio_data and pr.audio_data.path: cmd += ['-c:a', audio_codec, '-b:a', '192k']
            if is_shorts: cmd += ['-vf', 'scale=1080:1920:flags=lanczos', '-aspect', '9:16']
            cmd.append(output)
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            if result.returncode != 0: self.error.emit(f"FFmpeg error: {result.stderr[:500]}"); return
            for fp in frame_paths:
                try: os.unlink(fp)
                except: pass
            try: os.rmdir(tmp_dir)
            except: pass
            self.progress.emit(100, "Export complete!"); self.finished.emit(output)
        except Exception as e: self.error.emit(str(e))

# ═══════════════════════════════════════════════════════════════
#  MAIN WINDOW (with Alignment & Audio Playback Fixes)
# ═══════════════════════════════════════════════════════════════
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🎹 Piano Roll Studio")
        self.setMinimumSize(1100, 700)
        self.setStyleSheet(self._main_stylesheet())

        self.playlist: List[PlaylistEntry] = []
        self.current_idx = -1
        self.playing = False
        self.speed = 1.0
        self.last_tick = 0.0
        self.export_worker = None

        self._build_ui()
        self._setup_timers()
        self._setup_shortcuts()
        
        # FIX: Initialize Media Player for audio playback
        self.media_player = None
        self.audio_output = None
        if HAS_QTMULTIMEDIA:
            try:
                self.media_player = QMediaPlayer(self)
                self.audio_output = QAudioOutput()
                self.media_player.setAudioOutput(self.audio_output)
                self.audio_output.setVolume(0.8)
            except Exception as e:
                print(f"QMediaPlayer init error: {e}")
                self.media_player = None
                self.audio_output = None

    def _main_stylesheet(self):
        return """
        QMainWindow { background: #0d0d1a; }
        QWidget { background: #0d0d1a; color: #e0e0e8; font-family: "SF Pro", "Segoe UI", Arial; }
        QPushButton { background: #1a1a3e; border: 1px solid #2a2a50; border-radius: 8px;
                      padding: 8px 16px; color: #e0e0e8; font-weight: 600; font-size: 12px; }
        QPushButton:hover { background: #2a2a5e; border-color: #4a4a8e; }
        QPushButton:pressed { background: #0d0d2e; }
        QPushButton#playBtn { background: #2a6a2a; border-color: #3a8a3a; font-size: 14px; padding: 10px 24px; }
        QPushButton#playBtn:hover { background: #3a8a3a; }
        QPushButton#settingsBtn { background: #3a3a6e; border-color: #5a5aae; }
        QPushButton#settingsBtn:hover { background: #4a4a8e; }
        QPushButton#exportBtn { background: #6a3a2a; border-color: #8a5a3a; }
        QPushButton#exportBtn:hover { background: #8a5a3a; }
        QComboBox { background: #1a1a3e; border: 1px solid #2a2a50; border-radius: 6px;
                    padding: 6px 12px; color: #e0e0e8; min-height: 24px; }
        QComboBox::drop-down { border: none; width: 20px; }
        QComboBox QAbstractItemView { background: #1a1a3e; color: #e0e0e8;
                                      selection-background-color: #3a3a7a; }
        QSlider::groove:horizontal { background: #1e1e40; height: 6px; border-radius: 3px; }
        QSlider::handle:horizontal { background: #6060c0; width: 16px; height: 16px;
                                     margin: -5px 0; border-radius: 8px; }
        QSlider::sub-page:horizontal { background: #4040a0; border-radius: 3px; }
        QLabel { color: #a0a0c0; font-size: 11px; }
        QListWidget { background: #12122a; border: 1px solid #2a2a50; border-radius: 6px;
                      color: #c0c0d8; font-size: 12px; }
        QListWidget::item:selected { background: #2a2a6e; }
        QListWidget::item:hover { background: #1a1a4e; }
        QStatusBar { background: #0a0a18; color: #606080; font-size: 11px; }
        """

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)

        # Left sidebar
        sidebar = QWidget(); sidebar.setFixedWidth(220)
        sb_layout = QVBoxLayout(sidebar); sb_layout.setContentsMargins(0,0,0,0); sb_layout.setSpacing(6)

        title_lbl = QLabel("🎹 Piano Roll Studio v8")
        title_lbl.setFont(QFont("SF Pro", 13, QFont.Bold))
        title_lbl.setStyleSheet("color: #a0a0ff; padding: 4px;")
        sb_layout.addWidget(title_lbl)

        btn_row = QHBoxLayout()
        add_btn = QPushButton("＋ Add"); add_btn.clicked.connect(self._add_files)
        add_folder_btn = QPushButton("📁 Folder"); add_folder_btn.clicked.connect(self._add_folder)
        btn_row.addWidget(add_btn); btn_row.addWidget(add_folder_btn)
        sb_layout.addLayout(btn_row)

        self.playlist_widget = QListWidget()
        self.playlist_widget.currentRowChanged.connect(self._on_playlist_select)
        sb_layout.addWidget(self.playlist_widget, 1)

        self.info_label = QLabel("No file loaded")
        self.info_label.setWordWrap(True)
        self.info_label.setStyleSheet("color: #8080a0; padding: 8px; background: #0f0f22; border-radius: 6px;")
        sb_layout.addWidget(self.info_label)

        main_layout.addWidget(sidebar)

        # Right: piano roll + controls
        right = QWidget()
        right_layout = QVBoxLayout(right); right_layout.setContentsMargins(0,0,0,0); right_layout.setSpacing(6)

        ctrl_bar = QWidget(); ctrl_bar.setFixedHeight(52)
        ctrl_layout = QHBoxLayout(ctrl_bar); ctrl_layout.setContentsMargins(4,4,4,4); ctrl_layout.setSpacing(8)

        self.play_btn = QPushButton("▶ Play"); self.play_btn.setObjectName("playBtn")
        self.play_btn.clicked.connect(self._toggle_play)
        ctrl_layout.addWidget(self.play_btn)

        self.stop_btn = QPushButton("■"); self.stop_btn.setFixedWidth(36)
        self.stop_btn.clicked.connect(self._stop)
        ctrl_layout.addWidget(self.stop_btn)

        self.seek_slider = QSlider(Qt.Horizontal)
        self.seek_slider.setRange(0, 1000)
        self.seek_slider.sliderMoved.connect(self._on_seek)
        ctrl_layout.addWidget(self.seek_slider, 1)

        self.time_label = QLabel("0:00 / 0:00")
        self.time_label.setFixedWidth(110)
        self.time_label.setAlignment(Qt.AlignCenter)
        ctrl_layout.addWidget(self.time_label)

        self.speed_label = QLabel("1.0×")
        self.speed_label.setFixedWidth(40)
        self.speed_label.setAlignment(Qt.AlignCenter)
        self.speed_label.setStyleSheet("color: #8080c0; font-weight: bold;")
        ctrl_layout.addWidget(self.speed_label)

        self.preset_combo = QComboBox()
        self.preset_combo.addItems(list(PRESETS.keys()))
        self.preset_combo.setCurrentText("Neon")
        self.preset_combo.currentTextChanged.connect(self._on_preset_changed)
        ctrl_layout.addWidget(self.preset_combo)

        settings_btn = QPushButton("⚙ Settings"); settings_btn.setObjectName("settingsBtn")
        settings_btn.clicked.connect(self._open_settings)
        ctrl_layout.addWidget(settings_btn)

        export_btn = QPushButton("📹 Export"); export_btn.setObjectName("exportBtn")
        export_btn.clicked.connect(self._open_export)
        ctrl_layout.addWidget(export_btn)

        right_layout.addWidget(ctrl_bar)

        roll_container = QWidget()
        roll_layout = QVBoxLayout(roll_container); roll_layout.setContentsMargins(0,0,0,0); roll_layout.setSpacing(0)

        self.piano_roll = ProPianoRoll()
        self.piano_roll.seek_requested.connect(self._seek_to_time)
        roll_layout.addWidget(self.piano_roll, 1)

        self.keyboard = ProKeyboard()
        self.keyboard.setFixedHeight(96)
        roll_layout.addWidget(self.keyboard)

        right_layout.addWidget(roll_container, 1)
        main_layout.addWidget(right, 1)

        self.statusBar().showMessage("Ready — Drop MIDI/audio files or click Add")

    def _setup_timers(self):
        self.play_timer = QTimer()
        self.play_timer.setInterval(16) 
        self.play_timer.timeout.connect(self._tick)

    def _setup_shortcuts(self):
        pass

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Space: self._toggle_play()
        elif e.key() == Qt.Key_Left: self._seek_relative(-2)
        elif e.key() == Qt.Key_Right: self._seek_relative(2)
        elif e.key() == Qt.Key_Up:
            self.speed = min(3.0, self.speed + 0.25); self.speed_label.setText(f"{self.speed:.2f}×")
            self.piano_roll._speed_mult = self.speed
            if self.media_player: self.media_player.setPlaybackRate(self.speed)
        elif e.key() == Qt.Key_Down:
            self.speed = max(0.25, self.speed - 0.25); self.speed_label.setText(f"{self.speed:.2f}×")
            self.piano_roll._speed_mult = self.speed
            if self.media_player: self.media_player.setPlaybackRate(self.speed)
        elif e.key() == Qt.Key_R: self._stop()
        elif e.key() == Qt.Key_S: self._open_settings()
        elif e.key() == Qt.Key_E: self._open_export()
        elif e.key() == Qt.Key_F:
            if self.isFullScreen(): self.showNormal()
            else: self.showFullScreen()
        elif Qt.Key_1 <= e.key() <= Qt.Key_9:
            idx = e.key() - Qt.Key_1; presets = list(PRESETS.keys())
            if idx < len(presets): self.preset_combo.setCurrentText(presets[idx])
        elif e.key() == Qt.Key_0:
            presets = list(PRESETS.keys())
            if len(presets) > 9: self.preset_combo.setCurrentText(presets[9])
        else: super().keyPressEvent(e)

    def _add_files(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Add Music Files", "", FILE_FILTER)
        for f in files: self._load_file(f)

    def _add_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Select Music Folder")
        if d:
            for f in scan_directory(d): self._load_file(f)

    def _load_file(self, filepath):
        ext = os.path.splitext(filepath)[1].lower()
        entry = PlaylistEntry(); entry.name = os.path.basename(filepath); entry.file_size = os.path.getsize(filepath)

        if ext in MIDI_EXTS:
            entry.midi_path = filepath; entry.file_type = "midi"
            try:
                notes, dur, info, tmap, tpb = parse_midi(filepath)
                entry.notes = notes; entry.duration = dur; entry.track_info = info; entry.tempo_map = tmap
                entry.ticks_per_beat = tpb; entry.note_count = len(notes)
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to parse MIDI:\n{e}"); return
        elif ext in AUDIO_EXTS:
            entry.audio_path = filepath; entry.file_type = "audio"
            try:
                ad = parse_audio_file(filepath); entry.audio_data = ad; entry.duration = ad.duration
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to load audio:\n{e}"); return
        else: return

        self.playlist.append(entry)
        item = QListWidgetItem(f"{'🎵' if entry.file_type=='midi' else '🎧'} {entry.name}")
        self.playlist_widget.addItem(item)

    def _on_playlist_select(self, idx):
        if idx < 0 or idx >= len(self.playlist): return
        self.current_idx = idx; entry = self.playlist[idx]; self._stop()

        if entry.file_type == "midi" and entry.notes:
            beats = generate_beat_times(entry.tempo_map, entry.ticks_per_beat, entry.duration)
            self.piano_roll.set_notes(entry.notes, entry.duration, beats)
            
            self.keyboard.min_note = self.piano_roll.min_note
            self.keyboard.max_note = self.piano_roll.max_note
            self.keyboard.set_active(set())

            if entry.audio_data is None:
                self.statusBar().showMessage("Synthesizing audio... (may take a moment)")
                QApplication.processEvents()
                audio_path = synthesize_midi_audio(entry.notes, entry.duration)
                if audio_path:
                    entry.audio_data = parse_audio_file(audio_path); entry.audio_path = audio_path
                    self.statusBar().showMessage("Audio synthesized!", 3000)
                else:
                    self.statusBar().showMessage("Audio synthesis failed (NumPy required)", 5000)

            # FIX: Feed audio file to QMediaPlayer
            if entry.audio_path and self.media_player:
                abs_path = os.path.abspath(entry.audio_path)
                if os.path.exists(abs_path): self.media_player.setSource(QUrl.fromLocalFile(abs_path))

            info_text = (f"📝 {entry.name}\n🎵 {entry.note_count} notes | {len(entry.track_info)} tracks\n⏱ {fmt_time(entry.duration)}")
                        
        elif entry.file_type == "audio" and entry.audio_data:
            self.piano_roll.set_audio_data(entry.audio_data)
            # FIX: Feed audio file to QMediaPlayer
            if entry.audio_path and self.media_player:
                abs_path = os.path.abspath(entry.audio_path)
                if os.path.exists(abs_path): self.media_player.setSource(QUrl.fromLocalFile(abs_path))
            info_text = (f"🎧 {entry.name}\n⏱ {fmt_time(entry.duration)}\n📊 {entry.audio_data.sample_rate}Hz {entry.audio_data.channels}ch")
        else:
            info_text = "Unsupported or empty file"

        self.info_label.setText(info_text)
        self.seek_slider.setRange(0, max(1, int(entry.duration * 10)))
        self.piano_roll.current_time = 0.0
        self._update_time_label(); self.piano_roll.update(); self.keyboard.update()

    def _toggle_play(self):
        if self.playing:
            self.play_timer.stop(); self.play_btn.setText("▶ Play"); self.playing = False
            if self.media_player: self.media_player.pause()
        else:
            if self.current_idx < 0 or not self.playlist: return
            entry = self.playlist[self.current_idx]
            if self.piano_roll.current_time >= entry.duration - 0.1:
                self.piano_roll.current_time = 0.0; self.piano_roll._combo = 0; self.piano_roll._streak_notes.clear()
            self.last_tick = time.time(); self.play_timer.start(); self.play_btn.setText("⏸ Pause"); self.playing = True
            
            # FIX: Play the audio source and sync speed
            if self.media_player and self.media_player.source():
                self.media_player.setPlaybackRate(self.speed)
                self.media_player.setPosition(int(self.piano_roll.current_time * 1000))
                self.media_player.play()

    def _stop(self):
        self.play_timer.stop(); self.playing = False; self.play_btn.setText("▶ Play")
        self.piano_roll.current_time = 0.0; self.piano_roll._combo = 0; self.piano_roll._streak_notes.clear()
        self.keyboard.set_active(set())
        self.seek_slider.setValue(0)
        self._update_time_label()
        self.piano_roll.update()
        self.keyboard.update()
        if self.media_player: self.media_player.stop()

    def _tick(self):
        if self.current_idx < 0: return
        entry = self.playlist[self.current_idx]
        
        now = time.time()
        dt = (now - self.last_tick) * self.speed
        self.last_tick = now
        
        old_time = self.piano_roll.current_time
        new_time = old_time + dt
        
        active_notes = set()
        note_colors = {}
        for ne in entry.notes:
            if ne.start_time <= new_time < ne.start_time + ne.duration:
                active_notes.add(ne.note)
                c = self.piano_roll._note_color(ne)
                note_colors[ne.note] = c
                if old_time < ne.start_time <= new_time:
                    self.piano_roll.trigger_note(ne)
                    self.keyboard.emit_ripple(ne.note, c)
        
        self.keyboard.set_active(active_notes, note_colors)
        self.keyboard.advance(dt)
        self.piano_roll.particles.update(dt)
        
        self.piano_roll.set_time(new_time)
        
        self.seek_slider.blockSignals(True)
        self.seek_slider.setValue(int(new_time * 10))
        self.seek_slider.blockSignals(False)
        self._update_time_label()
        
        if new_time >= entry.duration: self._stop()

    def _on_seek(self, val):
        if self.current_idx < 0: return
        t = val / 10.0; self.piano_roll.set_time(t); self._update_time_label()
        self.keyboard.set_active(set())
        if self.media_player: self.media_player.setPosition(int(t * 1000))

    def _seek_relative(self, seconds):
        if self.current_idx < 0: return
        entry = self.playlist[self.current_idx]
        t = max(0, min(self.piano_roll.current_time + seconds, entry.duration))
        self.piano_roll.set_time(t); self.seek_slider.setValue(int(t * 10)); self._update_time_label()
        if self.media_player: self.media_player.setPosition(int(t * 1000))

    def _seek_to_time(self, t):
        if self.current_idx < 0: return
        self.piano_roll.set_time(t); self.seek_slider.setValue(int(t * 10)); self._update_time_label()
        if self.media_player: self.media_player.setPosition(int(t * 1000))

    def _update_time_label(self):
        ct = self.piano_roll.current_time
        dur = self.piano_roll.total_duration if self.piano_roll.total_duration > 0 else 0
        self.time_label.setText(f"{fmt_time(ct)} / {fmt_time(dur)}")

    def _on_preset_changed(self, name):
        if name in PRESETS:
            pr = PRESETS[name]; self.piano_roll.preset = pr; self.keyboard.preset = pr
            self.piano_roll.starfield = StarField(pr.star_count); self.piano_roll.update(); self.keyboard.update()

    def _open_settings(self):
        dlg = SettingsDialog(self, self.preset_combo.currentText())
        dlg.settings_changed.connect(self._apply_settings); dlg.exec()

    def _apply_settings(self, settings):
        self.piano_roll.apply_settings(settings)
        pn = settings.get('preset_name')
        if pn and pn != self.preset_combo.currentText():
            self.preset_combo.blockSignals(True); self.preset_combo.setCurrentText(pn); self.preset_combo.blockSignals(False)
        speed = settings.get('default_speed', 1.0)
        self.speed = speed; self.speed_label.setText(f"{speed:.2f}×"); self.piano_roll._speed_mult = speed
        if self.media_player: self.media_player.setPlaybackRate(self.speed)
        self.statusBar().showMessage("✓ Settings applied", 3000)

    def _open_export(self):
        if self.current_idx < 0 or not self.playlist:
            QMessageBox.warning(self, "Export", "Please load a file first."); return
        
        entry = self.playlist[self.current_idx]
        dlg = SettingsDialog(self, self.preset_combo.currentText()); dlg.tabs.setCurrentIndex(3)
        if dlg.exec() != QDialog.Accepted: return
        
        settings = dlg.get_settings(); self._apply_settings(settings)
        
        fmt = settings.get('export_format', 'mp4')
        default_name = os.path.splitext(entry.name)[0] + f".{fmt}"
        
        output_path, _ = QFileDialog.getSaveFileName(
            self, "Export Video", default_name, f"Video (*.{fmt});;All (*)"
        )
        if not output_path: return
        
        progress = QProgressDialog("Exporting video...", "Cancel", 0, 100, self)
        progress.setWindowTitle("📹 Exporting"); progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0); progress.setAutoClose(False); progress.setAutoReset(False)
        progress.setStyleSheet("""
            QProgressDialog { background: #0f0f1e; color: #e0e0e8; }
            QLabel { color: #e0e0e8; }
            QPushButton { background: #2a2a6e; color: #e0e0e8; border: none; border-radius: 6px; padding: 6px 16px; }
            QProgressBar { background: #1a1a3e; border: 1px solid #2a2a50; border-radius: 4px; text-align: center; color: #e0e0e8; }
            QProgressBar::chunk { background: #4a4ac0; border-radius: 3px; }
        """)
        
        self.export_worker = ExportWorker(self.piano_roll, settings, output_path)
        self.export_worker.progress.connect(lambda v, s: (progress.setValue(v), progress.setLabelText(s)))
        self.export_worker.finished.connect(lambda p: (progress.close(), QMessageBox.information(self, "✓ Export Complete", f"Video saved to:\n{p}"), self.statusBar().showMessage(f"Exported: {p}", 5000)))
        self.export_worker.error.connect(lambda e: (progress.close(), QMessageBox.critical(self, "Export Error", str(e)), self.statusBar().showMessage(f"Export error: {e}", 5000)))
        progress.canceled.connect(self.export_worker.cancel)
        
        self.export_worker.start(); progress.exec()

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls(): e.acceptProposedAction()
    
    def dropEvent(self, e):
        for url in e.mimeData().urls():
            path = url.toLocalFile()
            if os.path.splitext(path)[1].lower() in ALL_EXTS: self._load_file(path)


# ═══════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    # Set dark palette
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(13, 13, 26))
    palette.setColor(QPalette.WindowText, QColor(224, 224, 232))
    palette.setColor(QPalette.Base, QColor(18, 18, 42))
    palette.setColor(QPalette.AlternateBase, QColor(26, 26, 62))
    palette.setColor(QPalette.ToolTipBase, QColor(224, 224, 232))
    palette.setColor(QPalette.ToolTipText, QColor(224, 224, 232))
    palette.setColor(QPalette.Text, QColor(224, 224, 232))
    palette.setColor(QPalette.Button, QColor(26, 26, 62))
    palette.setColor(QPalette.ButtonText, QColor(224, 224, 232))
    palette.setColor(QPalette.BrightText, QColor(255, 50, 50))
    palette.setColor(QPalette.Link, QColor(100, 100, 200))
    palette.setColor(QPalette.Highlight, QColor(66, 66, 150))
    palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
    app.setPalette(palette)

    window = MainWindow()
    window.show()
    
    # Cleanup on exit
    app.aboutToQuit.connect(_cleanup_temps)
    
    sys.exit(app.exec())