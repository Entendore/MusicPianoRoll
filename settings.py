import logging, json
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QScrollArea, QGroupBox, QFormLayout, QComboBox, QSlider, QSpinBox, QDoubleSpinBox, QCheckBox, QPushButton, QLineEdit, QFrame, QFileDialog)
from presets import PRESETS
from data import YOUTUBE_PRESETS, HAS_CUPY, HAS_NUMPY, HAS_NUMBA, HAS_PYDUB, HAS_QTMULTIMEDIA, HAS_MIDO
from parsers import check_ffmpeg
logger = logging.getLogger(__name__)

_ffmpeg_ok = None
def _ff(): 
    global _ffmpeg_ok
    if _ffmpeg_ok is None: _ffmpeg_ok = check_ffmpeg()
    return _ffmpeg_ok

class SettingsTab(QWidget):
    settings_changed = Signal(dict)
    def __init__(self, parent=None, preset_name="Neon"):
        super().__init__(parent); self._building = True; self._init_ui(preset_name); self._building = False

    def _init_ui(self, pn):
        ml = QVBoxLayout(self); ml.setContentsMargins(0,0,0,0); ml.setSpacing(0)
        scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setFrameShape(QFrame.NoFrame)
        c = QWidget(); ly = QVBoxLayout(c); ly.setSpacing(10); ly.setContentsMargins(16,16,16,16)
        ly.addWidget(self._build_preset_group(pn)); ly.addWidget(self._build_bg_group()); ly.addWidget(self._build_note_group())
        ly.addWidget(self._build_post_group()); ly.addWidget(self._build_playback_group()); ly.addWidget(self._build_export_group())
        ly.addWidget(self._build_sys_group()); ly.addStretch(); scroll.setWidget(c); ml.addWidget(scroll, 1)
        bb = QWidget(); bb.setObjectName("settingsBottomBar"); bl = QHBoxLayout(bb); bl.setContentsMargins(16,10,16,10)
        rst = QPushButton("Reset"); rst.clicked.connect(self._reset); bl.addWidget(rst); bl.addStretch()
        app = QPushButton("Apply"); app.setObjectName("applyBtn"); app.clicked.connect(self._apply); bl.addWidget(app); ml.addWidget(bb)

    def _sl(self, mn, mx, val):
        sl = QSlider(Qt.Horizontal); sl.setRange(mn, mx); sl.setValue(val)
        lbl = QLabel(str(val)); lbl.setMinimumWidth(36); lbl.setAlignment(Qt.AlignCenter)
        lbl.setStyleSheet("color: #68687a; font-weight: 600; font-size: 11px; background: #1c1c22; border-radius: 4px; padding: 2px 4px;")
        w = QWidget(); h = QHBoxLayout(w); h.setContentsMargins(0,0,0,0); h.setSpacing(8); h.addWidget(sl, 1); h.addWidget(lbl)
        sl.valueChanged.connect(lambda v, l=lbl: l.setText(str(v)))
        return {'widget': w, 'slider': sl, 'label': lbl}

    def _build_preset_group(self, pn):
        g = QGroupBox("Visual Preset"); fl = QFormLayout(g)
        self.preset_combo = QComboBox(); self.preset_combo.addItems(list(PRESETS.keys())); self.preset_combo.setCurrentText(pn)
        self.preset_combo.currentTextChanged.connect(self._on_preset_changed); fl.addRow("Preset:", self.preset_combo)
        exp_btn = QPushButton("Export Preset"); exp_btn.clicked.connect(self._export_preset)
        imp_btn = QPushButton("Import Preset"); imp_btn.clicked.connect(self._import_preset)
        row = QHBoxLayout(); row.addWidget(exp_btn); row.addWidget(imp_btn); fl.addRow("Sharing:", row)
        return g

    def _build_bg_group(self):
        g = QGroupBox("Background & Scene"); fl = QFormLayout(g)
        self.bg_pulse = self._sl(0, 100, 30); fl.addRow("Beat Pulse:", self.bg_pulse['widget'])
        self.star_count = self._sl(0, 500, 60); fl.addRow("Star Count:", self.star_count['widget'])
        self.vignette = self._sl(0, 100, 40); fl.addRow("Vignette:", self.vignette['widget'])
        self.reflection = QCheckBox("Reflection"); fl.addRow("", self.reflection)
        return g

    def _build_note_group(self):
        g = QGroupBox("Notes & Trails"); fl = QFormLayout(g)
        self.note_glow = self._sl(0, 300, 250); fl.addRow("Glow:", self.note_glow['widget'])
        self.trail_length = self._sl(0, 150, 40); fl.addRow("Trail Length:", self.trail_length['widget'])
        self.trail_opacity = self._sl(0, 100, 35); fl.addRow("Trail Opacity:", self.trail_opacity['widget'])
        self.light_streak = QCheckBox("Light Streak"); self.light_streak.setChecked(True); fl.addRow("", self.light_streak)
        self.pulse_on_beat = QCheckBox("Pulse on Beat"); self.pulse_on_beat.setChecked(True); fl.addRow("", self.pulse_on_beat)
        return g

    def _build_post_group(self):
        g = QGroupBox("Post-Processing"); fl = QFormLayout(g)
        self.scanlines = self._sl(0, 100, 5); fl.addRow("Scanlines:", self.scanlines['widget'])
        self.film_grain = self._sl(0, 100, 2); fl.addRow("Film Grain:", self.film_grain['widget'])
        return g

    def _build_playback_group(self):
        g = QGroupBox("Playback"); fl = QFormLayout(g)
        self.default_speed = QDoubleSpinBox(); self.default_speed.setRange(0.25, 3.0); self.default_speed.setValue(1.0); self.default_speed.setSingleStep(0.25)
        fl.addRow("Default Speed:", self.default_speed)
        self.pps_slider = self._sl(50, 600, 220); fl.addRow("Scroll Speed:", self.pps_slider['widget'])
        self.loop_check = QCheckBox("Loop Playback"); fl.addRow("", self.loop_check)
        self.auto_adv = QCheckBox("Auto-Advance Playlist"); fl.addRow("", self.auto_adv)
        return g

    def _build_export_group(self):
        g = QGroupBox("Export"); fl = QFormLayout(g)
        self.export_preset_combo = QComboBox()
        for ep in YOUTUBE_PRESETS: self.export_preset_combo.addItem(f"{ep.name} — {ep.description}", ep.name)
        fl.addRow("Format:", self.export_preset_combo)
        self.export_width = QSpinBox(); self.export_width.setRange(256, 7680); self.export_width.setValue(1920); fl.addRow("Width:", self.export_width)
        self.export_height = QSpinBox(); self.export_height.setRange(256, 4320); self.export_height.setValue(1080); fl.addRow("Height:", self.export_height)
        self.export_fps = QSpinBox(); self.export_fps.setRange(24, 120); self.export_fps.setValue(60); fl.addRow("FPS:", self.export_fps)
        return g

    def _build_sys_group(self):
        g = QGroupBox("System Info"); fl = QFormLayout(g)
        for n, a in [("mido", HAS_MIDO), ("NumPy", HAS_NUMPY), ("Numba", HAS_NUMBA), ("CuPy", HAS_CUPY), ("pydub", HAS_PYDUB), ("QtMedia", HAS_QTMULTIMEDIA), ("FFmpeg", _ff())]:
            lbl = QLabel(("✓ " if a else "✗ ") + n); lbl.setStyleSheet(f"color: {'#5aad6a' if a else '#b55a5a'};"); fl.addRow("", lbl)
        return g

    def _on_preset_changed(self, name):
        if self._building: return
        pr = PRESETS.get(name)
        if not pr: return
        self._building = True
        self.bg_pulse['slider'].setValue(int(pr.bg_pulse*100)); self.star_count['slider'].setValue(pr.star_count)
        self.vignette['slider'].setValue(int(pr.vignette*100)); self.reflection.setChecked(pr.reflection)
        self.note_glow['slider'].setValue(int(pr.note_glow*100)); self.trail_length['slider'].setValue(int(pr.trail_length))
        self.trail_opacity['slider'].setValue(int(pr.trail_opacity*100)); self.light_streak.setChecked(pr.light_streak)
        self.pulse_on_beat.setChecked(pr.pulse_on_beat); self.scanlines['slider'].setValue(int(pr.scanlines*1000))
        self.film_grain['slider'].setValue(int(pr.film_grain*2000))
        self._building = False

    def _export_preset(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export Preset", "", "JSON (*.json)")
        if not path: return
        data = {"name": self.preset_combo.currentText(), "bg_pulse": self.bg_pulse['slider'].value()/100.0, "star_count": self.star_count['slider'].value(), "vignette": self.vignette['slider'].value()/100.0}
        with open(path, 'w') as f: json.dump(data, f, indent=2)

    def _import_preset(self):
        path, _ = QFileDialog.getOpenFileName(self, "Import Preset", "", "JSON (*.json)")
        if not path: return
        with open(path) as f: data = json.load(f)
        self._building = True
        self.bg_pulse['slider'].setValue(int(data.get('bg_pulse', 0.3)*100))
        self.star_count['slider'].setValue(data.get('star_count', 60))
        self.vignette['slider'].setValue(int(data.get('vignette', 0.4)*100))
        self._building = False; self._apply()

    def _reset(self): self._on_preset_changed("Neon"); self._apply()
    def _apply(self):
        if self._building: return
        self.settings_changed.emit({
            'preset_name': self.preset_combo.currentText(),
            'bg_pulse': self.bg_pulse['slider'].value()/100.0, 'star_count': self.star_count['slider'].value(),
            'vignette': self.vignette['slider'].value()/100.0, 'reflection': self.reflection.isChecked(),
            'note_glow': self.note_glow['slider'].value()/100.0, 'trail_length': self.trail_length['slider'].value(),
            'trail_opacity': self.trail_opacity['slider'].value()/100.0, 'light_streak': self.light_streak.isChecked(),
            'pulse_on_beat': self.pulse_on_beat.isChecked(), 'scanlines': self.scanlines['slider'].value()/1000.0,
            'film_grain': self.film_grain['slider'].value()/2000.0,
            'default_speed': self.default_speed.value(), 'pps': self.pps_slider['slider'].value(),
            'loop': self.loop_check.isChecked(), 'auto_advance': self.auto_adv.isChecked(),
        })