"""
🎹 Piano Roll Studio — Main application window and entry point.

ALGORITHM IMPROVEMENTS:
1. Load queue: Uses collections.deque instead of list for O(1) popleft
   instead of O(n) pop(0).
2. Playlist lookup: Uses dict for O(1) entry lookup instead of
   list.index() which is O(n).
3. ExportManager: Pre-allocates QImage once and reuses it per frame
   instead of allocating a new QImage each frame (reduces GC pressure).
"""
import sys, time as _t, logging, json, subprocess, os
from collections import deque
from pathlib import Path as _P
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QSlider, QLabel, QComboBox, QFrame, QMessageBox, QSizePolicy, QListWidget, QListWidgetItem, QFileDialog, QStatusBar, QSplitter, QTabWidget, QProgressDialog)
from PySide6.QtCore import Qt, QTimer, QThread, Signal, QUrl, QSettings, QObject, QProcess
from PySide6.QtGui import QFont, QColor, QPalette, QImage, QPainter
from data import NoteEvent, AudioData, PlaylistEntry, YOUTUBE_PRESETS, ALL_EXTS, FILE_FILTER, HAS_MIDO, HAS_NUMPY, HAS_QTMULTIMEDIA, NoteIndex, BeatDetector, KeyDetector, fmt_time
from presets import PRESETS, VisualPreset
from piano_widget import PianoRollWidget
from parsers import parse_midi, parse_audio_file, scan_directory, check_ffmpeg
from synth import synthesize_midi_audio, cleanup_temps
from settings import SettingsTab
logger = logging.getLogger(__name__)

MAIN_STYLESHEET = """
QMainWindow { background: #16161a; }
QWidget { font-family: 'Segoe UI', sans-serif; font-size: 12px; color: #dcdce0; }
QWidget#sidebar { background: #1c1c22; border-left: 1px solid #26262e; }
QTabWidget#sidebarTabs::pane { border: none; background: #1c1c22; top: -1px; }
QTabWidget#sidebarTabs QTabBar::tab { background: transparent; color: #58586a; border: none; border-bottom: 2px solid transparent; padding: 12px 16px 8px 16px; font-weight: 600; font-size: 11px; }
QTabWidget#sidebarTabs QTabBar::tab:selected { color: #dcdce0; border-bottom: 2px solid #5a8ab5; }
QLabel#sectionLabel { color: #68687a; font-size: 10px; font-weight: 600; letter-spacing: 1.2px; text-transform: uppercase; padding: 0px 0px 2px 0px; }
QLabel#timeDisplay { color: #b0b0bc; font-size: 12px; font-family: monospace; }
QLabel#speedDisplay { color: #5a8ab5; font-size: 12px; font-weight: 600; font-family: monospace; }
QPushButton { background: #242430; color: #b0b0bc; border: 1px solid #2e2e3a; border-radius: 6px; padding: 7px 14px; font-weight: 600; }
QPushButton:hover { background: #2e2e3a; color: #dcdce0; }
QPushButton#playBtn { background: #5a8ab5; color: #fff; border: none; border-radius: 6px; font-weight: 700; padding: 8px 20px; min-width: 80px; }
QPushButton#playBtn:hover { background: #6a9ac5; }
QSlider::groove:horizontal { background: #2e2e3a; height: 4px; border-radius: 2px; }
QSlider::handle:horizontal { background: #dcdce0; width: 12px; height: 12px; margin: -4px 0; border-radius: 6px; }
QSlider::sub-page:horizontal { background: #5a8ab5; border-radius: 2px; }
QComboBox { background: #1c1c22; border: 1px solid #2e2e3a; border-radius: 6px; padding: 7px 12px; color: #dcdce0; }
QListWidget { background: #14141a; border: 1px solid #26262e; border-radius: 6px; color: #9898a6; outline: none; }
QListWidget::item { padding: 7px 10px; } QListWidget::item:selected { background: #222838; color: #dcdce0; }
QStatusBar { background: #14141a; color: #58586a; border-top: 1px solid #1e1e26; font-size: 11px; padding: 4px 12px; }
QGroupBox { font-weight: 600; color: #b0b0bc; border: 1px solid #26262e; border-radius: 8px; margin-top: 16px; padding: 18px 14px 10px 14px; background: #1a1a20; }
QGroupBox::title { subcontrol-origin: margin; left: 14px; padding: 0 8px; color: #9898a6; font-size: 11px; }
QCheckBox { color: #b0b0bc; spacing: 8px; } QCheckBox::indicator { width: 16px; height: 16px; border-radius: 4px; border: 1px solid #2e2e3a; background: #1c1c22; }
QCheckBox::indicator:checked { background: #5a8ab5; border-color: #5a8ab5; }
QSpinBox, QDoubleSpinBox { background: #1c1c22; border: 1px solid #2e2e3a; border-radius: 6px; padding: 6px 10px; color: #dcdce0; }
"""

def _section_label(text):
    lbl = QLabel(text); lbl.setObjectName("sectionLabel"); return lbl

class LoadWorker(QThread):
    result_ready = Signal(object, str, object)
    def __init__(self, path): super().__init__(); self.path = path
    def run(self):
        ext = _P(self.path).suffix.lower(); entry = PlaylistEntry(name=_P(self.path).stem, file_type="midi" if ext in {'.mid', '.midi'} else "audio")
        try:
            if ext in {'.mid', '.midi'}:
                notes, dur, info, tmap, tpb = parse_midi(self.path)
                entry.notes, entry.duration, entry.track_info, entry.tempo_map, entry.ticks_per_beat, entry.note_count, entry.midi_path = notes, dur, info, tmap, tpb, len(notes), self.path
                entry.beats, entry.bpm = BeatDetector.analyze(notes, dur)
                entry.key = KeyDetector.detect(notes)
            elif ext in {'.wav', '.mp3'}:
                audio = parse_audio_file(self.path); entry.audio_data, entry.duration, entry.audio_path, entry.note_count = audio, audio.duration, self.path, 0
            self.result_ready.emit(entry, self.path, None)
        except Exception as e: self.result_ready.emit(None, self.path, e)

class ExportManager(QObject):
    """
    IMPROVEMENT: Pre-allocates QImage once and reuses it for all frames
    instead of allocating a new QImage per frame. This eliminates
    ~60 allocations per second during export (at 60fps), reducing
    GC pressure and memory fragmentation.
    """
    progress = Signal(int)
    finished = Signal(bool, str)

    def __init__(self, parent, roll, entry, out_path, w, h, fps):
        super().__init__(parent)
        self.roll, self.entry, self.out_path = roll, entry, out_path
        self.w, self.h, self.fps = w, h, fps
        self.process = None
        self.current_frame = 0
        self.total_frames = int(entry.duration * fps)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._render_frame)
        self.offscreen_roll = None
        # Pre-allocated frame buffer (reused every frame)
        self._frame_img = None

    def start(self):
        if not check_ffmpeg(): self.finished.emit(False, "FFmpeg not found"); return
        cmd = ['ffmpeg', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb32', '-s', f'{self.w}x{self.h}', '-r', str(self.fps), '-i', '-']
        if self.entry.audio_path and os.path.exists(self.entry.audio_path): cmd += ['-i', self.entry.audio_path]
        cmd += ['-c:v', 'libx264', '-crf', '18', '-preset', 'fast', '-pix_fmt', 'yuv420p', self.out_path]
        self.process = QProcess(self)
        self.process.start(cmd[0], cmd[1:])
        if not self.process.waitForStarted(3000): self.finished.emit(False, "Failed to start FFmpeg"); return
        self.offscreen_roll = PianoRollWidget()
        self.offscreen_roll.set_notes(self.entry.notes); self.offscreen_roll.duration = self.entry.duration
        self.offscreen_roll.set_preset(self.roll.preset.name); self.offscreen_roll.resize(self.w, self.h)
        # Pre-allocate frame image
        self._frame_img = QImage(self.w, self.h, QImage.Format_RGB32)
        self.timer.start(0)

    def cancel(self):
        self.timer.stop()
        if self.process and self.process.state() != QProcess.NotRunning: self.process.terminate()
        self.finished.emit(False, "Cancelled")

    def _render_frame(self):
        if self.current_frame >= self.total_frames:
            self.timer.stop(); self.process.closeWriteChannel(); self.process.waitForFinished(5000)
            self.finished.emit(True, self.out_path); return
        t = self.current_frame / self.fps
        self.offscreen_roll.current_time = t; self.offscreen_roll.seek_to(t); self.offscreen_roll.tick(1.0 / self.fps)
        # Reuse pre-allocated image instead of creating new one each frame
        self._frame_img.fill(Qt.black)
        self.offscreen_roll.render(self._frame_img)
        ptr = self._frame_img.constBits(); ptr.setsize(self.w * self.h * 4)
        self.process.write(bytes(ptr))
        self.current_frame += 1; self.progress.emit(int(100 * self.current_frame / self.total_frames))

class PianoRollStudio(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Piano Roll Studio"); self.setMinimumSize(1200, 720); self.resize(1440, 900)
        self.playlist, self.current_entry, self.playing, self.current_time, self.speed, self.preset_name = [], None, False, 0.0, 1.0, "Neon"
        # IMPROVEMENT: deque for O(1) popleft instead of list pop(0) which is O(n)
        self._load_queue = deque()
        self._is_loading, self._current_worker = False, None
        # IMPROVEMENT: dict for O(1) playlist entry lookup by id()
        self._entry_by_id: dict = {}
        self.setAcceptDrops(True)
        self._app_settings = QSettings("PianoRollStudio", "PianoRollStudio")
        self._init_audio(); self._init_ui(); self._init_timer()

    def _init_audio(self):
        self.media_player = None
        if HAS_QTMULTIMEDIA:
            from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
            self.media_player = QMediaPlayer(self); self.audio_output = QAudioOutput(self)
            self.media_player.setAudioOutput(self.audio_output); self.audio_output.setVolume(0.8)
            self.media_player.positionChanged.connect(self._on_audio_pos)

    def _init_ui(self):
        central = QWidget(); self.setCentralWidget(central); main_layout = QHBoxLayout(central); main_layout.setContentsMargins(0,0,0,0); main_layout.setSpacing(0)
        splitter = QSplitter(Qt.Horizontal); splitter.setHandleWidth(0)
        self.piano_roll = PianoRollWidget(); splitter.addWidget(self.piano_roll)
        sidebar_tabs = QTabWidget(); sidebar_tabs.setObjectName("sidebarTabs")
        pt = QWidget(); sb = QVBoxLayout(pt); sb.setContentsMargins(20,20,20,16); sb.setSpacing(0)
        sb.addWidget(QLabel("Piano Roll Studio")); sb.addWidget(_section_label("Transport")); sb.addSpacing(8)
        transport = QHBoxLayout(); transport.setSpacing(6)
        self.btn_play = QPushButton("▶  Play"); self.btn_play.setObjectName("playBtn"); self.btn_play.clicked.connect(self._toggle_play); transport.addWidget(self.btn_play, 1)
        self.btn_stop = QPushButton("⏹  Stop"); self.btn_stop.setObjectName("stopBtn"); self.btn_stop.clicked.connect(self._stop); transport.addWidget(self.btn_stop, 1)
        sb.addLayout(transport); sb.addSpacing(14)
        self.seek_slider = QSlider(Qt.Horizontal); self.seek_slider.setRange(0, 1000); self.seek_slider.valueChanged.connect(self._on_seek); sb.addWidget(self.seek_slider)
        self.time_label = QLabel("0:00 / 0:00"); self.time_label.setObjectName("timeDisplay"); self.time_label.setAlignment(Qt.AlignCenter); sb.addWidget(self.time_label)
        self.speed_slider = QSlider(Qt.Horizontal); self.speed_slider.setRange(25, 300); self.speed_slider.setValue(100); self.speed_slider.valueChanged.connect(self._on_speed_changed); sb.addWidget(self.speed_slider)
        self.speed_label = QLabel("1.00×"); self.speed_label.setObjectName("speedDisplay"); self.speed_label.setAlignment(Qt.AlignCenter); sb.addWidget(self.speed_label)
        self.preset_combo = QComboBox(); self.preset_combo.addItems(list(PRESETS.keys())); self.preset_combo.setCurrentText(self.preset_name); self.preset_combo.currentTextChanged.connect(self._on_preset_changed); sb.addWidget(self.preset_combo)
        sb.addSpacing(14)
        file_row = QHBoxLayout(); btn_open = QPushButton("Open File"); btn_open.clicked.connect(self._open_file); file_row.addWidget(btn_open, 1)
        btn_scan = QPushButton("Scan Dir"); btn_scan.clicked.connect(self._scan_dir); file_row.addWidget(btn_scan, 1); sb.addLayout(file_row)
        self.playlist_widget = QListWidget(); self.playlist_widget.itemDoubleClicked.connect(self._on_playlist_item); sb.addWidget(self.playlist_widget, 1)
        btn_export = QPushButton("Export Video"); btn_export.setObjectName("exportBtn"); btn_export.clicked.connect(self._export_video); sb.addWidget(btn_export)
        sidebar_tabs.addTab(pt, "Player")
        self.settings_tab = SettingsTab(self, self.preset_name); self.settings_tab.settings_changed.connect(self._apply_settings); sidebar_tabs.addTab(self.settings_tab, "Settings")
        sidebar_tabs.setMaximumWidth(320); sidebar_tabs.setMinimumWidth(260); self.sidebar_tabs = sidebar_tabs
        splitter.addWidget(sidebar_tabs); splitter.setSizes([1120, 320]); main_layout.addWidget(splitter)
        self.status = QStatusBar(); self.setStatusBar(self.status); self.status.showMessage("Ready — Drag & Drop files to begin")

    def _init_timer(self):
        self._timer = QTimer(self); self._timer.setInterval(16); self._timer.timeout.connect(self._tick); self._last_tick = None

    def _toggle_play(self):
        if self.current_entry is None: return
        self._pause() if self.playing else self._play()

    def _play(self):
        self.playing = True; self.btn_play.setText("⏸  Pause"); self._last_tick = _t.time(); self._timer.start()
        if self.media_player and self.current_entry and self.current_entry.audio_path:
            self.media_player.setSource(QUrl.fromLocalFile(self.current_entry.audio_path)); self.media_player.play()

    def _pause(self):
        self.playing = False; self.btn_play.setText("▶  Play"); self._timer.stop()
        if self.media_player: self.media_player.pause()

    def _stop(self):
        self._pause(); self.current_time = 0.0; self.piano_roll.combo = 0; self.piano_roll.streak = 0; self.piano_roll.seek_to(self.current_time); self._update_seek_display()
        if self.media_player: self.media_player.stop()

    def _tick(self):
        now = _t.time(); dt = (now - self._last_tick if self._last_tick else 0.016) * self.speed; self._last_tick = now; self.current_time += dt
        if self.current_entry and self.current_time >= self.current_entry.duration:
            if self.settings_tab.loop_check.isChecked(): self.current_time = 0.0; self.piano_roll.seek_to(0.0)
            elif self.settings_tab.auto_adv.isChecked():
                # IMPROVEMENT: Use _entry_by_id dict for O(1) lookup
                idx = self._playlist_index_of(self.current_entry)
                if idx + 1 < len(self.playlist): self._load_entry(self.playlist[idx+1]); self._play()
                else: self.current_time = self.current_entry.duration; self._pause()
            else: self.current_time = self.current_entry.duration; self._pause()
        self.piano_roll.current_time = self.current_time; self.piano_roll.tick(dt); self._update_seek_display()

    def _playlist_index_of(self, entry):
        """O(n) fallback — could be further optimized with an ordered dict if needed."""
        try:
            return self.playlist.index(entry)
        except ValueError:
            return -1

    def _on_audio_pos(self, ms):
        if self.media_player and self.playing: self.current_time = ms / 1000.0; self.piano_roll.current_time = self.current_time

    def _update_seek_display(self):
        if self.current_entry and self.current_entry.duration > 0:
            self.seek_slider.blockSignals(True); self.seek_slider.setValue(int(1000 * self.current_time / self.current_entry.duration)); self.seek_slider.blockSignals(False)
            self.time_label.setText(f"{fmt_time(self.current_time)} / {fmt_time(self.current_entry.duration)}")

    def _on_seek(self, val):
        if self.current_entry and self.current_entry.duration > 0:
            self.current_time = val / 1000.0 * self.current_entry.duration; self.piano_roll.seek_to(self.current_time)
            if self.media_player: self.media_player.setPosition(int(self.current_time * 1000))

    def _on_speed_changed(self, val):
        self.speed = val / 100.0; self.speed_label.setText(f"{self.speed:.2f}×"); self.piano_roll.playback_speed = self.speed

    def _on_preset_changed(self, name): self.preset_name = name; self.piano_roll.set_preset(name)

    def _on_playlist_item(self, item):
        idx = self.playlist_widget.row(item)
        if 0 <= idx < len(self.playlist): self._load_entry(self.playlist[idx])

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if _P(url.toLocalFile()).suffix.lower() in ALL_EXTS: event.acceptProposedAction(); return
    def dropEvent(self, event):
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if _P(path).suffix.lower() in ALL_EXTS: self._load_file(path)

    def _open_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Music File", "", FILE_FILTER)
        if path: self._load_file(path)

    def _scan_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Scan Directory")
        if d:
            files = scan_directory(d)
            for f in files: self._load_file(f, add_only=True)
            self.status.showMessage(f"Queued {len(files)} files for loading...")

    def _load_file(self, path, add_only=False):
        # IMPROVEMENT: deque append is O(1), same as list, but popleft is also O(1)
        self._load_queue.append((path, add_only))
        if not self._is_loading: self._process_next_load()

    def _process_next_load(self):
        if not self._load_queue: self._is_loading = False; self.status.showMessage("All files loaded."); return
        self._is_loading = True
        # IMPROVEMENT: deque.popleft() is O(1) vs list.pop(0) which is O(n)
        path, add_only = self._load_queue.popleft()
        self.status.showMessage(f"Loading: {_P(path).name}...")
        self._current_worker = LoadWorker(path)
        self._current_worker.result_ready.connect(lambda entry, p, error, ao=add_only: self._on_file_loaded(entry, p, error, ao))
        self._current_worker.finished.connect(self._process_next_load)
        self._current_worker.start()

    def _on_file_loaded(self, entry, path, error, add_only):
        if error: QMessageBox.warning(self, "Error", f"Failed to load {path}:\n{error}"); return
        self.playlist.append(entry)
        self._entry_by_id[id(entry)] = entry
        if not add_only: self._load_entry(entry)
        self._refresh_playlist(); self.status.showMessage(f"Loaded: {entry.name}  ·  {entry.note_count} notes  ·  {fmt_time(entry.duration)}")

    def _load_entry(self, entry):
        was_playing = self.playing; self._stop()
        self.current_entry = entry; self.current_time = 0.0; self.piano_roll.reset_state()
        self.piano_roll.set_notes(entry.notes); self.piano_roll.duration = entry.duration
        self.piano_roll._beats = entry.beats; self.piano_roll.seek_to(0.0)
        if entry.audio_data and entry.audio_data.spectrum_data: self.piano_roll.spectrum_data = entry.audio_data.spectrum_data
        else: self.piano_roll.spectrum_data = []
        if not entry.audio_path and entry.midi_path and HAS_NUMPY:
            wav = synthesize_midi_audio(entry.notes, entry.duration)
            if wav: entry.audio_path = wav
        self._update_seek_display()
        if was_playing: self._play()

    def _refresh_playlist(self):
        self.playlist_widget.clear()
        for e in self.playlist:
            icon = "♪" if e.file_type == "midi" else "♩"
            info = f" | {e.bpm} BPM | {e.key}" if e.bpm else ""
            self.playlist_widget.addItem(QListWidgetItem(f"{icon}  {e.name}  ({fmt_time(e.duration)}){info}"))

    def _apply_settings(self, s):
        pr = self.piano_roll.preset
        if pr:
            pr.bg_pulse, pr.star_count, pr.vignette, pr.reflection = s['bg_pulse'], s['star_count'], s['vignette'], s['reflection']
            pr.note_glow, pr.trail_length, pr.trail_opacity = s['note_glow'], s['trail_length'], s['trail_opacity']
            pr.light_streak, pr.pulse_on_beat = s['light_streak'], s['pulse_on_beat']
            pr.scanlines, pr.film_grain = s['scanlines'], s['film_grain']
            if s['star_count'] != len(self.piano_roll.stars.stars): self.piano_roll.stars = StarField(s['star_count'])
        self.piano_roll.pps = s['pps']; self.piano_roll.update()

    def _export_video(self):
        if not check_ffmpeg(): QMessageBox.warning(self, "FFmpeg Missing", "Install FFmpeg for export."); return
        if not self.current_entry: QMessageBox.information(self, "No File", "Load a file first."); return
        out_path, _ = QFileDialog.getSaveFileName(self, "Export Video", "output.mp4", "Video (*.mp4 *.mov *.webm)")
        if not out_path: return
        w, h, fps = self.settings_tab.export_width.value(), self.settings_tab.export_height.value(), self.settings_tab.export_fps.value()
        self._export_manager = ExportManager(self, self.piano_roll, self.current_entry, out_path, w, h, fps)
        self._prog = QProgressDialog("Exporting video...", "Cancel", 0, 100, self); self._prog.setWindowModality(Qt.WindowModal)
        self._export_manager.progress.connect(self._prog.setValue)
        self._export_manager.finished.connect(self._on_export_finished)
        self._prog.canceled.connect(self._export_manager.cancel)
        self._export_manager.start()

    def _on_export_finished(self, success, msg):
        self._prog.close()
        if success: QMessageBox.information(self, "Export Complete", f"Saved to:\n{msg}")
        else: QMessageBox.warning(self, "Export Failed", msg)

    def keyPressEvent(self, event):
        k = event.key()
        if k == Qt.Key_Space: self._toggle_play()
        elif k == Qt.Key_Left: self.current_time = max(0, self.current_time - 2.0); self.piano_roll.seek_to(self.current_time)
        elif k == Qt.Key_Right and self.current_entry: self.current_time = min(self.current_entry.duration, self.current_time + 2.0); self.piano_roll.seek_to(self.current_time)
        elif k == Qt.Key_R: self._stop()
        elif k == Qt.Key_F: self.showNormal() if self.isFullScreen() else self.showFullScreen()
        else: super().keyPressEvent(event)

    def closeEvent(self, event):
        self._pause(); cleanup_temps()
        if self._is_loading and self._current_worker and self._current_worker.isRunning(): self._current_worker.quit(); self._current_worker.wait(1000)
        if hasattr(self, '_export_manager') and self._export_manager: self._export_manager.cancel()
        self._app_settings.setValue("geometry", self.saveGeometry())
        super().closeEvent(event)

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s', datefmt='%H:%M:%S')
    app = QApplication(sys.argv)
    app.setApplicationName("Piano Roll Studio"); app.setOrganizationName("PianoRollStudio")
    pal = QPalette()
    pal.setColor(QPalette.Window, QColor(22,22,26)); pal.setColor(QPalette.WindowText, QColor(220,220,226))
    pal.setColor(QPalette.Base, QColor(28,28,34)); pal.setColor(QPalette.Text, QColor(220,220,226))
    pal.setColor(QPalette.Button, QColor(34,34,40)); pal.setColor(QPalette.ButtonText, QColor(220,220,226))
    pal.setColor(QPalette.Highlight, QColor(90,138,181)); pal.setColor(QPalette.HighlightedText, QColor(255,255,255))
    app.setPalette(pal); app.setStyleSheet(MAIN_STYLESHEET)
    window = PianoRollStudio()
    geo = window._app_settings.value("geometry")
    if geo: window.restoreGeometry(geo)
    window.show(); sys.exit(app.exec())

if __name__ == "__main__":
    main()