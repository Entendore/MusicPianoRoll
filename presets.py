from dataclasses import dataclass
from typing import List, Tuple, Union
from PySide6.QtGui import QColor
from data import color_c as _c

@dataclass
class VisualPreset:
    name: str; bg1: QColor; bg2: QColor; lane_white: QColor; lane_black: QColor; grid_line: QColor; c_line: QColor
    note_colors: Union[List[QColor], str]; note_glow: float; note_radius: float; note_border: bool
    trail_length: float; trail_opacity: float; particle_count: int; particle_speed: float; particle_size: float
    kb_wt: QColor; kb_wb: QColor; kb_bt: QColor; kb_bb: QColor; kb_border: QColor; kb_active_glow: float
    show_spectrum: bool; spectrum_style: str; show_combo: bool; show_streaks: bool; show_speed: bool
    screen_shake: float; bg_pulse: float; camera_zoom_range: Tuple[float, float]
    is_rainbow: bool = False; star_count: int = 0; star_color: QColor = QColor(255,255,255,15)
    reflection: bool = False; vignette: float = 0.0; bloom: float = 0.0; chromatic_aberration: float = 0.0
    scanlines: float = 0.0; film_grain: float = 0.0; light_streak: bool = False; pulse_on_beat: bool = False
    note_splash: bool = False; floating_labels: bool = False; hdr_tonemap: float = 0.0

PRESETS = {}
PRESETS["Minimal"] = VisualPreset(name="Minimal", bg1=_c(18,18,28), bg2=_c(12,12,22), lane_white=_c(28,28,42), lane_black=_c(16,16,26), grid_line=_c(38,38,55), c_line=_c(50,50,70), note_colors=[_c(100,160,255),_c(255,100,120),_c(100,230,160),_c(255,200,80),_c(180,120,255),_c(255,160,80)], note_glow=0.4, note_radius=4, note_border=False, trail_length=0, trail_opacity=0, particle_count=0, particle_speed=0, particle_size=0, kb_wt=_c(245,245,250), kb_wb=_c(220,220,230), kb_bt=_c(55,55,65), kb_bb=_c(22,22,30), kb_border=_c(160,160,175), kb_active_glow=0.6, show_spectrum=False, spectrum_style='bars', show_combo=False, show_streaks=False, show_speed=False, screen_shake=0, bg_pulse=0, camera_zoom_range=(1.0,1.0))
PRESETS["Neon"] = VisualPreset(name="Neon", bg1=_c(2,2,10), bg2=_c(0,0,5), lane_white=_c(8,8,22), lane_black=_c(3,3,10), grid_line=_c(15,15,35), c_line=_c(25,25,55), note_colors=[_c(0,255,100),_c(255,0,110),_c(0,180,255),_c(255,255,0),_c(255,0,255),_c(0,255,255)], note_glow=2.5, note_radius=3, note_border=True, trail_length=40, trail_opacity=0.35, particle_count=12, particle_speed=80, particle_size=4, kb_wt=_c(15,15,30), kb_wb=_c(8,8,18), kb_bt=_c(5,5,12), kb_bb=_c(2,2,6), kb_border=_c(0,100,50), kb_active_glow=2.0, show_spectrum=True, spectrum_style='bars', show_combo=True, show_streaks=True, show_speed=True, screen_shake=2.0, bg_pulse=0.3, camera_zoom_range=(1.0,1.05), star_count=60, star_color=_c(0,255,100,12), vignette=0.4, light_streak=True, pulse_on_beat=True, note_splash=True, floating_labels=True)
PRESETS["Rainbow"] = VisualPreset(name="Rainbow", bg1=_c(8,5,18), bg2=_c(4,2,12), lane_white=_c(18,12,32), lane_black=_c(10,6,20), grid_line=_c(30,20,50), c_line=_c(50,35,75), note_colors='rainbow', note_glow=2.0, note_radius=5, note_border=True, trail_length=60, trail_opacity=0.4, particle_count=10, particle_speed=60, particle_size=3.5, kb_wt=_c(35,25,55), kb_wb=_c(22,15,40), kb_bt=_c(45,25,65), kb_bb=_c(18,10,30), kb_border=_c(90,60,140), kb_active_glow=1.8, show_spectrum=True, spectrum_style='circular', show_combo=True, show_streaks=True, show_speed=True, screen_shake=3.0, bg_pulse=0.5, camera_zoom_range=(0.98,1.06), is_rainbow=True, star_count=100, star_color=_c(200,100,255,18), reflection=True, vignette=0.3, light_streak=True, pulse_on_beat=True, note_splash=True, scanlines=0.02, film_grain=0.01)