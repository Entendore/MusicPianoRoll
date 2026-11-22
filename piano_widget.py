"""
Custom QWidget that renders the enhanced piano roll.
Major visual upgrades: aurora, light pillars, crystal notes,
connection arcs, depth fog, velocity-aware rendering.
"""
import math, random, bisect, logging
from PySide6.QtWidgets import QWidget, QSizePolicy
from PySide6.QtCore import Qt, QRectF, QPointF
from PySide6.QtGui import (QPainter, QBrush, QPen, QFont, QColor, QLinearGradient,
                            QRadialGradient, QPixmap, QPainterPath, QImage, QConicalGradient)
from data import is_black, note_name, fmt_time, color_hsv, cached_color_hsv, NoteIndex, HAS_NUMPY
if HAS_NUMPY:
    from data import np
from presets import PRESETS, VisualPreset
from effects import (ParticleEngine, StarField, RippleSystem, NoteSplashSystem,
                      AuroraRenderer, LightPillarSystem, ConnectionArcSystem)

logger = logging.getLogger(__name__)
MIN_NOTE, MAX_NOTE, NOTE_RANGE = 21, 108, 87

CHANNEL_COLORS = [
    QColor(65, 105, 225), QColor(50, 205, 50), QColor(255, 99, 71), QColor(255, 215, 0),
    QColor(186, 85, 211), QColor(0, 206, 209), QColor(255, 105, 180), QColor(244, 164, 96),
    QColor(127, 255, 212), QColor(220, 20, 60), QColor(100, 149, 237), QColor(255, 165, 0),
    QColor(144, 238, 144), QColor(218, 112, 214), QColor(64, 224, 208), QColor(245, 222, 179)
]


class SmartCamera:
    def __init__(self):
        self.center_note, self.zoom = 64.0, 1.0
        self.target_center, self.target_zoom = 64.0, 1.0

    def update(self, dt, active_notes):
        if active_notes:
            self.target_center = (min(active_notes) + max(active_notes)) / 2.0
            span = max(active_notes) - min(active_notes)
            self.target_zoom = 1.3 if span < 12 else (1.1 if span < 36 else 0.95)
        t = 1.0 - math.exp(-3.0 * dt)
        self.center_note += (self.target_center - self.center_note) * t
        self.zoom += (self.target_zoom - self.zoom) * t


class PianoRollWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(640, 360)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.notes, self.current_time, self.duration, self.pps, self.playback_speed = [], 0.0, 0.0, 220, 1.0
        self.preset = PRESETS.get("Neon") or PRESETS["Minimal"]
        self.orientation = 'vertical'
        self.particles = ParticleEngine(3000)
        self.stars = StarField(60)
        self.ripples = RippleSystem()
        self.splashes = NoteSplashSystem()
        # ── New effect systems ──
        self.aurora = AuroraRenderer(5)
        self.light_pillars = LightPillarSystem()
        self.arcs = ConnectionArcSystem()
        self.active_notes, self._prev_active, self.combo, self.streak = set(), set(), 0, 0
        self.spectrum_data, self._anim_time, self._note_index = [], 0.0, None
        self.camera = SmartCamera()
        self.watermark_text = ""
        self._beats = []
        self._zoom_level, self._dragging, self._last_mouse_x = 1.0, False, 0
        self._smooth_spectrum = []
        self._active_events_map: dict = {}
        self._scanline_cache = None
        self._scanline_cache_key = None
        # Track previous active for pillar deactivation
        self._prev_active_for_pillars = set()

    def set_preset(self, name):
        if name in PRESETS:
            self.preset = PRESETS[name]
            self.stars = StarField(self.preset.star_count)
            self._scanline_cache = None

    def set_notes(self, notes):
        self.notes = notes
        self._note_index = NoteIndex(notes)

    def seek_to(self, time):
        self.current_time = time
        self.active_notes = self._note_index.active_at(time) if self._note_index else set()
        self._active_events_map = self._note_index.active_events_map_at(time) if self._note_index else {}
        self.update()

    def reset_state(self):
        self.active_notes, self._prev_active = set(), set()
        self._prev_active_for_pillars = set()
        self._active_events_map = {}
        self.combo, self.streak, self.current_time, self._anim_time = 0, 0, 0.0, 0.0
        self.particles.active_count = 0
        self.ripples.ripples.clear()
        self.splashes.splashes.clear()
        self.light_pillars.pillars.clear()
        self.arcs.arcs.clear()
        self._smooth_spectrum = []

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
        w, h, pr = self.width(), self.height(), self.preset
        if pr is None:
            painter.fillRect(0, 0, w, h, QColor(0, 0, 0))
            painter.end(); return

        # ── 1. Background gradient ──
        bg = QLinearGradient(0, 0, 0, h)
        bg.setColorAt(0, pr.bg1); bg.setColorAt(1, pr.bg2)
        painter.fillRect(0, 0, w, h, QBrush(bg))

        # ── 2. Aurora (behind stars) ──
        aurora_intensity = getattr(pr, 'aurora_intensity', 0.0)
        if aurora_intensity > 0:
            self.aurora.draw(painter, w, h, self._anim_time, aurora_intensity)

        # ── 3. Stars ──
        if pr.star_count > 0:
            self.stars.draw(painter, w, h, self._anim_time, pr.star_color, pr.is_rainbow)

        # ── 4. Main roll content ──
        if self.orientation == 'vertical':
            self._paint_vertical(painter, w, h, pr)
        else:
            self._paint_horizontal(painter, w, h, pr)

        # ── 5. Post-processing ──
        if pr.vignette > 0: self._draw_vignette(painter, w, h, pr)
        if pr.scanlines > 0: self._draw_scanlines(painter, w, h, pr)
        if pr.film_grain > 0: self._draw_film_grain(painter, w, h, pr)
        if self.watermark_text: self._draw_watermark(painter, w, h)

        # ── 6. Overlay effects ──
        self.particles.draw(painter)
        self.arcs.draw(painter)
        self.ripples.draw(painter)
        self.splashes.draw(painter)
        self._paint_hud(painter, w, h, pr)
        painter.end()

    # ══════════════════════════════════════════════════════════
    #  VERTICAL RENDERING — Enhanced
    # ══════════════════════════════════════════════════════════
    def _paint_vertical(self, painter, w, h, pr):
        kb_h = 120
        roll_h = h - kb_h
        lane_w = w / NOTE_RANGE
        t = self.current_time
        y_hit = roll_h * 0.88

        # ── Lane backgrounds with depth ──
        roll_bg = QLinearGradient(0, 0, 0, roll_h)
        roll_bg.setColorAt(0, pr.bg2.darker(115))
        roll_bg.setColorAt(0.6, QColor(pr.bg2.red() + 5, pr.bg2.green() + 5, pr.bg2.blue() + 12))
        roll_bg.setColorAt(1, pr.bg1.lighter(108))
        painter.fillRect(0, 0, w, roll_h, QBrush(roll_bg))

        # Subtle lane shading
        for n in range(MIN_NOTE, MAX_NOTE + 1):
            x = (n - MIN_NOTE) * lane_w
            if is_black(n):
                painter.fillRect(int(x), 0, max(int(lane_w) + 1, 1), roll_h, QColor(0, 0, 0, 20))
            else:
                painter.fillRect(int(x), 0, max(int(lane_w) + 1, 1), roll_h, QColor(255, 255, 255, 5))

        # Octave separators
        for n in range(MIN_NOTE, MAX_NOTE + 1):
            if n % 12 == 0:
                x = (n - MIN_NOTE) * lane_w
                painter.setPen(QPen(QColor(255, 255, 255, 22), 1))
                painter.drawLine(int(x), 0, int(x), roll_h)

        # ── Spectrum (behind notes) ──
        if pr.show_spectrum and self.spectrum_data:
            self._draw_spectrum_vertical(painter, w, h, roll_h, pr)

        # ── Beat pulse ──
        if pr.pulse_on_beat and pr.bg_pulse > 0:
            self._draw_beat_pulse(painter, w, h, pr, t)

        # ── Light pillars (behind notes) ──
        self.light_pillars.draw(painter, roll_h, kb_h, lane_w)

        # ── Hit line (enhanced) ──
        self._draw_hit_line(painter, w, y_hit, pr, t)

        # ── Notes ──
        t_start = t - (y_hit / self.pps)
        t_end = t + ((roll_h - y_hit) / self.pps)
        visible = self._note_index.visible_in(t_start, t_end) if self._note_index else []
        visible_sorted = sorted(visible, key=lambda ne: ne.note)

        for note_evt in visible_sorted:
            self._draw_note_vertical(painter, note_evt, lane_w, y_hit, roll_h, t, pr)

        # ── Keyboard ──
        self._draw_keyboard_vertical(painter, w, h, roll_h, kb_h, lane_w, t, pr)

    def _draw_hit_line(self, painter, w, y_hit, pr, t):
        """Enhanced hit line with wide glow and pulse."""
        # Wide atmospheric glow
        glow_h = 40
        glow = QLinearGradient(0, int(y_hit) - glow_h, 0, int(y_hit) + glow_h)
        pulse = 0.7 + 0.3 * math.sin(t * 2.0)
        glow.setColorAt(0.0, QColor(255, 255, 255, 0))
        glow.setColorAt(0.3, QColor(255, 255, 255, int(4 * pulse)))
        glow.setColorAt(0.45, QColor(255, 255, 255, int(10 * pulse)))
        glow.setColorAt(0.5, QColor(255, 255, 255, int(20 * pulse)))
        glow.setColorAt(0.55, QColor(255, 255, 255, int(10 * pulse)))
        glow.setColorAt(0.7, QColor(255, 255, 255, int(4 * pulse)))
        glow.setColorAt(1.0, QColor(255, 255, 255, 0))
        painter.fillRect(0, int(y_hit) - glow_h, w, glow_h * 2, QBrush(glow))

        # Core line with color tint from active notes
        active_count = len(self.active_notes)
        if active_count > 0 and self._active_events_map:
            # Blend active note colors for the hit line tint
            sample_evt = next(iter(self._active_events_map.values()))
            tint = self._note_color_by_channel(sample_evt, pr)
            line_color = QColor(
                min(255, 200 + tint.red() // 5),
                min(255, 200 + tint.green() // 5),
                min(255, 200 + tint.blue() // 5),
                int(180 * pulse)
            )
        else:
            line_color = QColor(255, 255, 255, int(160 * pulse))

        painter.setPen(QPen(line_color, 2.5))
        painter.drawLine(0, int(y_hit), w, int(y_hit))

        # Bright highlight
        painter.setPen(QPen(QColor(255, 255, 255, int(60 * pulse)), 1))
        painter.drawLine(0, int(y_hit) - 1, w, int(y_hit) - 1)

    def _draw_note_vertical(self, painter, note_evt, lane_w, y_hit, roll_h, t, pr):
        """
        Enhanced note rendering with crystal/glass effect,
        velocity-aware brightness, and depth fog.
        """
        n = note_evt.note
        x = (n - MIN_NOTE) * lane_w
        y_b = y_hit - (note_evt.start_time - t) * self.pps
        y_t = y_hit - (note_evt.start_time + note_evt.duration - t) * self.pps
        if y_b < -50 or y_t > roll_h + 50:
            return

        dt_top = max(0, int(y_t))
        dt_bot = min(roll_h, int(y_b))
        pad = max(1, int(lane_w * 0.10))
        nw = max(int(lane_w) - pad, 2)
        x_pad = int(x + pad / 2)
        nh = max(dt_bot - dt_top, 2)

        nc = self._note_color_by_channel(note_evt, pr)
        is_active = self._is_note_active(n, note_evt, t)

        # ── Velocity-based brightness ──
        vel_ratio = note_evt.velocity / 127.0
        brightness = 0.6 + 0.4 * vel_ratio  # 60%-100% brightness range

        # ── Depth fog: notes far from hit line are slightly faded ──
        note_center_y = (dt_top + dt_bot) / 2.0
        dist_from_hit = abs(note_center_y - y_hit) / roll_h
        fog = max(0.3, 1.0 - dist_from_hit * 0.5)

        # Apply fog and velocity to color
        nc_fogged = QColor(
            int(nc.red() * fog * brightness),
            int(nc.green() * fog * brightness),
            int(nc.blue() * fog * brightness),
            nc.alpha()
        )

        # ── Drop shadow ──
        shadow_offset = 3 if not is_active else 4
        shadow_alpha = int((50 if not is_active else 80) * fog)
        painter.setBrush(QBrush(QColor(0, 0, 0, shadow_alpha)))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(x_pad + shadow_offset, dt_top + shadow_offset,
                                nw, nh, int(pr.note_radius), int(pr.note_radius))

        # ── Note glow ──
        if pr.note_glow > 0:
            center_x = x_pad + nw / 2
            center_y = (dt_top + dt_bot) / 2
            glow_r = max(nw, nh) * pr.note_glow * 0.25
            glow = QRadialGradient(center_x, center_y, max(glow_r, 1))
            gc = QColor(nc_fogged)
            glow_alpha = int((60 if is_active else 40) * fog)
            gc.setAlpha(glow_alpha)
            glow.setColorAt(0, gc)
            gc2 = QColor(nc_fogged); gc2.setAlpha(0)
            glow.setColorAt(1, gc2)
            painter.setBrush(QBrush(glow)); painter.setPen(Qt.NoPen)
            ext = int(nw * 0.6)
            painter.drawRoundedRect(x_pad - ext, dt_top - ext,
                                    nw + ext * 2, nh + ext * 2,
                                    int(pr.note_radius * 3), int(pr.note_radius * 3))

        # ── Crystal/glass gradient fill ──
        note_grad = QLinearGradient(x_pad, dt_top, x_pad, dt_bot)
        if is_active:
            # Active: bright, luminous
            note_grad.setColorAt(0.0, nc_fogged.lighter(170))
            note_grad.setColorAt(0.08, nc_fogged.lighter(150))
            note_grad.setColorAt(0.20, nc_fogged.lighter(130))
            note_grad.setColorAt(0.80, nc_fogged.lighter(110))
            note_grad.setColorAt(0.92, nc_fogged.lighter(95))
            note_grad.setColorAt(1.0, nc_fogged.darker(115))
        else:
            # Inactive: glass-like with internal reflection
            note_grad.setColorAt(0.0, nc_fogged.lighter(145))
            note_grad.setColorAt(0.06, nc_fogged.lighter(120))
            note_grad.setColorAt(0.15, nc_fogged.lighter(108))
            note_grad.setColorAt(0.50, nc_fogged)
            note_grad.setColorAt(0.85, nc_fogged.darker(108))
            note_grad.setColorAt(0.92, nc_fogged.darker(90))
            note_grad.setColorAt(1.0, nc_fogged.darker(125))

        painter.setBrush(QBrush(note_grad))
        if pr.note_border:
            border_c = nc_fogged.darker(150) if not is_active else nc_fogged.darker(120)
            painter.setPen(QPen(border_c, 1))
        else:
            painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(x_pad, dt_top, nw, nh, int(pr.note_radius), int(pr.note_radius))

        # ── Top highlight strip (glass reflection) ──
        if nh > 5:
            hl_h = min(4, nh // 3)
            hl_grad = QLinearGradient(x_pad, dt_top, x_pad, dt_top + hl_h)
            hl_alpha = int((100 if is_active else 65) * fog * brightness)
            hl_grad.setColorAt(0, QColor(255, 255, 255, hl_alpha))
            hl_grad.setColorAt(0.5, QColor(255, 255, 255, int(hl_alpha * 0.5)))
            hl_grad.setColorAt(1, QColor(255, 255, 255, 0))
            painter.setBrush(QBrush(hl_grad)); painter.setPen(Qt.NoPen)
            inset = max(1, int(pr.note_radius))
            painter.drawRoundedRect(x_pad + 1, dt_top + 1, nw - 2, hl_h, inset, inset)

        # ── Internal reflection line (glass effect) ──
        if nh > 12 and nw > 4:
            ref_y = dt_top + int(nh * 0.35)
            ref_h = max(1, nh // 8)
            ref_grad = QLinearGradient(x_pad, ref_y, x_pad, ref_y + ref_h)
            ref_alpha = int((35 if is_active else 20) * fog)
            ref_grad.setColorAt(0, QColor(255, 255, 255, 0))
            ref_grad.setColorAt(0.5, QColor(255, 255, 255, ref_alpha))
            ref_grad.setColorAt(1, QColor(255, 255, 255, 0))
            painter.setBrush(QBrush(ref_grad)); painter.setPen(Qt.NoPen)
            painter.drawRoundedRect(x_pad + 2, ref_y, nw - 4, ref_h, 1, 1)

        # ── Active flash ──
        if is_active and nh > 4:
            flash_h = min(10, nh // 2)
            flash_grad = QLinearGradient(x_pad, dt_top, x_pad, dt_top + flash_h)
            flash_grad.setColorAt(0, QColor(255, 255, 255, 70))
            flash_grad.setColorAt(1, QColor(255, 255, 255, 0))
            painter.setBrush(QBrush(flash_grad)); painter.setPen(Qt.NoPen)
            painter.drawRoundedRect(x_pad, dt_top, nw, flash_h, int(pr.note_radius), int(pr.note_radius))

        # ── Trail ──
        if pr.trail_length > 0 and pr.trail_opacity > 0 and is_active:
            trail_h = int(pr.trail_length)
            trail_grad = QLinearGradient(x_pad, dt_top, x_pad, dt_top - trail_h)
            tc = QColor(nc); tc.setAlpha(int(pr.trail_opacity * 140 * brightness))
            trail_grad.setColorAt(0, tc)
            tc2 = QColor(nc); tc2.setAlpha(0)
            trail_grad.setColorAt(1, tc2)
            painter.setBrush(QBrush(trail_grad)); painter.setPen(Qt.NoPen)
            painter.drawRoundedRect(x_pad, dt_top - trail_h, nw, trail_h,
                                    int(pr.note_radius), int(pr.note_radius))

    def _draw_keyboard_vertical(self, painter, w, h, roll_h, kb_h, lane_w, t, pr):
        """Enhanced 3D keyboard with dramatic underglow lighting."""
        painter.setPen(Qt.NoPen)

        # Keyboard background
        kb_bg = QLinearGradient(0, roll_h, 0, h)
        kb_bg.setColorAt(0, QColor(10, 10, 16))
        kb_bg.setColorAt(0.02, QColor(14, 14, 20))
        kb_bg.setColorAt(1, QColor(6, 6, 12))
        painter.fillRect(0, roll_h, w, kb_h, QBrush(kb_bg))

        # Top shadow
        border_shadow = QLinearGradient(0, roll_h, 0, roll_h + 14)
        border_shadow.setColorAt(0, QColor(0, 0, 0, 120))
        border_shadow.setColorAt(1, QColor(0, 0, 0, 0))
        painter.fillRect(0, roll_h, w, 14, QBrush(border_shadow))

        # ── White Keys ──
        for n in range(MIN_NOTE, MAX_NOTE + 1):
            if is_black(n): continue
            x = (n - MIN_NOTE) * lane_w
            kw = max(int(lane_w) + 1, 1)
            is_active = n in self.active_notes
            press_offset = 2 if is_active else 0
            key_y_start = roll_h + press_offset
            key_h = kb_h - press_offset

            if is_active:
                active_color = self._get_active_key_color(n, t, pr)

                # ── Dramatic underglow ──
                # Wide ambient glow
                glow_grad = QRadialGradient(QPointF(x + lane_w / 2, roll_h + kb_h * 0.5), kb_h * 1.2)
                glow_grad.setColorAt(0, QColor(active_color.red(), active_color.green(), active_color.blue(), 50))
                glow_grad.setColorAt(0.5, QColor(active_color.red(), active_color.green(), active_color.blue(), 20))
                glow_grad.setColorAt(1, QColor(active_color.red(), active_color.green(), active_color.blue(), 0))
                painter.setBrush(QBrush(glow_grad))
                painter.drawRect(int(x) - 4, roll_h, kw + 8, kb_h)

                # Lit key with rich gradient
                key_grad = QLinearGradient(int(x), key_y_start, int(x), roll_h + key_h)
                key_grad.setColorAt(0.0, active_color.lighter(160))
                key_grad.setColorAt(0.15, active_color.lighter(140))
                key_grad.setColorAt(0.5, active_color.lighter(120))
                key_grad.setColorAt(0.85, active_color.lighter(105))
                key_grad.setColorAt(1.0, active_color.darker(115))
                painter.setBrush(QBrush(key_grad))
            else:
                # Realistic 3D white key
                key_grad = QLinearGradient(int(x), key_y_start, int(x), roll_h + key_h)
                key_grad.setColorAt(0.0, QColor(200, 203, 215))
                key_grad.setColorAt(0.02, QColor(235, 238, 248))
                key_grad.setColorAt(0.05, QColor(242, 244, 252))
                key_grad.setColorAt(0.65, QColor(236, 238, 246))
                key_grad.setColorAt(0.92, QColor(222, 224, 234))
                key_grad.setColorAt(0.97, QColor(210, 212, 224))
                key_grad.setColorAt(1.0, QColor(185, 188, 202))
                painter.setBrush(QBrush(key_grad))

            painter.drawRect(int(x), key_y_start, kw, key_h)

            # Edge lines for depth
            painter.setPen(QPen(QColor(0, 0, 0, 35), 1))
            painter.drawLine(int(x), key_y_start, int(x), roll_h + key_h)
            painter.setPen(QPen(QColor(255, 255, 255, 18), 1))
            painter.drawLine(int(x) + 1, key_y_start, int(x) + 1, roll_h + key_h)
            painter.setPen(Qt.NoPen)

            # Note label
            if n % 12 == 0:
                label_color = QColor(40, 40, 60, 180) if not is_active else QColor(255, 255, 255, 230)
                painter.setPen(label_color)
                painter.setFont(QFont("SF Pro", 7, QFont.Bold))
                painter.drawText(int(x + 2), h - 5, note_name(n))
                painter.setPen(Qt.NoPen)

        # ── Black Keys ──
        for n in range(MIN_NOTE, MAX_NOTE + 1):
            if not is_black(n): continue
            x = (n - MIN_NOTE) * lane_w
            is_active = n in self.active_notes
            bk_w = max(int(lane_w * 0.58), 2)
            bk_offset = int((lane_w - bk_w) / 2)
            bk_h_actual = int(kb_h * 0.62)
            press_offset = 2 if is_active else 0
            bk_y_start = roll_h + press_offset

            # Shadow
            shadow_alpha = 55 if not is_active else 85
            painter.setBrush(QColor(0, 0, 0, shadow_alpha))
            painter.drawRoundedRect(int(x) + bk_offset + 2, bk_y_start + 2, bk_w, bk_h_actual, 2, 2)

            if is_active:
                active_color = self._get_active_key_color(n, t, pr)
                # Underglow for black key
                glow_grad = QRadialGradient(QPointF(x + lane_w / 2, roll_h + bk_h_actual * 0.5), bk_h_actual)
                glow_grad.setColorAt(0, QColor(active_color.red(), active_color.green(), active_color.blue(), 40))
                glow_grad.setColorAt(1, QColor(active_color.red(), active_color.green(), active_color.blue(), 0))
                painter.setBrush(QBrush(glow_grad))
                painter.drawRect(int(x) - 2, roll_h, bk_w + 4, bk_h_actual + 2)

                key_grad = QLinearGradient(int(x) + bk_offset, bk_y_start, int(x) + bk_offset, bk_y_start + bk_h_actual)
                key_grad.setColorAt(0.0, active_color.lighter(150))
                key_grad.setColorAt(0.2, active_color.lighter(130))
                key_grad.setColorAt(0.6, active_color.lighter(110))
                key_grad.setColorAt(1.0, active_color.darker(140))
                painter.setBrush(QBrush(key_grad))
            else:
                key_grad = QLinearGradient(int(x) + bk_offset, bk_y_start, int(x) + bk_offset, bk_y_start + bk_h_actual)
                key_grad.setColorAt(0.0, QColor(58, 58, 72))
                key_grad.setColorAt(0.08, QColor(44, 44, 58))
                key_grad.setColorAt(0.4, QColor(30, 30, 42))
                key_grad.setColorAt(0.8, QColor(20, 20, 32))
                key_grad.setColorAt(1.0, QColor(12, 12, 22))
                painter.setBrush(QBrush(key_grad))

            painter.drawRoundedRect(int(x) + bk_offset, bk_y_start, bk_w, bk_h_actual, 3, 3)

            # Glossy highlight
            if bk_h_actual > 10:
                hl_grad = QLinearGradient(int(x) + bk_offset, bk_y_start,
                                          int(x) + bk_offset, bk_y_start + bk_h_actual * 0.2)
                hl_alpha = 40 if not is_active else 70
                hl_grad.setColorAt(0, QColor(255, 255, 255, hl_alpha))
                hl_grad.setColorAt(1, QColor(255, 255, 255, 0))
                painter.setBrush(QBrush(hl_grad)); painter.setPen(Qt.NoPen)
                painter.drawRoundedRect(int(x) + bk_offset + 2, bk_y_start + 1, bk_w - 4, int(bk_h_actual * 0.2), 2, 2)

    def _get_active_key_color(self, n, t, pr):
        evt = self._active_events_map.get(n)
        if evt is not None:
            return self._note_color_by_channel(evt, pr)
        return QColor(100, 180, 255)

    def _draw_spectrum_vertical(self, painter, w, h, roll_h, pr):
        idx = min(int(self.current_time * 30), len(self.spectrum_data) - 1)
        if idx < 0: return
        frame = self.spectrum_data[idx]
        if not frame: return

        if HAS_NUMPY and len(frame) > 0:
            frame_arr = np.array(frame, dtype=np.float64)
            if len(self._smooth_spectrum) != len(frame):
                self._smooth_spectrum = frame_arr.copy()
            else:
                smooth = np.array(self._smooth_spectrum, dtype=np.float64)
                smooth += (frame_arr - smooth) * 0.3
                self._smooth_spectrum = smooth
        else:
            if len(self._smooth_spectrum) != len(frame):
                self._smooth_spectrum = list(frame)
            else:
                for i in range(len(frame)):
                    self._smooth_spectrum[i] += (frame[i] - self._smooth_spectrum[i]) * 0.3

        n_bins = len(self._smooth_spectrum)
        bar_w = w / n_bins
        max_h = roll_h * 0.18

        for b in range(n_bins):
            val = float(self._smooth_spectrum[b])
            bh = val * max_h
            if bh < 1: continue
            bx = int(b * bar_w)
            bw = max(int(bar_w) - 1, 1)

            if pr.is_rainbow:
                c = color_hsv(b / n_bins, 0.7, 0.9, 70)
            else:
                c = QColor(pr.c_line)
                c.setAlpha(70)

            # Gradient bar with glow
            bar_grad = QLinearGradient(bx, roll_h - bh, bx, roll_h)
            c_top = QColor(c); c_top.setAlpha(25)
            c_bot = QColor(c); c_bot.setAlpha(80)
            bar_grad.setColorAt(0, c_top)
            bar_grad.setColorAt(1, c_bot)

            painter.setBrush(QBrush(bar_grad)); painter.setPen(Qt.NoPen)
            painter.drawRoundedRect(bx, int(roll_h - bh), bw, int(bh), 2, 2)

            # Mirror reflection (subtle)
            mirror_h = bh * 0.3
            mirror_grad = QLinearGradient(bx, roll_h, bx, roll_h + mirror_h)
            mc = QColor(c); mc.setAlpha(15)
            mirror_grad.setColorAt(0, mc)
            mirror_grad.setColorAt(1, QColor(c.red(), c.green(), c.blue(), 0))
            painter.setBrush(QBrush(mirror_grad))
            painter.drawRect(bx, int(roll_h), bw, int(mirror_h))

    # ══════════════════════════════════════════════════════════
    #  HORIZONTAL RENDERING
    # ══════════════════════════════════════════════════════════
    def _paint_horizontal(self, painter, w, h, pr):
        kb_h, roll_h, lane_h = 90, h - 90, (h - 90) / NOTE_RANGE
        t, x_center = self.current_time, w * 0.38

        for n in range(MIN_NOTE, MAX_NOTE + 1):
            y = roll_h - (n - MIN_NOTE) * lane_h
            base = pr.lane_black if is_black(n) else pr.lane_white
            lane_grad = QLinearGradient(0, int(y), 0, int(y + lane_h))
            lane_grad.setColorAt(0, base.lighter(103))
            lane_grad.setColorAt(1, base)
            painter.fillRect(0, int(y), w, max(int(lane_h) + 1, 1), QBrush(lane_grad))

        for n in range(MIN_NOTE, MAX_NOTE + 1):
            if n % 12 == 0:
                y = roll_h - (n - MIN_NOTE) * lane_h
                painter.setPen(QPen(QColor(255, 255, 255, 15), 1))
                painter.drawLine(0, int(y), w, int(y))

        painter.setPen(QPen(QColor(255, 255, 255, 80), 2))
        painter.drawLine(int(x_center), 0, int(x_center), roll_h)

        if pr.show_spectrum and self.spectrum_data:
            self._draw_spectrum(painter, w, h, roll_h, pr)
        if pr.pulse_on_beat and pr.bg_pulse > 0:
            self._draw_beat_pulse(painter, w, h, pr, t)

        t_start = t - (x_center / self.pps)
        t_end = t + ((w - x_center) / self.pps)
        visible = self._note_index.visible_in(t_start, t_end) if self._note_index else []

        for note_evt in visible:
            n = note_evt.note
            x_s = x_center + (note_evt.start_time - t) * self.pps
            x_e = x_center + (note_evt.start_time + note_evt.duration - t) * self.pps
            if x_e < 0 or x_s > w: continue

            y = roll_h - (n - MIN_NOTE) * lane_h
            nw_val = max(int(x_e - x_s), 2)
            nh_val = max(int(lane_h) - 1, 2)
            nc = self._note_color_by_channel(note_evt, pr)
            is_active = self._is_note_active(n, note_evt, t)
            vel_ratio = note_evt.velocity / 127.0

            painter.setBrush(QBrush(QColor(0, 0, 0, 35))); painter.setPen(Qt.NoPen)
            painter.drawRoundedRect(int(x_s) + 2, int(y) + 2, nw_val, nh_val,
                                    int(pr.note_radius), int(pr.note_radius))

            note_grad = QLinearGradient(int(x_s), int(y), int(x_s), int(y + nh_val))
            if is_active:
                note_grad.setColorAt(0, nc.lighter(170))
                note_grad.setColorAt(0.08, nc.lighter(150))
                note_grad.setColorAt(0.85, nc.lighter(110))
                note_grad.setColorAt(1, nc.darker(115))
            else:
                note_grad.setColorAt(0, nc.lighter(145))
                note_grad.setColorAt(0.08, nc.lighter(120))
                note_grad.setColorAt(0.5, nc)
                note_grad.setColorAt(1, nc.darker(125))

            if pr.trail_length > 0 and pr.trail_opacity > 0:
                trail_grad = QLinearGradient(int(x_s - pr.trail_length), int(y), int(x_s), int(y))
                tc = QColor(nc); tc.setAlpha(int(pr.trail_opacity * 140))
                tc2 = QColor(nc); tc2.setAlpha(0)
                trail_grad.setColorAt(0, tc2); trail_grad.setColorAt(1, tc)
                painter.setBrush(QBrush(trail_grad)); painter.setPen(Qt.NoPen)
                painter.drawRoundedRect(int(x_s - pr.trail_length), int(y),
                                        int(pr.trail_length), nh_val, 2, 2)

            if pr.light_streak and is_active:
                time_since_hit = t - note_evt.start_time
                if 0 <= time_since_hit < 0.2:
                    alpha = int((1 - time_since_hit / 0.2)**2 * 120)
                    grad = QLinearGradient(0, int(y + lane_h / 2 - 15), 0, int(y + lane_h / 2 + 15))
                    sc = QColor(nc); sc.setAlpha(alpha)
                    grad.setColorAt(0, QColor(nc.red(), nc.green(), nc.blue(), 0))
                    grad.setColorAt(0.5, sc)
                    grad.setColorAt(1, QColor(nc.red(), nc.green(), nc.blue(), 0))
                    painter.fillRect(0, int(y + lane_h / 2 - 15), w, 30, QBrush(grad))

            if pr.note_glow > 0:
                self._draw_note_glow_h(painter, x_s, y, nw_val, nh_val, nc, pr)

            painter.setBrush(QBrush(note_grad))
            pen = QPen(nc.darker(140), 1) if pr.note_border else Qt.NoPen
            painter.setPen(pen)
            painter.drawRoundedRect(int(x_s), int(y), nw_val, nh_val,
                                    int(pr.note_radius), int(pr.note_radius))

            if nh_val > 4:
                hl_h = min(2, nh_val // 4)
                hl = QLinearGradient(int(x_s), int(y), int(x_s), int(y) + hl_h)
                hl.setColorAt(0, QColor(255, 255, 255, 55))
                hl.setColorAt(1, QColor(255, 255, 255, 0))
                painter.setBrush(QBrush(hl)); painter.setPen(Qt.NoPen)
                painter.drawRect(int(x_s) + 1, int(y) + 1, nw_val - 2, hl_h)

        if pr.reflection:
            self._draw_reflection_h(painter, w, h, roll_h, kb_h, pr, t, x_center, lane_h)

        # Keyboard (horizontal)
        painter.setPen(Qt.NoPen)
        for n in range(MIN_NOTE, MAX_NOTE + 1):
            y = roll_h - (n - MIN_NOTE) * lane_h
            is_active = n in self.active_notes
            if is_black(n):
                col = (self._get_active_key_color(n, t, pr).lighter(160) if is_active else pr.kb_bt)
            else:
                col = (self._get_active_key_color(n, t, pr).lighter(130) if is_active else pr.kb_wt)
            painter.setBrush(QBrush(col))
            painter.drawRect(0, int(y), 50, max(int(lane_h) + 1, 1))
            if not is_black(n) and n % 12 == 0:
                painter.setPen(QColor(120, 120, 140, 150))
                painter.setFont(QFont("SF Pro", 7))
                painter.drawText(5, int(y + lane_h / 2 + 4), note_name(n))
                painter.setPen(Qt.NoPen)

    # ══════════════════════════════════════════════════════════
    #  POST PROCESSING & HELPERS
    # ══════════════════════════════════════════════════════════
    def _note_color_by_channel(self, note_evt, pr):
        if pr.is_rainbow or pr.note_colors == 'rainbow':
            return cached_color_hsv((note_evt.note - MIN_NOTE) / NOTE_RANGE, 0.85, 0.9)
        if isinstance(pr.note_colors, list):
            ci = (note_evt.note - MIN_NOTE) % len(pr.note_colors)
            return QColor(pr.note_colors[ci])
        return CHANNEL_COLORS[note_evt.channel % len(CHANNEL_COLORS)]

    @staticmethod
    def _is_note_active(n, note_evt, t):
        return note_evt.note == n and note_evt.start_time <= t <= note_evt.start_time + note_evt.duration

    def _draw_note_glow_h(self, p, x_s, y, nw, nh, nc, pr):
        glow = QRadialGradient(x_s + nw / 2, y + nh / 2, max(nw * pr.note_glow * 0.5, 1))
        gc = QColor(nc); gc.setAlpha(40)
        glow.setColorAt(0, gc)
        gc2 = QColor(nc); gc2.setAlpha(0)
        glow.setColorAt(1, gc2)
        p.setBrush(QBrush(glow)); p.setPen(Qt.NoPen)
        p.drawRoundedRect(int(x_s - nw * 0.3), int(y - nh * 0.5),
                          int(nw * 1.6), int(nh * 2),
                          int(pr.note_radius * 2), int(pr.note_radius * 2))

    def _draw_spectrum(self, p, w, h, roll_h, pr):
        idx = min(int(self.current_time * 30), len(self.spectrum_data) - 1)
        if idx < 0: return
        frame = self.spectrum_data[idx]
        if not frame: return
        if HAS_NUMPY and len(frame) > 0:
            frame_arr = np.array(frame, dtype=np.float64)
            if len(self._smooth_spectrum) != len(frame):
                self._smooth_spectrum = frame_arr.copy()
            else:
                smooth = np.array(self._smooth_spectrum, dtype=np.float64)
                smooth += (frame_arr - smooth) * 0.3
                self._smooth_spectrum = smooth
        else:
            if len(self._smooth_spectrum) != len(frame):
                self._smooth_spectrum = list(frame)
            else:
                for i in range(len(frame)):
                    self._smooth_spectrum[i] += (frame[i] - self._smooth_spectrum[i]) * 0.3
        n_bins, bar_w = len(self._smooth_spectrum), w / len(self._smooth_spectrum)
        for b in range(n_bins):
            val = float(self._smooth_spectrum[b])
            bh = val * roll_h * 0.25
            if bh < 1: continue
            c = color_hsv(b / n_bins, 0.7, 0.8, 80) if pr.is_rainbow else QColor(pr.c_line)
            c.setAlpha(80)
            bar_grad = QLinearGradient(int(b * bar_w), int(roll_h - bh), int(b * bar_w), int(roll_h))
            c_top = QColor(c); c_top.setAlpha(40)
            bar_grad.setColorAt(0, c_top)
            bar_grad.setColorAt(1, c)
            p.setBrush(QBrush(bar_grad)); p.setPen(Qt.NoPen)
            p.drawRoundedRect(int(b * bar_w), int(roll_h - bh), max(int(bar_w) - 1, 1), int(bh), 2, 2)

    def _draw_vignette(self, p, w, h, pr):
        grad = QRadialGradient(w / 2, h / 2, max(w, h) * 0.55)
        grad.setColorAt(0, QColor(0, 0, 0, 0))
        grad.setColorAt(0.7, QColor(0, 0, 0, int(pr.vignette * 60)))
        grad.setColorAt(1, QColor(0, 0, 0, int(pr.vignette * 220)))
        p.fillRect(0, 0, w, h, QBrush(grad))

    def _draw_scanlines(self, p, w, h, pr):
        cache_key = (w, h, int(pr.scanlines * 1000))
        if self._scanline_cache_key != cache_key or self._scanline_cache is None:
            self._scanline_cache = QPixmap(w, h)
            self._scanline_cache.fill(Qt.transparent)
            cache_painter = QPainter(self._scanline_cache)
            cache_painter.setPen(Qt.NoPen)
            alpha = int(pr.scanlines * 200)
            for y in range(0, h, 3):
                cache_painter.setBrush(QColor(0, 0, 0, alpha))
                cache_painter.drawRect(0, y, w, 1)
            cache_painter.end()
            self._scanline_cache_key = cache_key
        p.drawPixmap(0, 0, self._scanline_cache)

    def _draw_film_grain(self, p, w, h, pr):
        grain_density = pr.film_grain * 0.01
        n_pixels = int(w * h * grain_density)
        if n_pixels <= 0: return
        if HAS_NUMPY:
            img = QImage(w, h, QImage.Format_ARGB32)
            img.fill(Qt.transparent)
            xs = np.random.randint(0, w, size=n_pixels)
            ys = np.random.randint(0, h, size=n_pixels)
            alphas = np.random.randint(0, int(pr.film_grain * 80), size=n_pixels)
            ptr = img.bits()
            ptr.setsize(w * h * 4)
            arr = np.frombuffer(ptr, dtype=np.uint8).reshape(h, w, 4)
            for i in range(n_pixels):
                a = int(alphas[i])
                arr[ys[i], xs[i]] = [255, 255, 255, a]
            p.drawImage(0, 0, img)
        else:
            a = int(pr.film_grain * 80)
            p.setPen(Qt.NoPen)
            for _ in range(min(n_pixels, 500)):
                p.setBrush(QColor(255, 255, 255, random.randint(0, a)))
                p.drawPoint(random.randint(0, w - 1), random.randint(0, h - 1))

    def _draw_beat_pulse(self, p, w, h, pr, t):
        if not self._beats: return
        idx = bisect.bisect_left(self._beats, t)
        candidates = []
        if idx < len(self._beats): candidates.append(self._beats[idx])
        if idx > 0: candidates.append(self._beats[idx - 1])
        if not candidates: return
        nearest = min(candidates, key=lambda b: abs(t - b))
        diff = abs(t - nearest)
        if diff < 0.3:
            intensity = (1 - diff / 0.3) ** 2 * pr.bg_pulse
            cx, cy = w / 2, h / 2
            pulse_grad = QRadialGradient(cx, cy, max(w, h) * 0.5)
            pulse_grad.setColorAt(0, QColor(255, 255, 255, int(intensity * 35)))
            pulse_grad.setColorAt(0.5, QColor(255, 255, 255, int(intensity * 15)))
            pulse_grad.setColorAt(1, QColor(255, 255, 255, 0))
            p.fillRect(0, 0, w, h, QBrush(pulse_grad))

    def _draw_reflection_h(self, p, w, h, roll_h, kb_h, pr, t, x_center, lane_h):
        p.save()
        p.setClipRect(0, roll_h, w, kb_h)
        p.translate(0, 2 * roll_h)
        p.scale(1, -1)
        p.setOpacity(0.12)
        for note_evt in (self._note_index.visible_in(t - 5, t + 5) if self._note_index else []):
            n = note_evt.note
            x_s = x_center + (note_evt.start_time - t) * self.pps
            x_e = x_center + (note_evt.start_time + note_evt.duration - t) * self.pps
            if x_e < 0 or x_s > w: continue
            y = roll_h - (n - MIN_NOTE) * lane_h
            nw, nh = max(int(x_e - x_s), 2), max(int(lane_h) - 1, 2)
            nc = self._note_color_by_channel(note_evt, pr)
            p.setBrush(QBrush(nc)); p.setPen(Qt.NoPen)
            p.drawRoundedRect(int(x_s), int(y), nw, nh, int(pr.note_radius), int(pr.note_radius))
        p.restore()

    def _draw_watermark(self, p, w, h):
        p.setPen(QColor(255, 255, 255, 40))
        p.setFont(QFont("SF Pro", 14, QFont.Bold))
        p.drawText(w - 200, h - 20, self.watermark_text)

    def _paint_hud(self, painter, w, h, pr):
        if pr.show_speed:
            painter.setPen(QColor(200, 200, 220, 160))
            painter.setFont(QFont("SF Pro", 12))
            painter.drawText(w - 180, 50, f"Speed: {self.playback_speed:.2f}×")

        # Combo display
        if self.combo > 1 and getattr(pr, 'show_combo', False):
            combo_text = f"×{self.combo}"
            painter.setPen(QColor(255, 220, 80, 200))
            painter.setFont(QFont("SF Pro", 22, QFont.Bold))
            painter.drawText(w - 180, 85, combo_text)

        time_text = f"{fmt_time(self.current_time)} / {fmt_time(self.duration)}"
        painter.setFont(QFont("SF Pro", 11))
        fm = painter.fontMetrics()
        tw = fm.horizontalAdvance(time_text) + 16
        painter.setBrush(QColor(0, 0, 0, 60))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(54, h - 26, tw, 20, 4, 4)
        painter.setPen(QColor(200, 200, 220, 180))
        painter.drawText(62, h - 10, time_text)

    # ══════════════════════════════════════════════════════════
    #  INTERACTION & ANIMATION
    # ══════════════════════════════════════════════════════════
    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        if event.modifiers() & Qt.ControlModifier:
            factor = 1.1 if delta > 0 else 0.9
            self._zoom_level = max(0.25, min(8.0, self._zoom_level * factor))
            self.pps = int(220 * self._zoom_level)
        else:
            self.current_time = max(0, self.current_time - delta * 0.5 / self.pps)
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._dragging = True
            self._last_mouse_x = event.position().x()

    def mouseMoveEvent(self, event):
        if self._dragging:
            dx = event.position().x() - self._last_mouse_x
            self._last_mouse_x = event.position().x()
            self.current_time = max(0, self.current_time - dx / self.pps)
            self.update()

    def mouseReleaseEvent(self, event):
        self._dragging = False

    def tick(self, dt):
        self._anim_time += dt
        self.particles.update(dt)
        self.ripples.update(dt)
        self.splashes.update(dt)
        self.stars.update(dt)
        self.arcs.update(dt)

        self._prev_active = self.active_notes.copy()
        self.active_notes = self._note_index.active_at(self.current_time) if self._note_index else set()
        self._active_events_map = self._note_index.active_events_map_at(self.current_time) if self._note_index else {}

        t, pr = self.current_time, self.preset
        new_notes = self.active_notes - self._prev_active
        released_notes = self._prev_active - self.active_notes

        # ── Update light pillars ──
        if pr and self.orientation == 'vertical':
            lane_w = self.width() / NOTE_RANGE
            kb_h = 120
            roll_h = self.height() - kb_h

            for pitch in new_notes:
                note_evt = self._active_events_map.get(pitch)
                if note_evt is not None:
                    nc = self._note_color_by_channel(note_evt, pr)
                    nx = (pitch - MIN_NOTE) * lane_w + lane_w / 2
                    self.light_pillars.activate(pitch, nx, nc, roll_h,
                                                intensity=0.5 + 0.5 * (note_evt.velocity / 127.0))

            for pitch in released_notes:
                self.light_pillars.deactivate(pitch)

            self.light_pillars.update(dt, self.active_notes)

        # ── Emit effects for new notes ──
        if new_notes and pr:
            # Collect positions for connection arcs
            new_positions = []
            for pitch in new_notes:
                note_evt = self._active_events_map.get(pitch)
                if note_evt is None:
                    continue
                nc = self._note_color_by_channel(note_evt, pr)
                if self.orientation == 'vertical':
                    lane_w = self.width() / NOTE_RANGE
                    nx = (pitch - MIN_NOTE) * lane_w + lane_w / 2
                    ny = (self.height() - 120) * 0.88
                else:
                    nx = 50
                    ny = (self.height() - 90) - (pitch - MIN_NOTE) * ((self.height() - 90) / NOTE_RANGE)

                vel_factor = note_evt.velocity / 127.0
                new_positions.append((nx, ny, nc))

                if pr.particle_count > 0:
                    # Sparks
                    self.particles.emit(nx, ny, nc, count=pr.particle_count,
                                        speed=pr.particle_speed * (0.7 + 0.6 * vel_factor),
                                        size=pr.particle_size, ptype='spark')
                    # Orbs (softer, fewer)
                    self.particles.emit(nx, ny, nc.lighter(140),
                                        count=max(1, pr.particle_count // 3),
                                        speed=pr.particle_speed * 0.4,
                                        size=pr.particle_size * 1.8, ptype='orb')
                    # Ribbons (gentle floaters)
                    if vel_factor > 0.5:
                        self.particles.emit(nx, ny, nc.lighter(120),
                                            count=2,
                                            speed=pr.particle_speed * 0.3,
                                            size=pr.particle_size * 2.0, ptype='ribbon')
                if pr.note_splash:
                    self.splashes.emit(nx + 10, ny - 10, note_name(pitch), nc, size=12)
                self.ripples.emit(nx, ny, nc, mr=45, life=0.5)

            # ── Connection arcs between simultaneous notes ──
            if len(new_positions) >= 2 and getattr(pr, 'connection_arcs', False):
                for i in range(len(new_positions)):
                    for j in range(i + 1, len(new_positions)):
                        x1, y1, c1 = new_positions[i]
                        x2, y2, c2 = new_positions[j]
                        if abs(x2 - x1) < self.width() * 0.5:  # Only connect nearby notes
                            blend = QColor((c1.red() + c2.red()) // 2,
                                           (c1.green() + c2.green()) // 2,
                                           (c1.blue() + c2.blue()) // 2)
                            self.arcs.emit(x1, y1, x2, y2, blend, alpha=60)

        if self.active_notes and new_notes:
            self.combo += len(new_notes)
            self.streak += 1
        elif not self.active_notes and self._prev_active:
            self.streak = 0

        self.camera.update(dt, self.active_notes)
        self.update()