"""
Particle engine, star field, ripple system, note-splash system,
aurora renderer, light pillar system, and connection arc system.
"""
import math
import random
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from PySide6.QtGui import QColor, QBrush, QPen, QFont, QPainter, QLinearGradient, QRadialGradient, QPainterPath
from PySide6.QtCore import Qt, QPointF
from data import color_hsv as _hsv, HAS_NUMPY
if HAS_NUMPY:
    from data import np


class ParticleEngine:
    """Struct-of-Arrays particle engine with batched drawing."""
    def __init__(self, max_p=3000):
        self.max_p = max_p
        self.px = [0.0] * max_p
        self.py = [0.0] * max_p
        self.vx = [0.0] * max_p
        self.vy = [0.0] * max_p
        self.life = [0.0] * max_p
        self.max_life = [1.0] * max_p
        self.cr = [0] * max_p
        self.cg = [0] * max_p
        self.cb = [0] * max_p
        self.ca = [0] * max_p
        self.size = [1.0] * max_p
        self.ptype = [0] * max_p  # 0=spark, 1=orb, 2=ribbon
        self.trails: List[Optional[list]] = [None] * max_p
        self.active_count: int = 0

    def emit(self, x, y, color, count=6, speed=50, life=0.7, size=3, ptype='spark'):
        ptype_id = {'spark': 0, 'orb': 1, 'ribbon': 2}.get(ptype, 0)
        for _ in range(count):
            if self.active_count >= self.max_p:
                break
            i = self.active_count
            a = random.uniform(0, 2 * math.pi)
            s = random.uniform(speed * 0.3, speed)
            ml = random.uniform(life * 0.4, life)
            alpha = random.randint(140, 255)
            self.px[i] = x
            self.py[i] = y
            self.vx[i] = math.cos(a) * s
            self.vy[i] = -abs(math.sin(a) * s) * 0.8 if ptype_id != 2 else math.sin(a) * s * 0.3
            self.life[i] = ml
            self.max_life[i] = ml
            self.cr[i] = color.red()
            self.cg[i] = color.green()
            self.cb[i] = color.blue()
            self.ca[i] = alpha
            self.size[i] = random.uniform(size * 0.5, size * 1.5)
            self.ptype[i] = ptype_id
            self.trails[i] = [] if ptype_id in (0, 1, 2) else None
            self.active_count += 1

    def update(self, dt):
        write = 0
        for i in range(self.active_count):
            self.life[i] -= dt
            if self.life[i] <= 0:
                continue
            trail = self.trails[i]
            if trail is not None:
                ratio = self.life[i] / self.max_life[i]
                trail.append((self.px[i], self.py[i], ratio))
                max_trail = 20 if self.ptype[i] == 2 else 16
                if len(trail) > max_trail:
                    trail.pop(0)
            self.px[i] += self.vx[i] * dt
            self.py[i] += self.vy[i] * dt
            if self.ptype[i] == 2:  # ribbon: gentle float up
                self.vy[i] -= 30 * dt
                self.vx[i] *= 0.99
            else:
                self.vy[i] += 120 * dt
            self.vx[i] *= 0.97
            if write != i:
                for arr in (self.px, self.py, self.vx, self.vy,
                            self.life, self.max_life, self.cr, self.cg,
                            self.cb, self.ca, self.size, self.ptype):
                    arr[write] = arr[i]
                self.trails[write] = self.trails[i]
            write += 1
        self.active_count = write

    def draw(self, painter: QPainter, ox=0, oy=0):
        ac = self.active_count
        _px, _py = self.px, self.py
        _cr, _cg, _cb, _ca = self.cr, self.cg, self.cb, self.ca
        _size, _life, _max_life = self.size, self.life, self.max_life
        _trails, _ptype = self.trails, self.ptype

        painter.setPen(Qt.NoPen)

        # Pass 1: Ribbons (draw first, behind everything)
        for i in range(ac):
            if _ptype[i] != 2:
                continue
            ratio = max(0, _life[i] / _max_life[i])
            alpha = int(200 * ratio)
            r, g, b = _cr[i], _cg[i], _cb[i]
            trail = _trails[i]
            if trail and len(trail) > 2:
                # Draw as a smooth curved ribbon
                path = QPainterPath()
                x0, y0, _ = trail[0]
                path.moveTo(x0 - ox, y0 - oy)
                for j in range(1, len(trail)):
                    x1, y1, _ = trail[j]
                    path.lineTo(x1 - ox, y1 - oy)
                tw = max(0.5, _size[i] * ratio * 2)
                tc = QColor(r, g, b, max(0, min(255, int(alpha * 0.6))))
                painter.setPen(QPen(tc, tw, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
                painter.setBrush(Qt.NoBrush)
                painter.drawPath(path)
                painter.setPen(Qt.NoPen)

        # Pass 2: Sparks
        for i in range(ac):
            if _ptype[i] != 0:
                continue
            ratio = max(0, _life[i] / _max_life[i])
            alpha = int(255 * ratio)
            r, g, b = _cr[i], _cg[i], _cb[i]
            sz = _size[i] * ratio
            trail = _trails[i]
            if trail and len(trail) > 1:
                for j in range(1, len(trail)):
                    t_ratio = j / len(trail)
                    ta = max(0, min(255, int(alpha * 0.35 * t_ratio)))
                    tw = max(0.5, sz * 0.5 * t_ratio)
                    painter.setPen(QPen(QColor(r, g, b, ta), tw, Qt.SolidLine, Qt.RoundCap))
                    x1, y1, _ = trail[j - 1]; x2, y2, _ = trail[j]
                    painter.drawLine(QPointF(x1 - ox, y1 - oy), QPointF(x2 - ox, y2 - oy))
                painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(QColor(r, g, b, max(0, min(255, alpha)))))
            painter.drawEllipse(QPointF(_px[i] - ox, _py[i] - oy), sz, sz)

        # Pass 3: Orbs
        for i in range(ac):
            if _ptype[i] != 1:
                continue
            ratio = max(0, _life[i] / _max_life[i])
            alpha = int(255 * ratio)
            r, g, b = _cr[i], _cg[i], _cb[i]
            sz = _size[i] * ratio
            cx, cy = _px[i] - ox, _py[i] - oy
            if sz > 1.5:
                painter.setBrush(QBrush(QColor(r, g, b, max(0, min(80, int(alpha * 0.3))))))
                painter.drawEllipse(QPointF(cx, cy), sz * 2.5, sz * 2.5)
                painter.setBrush(QBrush(QColor(r, g, b, max(0, min(255, alpha)))))
                painter.drawEllipse(QPointF(cx, cy), sz * 0.8, sz * 0.8)
            else:
                painter.setBrush(QBrush(QColor(r, g, b, max(0, min(255, alpha)))))
                painter.drawEllipse(QPointF(cx, cy), sz, sz)


@dataclass
class ShootingStar:
    x: float; y: float; vx: float; vy: float
    life: float; max_life: float; size: float


class AuroraRenderer:
    """
    Renders flowing aurora/northern lights bands across the background.
    Uses layered semi-transparent gradient bands with sine-wave distortion.
    Computationally cheap — just a few gradient rectangles per frame.
    """
    def __init__(self, band_count=5):
        self.bands = []
        for i in range(band_count):
            self.bands.append({
                'y_base': random.uniform(0.05, 0.5),
                'amplitude': random.uniform(0.03, 0.08),
                'frequency': random.uniform(0.3, 0.8),
                'speed': random.uniform(0.08, 0.25),
                'phase': random.uniform(0, 6.28),
                'hue': random.uniform(0.45, 0.75),  # Blue-cyan-purple range
                'saturation': random.uniform(0.5, 0.9),
                'thickness': random.uniform(0.04, 0.10),
                'alpha_base': random.randint(8, 22),
            })

    def draw(self, painter: QPainter, w, h, t, intensity=1.0):
        painter.setPen(Qt.NoPen)
        for band in self.bands:
            phase = band['phase'] + t * band['speed']
            y_center = band['y_base'] + math.sin(phase) * band['amplitude']
            thickness = band['thickness']
            alpha = int(band['alpha_base'] * intensity)
            if alpha < 2:
                continue

            # Draw the band as a horizontal gradient
            y_top = int((y_center - thickness / 2) * h)
            y_bot = int((y_center + thickness / 2) * h)
            band_h = max(y_bot - y_top, 1)

            # Horizontal color shift
            grad = QLinearGradient(0, y_top, w, y_bot)
            hue1 = (band['hue'] + math.sin(phase * 0.7) * 0.05) % 1.0
            hue2 = (hue1 + 0.08) % 1.0
            c1 = _hsv(hue1, band['saturation'], 0.6, alpha)
            c2 = _hsv(hue2, band['saturation'] * 0.8, 0.4, alpha // 2)
            c3 = _hsv((hue1 + hue2) / 2, band['saturation'] * 0.6, 0.5, alpha)

            grad.setColorAt(0.0, QColor(c1.red(), c1.green(), c1.blue(), 0))
            grad.setColorAt(0.15, c1)
            grad.setColorAt(0.5, c3)
            grad.setColorAt(0.85, c2)
            grad.setColorAt(1.0, QColor(c2.red(), c2.green(), c2.blue(), 0))

            # Vertical fade: soft edges top and bottom
            vgrad = QLinearGradient(0, y_top, 0, y_bot)
            vgrad.setColorAt(0.0, QColor(255, 255, 255, 0))
            vgrad.setColorAt(0.2, QColor(255, 255, 255, 180))
            vgrad.setColorAt(0.5, QColor(255, 255, 255, 255))
            vgrad.setColorAt(0.8, QColor(255, 255, 255, 180))
            vgrad.setColorAt(1.0, QColor(255, 255, 255, 0))

            # Combine: draw horizontal gradient, masked by vertical
            painter.setBrush(QBrush(grad))
            painter.drawRect(0, y_top, w, band_h)


class LightPillarSystem:
    """
    Draws dramatic vertical light beams when keys are pressed.
    Each pillar is a semi-transparent gradient column that shoots up
    from the keyboard through the roll area.
    """
    def __init__(self):
        self.pillars: Dict[int, dict] = {}  # note -> pillar info

    def activate(self, note, x, color, roll_h, intensity=1.0):
        """Activate a light pillar for a note."""
        self.pillars[note] = {
            'x': x,
            'color': QColor(color),
            'intensity': intensity,
            'age': 0.0,
        }

    def deactivate(self, note):
        """Start fading out a pillar."""
        if note in self.pillars:
            self.pillars[note]['fading'] = True

    def update(self, dt, active_notes):
        # Remove pillars for notes no longer active
        to_remove = []
        for note, p in self.pillars.items():
            p['age'] += dt
            if p.get('fading'):
                p['intensity'] -= dt * 4.0
                if p['intensity'] <= 0:
                    to_remove.append(note)
            elif note not in active_notes:
                p['fading'] = True
        for n in to_remove:
            del self.pillars[n]

    def draw(self, painter: QPainter, roll_h, kb_h, lane_w):
        painter.setPen(Qt.NoPen)
        for note, p in self.pillars.items():
            intensity = max(0, min(1, p['intensity']))
            if intensity < 0.01:
                continue
            c = p['color']
            x = p['x']
            w = max(lane_w * 1.2, 4)

            # Main pillar: gradient from keyboard up, fading out
            grad = QLinearGradient(x, roll_h, x, 0)
            alpha_base = int(35 * intensity)
            grad.setColorAt(0.0, QColor(c.red(), c.green(), c.blue(), alpha_base))
            grad.setColorAt(0.1, QColor(c.red(), c.green(), c.blue(), int(alpha_base * 0.8)))
            grad.setColorAt(0.5, QColor(c.red(), c.green(), c.blue(), int(alpha_base * 0.3)))
            grad.setColorAt(1.0, QColor(c.red(), c.green(), c.blue(), 0))

            painter.setBrush(QBrush(grad))
            painter.drawRect(int(x - w / 2), 0, int(w), roll_h)

            # Bright core near keyboard
            core_h = int(roll_h * 0.15)
            core_grad = QLinearGradient(x, roll_h, x, roll_h - core_h)
            core_alpha = int(60 * intensity)
            core_grad.setColorAt(0.0, QColor(c.red(), c.green(), c.blue(), core_alpha))
            core_grad.setColorAt(1.0, QColor(c.red(), c.green(), c.blue(), 0))
            painter.setBrush(QBrush(core_grad))
            painter.drawRect(int(x - w * 0.6), roll_h - core_h, int(w * 1.2), core_h)


class ConnectionArcSystem:
    """
    Draws curved arcs connecting notes that are played simultaneously
    or in rapid succession, visualizing musical phrases.
    """
    def __init__(self):
        self.arcs: List[list] = []  # [x1, y1, x2, y2, color, alpha, max_alpha]

    def emit(self, x1, y1, x2, y2, color, alpha=80):
        self.arcs.append([x1, y1, x2, y2, QColor(color), alpha, alpha])

    def update(self, dt):
        write = 0
        for arc in self.arcs:
            arc[5] -= dt * 120  # Fade speed
            if arc[5] > 0:
                self.arcs[write] = arc
                write += 1
        del self.arcs[write:]

    def draw(self, painter: QPainter):
        for arc in self.arcs:
            x1, y1, x2, y2, color, alpha, max_alpha = arc
            if alpha < 2:
                continue
            c = QColor(color)
            c.setAlpha(max(0, min(255, int(alpha))))

            # Draw a quadratic bezier arc
            mid_x = (x1 + x2) / 2
            mid_y = min(y1, y2) - abs(x2 - x1) * 0.15  # Arc above

            path = QPainterPath()
            path.moveTo(x1, y1)
            path.quadTo(mid_x, mid_y, x2, y2)

            painter.setPen(QPen(c, 1.5, Qt.SolidLine, Qt.RoundCap))
            painter.setBrush(Qt.NoBrush)
            painter.drawPath(path)


class StarField:
    def __init__(self, count=100):
        self.stars = tuple(
            (random.uniform(0, 1), random.uniform(0, 1), random.uniform(0.2, 1.0),
             random.uniform(0.5, 2.5), random.uniform(0, 6.28))
            for _ in range(count)
        )
        self.shooting_stars: List[ShootingStar] = []
        self._shoot_timer = random.uniform(2, 6)

    def update(self, dt):
        self._shoot_timer -= dt
        if self._shoot_timer <= 0:
            self._shoot_timer = random.uniform(3, 8)
            sx = random.uniform(0.1, 0.9)
            sy = random.uniform(0.05, 0.4)
            angle = random.uniform(0.3, 0.8)
            speed = random.uniform(0.4, 0.9)
            self.shooting_stars.append(ShootingStar(
                sx, sy, math.cos(angle) * speed, math.sin(angle) * speed,
                random.uniform(0.6, 1.2), 1.0, random.uniform(1.5, 3.0)
            ))
        write = 0
        for ss in self.shooting_stars:
            ss.life -= dt
            if ss.life > 0:
                ss.x += ss.vx * dt
                ss.y += ss.vy * dt
                self.shooting_stars[write] = ss
                write += 1
        del self.shooting_stars[write:]

    def draw(self, painter: QPainter, w, h, t, color, is_rainbow=False):
        painter.setPen(Qt.NoPen)
        for sx, sy, br, sz, phase in self.stars:
            twinkle = 0.5 + 0.5 * math.sin(t * (1.2 + br) + phase)
            c = _hsv((sx + t * 0.04) % 1, 0.5, br * 0.4, int(br * 50 * twinkle)) if is_rainbow else QColor(color)
            if not is_rainbow:
                a = max(0, min(255, int(br * color.alpha() * twinkle)))
                c.setAlpha(a)
            dx = math.sin(t * 0.08 + phase) * 3
            dy = math.cos(t * 0.06 + phase) * 2
            painter.setBrush(QBrush(c))
            if br > 0.7:
                gc = QColor(c); gc.setAlpha(max(0, gc.alpha() // 3))
                painter.setBrush(QBrush(gc))
                painter.drawEllipse(QPointF(sx * w + dx, sy * h + dy), sz * 3, sz * 3)
                painter.setBrush(QBrush(c))
            painter.drawEllipse(QPointF(sx * w + dx, sy * h + dy), sz * twinkle, sz * twinkle)

        for ss in self.shooting_stars:
            ratio = max(0, ss.life / ss.max_life)
            alpha = int(200 * ratio)
            sc = _hsv(0.55, 0.3, 1.0, alpha) if is_rainbow else QColor(color)
            if not is_rainbow: sc.setAlpha(alpha)
            tail_len = 0.06
            x1, y1 = ss.x * w, ss.y * h
            x2 = (ss.x - ss.vx * tail_len) * w
            y2 = (ss.y - ss.vy * tail_len) * h
            grad = QLinearGradient(x1, y1, x2, y2)
            sc1 = QColor(sc); sc1.setAlpha(alpha)
            sc2 = QColor(sc); sc2.setAlpha(0)
            grad.setColorAt(0, sc1); grad.setColorAt(1, sc2)
            painter.setPen(QPen(QBrush(grad), ss.size * ratio, Qt.SolidLine, Qt.RoundCap))
            painter.drawLine(QPointF(x1, y1), QPointF(x2, y2))
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(QColor(255, 255, 255, alpha)))
            painter.drawEllipse(QPointF(x1, y1), ss.size * ratio * 0.7, ss.size * ratio * 0.7)


class RippleSystem:
    def __init__(self): self.ripples = []
    def emit(self, x, y, color, mr=35, life=0.45): self.ripples.append([x, y, 0.0, mr, life, life, QColor(color)])
    def update(self, dt):
        write = 0
        for r in self.ripples:
            r[4] -= dt
            if r[4] > 0:
                r[2] = r[3] * (1 - r[4] / r[5])
                self.ripples[write] = r
                write += 1
        del self.ripples[write:]
    def draw(self, painter: QPainter):
        for r in self.ripples:
            ratio = r[4] / r[5]
            alpha_outer = max(0, int(80 * ratio))
            alpha_inner = max(0, int(140 * ratio))
            c_outer = QColor(r[6]); c_outer.setAlpha(alpha_outer)
            painter.setPen(QPen(c_outer, 3)); painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(QPointF(r[0], r[1]), r[2], r[2] * 0.35)
            c_inner = QColor(r[6]); c_inner.setAlpha(alpha_inner)
            painter.setPen(QPen(c_inner, 1.5))
            inner_r = r[2] * 0.85
            painter.drawEllipse(QPointF(r[0], r[1]), inner_r, inner_r * 0.35)


class NoteSplashSystem:
    def __init__(self): self.splashes = []
    def emit(self, x, y, text, color, size=14): self.splashes.append([x, y, text, QColor(color), size, 1.0, 0.0])
    def update(self, dt):
        write = 0
        for s in self.splashes:
            s[5] -= dt * 1.2; s[6] += dt; s[1] -= dt * 50
            if s[5] > 0:
                self.splashes[write] = s
                write += 1
        del self.splashes[write:]
    def draw(self, painter: QPainter):
        for s in self.splashes:
            x, y, text, color, size, alpha, age = s
            scale = 1.0 + (1.0 - alpha) * 0.4
            font_size = int(size * scale)
            shadow_c = QColor(0, 0, 0, max(0, min(180, int(alpha * 180))))
            painter.setPen(shadow_c); painter.setFont(QFont("SF Pro", font_size, QFont.Bold))
            painter.drawText(int(x + 1), int(y + 1), text)
            c = QColor(color); c.setAlpha(max(0, min(255, int(alpha * 255))))
            painter.setPen(c)
            painter.drawText(int(x), int(y), text)