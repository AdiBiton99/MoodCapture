"""
overlay_ui.py — Floating overlay UI — ניתוח רגשות מתוכן מסך

עיצוב: Dark glassmorphism, כפתור מצלמה עגול צף, פאנל תוצאות compact.
טכנולוגיה: PyQt5

מחלקות ציבוריות:
    EmotionOverlay        — חלון-אם בלתי נראה (always-on-top),
                            מכיל FloatingCameraButton ו-ResultsPanel.
    FloatingCameraButton  — כפתור מצלמה עגול 52px, ניתן לגרירה, gradient/glow.
    ResultsPanel          — פאנל תוצאות compact (~360px), auto-close 10s, כרטיסיות פנים.

API ציבורי (תואם main.py):
    overlay.set_capture_callback(fn)
    overlay.set_region_callback(fn)
    overlay.display_image(image, faces)
    overlay.update_results(results_dict)
    overlay._reset_buttons()
    overlay.hide() / overlay.show()

הרצה עצמאית:
    python ui/overlay_ui.py
"""

import sys
import cv2
import numpy as np

# ──────────────────────────────────────────
# Foreground-window watching (Windows only)
# ──────────────────────────────────────────
# Used to auto-hide bboxes + card when the user switches away from the
# application that was captured. No-op on non-Windows.
if sys.platform == "win32":
    import ctypes
    try:
        _USER32 = ctypes.windll.user32
    except Exception:
        _USER32 = None
else:
    _USER32 = None


def _get_foreground_hwnd() -> int:
    """Return the OS-level handle of the currently foreground window, or 0."""
    if _USER32 is None:
        return 0
    try:
        return int(_USER32.GetForegroundWindow())
    except Exception:
        return 0


def _get_window_title(hwnd: int) -> str:
    """Return the title of an OS window handle, or empty string."""
    if _USER32 is None or not hwnd:
        return ""
    try:
        length = int(_USER32.GetWindowTextLengthW(hwnd))
        if length <= 0:
            return ""
        buf = ctypes.create_unicode_buffer(length + 1)
        _USER32.GetWindowTextW(hwnd, buf, length + 1)
        return buf.value or ""
    except Exception:
        return ""


def _get_window_process_id(hwnd: int) -> int:
    """Return the OS process id that owns a given hwnd, or 0."""
    if _USER32 is None or not hwnd:
        return 0
    try:
        pid = ctypes.c_ulong(0)
        _USER32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        return int(pid.value)
    except Exception:
        return 0
from PyQt5.QtWidgets import (
    QApplication, QWidget, QFrame,
    QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QScrollArea, QAbstractScrollArea, QProgressBar, QGraphicsDropShadowEffect,
    QSizePolicy,
)
from PyQt5.QtCore import (
    Qt, QPoint, QPointF, QRectF, QTimer, QPropertyAnimation, QEasingCurve,
    QSize, pyqtSignal, QThread,
)
from PyQt5.QtGui import (
    QColor, QPainter, QBrush, QPen, QFont, QFontMetrics,
    QPixmap, QImage, QLinearGradient, QRadialGradient,
    QPainterPath, QRegion,
)


# ──────────────────────────────────────────
# צבעים — Clean Beige / Warm Neutral palette
# ──────────────────────────────────────────
C_BG        = "#F4EDE4"          # soft warm cream
C_PANEL     = "#FDFAF7"          # clean warm white
C_CARD      = "#F9F5F0"          # light warm card
C_BORDER    = "#DDD0BF"          # taupe border
C_TEXT      = "#3A2D22"          # deep warm brown
C_DIM       = "#9A8878"          # muted taupe
C_PRIMARY   = "#A0806A"          # warm medium brown (coffee)
C_ACCENT    = "#7A5C44"          # deeper espresso brown
C_SUCCESS   = "#7AAE82"          # muted sage green
C_WARNING   = "#C4A070"          # warm sand
C_DANGER    = "#C26B5A"          # muted terracotta
C_SECONDARY = "#7A9BAD"          # muted slate blue

EMOTION_COLORS = {
    "happy":    "#4CAE6C",   # ירוק — כמו בפוסטר
    "sad":      "#4F82C8",   # כחול-פלדה
    "angry":    "#D05830",   # כתום-אדום
    "neutral":  "#8A8480",   # אפור חם
    "surprise": "#C87818",   # ענבר כהה
    "fear":     "#8850C8",   # סגול
    "disgust":  "#C84888",   # ורוד-מגנטה
}

ALL_EMOTIONS = ["happy", "neutral", "sad", "angry", "fear", "surprise", "disgust"]


# ──────────────────────────────────────────
# stylesheet גלובלי
# ──────────────────────────────────────────
GLOBAL_SS = f"""
QWidget {{
    background: transparent;
    color: {C_TEXT};
    font-family: 'Segoe UI', 'Inter', 'Poppins', Arial, sans-serif;
}}
QScrollArea  {{ border: none; background: transparent; }}
QScrollBar:vertical {{
    background: {C_BORDER}; width: 4px; border-radius: 2px;
}}
QScrollBar::handle:vertical {{
    background: {C_PRIMARY}88; border-radius: 2px; min-height: 20px;
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QProgressBar {{
    border: none; border-radius: 3px;
    background: {C_BORDER}; height: 4px;
}}
QProgressBar::chunk {{
    border-radius: 3px;
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
        stop:0 {C_SUCCESS}, stop:1 {C_PRIMARY});
}}
"""


# ══════════════════════════════════════════
# AnalysisWorker — thread לניתוח ברקע
# ══════════════════════════════════════════

class _AnalysisWorker(QThread):
    """
    מריץ callback ניתוח (capture + DeepFace) בthread נפרד.

    הסיגנלים נשלחים תמיד לthread הראשי (Qt queued connection) —
    כך ש-update_results / _reset_buttons בצד ה-overlay בטוחים.
    """

    done   = pyqtSignal()
    failed = pyqtSignal(str)

    def __init__(self, fn):
        super().__init__()
        self._fn = fn

    def run(self) -> None:
        try:
            self._fn()
        except Exception as e:
            self.failed.emit(str(e))
        else:
            self.done.emit()


# ══════════════════════════════════════════
# FloatingCameraButton
# ══════════════════════════════════════════

class FloatingCameraButton(QWidget):
    """
    כפתור מצלמה עגול 52px, צף ו-draggable.

    סיגנלים:
        capture_clicked   — לחיצה רגילה (full screen)
        region_clicked    — לחיצה ימנית / DoubleClick (select region)

    מצבים:
        idle       — gradient כחול-סגול + אייקון מצלמה
        processing — spinner (animation)
    """

    capture_clicked = pyqtSignal()
    region_clicked  = pyqtSignal()

    _BTN_SIZE    = 60
    _WIDGET_SIZE = 74   # larger than circle to fit X badge
    _ICON_SIZE   = 26
    _GLOW_IDLE   = 22
    _GLOW_HOVER  = 40
    _CLOSE_R     = 10   # X badge radius
    _CLOSE_CX    = 74 - _CLOSE_R - 1   # top-right corner X
    _CLOSE_CY    = _CLOSE_R + 1

    close_clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._drag_offset   = QPoint()
        self._dragging      = False
        self._processing    = False
        self._hover         = False
        self._hover_close   = False
        self._spin_angle    = 0

        # spinner timer
        self._spin_timer = QTimer(self)
        self._spin_timer.setInterval(30)
        self._spin_timer.timeout.connect(self._tick_spinner)

        self.setFixedSize(self._WIDGET_SIZE, self._WIDGET_SIZE)
        self.setCursor(Qt.PointingHandCursor)
        self.setAttribute(Qt.WA_TranslucentBackground)

        # glow effect — soft brown, symmetric (no offset)
        self._shadow = QGraphicsDropShadowEffect(self)
        self._shadow.setBlurRadius(self._GLOW_IDLE)
        self._shadow.setColor(QColor(140, 100, 70, 130))   # soft brown glow
        self._shadow.setOffset(0, 0)
        self.setGraphicsEffect(self._shadow)

        # tooltip
        self.setToolTip("Left-click: Capture Screen  |  Right-click: Select Region")

    # ── ציור ──────────────────────────────

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        ws = self._WIDGET_SIZE
        cx, cy = ws / 2, ws / 2
        r = self._BTN_SIZE / 2 - 3

        if self._processing:
            self._draw_spinner(p, cx, cy, r)
        else:
            self._draw_button(p, cx, cy, r)

        if self._hover:
            self._draw_close_badge(p)

        p.end()

    def _draw_button(self, p: QPainter, cx, cy, r) -> None:
        grad = QLinearGradient(0, 0, self._BTN_SIZE, self._BTN_SIZE)
        grad.setColorAt(0.0, QColor(C_PRIMARY))
        grad.setColorAt(1.0, QColor(C_ACCENT))

        scale = 1.08 if self._hover else 1.0
        sr = r * scale

        # use QRectF so the circle stays perfectly centered (no int() drift)
        p.setBrush(QBrush(grad))
        p.setPen(Qt.NoPen)
        p.drawEllipse(QRectF(cx - sr, cy - sr, sr * 2, sr * 2))

        # inner glow ring
        ring_pen = QPen(QColor(255, 255, 255, 40 if self._hover else 20))
        ring_pen.setWidth(2)
        p.setPen(ring_pen)
        p.setBrush(Qt.NoBrush)
        p.drawEllipse(QRectF(cx - sr + 2, cy - sr + 2, sr * 2 - 4, sr * 2 - 4))

        # אייקון מצלמה (פשוט, גיאומטרי)
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(QColor(255, 255, 255, 230)))
        self._draw_camera_icon(p, cx, cy, self._ICON_SIZE)

    def _draw_spinner(self, p: QPainter, cx, cy, r) -> None:
        # רקע לבן חצי-שקוף
        p.setBrush(QBrush(QColor(255, 255, 255, 200)))
        p.setPen(Qt.NoPen)
        p.drawEllipse(int(cx - r), int(cy - r), int(r * 2), int(r * 2))

        # קשת מסתובבת — soft blue
        pen = QPen(QColor(108, 142, 255, 220))
        pen.setWidth(4)
        pen.setCapStyle(Qt.RoundCap)
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)
        p.save()
        p.translate(cx, cy)
        p.rotate(self._spin_angle)
        arc_r = int(r * 0.65)
        p.drawArc(-arc_r, -arc_r, arc_r * 2, arc_r * 2, 0, 270 * 16)
        p.restore()

    @staticmethod
    def _draw_camera_icon(p: QPainter, cx, cy, size) -> None:
        """אייקון מצלמה — outline style (קווי מתאר), נקי ומודרני."""
        stroke_w = max(1.8, size * 0.085)
        pen = QPen(QColor(255, 255, 255, 235), stroke_w)
        pen.setJoinStyle(Qt.RoundJoin)
        pen.setCapStyle(Qt.RoundCap)
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)

        # ── גוף מצלמה ───────────────────────────────
        bw = size * 0.88
        bh = size * 0.58
        bx = cx - bw / 2
        by = cy - bh / 2 + size * 0.07
        body = QPainterPath()
        body.addRoundedRect(QRectF(bx, by, bw, bh), 4.5, 4.5)
        p.drawPath(body)

        # ── viewfinder bump (קטן, מרוכז) ────────────
        vw = size * 0.27
        vh = size * 0.15
        notch = QPainterPath()
        notch.addRoundedRect(QRectF(cx - vw / 2, by - vh + stroke_w * 0.5, vw, vh), 2.5, 2.5)
        p.drawPath(notch)

        # ── עדשה — ring ─────────────────────────────
        lens_r = size * 0.215
        mid_y  = by + bh * 0.52
        p.drawEllipse(QRectF(cx - lens_r, mid_y - lens_r, lens_r * 2, lens_r * 2))

        # ── עדשה — עיגול פנימי קטן ──────────────────
        inner_r = size * 0.10
        p.setBrush(QBrush(QColor(255, 255, 255, 200)))
        p.setPen(Qt.NoPen)
        p.drawEllipse(QRectF(cx - inner_r, mid_y - inner_r, inner_r * 2, inner_r * 2))

        # ── נורית (flash) — נקודה בפינה ימנית עליונה
        p.setPen(pen)
        p.setBrush(QBrush(QColor(255, 255, 255, 180)))
        flash_r = size * 0.065
        p.setPen(Qt.NoPen)
        p.drawEllipse(QRectF(
            bx + bw - flash_r * 2.4,
            by + bh * 0.18,
            flash_r * 2, flash_r * 2,
        ))

    def _draw_close_badge(self, p: QPainter) -> None:
        """X badge בפינה ימנית עליונה לסגירת האפליקציה."""
        cx = float(self._CLOSE_CX)
        cy = float(self._CLOSE_CY)
        r  = float(self._CLOSE_R)
        bg = QColor("#FC8181") if self._hover_close else QColor(60, 60, 80, 200)
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(bg))
        p.drawEllipse(QRectF(cx - r, cy - r, r * 2, r * 2))
        pen = QPen(QColor(255, 255, 255, 230), 1.8)
        pen.setCapStyle(Qt.RoundCap)
        p.setPen(pen)
        off = r * 0.42
        p.drawLine(QPointF(cx - off, cy - off), QPointF(cx + off, cy + off))
        p.drawLine(QPointF(cx + off, cy - off), QPointF(cx - off, cy + off))

    def _is_on_close(self, pos: QPoint) -> bool:
        dx = pos.x() - self._CLOSE_CX
        dy = pos.y() - self._CLOSE_CY
        return dx * dx + dy * dy <= self._CLOSE_R ** 2

    # ── spinner ────────────────────────────

    def _tick_spinner(self) -> None:
        self._spin_angle = (self._spin_angle + 12) % 360
        self.update()

    def set_processing(self, processing: bool) -> None:
        self._processing = processing
        if processing:
            self._shadow.setColor(QColor(100, 70, 50, 180))
            self._shadow.setBlurRadius(self._GLOW_HOVER)
            self._spin_timer.start()
        else:
            self._shadow.setColor(QColor(140, 100, 70, 130))
            self._shadow.setBlurRadius(self._GLOW_IDLE)
            self._spin_timer.stop()
        self.update()

    # ── עכבר ───────────────────────────────

    def enterEvent(self, event) -> None:
        self._hover = True
        if not self._processing:
            self._shadow.setColor(QColor(100, 70, 50, 180))
            self._shadow.setBlurRadius(self._GLOW_HOVER)
        self.update()

    def leaveEvent(self, event) -> None:
        self._hover       = False
        self._hover_close = False
        if not self._processing:
            self._shadow.setColor(QColor(140, 100, 70, 130))
            self._shadow.setBlurRadius(self._GLOW_IDLE)
        self.update()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            if not self._is_on_close(event.pos()):
                self._drag_offset = event.globalPos() - self.parent().pos()
            self._dragging = False
        elif event.button() == Qt.RightButton:
            self.region_clicked.emit()

    def mouseMoveEvent(self, event) -> None:
        prev = self._hover_close
        self._hover_close = self._is_on_close(event.pos())
        if prev != self._hover_close:
            self.update()
        if event.buttons() == Qt.LeftButton and not self._hover_close:
            self._dragging = True
            new_pos = event.globalPos() - self._drag_offset
            self.parent().move(new_pos)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            if self._is_on_close(event.pos()):
                self.close_clicked.emit()
            elif not self._dragging and not self._processing:
                self.capture_clicked.emit()
        self._dragging = False

    def mouseDoubleClickEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self.region_clicked.emit()


# ══════════════════════════════════════════
# FaceCard — כרטיסיית פנים (עיצוב מחדש)
# ══════════════════════════════════════════

class FaceCard(QFrame):
    """
    כרטיסיית פנים מודרנית:
      • תמונה עגולה של הפנים
      • Badge "✓ Identified"
      • שם רגש bold + אחוז גדול
      • Progress bar דק עם gradient
    """

    THUMB_SIZE = 58

    def __init__(
        self,
        face_index:   int,
        emotion:      str,
        confidence:   float,
        face_image:   np.ndarray = None,
        all_emotions: dict       = None,
        parent=None,
    ):
        super().__init__(parent)
        self._emotion    = (emotion or "unknown").lower()
        self._confidence = confidence
        self._index      = face_index
        self._face_image = face_image
        self._color      = EMOTION_COLORS.get(self._emotion, C_PRIMARY)
        self._build()
        self._style()

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(14, 12, 14, 11)
        root.setSpacing(10)

        # ── שורה עליונה: תמונה + מידע + badge ──────
        top = QHBoxLayout()
        top.setSpacing(12)

        thumb = self._make_thumb()
        top.addWidget(thumb, 0, Qt.AlignVCenter)

        info = QVBoxLayout()
        info.setSpacing(3)

        face_lbl = QLabel(f"FACE {self._index + 1}")
        face_lbl.setStyleSheet(
            f"font-size:12px; font-weight:700; color:{C_DIM}; letter-spacing:2px; background:transparent;"
        )

        # שם הרגש עם נקודה צבעונית
        emo_row = QHBoxLayout()
        emo_row.setSpacing(6)
        emo_row.setContentsMargins(0, 0, 0, 0)

        dot = QLabel("●")
        dot.setStyleSheet(f"font-size:9px; color:{self._color}; background:transparent;")

        emo_lbl = QLabel(self._emotion.capitalize())
        emo_lbl.setStyleSheet(
            f"font-size:20px; font-weight:900; color:{self._color}; background:transparent; letter-spacing:-0.5px;"
        )
        emo_row.addWidget(dot, 0, Qt.AlignVCenter)
        emo_row.addWidget(emo_lbl, 0, Qt.AlignVCenter)
        emo_row.addStretch()

        conf_lbl = QLabel(f"{self._confidence:.0%}")
        conf_lbl.setStyleSheet(
            f"font-size:12px; font-weight:600; color:{C_DIM}; background:transparent;"
        )

        info.addWidget(face_lbl)
        info.addLayout(emo_row)
        info.addWidget(conf_lbl)
        top.addLayout(info, 1)

        # Badge "✓ Identified"
        badge = QLabel("✓  Identified")
        badge.setAlignment(Qt.AlignCenter)
        badge.setFixedHeight(22)
        badge.setStyleSheet(f"""
            background: {self._color}1A;
            color: {self._color};
            border: 1px solid {self._color}55;
            border-radius: 10px;
            font-size: 9px;
            font-weight: 700;
            padding: 0 8px;
        """)
        top.addWidget(badge, 0, Qt.AlignTop)
        root.addLayout(top)

        # ── Progress bar ────────────────────────────
        bar = QProgressBar()
        bar.setRange(0, 100)
        bar.setValue(0)
        bar.setTextVisible(False)
        bar.setFixedHeight(5)
        bar.setStyleSheet(f"""
            QProgressBar {{
                border: none;
                border-radius: 3px;
                background: {C_BORDER};
            }}
            QProgressBar::chunk {{
                border-radius: 3px;
                background: {self._color};
            }}
        """)
        anim = QPropertyAnimation(bar, b"value")
        anim.setDuration(900)
        anim.setEasingCurve(QEasingCurve.OutCubic)
        anim.setEndValue(int(self._confidence * 100))
        QTimer.singleShot(120, anim.start)
        self._bar_anim = anim
        root.addWidget(bar)

    def _make_thumb(self) -> QLabel:
        lbl = QLabel()
        s = self.THUMB_SIZE
        lbl.setFixedSize(s, s)
        lbl.setAlignment(Qt.AlignCenter)

        if self._face_image is not None:
            pm = self._make_circular_pixmap(self._face_image, s)
            if pm:
                lbl.setPixmap(pm)
                lbl.setStyleSheet(f"""
                    border: 2px solid {self._color}88;
                    border-radius: {s // 2}px;
                    background: transparent;
                """)
                return lbl

        # fallback: ראשית הרגש על רקע gradient
        lbl.setText(self._emotion[:1].upper())
        lbl.setStyleSheet(f"""
            background: qradialgradient(
                cx:0.5, cy:0.5, radius:0.5,
                stop:0 {self._color}44, stop:1 {self._color}11
            );
            border: 2px solid {self._color}66;
            border-radius: {s // 2}px;
            font-size: 22px;
            font-weight: 800;
            color: {self._color};
        """)
        return lbl

    @staticmethod
    def _make_circular_pixmap(img: np.ndarray, size: int) -> "QPixmap | None":
        """חותך את התמונה לעיגול חלק."""
        try:
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif img.ndim == 3 and img.shape[2] == 1:
                img = cv2.cvtColor(img[:, :, 0], cv2.COLOR_GRAY2RGB)
            if img.dtype != np.uint8:
                img = (img * 255).clip(0, 255).astype(np.uint8)
            img = cv2.resize(img, (size, size))
            h, w, ch = img.shape
            source = QPixmap.fromImage(
                QImage(img.data, w, h, w * ch, QImage.Format_RGB888)
            )
            # ציור מעגלי
            result = QPixmap(size, size)
            result.fill(Qt.transparent)
            painter = QPainter(result)
            painter.setRenderHint(QPainter.Antialiasing)
            path = QPainterPath()
            path.addEllipse(0, 0, size, size)
            painter.setClipPath(path)
            painter.drawPixmap(0, 0, source)
            painter.end()
            return result
        except Exception:
            return None

    def _style(self) -> None:
        self.setStyleSheet(f"""
            FaceCard {{
                background: #FDFAF7;
                border: 1.5px solid {self._color}44;
                border-radius: 14px;
            }}
            FaceCard:hover {{
                background: #FFFFFF;
                border: 1.5px solid {self._color}99;
            }}
        """)


# ══════════════════════════════════════════
# ResultsPanel — פאנל תוצאות צף
# ══════════════════════════════════════════

class ResultsPanel(QFrame):
    """
    פאנל תוצאות dark-glass compact (~360px).
    נסגר אוטומטית לאחר AUTO_CLOSE_MS (10 שניות), או בלחיצת X.
    """

    AUTO_CLOSE_MS = 10_000
    PANEL_WIDTH   = 360

    def __init__(self, parent=None):
        super().__init__(parent)
        self._face_cards: list[FaceCard] = []
        self._auto_timer = QTimer(self)
        self._auto_timer.setSingleShot(True)
        self._auto_timer.timeout.connect(self.hide)

        # countdown label
        self._countdown_val = self.AUTO_CLOSE_MS // 1000
        self._countdown_timer = QTimer(self)
        self._countdown_timer.setInterval(1000)
        self._countdown_timer.timeout.connect(self._tick_countdown)

        self._build()
        self._style()
        self.setFixedWidth(self.PANEL_WIDTH)
        screen_h = QApplication.primaryScreen().geometry().height()
        self.setMaximumHeight(int(screen_h * 0.85))
        self.adjustSize()
        self.hide()

    # ── בנייה ─────────────────────────────

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ══ HEADER BAR ══════════════════════════════
        header_frame = QFrame()
        header_frame.setStyleSheet(f"""
            QFrame {{
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 #F4EDE4, stop:1 #EFE6DA
                );
                border-top-left-radius: 20px;
                border-top-right-radius: 20px;
                border-bottom: 1px solid {C_BORDER};
            }}
        """)
        hbox = QHBoxLayout(header_frame)
        hbox.setContentsMargins(16, 12, 12, 12)
        hbox.setSpacing(8)

        title_lbl = QLabel("MoodCapture")
        title_lbl.setStyleSheet(
            f"font-size:15px; font-weight:800; color:{C_TEXT}; letter-spacing:-0.3px;"
        )

        hbox.addWidget(title_lbl)
        hbox.addStretch()

        # countdown label
        self._countdown_lbl = QLabel("")
        self._countdown_lbl.setStyleSheet(f"font-size:10px; color:{C_DIM}; background:transparent;")
        hbox.addWidget(self._countdown_lbl)

        # ✕ close
        close_btn = QPushButton("✕")
        close_btn.setFixedSize(26, 26)
        close_btn.setCursor(Qt.PointingHandCursor)
        close_btn.setStyleSheet(f"""
            QPushButton {{
                background: transparent;
                color: {C_DIM};
                border: none;
                border-radius: 13px;
                font-size: 11px;
            }}
            QPushButton:hover {{
                background: {C_DANGER}22;
                color: {C_DANGER};
            }}
        """)
        close_btn.clicked.connect(self.hide)
        hbox.addWidget(close_btn)
        root.addWidget(header_frame)

        # ══ STATS STRIP ══════════════════════════════
        stats_frame = QFrame()
        stats_frame.setStyleSheet(f"""
            QFrame {{
                background: #F2E8DE;
                border-bottom: 1px solid {C_BORDER};
            }}
        """)
        sbox = QHBoxLayout(stats_frame)
        sbox.setContentsMargins(16, 10, 16, 10)
        sbox.setSpacing(0)

        # Total Faces block
        tot_col = QVBoxLayout()
        tot_col.setSpacing(2)
        tot_title = QLabel("Total Faces")
        tot_title.setStyleSheet(
            f"font-size:9px; font-weight:600; color:{C_DIM}; letter-spacing:1.2px; text-transform:uppercase;"
        )
        self._total_count = QLabel("—")
        self._total_count.setStyleSheet(
            f"font-size:24px; font-weight:800; color:{C_PRIMARY};"
        )
        tot_col.addWidget(tot_title)
        tot_col.addWidget(self._total_count)
        sbox.addLayout(tot_col)

        sbox.addStretch()

        root.addWidget(stats_frame)

        # ══ CARDS AREA (no scroll — panel grows dynamically) ═════════
        cards_container = QWidget()
        cards_container.setStyleSheet("background: transparent;")
        cards_container.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self._cards_layout = QVBoxLayout(cards_container)
        self._cards_layout.setContentsMargins(12, 12, 12, 12)
        self._cards_layout.setSpacing(8)

        self._placeholder = self._make_placeholder()
        self._cards_layout.addWidget(self._placeholder)

        self._cards_container = cards_container
        root.addWidget(cards_container)

        # ══ FOOTER: Select Region ════════════════════
        footer_frame = QFrame()
        footer_frame.setStyleSheet(f"""
            QFrame {{
                background: #F4EDE4;
                border-top: 1px solid {C_BORDER};
                border-bottom-left-radius: 20px;
                border-bottom-right-radius: 20px;
            }}
        """)
        fbox = QHBoxLayout(footer_frame)
        fbox.setContentsMargins(14, 10, 14, 10)

        self._region_btn = QPushButton("⊡  Select Region")
        self._region_btn.setCursor(Qt.PointingHandCursor)
        self._region_btn.setStyleSheet(f"""
            QPushButton {{
                background: white;
                color: {C_DIM};
                border: 1px solid {C_BORDER};
                border-radius: 10px;
                font-size: 11px;
                font-weight: 600;
                padding: 7px 18px;
            }}
            QPushButton:hover {{
                border-color: {C_PRIMARY};
                color: {C_PRIMARY};
                background: {C_PRIMARY}11;
            }}
        """)
        fbox.addStretch()
        fbox.addWidget(self._region_btn)
        fbox.addStretch()
        root.addWidget(footer_frame)

    @staticmethod
    def _make_placeholder() -> QLabel:
        lbl = QLabel("😶  No faces detected")
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setStyleSheet(
            f"font-size:13px; color:{C_DIM}; padding:32px 0; background:transparent;"
        )
        return lbl

    def _style(self) -> None:
        self.setStyleSheet(f"""
            ResultsPanel {{
                background: rgba(253, 250, 247, 0.97);
                border: 1px solid {C_BORDER};
                border-radius: 20px;
            }}
        """)
        # soft warm brown shadow
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(44)
        shadow.setColor(QColor(120, 85, 60, 55))
        shadow.setOffset(0, 10)
        self.setGraphicsEffect(shadow)
        self.setCursor(Qt.SizeAllCursor)

    # ── גרירה ──────────────────────────────

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self._drag_offset = event.globalPos() - self.mapToGlobal(QPoint(0, 0))
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if event.buttons() & Qt.LeftButton and hasattr(self, '_drag_offset') and not self._drag_offset.isNull():
            new_global = event.globalPos() - self._drag_offset
            # ממפים לקואורדינטות של ה-parent
            new_local = self.parent().mapFromGlobal(new_global)
            # מגבילים לגבולות המסך (parent = full-screen overlay)
            parent_rect = self.parent().rect()
            x = max(0, min(new_local.x(), parent_rect.width()  - self.width()))
            y = max(0, min(new_local.y(), parent_rect.height() - self.height()))
            self.move(x, y)
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self._drag_offset = QPoint()
        super().mouseReleaseEvent(event)

    # hover אינו משפיע על הטיימר — הספירה רצה ברציפות

    # ── API ────────────────────────────────

    def show_results(self, results: dict) -> None:
        """קבל dict מ-EmotionAnalysisService, הצג ואתחל timer."""
        faces      = results.get("faces", [])
        final_emo  = results.get("final_emotion") or "—"
        final_conf = results.get("confidence", 0.0)
        n          = len(faces)

        # Total Faces counter
        self._total_count.setText(str(n) if n else "0")

        # כרטיסיות פנים
        self._clear_cards()
        if not faces:
            self._placeholder.show()
            self.adjustSize()
            self._reposition_on_parent()
        else:
            self._placeholder.hide()
            for i, fd in enumerate(faces):
                QTimer.singleShot(
                    i * 100,
                    lambda idx=i, d=fd: self._add_card(idx, d),
                )
            # מרחיב את הפאנל אחרי שכל הכרטיסיות נוצרו
            QTimer.singleShot(len(faces) * 100 + 80, self._fit_scroll_to_cards)

        self._restart_auto_close()
        self.show()

    def _add_card(self, idx: int, face_data: dict) -> None:
        card = FaceCard(
            face_index=idx,
            emotion=face_data.get("emotion", "unknown"),
            confidence=face_data.get("confidence", 0.0),
            face_image=face_data.get("face_image"),
            all_emotions=face_data.get("all_emotions"),
        )
        self._face_cards.append(card)
        self._cards_layout.addWidget(card)

    def _clear_cards(self) -> None:
        for c in self._face_cards:
            c.setParent(None)
            c.deleteLater()
        self._face_cards.clear()

    def _fit_scroll_to_cards(self) -> None:
        """מרחיב את הפאנל להכיל את כל הכרטיסיות בלי גלילה."""
        self.adjustSize()
        self._reposition_on_parent()

    def _reposition_on_parent(self) -> None:
        overlay = self.parent()
        if overlay and hasattr(overlay, '_reposition_panel'):
            overlay._reposition_panel()

    def _restart_auto_close(self) -> None:
        self._auto_timer.stop()
        self._countdown_timer.stop()
        self._countdown_val = self.AUTO_CLOSE_MS // 1000
        self._countdown_lbl.setText(f"{self._countdown_val}s")
        self._auto_timer.start(self.AUTO_CLOSE_MS)
        self._countdown_timer.start()

    def _tick_countdown(self) -> None:
        self._countdown_val -= 1
        self._countdown_lbl.setText(f"{self._countdown_val}s")
        if self._countdown_val <= 0:
            self._countdown_timer.stop()

    def stop_auto_close(self) -> None:
        self._auto_timer.stop()
        self._countdown_timer.stop()
        self._countdown_lbl.setText("")


# ══════════════════════════════════════════
# _AnnotatedImageCanvas — ציור מסגרות זיהוי
# ══════════════════════════════════════════

class _AnnotatedImageCanvas(QWidget):
    """מצייר את צילום המסך + תיבות זיהוי צבעוניות עם QPainter."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pixmap: "QPixmap | None" = None
        self._faces:  list  = []
        self._scale_x: float = 1.0
        self._scale_y: float = 1.0

    def set_content(self, pm: QPixmap, faces: list, sx: float, sy: float) -> None:
        self._pixmap  = pm
        self._faces   = faces
        self._scale_x = sx
        self._scale_y = sy
        self.setFixedSize(pm.size())
        self.update()

    def paintEvent(self, event) -> None:
        if not self._pixmap:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.TextAntialiasing)
        p.drawPixmap(0, 0, self._pixmap)

        for idx, face in enumerate(self._faces):
            bbox = face.get("bbox")
            if not bbox or len(bbox) < 4:
                continue
            fx, fy, fw, fh = bbox
            fx = int(fx * self._scale_x)
            fy = int(fy * self._scale_y)
            fw = int(fw * self._scale_x)
            fh = int(fh * self._scale_y)

            emotion    = (face.get("emotion") or "unknown").lower()
            confidence = face.get("confidence", 0.0)
            hex_col    = EMOTION_COLORS.get(emotion, C_PRIMARY)
            color      = QColor(hex_col)

            # faint fill
            fill = QColor(color); fill.setAlpha(22)
            p.fillRect(QRectF(fx, fy, fw, fh), fill)

            # thin full-rect border
            p.setPen(QPen(QColor(hex_col + "77"), 1.5))
            p.setBrush(Qt.NoBrush)
            p.drawRoundedRect(QRectF(fx, fy, fw, fh), 5, 5)

            # corner bracket accents
            bk = max(10, min(fw, fh) // 6)
            pen = QPen(color, 3.0); pen.setCapStyle(Qt.RoundCap)
            p.setPen(pen)
            for (x0, y0, x1, y1) in [
                (fx + bk, fy,      fx,      fy),
                (fx,      fy,      fx,      fy + bk),
                (fx+fw-bk,fy,      fx+fw,   fy),
                (fx+fw,   fy,      fx+fw,   fy + bk),
                (fx,      fy+fh-bk,fx,      fy+fh),
                (fx,      fy+fh,   fx + bk, fy+fh),
                (fx+fw-bk,fy+fh,   fx+fw,   fy+fh),
                (fx+fw,   fy+fh-bk,fx+fw,   fy+fh),
            ]:
                p.drawLine(QPointF(x0, y0), QPointF(x1, y1))

            # label above bbox
            face_txt = f"Face {idx + 1}"
            conf_txt = f"{emotion.capitalize()}  {confidence:.0%}"

            fnt_b = QFont("Segoe UI", 8); fnt_b.setWeight(QFont.Bold)
            fnt_r = QFont("Segoe UI", 7)
            fm_b  = QFontMetrics(fnt_b)
            fm_r  = QFontMetrics(fnt_r)

            lw = max(fm_b.horizontalAdvance(face_txt),
                     fm_r.horizontalAdvance(conf_txt)) + 18
            lh = fm_b.height() + fm_r.height() + 10
            lx = fx
            ly = max(0, fy - lh - 4)

            # label background
            bg = QColor(C_PANEL); bg.setAlpha(225)
            p.setBrush(QBrush(bg)); p.setPen(Qt.NoPen)
            p.drawRoundedRect(QRectF(lx, ly, lw, lh), 6, 6)

            # color accent strip
            p.setBrush(QBrush(color))
            p.drawRoundedRect(QRectF(lx, ly, 4, lh), 2, 2)

            # face label (colored, bold)
            p.setFont(fnt_b); p.setPen(color)
            p.drawText(QRectF(lx + 8, ly + 3, lw - 10, fm_b.height() + 2),
                       Qt.AlignLeft | Qt.AlignVCenter, face_txt)

            # emotion + confidence (dark, regular)
            p.setFont(fnt_r); p.setPen(QColor(C_TEXT))
            p.drawText(QRectF(lx + 8, ly + fm_b.height() + 5, lw - 10, fm_r.height() + 2),
                       Qt.AlignLeft | Qt.AlignVCenter, conf_txt)

        p.end()


# ══════════════════════════════════════════
# AnnotatedImageViewer — חלון צף לצילום מסומן
# ══════════════════════════════════════════

class AnnotatedImageViewer(QFrame):
    """
    חלון צף המציג את הצילום עם תיבות זיהוי צבעוניות לכל פנים.
    נסגר אוטומטית אחרי 15 שניות או בלחיצת X.
    """

    AUTO_CLOSE_MS = 15_000
    MAX_W = 520
    MAX_H = 350

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build()
        self._style()
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(28)
        shadow.setColor(QColor(C_ACCENT + "44"))
        shadow.setOffset(0, 4)
        self.setGraphicsEffect(shadow)
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self.hide)
        self.hide()

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Header bar
        header = QFrame()
        header.setStyleSheet(f"""
            QFrame {{
                background: {C_PANEL};
                border-top-left-radius: 16px;
                border-top-right-radius: 16px;
                border-bottom: 1px solid {C_BORDER};
            }}
        """)
        hbox = QHBoxLayout(header)
        hbox.setContentsMargins(14, 9, 10, 9)

        title = QLabel("Detected Faces")
        title.setStyleSheet(
            f"font-size:12px; font-weight:700; color:{C_TEXT}; background:transparent;"
        )
        close_btn = QPushButton("✕")
        close_btn.setFixedSize(24, 24)
        close_btn.setCursor(Qt.PointingHandCursor)
        close_btn.setStyleSheet(f"""
            QPushButton {{
                background: transparent; color: {C_DIM};
                border: none; border-radius: 12px; font-size: 10px;
            }}
            QPushButton:hover {{ background: {C_DANGER}22; color: {C_DANGER}; }}
        """)
        close_btn.clicked.connect(self.hide)

        hbox.addWidget(title)
        hbox.addStretch()
        hbox.addWidget(close_btn)
        root.addWidget(header)

        # Image canvas
        self._canvas = _AnnotatedImageCanvas()
        root.addWidget(self._canvas)

    def _style(self) -> None:
        self.setStyleSheet(f"""
            AnnotatedImageViewer {{
                background: {C_PANEL};
                border-radius: 16px;
                border: 1px solid {C_BORDER};
            }}
        """)

    def show_image(self, image: "np.ndarray", faces: list) -> None:
        """Scales image, draws bboxes and shows the viewer."""
        if image is None:
            return
        pm = _numpy_to_pixmap(image)
        if pm is None:
            return
        scaled = pm.scaled(
            self.MAX_W, self.MAX_H,
            Qt.KeepAspectRatio, Qt.SmoothTransformation,
        )
        sx = scaled.width()  / max(image.shape[1], 1)
        sy = scaled.height() / max(image.shape[0], 1)
        self._canvas.set_content(scaled, faces, sx, sy)
        self.setFixedWidth(scaled.width())
        self.adjustSize()
        self.show()
        self.raise_()
        self._timer.start(self.AUTO_CLOSE_MS)


# ══════════════════════════════════════════
# NoFacesToast — הודעת חיווי קצרה
# ══════════════════════════════════════════

class NoFacesToast(QFrame):
    """
    Small auto-hiding banner ("No faces detected on screen") shown
    centered near the top of the screen when an analysis comes back
    empty. Auto-disappears after `AUTO_HIDE_MS`. Stays out of the way
    of the camera button.
    """

    AUTO_HIDE_MS = 3500

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(
            Qt.FramelessWindowHint |
            Qt.WindowStaysOnTopHint |
            Qt.Tool
        )
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)

        wrap = QHBoxLayout(self)
        wrap.setContentsMargins(0, 0, 0, 0)

        card = QFrame()
        card.setObjectName("NoFacesToastCard")
        card.setStyleSheet(f"""
            QFrame#NoFacesToastCard {{
                background: {C_PANEL};
                border: 1px solid {C_BORDER};
                border-radius: 14px;
            }}
        """)
        shadow = QGraphicsDropShadowEffect(card)
        shadow.setBlurRadius(28)
        shadow.setOffset(0, 6)
        shadow.setColor(QColor(0, 0, 0, 60))
        card.setGraphicsEffect(shadow)

        cv = QHBoxLayout(card)
        cv.setContentsMargins(16, 12, 18, 12)
        cv.setSpacing(10)

        icon = QLabel("⚠")
        icon.setStyleSheet(
            f"font-size: 18px; color: {C_WARNING}; background: transparent;"
        )
        title = QLabel("No faces detected on the captured screen.")
        title.setStyleSheet(
            f"font-size: 12px; font-weight: 700; color: {C_TEXT};"
            f"background: transparent;"
        )
        cv.addWidget(icon)
        cv.addWidget(title)

        wrap.addWidget(card)
        self.adjustSize()

        self._hide_timer = QTimer(self)
        self._hide_timer.setSingleShot(True)
        self._hide_timer.setInterval(self.AUTO_HIDE_MS)
        self._hide_timer.timeout.connect(self.hide)
        self.hide()

    def pop(self) -> None:
        """Show the toast and arm the auto-hide timer."""
        self.adjustSize()
        screen = QApplication.primaryScreen().geometry()
        x = (screen.width() - self.width()) // 2
        y = max(40, int(screen.height() * 0.08))
        self.move(x, y)
        self.show()
        self.raise_()
        self._hide_timer.start()


# ══════════════════════════════════════════
# BBoxScreenOverlay — תיבות זיהוי על המסך
# ══════════════════════════════════════════

class BBoxScreenOverlay(QWidget):
    """
    Transparent, full-screen overlay that draws face-detection bboxes on top
    of whatever is on the screen.

    Behavior:
      * Stays visible until a new analysis replaces it or the app closes.
      * Each bbox is CLICKABLE. Clicking emits `face_clicked(face_index)`.
      * Outside the bbox regions, the window is invisible AND click-through
        (achieved via setMask) so the user can still interact with the
        underlying application (browser, Zoom, etc).
      * Active (selected) bbox is rendered with thicker brackets + brighter
        border to provide visual feedback.
    """

    face_clicked = pyqtSignal(int)

    def __init__(self):
        super().__init__(
            None,
            Qt.FramelessWindowHint |
            Qt.WindowStaysOnTopHint |
            Qt.Tool,
        )
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_NoSystemBackground)
        # NOTE: NO WA_TransparentForMouseEvents. We use setMask() so only
        # the bbox rectangles receive clicks; everywhere else the window
        # is masked out and clicks pass to the OS / underlying app.

        screen = QApplication.primaryScreen().geometry()
        self.setGeometry(screen)

        self._faces:    list  = []
        self._offset_x: int   = 0
        self._offset_y: int   = 0
        self._scale_x:  float = 1.0
        self._scale_y:  float = 1.0
        self._active_face_idx = None   # int or None

        self.setCursor(Qt.PointingHandCursor)
        self.hide()

    # ── Public API ────────────────────────

    def show_faces(self, faces: list,
                   offset_x: int = 0, offset_y: int = 0,
                   scale_x: float = 1.0, scale_y: float = 1.0) -> None:
        """Show bboxes for the given faces. Stays visible until cleared."""
        self._faces            = [f for f in faces if f.get("bbox")]
        self._offset_x         = offset_x
        self._offset_y         = offset_y
        self._scale_x          = scale_x
        self._scale_y          = scale_y
        self._active_face_idx  = None
        self._refresh_mask()
        self.update()
        if self._faces:
            self.show()
            self.raise_()
        else:
            self.hide()

    def hide_faces(self) -> None:
        """Hide all bboxes immediately."""
        self._faces           = []
        self._active_face_idx = None
        self.clearMask()
        self.hide()

    def set_active_face_idx(self, idx) -> None:
        """Programmatically mark a face as active (highlighted)."""
        if idx == self._active_face_idx:
            return
        self._active_face_idx = idx
        self.update()

    def get_active_face_idx(self):
        return self._active_face_idx

    # ── Geometry / mask ───────────────────

    def _bbox_to_screen(self, bbox):
        x, y, w, h = bbox[:4]
        sx = self._scale_x if self._scale_x > 0 else 1.0
        sy = self._scale_y if self._scale_y > 0 else 1.0
        fx = int((x + self._offset_x) * sx)
        fy = int((y + self._offset_y) * sy)
        fw = int(w * sx)
        fh = int(h * sy)
        return fx, fy, fw, fh

    def _label_geometry_for_face(self, idx: int, fx: int, fy: int,
                                 fw: int, fh: int):
        """
        Return (lx, ly, tw, th, face_pt, emo_pt) for the in/around-bbox
        label of face #`idx`. Geometry is computed *only* from the face
        size, so very small bboxes get smaller fonts and labels placed
        ABOVE the bbox (when space allows) so they don't cover the face.

        Used by both `_refresh_mask` and `paintEvent` — they MUST agree
        on the final rectangle, otherwise the label gets clipped.
        """
        face = self._faces[idx]
        emotion = (face.get("emotion") or "unknown").lower()
        conf    = face.get("confidence", 0.0)
        face_text = f"FACE {idx + 1}"
        emo_text  = f"{emotion.capitalize()}  {conf:.0%}"

        face_size = max(20, min(fw, fh))
        scale = max(0.55, min(1.0, face_size / 180.0))
        face_pt = max(6, int(round(8 * scale)))
        emo_pt  = max(7, int(round(9 * scale)))

        face_fnt = QFont("Segoe UI", face_pt)
        face_fnt.setWeight(QFont.Bold)
        emo_fnt = QFont("Segoe UI", emo_pt)
        emo_fnt.setWeight(QFont.Bold)
        fm_face = QFontMetrics(face_fnt)
        fm_emo  = QFontMetrics(emo_fnt)

        pad_h = max(8, int(12 * scale))
        pad_v = max(6, int(10 * scale))

        tw = max(
            fm_face.horizontalAdvance(face_text),
            fm_emo.horizontalAdvance(emo_text),
        ) + pad_h * 2
        th = fm_face.height() + fm_emo.height() + pad_v

        # Prefer placing the label ABOVE the bbox so it doesn't cover the
        # actual face. If there is no room above (face is near top of
        # screen), tuck it INSIDE the top-left corner instead.
        screen_geo = self.geometry()
        gap = 4
        if fy - th - gap >= 0:
            lx = fx
            ly = fy - th - gap
        elif fy + fh + th + gap <= screen_geo.height():
            lx = fx
            ly = fy + fh + gap
        else:
            lx = fx + 4
            ly = fy + 4

        # Keep label fully on-screen horizontally too
        if lx + tw > screen_geo.width():
            lx = max(0, screen_geo.width() - tw - 2)
        if lx < 0:
            lx = 0

        return lx, ly, tw, th, face_pt, emo_pt

    def _refresh_mask(self) -> None:
        """
        Build a QRegion containing the bbox rectangles AND each face's
        label area (computed by `_label_geometry_for_face`). Areas
        outside this region are click-through and invisible.
        """
        if not self._faces:
            self.clearMask()
            return
        pad = 3
        region = QRegion()

        for idx, face in enumerate(self._faces):
            bbox = face.get("bbox")
            if not bbox:
                continue
            fx, fy, fw, fh = self._bbox_to_screen(bbox)
            region = region.united(
                QRegion(fx - pad, fy - pad, fw + 2 * pad, fh + 2 * pad)
            )

            lx, ly, tw, th, _, _ = self._label_geometry_for_face(
                idx, fx, fy, fw, fh
            )
            region = region.united(
                QRegion(lx - pad, ly - pad, tw + 2 * pad, th + 2 * pad)
            )
        self.setMask(region)

    # ── Mouse ─────────────────────────────

    def face_at_position(self, pos) -> "int | None":
        """
        Return the index of the face whose bbox contains the (screen-space)
        point `pos`, or None if no bbox is hit. Used by `EmotionOverlay`
        to forward clicks that the OS routed to the full-screen overlay
        window instead of this masked one.
        """
        for idx, face in enumerate(self._faces):
            bbox = face.get("bbox")
            if not bbox:
                continue
            fx, fy, fw, fh = self._bbox_to_screen(bbox)
            if fx <= pos.x() <= fx + fw and fy <= pos.y() <= fy + fh:
                return idx
        return None

    def mousePressEvent(self, event) -> None:
        if event.button() != Qt.LeftButton:
            super().mousePressEvent(event)
            return
        idx = self.face_at_position(event.pos())
        if idx is not None:
            self.set_active_face_idx(idx)
            self.face_clicked.emit(idx)
            event.accept()
            return
        super().mousePressEvent(event)

    # ── Painting ──────────────────────────

    def paintEvent(self, event) -> None:
        if not self._faces:
            return

        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.TextAntialiasing)

        for idx, face in enumerate(self._faces):
            bbox = face.get("bbox")
            if not bbox or len(bbox) < 4:
                continue

            fx, fy, fw, fh = self._bbox_to_screen(bbox)

            emotion   = (face.get("emotion") or "unknown").lower()
            conf      = face.get("confidence", 0.0)
            hex_col   = EMOTION_COLORS.get(emotion, C_PRIMARY)
            color     = QColor(hex_col)
            is_active = (idx == self._active_face_idx)

            # ── Fill ──
            fill = QColor(color)
            fill.setAlpha(40 if is_active else 18)
            p.fillRect(QRectF(fx, fy, fw, fh), fill)

            # ── Full-rect border ──
            border_pen = QPen(
                QColor(hex_col + ("CC" if is_active else "88")),
                3.0 if is_active else 2.0,
            )
            p.setPen(border_pen)
            p.setBrush(Qt.NoBrush)
            p.drawRoundedRect(QRectF(fx, fy, fw, fh), 6, 6)

            # ── Corner brackets ──
            bk = max(14, min(fw, fh) // 6)
            bracket = QPen(color, 5.0 if is_active else 3.5)
            bracket.setCapStyle(Qt.RoundCap)
            p.setPen(bracket)
            for (x0, y0, x1, y1) in [
                (fx + bk, fy,       fx,       fy),
                (fx,      fy,       fx,       fy + bk),
                (fx+fw-bk,fy,       fx + fw,  fy),
                (fx + fw, fy,       fx + fw,  fy + bk),
                (fx,      fy+fh-bk, fx,       fy + fh),
                (fx,      fy + fh,  fx + bk,  fy + fh),
                (fx+fw-bk,fy + fh,  fx + fw,  fy + fh),
                (fx + fw, fy+fh-bk, fx + fw,  fy + fh),
            ]:
                p.drawLine(QPointF(x0, y0), QPointF(x1, y1))

            # ── Two-line label (size + position scale with face) ──
            # Line 1: "FACE N" (so the user knows which face the AI is
            #         referring to when explaining)
            # Line 2: "<Emotion>  XX%"
            face_text = f"FACE {idx + 1}"
            emo_text  = f"{emotion.capitalize()}  {conf:.0%}"
            lx, ly, tw, th, face_pt, emo_pt = self._label_geometry_for_face(
                idx, fx, fy, fw, fh
            )
            face_fnt = QFont("Segoe UI", face_pt)
            face_fnt.setWeight(QFont.Bold)
            emo_fnt  = QFont("Segoe UI", emo_pt)
            emo_fnt.setWeight(QFont.Bold)
            fm_face = QFontMetrics(face_fnt)
            fm_emo  = QFontMetrics(emo_fnt)

            bg = QColor(C_PANEL)
            bg.setAlpha(235)
            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(bg))
            p.drawRoundedRect(QRectF(lx, ly, tw, th), 7, 7)

            p.setBrush(QBrush(color))
            p.drawRoundedRect(QRectF(lx, ly, 4, th), 2, 2)

            text_x = lx + 8
            text_w = tw - 12
            text_top = ly + 3
            p.setFont(face_fnt)
            p.setPen(color)
            p.drawText(
                QRectF(text_x, text_top, text_w, fm_face.height() + 2),
                Qt.AlignLeft | Qt.AlignVCenter,
                face_text,
            )

            p.setFont(emo_fnt)
            p.setPen(QColor(C_TEXT))
            p.drawText(
                QRectF(text_x, text_top + fm_face.height() + 2,
                       text_w, fm_emo.height() + 2),
                Qt.AlignLeft | Qt.AlignVCenter,
                emo_text,
            )

        p.end()


# ══════════════════════════════════════════
# EmotionOverlay — חלון-אם בלתי נראה
# ══════════════════════════════════════════

class EmotionOverlay(QWidget):
    """
    חלון-עטיפה always-on-top, frameless, שקוף.
    מכיל:
        • FloatingCameraButton (גרוע, top-right)
        • ResultsPanel (נפתח לשמאל/למטה מהכפתור)

    API תואם main.py:
        set_capture_callback(fn)
        set_region_callback(fn)
        display_image(image, faces)   ← thread-safe
        update_results(results)       ← thread-safe
        _reset_buttons()              ← thread-safe
        hide() / show()
    """

    # signals — בטוחים לשליחה מ-thread אחר
    _sig_update_results = pyqtSignal(dict)
    _sig_display_image  = pyqtSignal(object, list, int, int, int, int)
    _sig_reset          = pyqtSignal()
    # Explainable AI Emotion Assistant — thread-safe signals
    _sig_explain_loading = pyqtSignal(str, float)
    _sig_explain_text    = pyqtSignal(str)
    _sig_explain_error   = pyqtSignal(str)
    _sig_explain_prepare = pyqtSignal(dict)
    # Fires AFTER results are rendered, in the MAIN thread — safe place to
    # spawn downstream work that needs a Qt event loop (e.g. QTimer watchdogs).
    analysis_completed   = pyqtSignal(dict)
    # Forwarded from ExplanationCard — emitted on USER tab clicks only.
    # Payload is "overall" or "face:N".
    face_explanation_requested = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._capture_callback  = None
        self._region_callback   = None
        self._last_screenshot: "np.ndarray | None" = None
        self._last_faces:  list = []
        self._last_offset: tuple = (0, 0)
        self._last_reference_size: tuple = (0, 0)
        self._worker: "_AnalysisWorker | None" = None

        # Auto-hide watcher: remembers which OS window was captured so the
        # overlays vanish when the user switches to a different application
        # OR even to a different browser-tab (same hwnd, different title).
        self._captured_hwnd  = 0
        self._captured_title = ""
        self._captured_pid   = 0
        # Continuous background poll: records the user's *external*
        # foreground window every 250ms. This way, even when she clicks
        # our floating button (which may briefly promote our overlay to
        # the OS-foreground), we always know which "real" app she was
        # looking at last. Used as a robust fallback in `_on_capture`
        # and `_on_region`.
        self._last_external_hwnd  = 0
        self._last_external_title = ""
        self._fg_poll_timer = QTimer(self)
        self._fg_poll_timer.setInterval(250)
        self._fg_poll_timer.timeout.connect(self._poll_external_foreground)
        self._fg_poll_timer.start()

        # Pixel-content watcher: detects when the captured region's
        # underlying content changes (e.g. the user navigates to the
        # next photo *in the same window*, so the hwnd / title don't
        # change but the visible image does). When MAE between the
        # current thumbnail and the reference exceeds a threshold, we
        # auto-close the overlays.
        self._pixel_reference: "np.ndarray | None" = None
        self._pixel_check_region: "tuple[int, int, int, int] | None" = None
        self._pixel_check_timer = QTimer(self)
        self._pixel_check_timer.setInterval(800)
        self._pixel_check_timer.timeout.connect(self._check_pixel_change)

        self._fg_watch_timer = QTimer(self)
        self._fg_watch_timer.setInterval(400)
        self._fg_watch_timer.timeout.connect(self._check_foreground)
        # Hard safety-net: hide overlays after this many ms no matter what.
        self._fg_max_timer = QTimer(self)
        self._fg_max_timer.setSingleShot(True)
        self._fg_max_timer.setInterval(25_000)
        self._fg_max_timer.timeout.connect(
            lambda: self._auto_close_overlays("max watch time")
        )
        # Pending auto-close (debounced focus-loss). Fires shortly after
        # focus leaves our overlay; cancelled if focus comes back.
        self._fg_pending_close = QTimer(self)
        self._fg_pending_close.setSingleShot(True)
        self._fg_pending_close.setInterval(350)
        self._fg_pending_close.timeout.connect(
            lambda: self._auto_close_overlays("focus left overlay")
        )

        # Qt-level signal: fires when our app goes inactive / active.
        try:
            app = QApplication.instance()
            if app is not None:
                app.applicationStateChanged.connect(self._on_app_state_changed)
                app.focusWindowChanged.connect(self._on_focus_window_changed)
        except Exception as exc:
            print(f"[overlay] could not hook app state: {exc}")

        self._setup_window()
        self._build_ui()
        self.setStyleSheet(GLOBAL_SS)

        # חיבור signals → slots ב-main thread
        self._sig_update_results.connect(self._do_update_results)
        self._sig_display_image.connect(self._do_display_image)
        self._sig_reset.connect(self._do_reset)
        # Explainable AI — connect on main thread
        self._sig_explain_loading.connect(self._do_explain_loading)
        self._sig_explain_text.connect(self._do_explain_text)
        self._sig_explain_error.connect(self._do_explain_error)
        self._sig_explain_prepare.connect(self._do_explain_prepare)

    # ── הגדרת חלון ────────────────────────

    def _setup_window(self) -> None:
        self.setWindowFlags(
            Qt.FramelessWindowHint |
            Qt.WindowStaysOnTopHint |
            Qt.Tool
        )
        self.setAttribute(Qt.WA_TranslucentBackground)

        # כיסוי מסך מלא (לצורך מיקום חופשי של ילדים)
        screen = QApplication.primaryScreen().geometry()
        self.setGeometry(screen)

    # ── Mouse routing ──────────────────────
    def mousePressEvent(self, event) -> None:
        """
        EmotionOverlay is a full-screen always-on-top window. Three kinds
        of clicks can happen on it:

        1. Click on a face's bbox  → open the explanation card for that
           face.
        2. Click on a child widget (camera button, results panel,
           explanation card, …) → let Qt deliver the event normally so
           the widget handles it.
        3. Click on a transparent area (none of the above) → the user
           clearly wants to interact with the app *underneath* our
           overlay. We dismiss the analysis so she can do that, and the
           next click will land on the actual underlying app.
        """
        try:
            if event.button() == Qt.LeftButton:
                # ─ 1. bbox check ─
                if (
                    hasattr(self, "_bbox_overlay")
                    and self._bbox_overlay is not None
                    and self._bbox_overlay.isVisible()
                ):
                    global_pos = event.globalPos()
                    bbox_pos = self._bbox_overlay.mapFromGlobal(global_pos)
                    idx = self._bbox_overlay.face_at_position(bbox_pos)
                    if idx is not None:
                        print(
                            f"[overlay] mousePressEvent at "
                            f"global=({global_pos.x()},{global_pos.y()}) "
                            f"→ face_at_position={idx}"
                        )
                        self._bbox_overlay.set_active_face_idx(idx)
                        self._on_bbox_clicked(idx)
                        event.accept()
                        return

                # ─ 3. dismiss-on-empty-click ─
                # Only do this when an analysis is *active*, otherwise we'd
                # interfere with regular interactions.
                analysis_active = (
                    (hasattr(self, "_bbox_overlay")
                     and self._bbox_overlay is not None
                     and self._bbox_overlay.isVisible())
                    or (hasattr(self, "_explanation")
                        and self._explanation is not None
                        and self._explanation.isVisible())
                )
                child = self.childAt(event.pos())
                if analysis_active and child is None:
                    print("[overlay] click on transparent area → dismissing analysis")
                    self._clear_active_analysis(reason="user clicked outside")
                    event.accept()
                    return
        except Exception as exc:
            print(f"[overlay] mousePressEvent forwarding failed: {exc}")
        super().mousePressEvent(event)

    # ── בנייה ─────────────────────────────

    def _build_ui(self) -> None:
        screen = QApplication.primaryScreen().geometry()
        sw, sh = screen.width(), screen.height()

        # כפתור מצלמה — פינה ימנית תחתונה
        self._btn = FloatingCameraButton(self)
        btn_size  = self._btn._BTN_SIZE
        self._btn.move(sw - btn_size - 40, sh - btn_size - 60)

        self._btn.capture_clicked.connect(self._on_capture)
        self._btn.region_clicked.connect(self._on_region)
        self._btn.close_clicked.connect(QApplication.instance().quit)

        # פאנל תוצאות
        self._panel = ResultsPanel(self)
        self._panel._region_btn.clicked.connect(self._on_region)

        # viewer צילום מסך מסומן (popup)
        self._viewer = AnnotatedImageViewer(self)

        # שכבת תיבות זיהוי — חלון עצמאי שקוף מעל הכל
        self._bbox_overlay = BBoxScreenOverlay()

        # Toast for "no faces detected" — separate top-level widget so
        # it shows above any application without competing with the
        # bbox overlay's mask.
        self._no_faces_toast = NoFacesToast()

        # Explainable AI — floating draggable explanation card
        from ui.explanation_card import ExplanationCard
        self._explanation = ExplanationCard(self)
        # Route card tab clicks AND screen bbox clicks through the same
        # handler so the bbox highlight and the card chip stay in sync,
        # then forward to the controller (main.py).
        self._explanation.face_selected.connect(self._on_card_tab_selected)
        # When the user dismisses the card with ×, also clear the bboxes
        # from the screen — the entire analysis goes away together.
        self._explanation.closed.connect(self._on_explanation_closed)
        self._bbox_overlay.face_clicked.connect(self._on_bbox_clicked)

        self._reposition_panel()

    def _reposition_panel(self) -> None:
        """ממקם את הפאנל שמאלה ומעל הכפתור — גובה דינמי."""
        self._panel.adjustSize()
        btn_pos = self._btn.pos()
        panel_w = self._panel.PANEL_WIDTH
        panel_h = max(self._panel.height(), 120)
        screen  = QApplication.primaryScreen().geometry()
        x = max(10, btn_pos.x() - panel_w - 12)
        # מנסה למקם מעל הכפתור; אם לא נכנס — שם מתחתיו
        y_above = btn_pos.y() - panel_h + self._btn._BTN_SIZE
        y = y_above if y_above >= 10 else btn_pos.y() + self._btn._BTN_SIZE + 8
        # וידוא שלא יוצא מגבולות המסך
        y = max(10, min(y, screen.height() - panel_h - 10))
        self._panel.move(x, y)

    # ── API ציבורי ─────────────────────────

    def set_capture_callback(self, callback) -> None:
        self._capture_callback = callback

    def set_region_callback(self, callback) -> None:
        self._region_callback = callback

    # ── API ציבורי — thread-safe ────────────

    def display_image(self, image: np.ndarray, faces: list = None,
                      region_offset: tuple = (0, 0),
                      reference_size: tuple | None = None) -> None:
        """שומר את הצילום + offset של אזור הלכידה — thread-safe."""
        ox, oy = (region_offset or (0, 0))
        if image is not None and image.size:
            ref_w, ref_h = reference_size or (image.shape[1], image.shape[0])
        else:
            ref_w, ref_h = reference_size or (0, 0)
        self._sig_display_image.emit(
            image,
            faces or [],
            int(ox),
            int(oy),
            int(ref_w),
            int(ref_h),
        )

    def update_results(self, results: dict) -> None:
        """מציג פאנל תוצאות — thread-safe."""
        self._sig_update_results.emit(results)

    def _reset_buttons(self) -> None:
        """מאפס מצב עיבוד — thread-safe."""
        self._sig_reset.emit()

    # ── Explainable AI public API — thread-safe ───────────

    def prepare_explanation(self, results: dict) -> None:
        """
        Initialize the explanation card for a new analysis result.
        Builds the tab strip and sets the active tab to 'overall'.
        Thread-safe.
        """
        self._sig_explain_prepare.emit(results or {})

    def show_explanation_loading(self, emotion: str = "", confidence: float = 0.0) -> None:
        """
        Open the explanation card body in 'generating...' state.

        Args are optional (and ignored when the card was already prepared via
        `prepare_explanation`) — they exist for legacy callers and tests.
        Thread-safe.
        """
        try:
            conf = float(confidence)
        except (TypeError, ValueError):
            conf = 0.0
        self._sig_explain_loading.emit(emotion or "", conf)

    def update_explanation(self, text: str) -> None:
        """Replace the card's text with the final explanation. Thread-safe."""
        self._sig_explain_text.emit(text or "")

    def show_explanation_error(self, reason: str) -> None:
        """Show a soft-error state on the card. Thread-safe."""
        self._sig_explain_error.emit(reason or "Explanation unavailable.")

    def hide_explanation(self) -> None:
        """Force-hide the card from the main thread."""
        if hasattr(self, "_explanation") and self._explanation is not None:
            self._explanation.hide_card()

    def get_active_explanation_tab(self) -> str:
        """Return the tab currently visible on the card ('overall' or 'face:N')."""
        if hasattr(self, "_explanation") and self._explanation is not None:
            return self._explanation.get_active_tab()
        return "overall"

    # ── slots פנימיים (תמיד ב-main thread) ─

    def _do_display_image(self, image: np.ndarray, faces: list,
                          offset_x: int, offset_y: int,
                          reference_width: int, reference_height: int) -> None:
        self._last_screenshot = image
        self._last_faces      = faces or []
        self._last_offset     = (offset_x, offset_y)
        self._last_reference_size = (reference_width, reference_height)

    def _do_update_results(self, results: dict) -> None:
        # פאנל צדדי מוסתר — רק תיבות זיהוי על המסך
        self._panel.hide()

        faces = results.get("faces", [])
        if not faces:
            # No faces in the analyzed image — show a small toast and
            # bail. Don't start any auto-close watchers (there's
            # nothing to watch).
            self._show_no_faces_toast()
            try:
                if hasattr(self, "_bbox_overlay") and self._bbox_overlay is not None:
                    self._bbox_overlay.hide_faces()
            except Exception:
                pass
            try:
                if hasattr(self, "_explanation") and self._explanation is not None:
                    self._explanation.hide()
            except Exception:
                pass
            self.show()
            self.raise_()
            return

        if faces:
            ox, oy = self._last_offset
            screen   = QApplication.primaryScreen()
            scr_geo  = screen.geometry()
            img      = self._last_screenshot
            img_w    = img.shape[1] if img is not None else 0
            img_h    = img.shape[0] if img is not None else 0
            ref_w, ref_h = self._last_reference_size
            if not ref_w or not ref_h:
                ref_w, ref_h = img_w, img_h
            ovl_geo  = self._bbox_overlay.geometry()
            print(f"[DEBUG] screen={scr_geo.width()}x{scr_geo.height()} "
                  f"img={img_w}x{img_h} ref={ref_w}x{ref_h} "
                  f"overlay={ovl_geo.width()}x{ovl_geo.height()} "
                  f"offset=({ox},{oy})")
            print(f"[DEBUG] raw bboxes={[f.get('bbox') for f in faces]}")

            sx = scr_geo.width() / ref_w if ref_w else 1.0
            sy = scr_geo.height() / ref_h if ref_h else 1.0
            print(f"[DEBUG] scale_x={sx:.4f} scale_y={sy:.4f}")

            self._bbox_overlay.show_faces(
                faces,
                offset_x=ox,
                offset_y=oy,
                scale_x=sx,
                scale_y=sy,
            )
            self._btn.raise_()
        else:
            self._bbox_overlay.hide_faces()

        self.show()
        self.raise_()

        # Start auto-close watching: hide overlays when the user switches
        # away from the captured application (so the analysis doesn't follow
        # the user across unrelated apps like Canva, browsers, etc).
        # We start the timers whenever there ARE faces, even if we couldn't
        # capture the original foreground hwnd — the max-timer is the hard
        # safety-net that guarantees the overlays go away.
        if faces:
            self._fg_watch_timer.start()
            self._fg_max_timer.start()
            print(
                f"[overlay] watcher started "
                f"(hwnd={self._captured_hwnd}, max={self._fg_max_timer.interval()}ms)"
            )

            # Pixel-change watcher: handles the case where the underlying
            # app stays the same (same hwnd / title) but the user navigates
            # between photos inside it (e.g. Photos app, image slideshow,
            # social feed). We snapshot the captured region a moment after
            # the bboxes have rendered, then poll for changes every ~1.5s.
            ox = self._last_offset[0]
            oy = self._last_offset[1]
            cap_w = img.shape[1] if img is not None else 0
            cap_h = img.shape[0] if img is not None else 0
            if cap_w > 0 and cap_h > 0:
                self._pixel_check_region = (int(ox), int(oy), int(cap_w), int(cap_h))
                self._pixel_reference = None
                QTimer.singleShot(450, self._take_pixel_reference)
                self._pixel_check_timer.start()

        # Notify listeners (e.g. Explainable AI) — we are in the main thread.
        try:
            self.analysis_completed.emit(results)
        except Exception as exc:
            print(f"[overlay] analysis_completed emit failed: {exc}")

    def _reposition_viewer(self) -> None:
        """מציב את ה-viewer מעל פאנל התוצאות."""
        self._viewer.adjustSize()
        btn_pos  = self._btn.pos()
        panel_w  = self._panel.PANEL_WIDTH
        viewer_h = self._viewer.height()
        viewer_w = self._viewer.width()
        screen   = QApplication.primaryScreen().geometry()

        # מיישר עם הצד השמאלי של הפאנל
        x = max(10, btn_pos.x() - panel_w - 12)
        x = min(x, screen.width() - viewer_w - 10)

        # ממוקם מעל הפאנל
        panel_top = self._panel.y()
        y = max(10, panel_top - viewer_h - 10)

        self._viewer.move(x, y)

    def _do_reset(self) -> None:
        self._btn.set_processing(False)

    # ── Face selection routing (bbox click ↔ card tab) ────

    def _on_explanation_closed(self) -> None:
        """Card was dismissed by the user — also clear the bboxes."""
        self._fg_watch_timer.stop()
        self._fg_max_timer.stop()
        self._fg_pending_close.stop()
        self._pixel_check_timer.stop()
        self._pixel_reference = None
        self._pixel_check_region = None
        self._captured_hwnd  = 0
        self._captured_title = ""
        self._captured_pid   = 0
        if hasattr(self, "_bbox_overlay") and self._bbox_overlay is not None:
            self._bbox_overlay.hide_faces()

    # ── Foreground-window watcher ─────────────────────────

    def _own_hwnds(self) -> set:
        """
        Return the OS window-handles that belong to *our* Qt application.
        Includes EmotionOverlay, BBoxScreenOverlay, ExplanationCard, the
        ResultsPanel, the AnnotatedImageViewer AND the RegionSelector
        widget that's only alive during region capture — so the
        foreground-watcher / external-fg poller never accidentally treats
        one of our own helper windows as "the user's app".
        """
        my_hwnds: set = set()
        try:
            app = QApplication.instance()
            if app is not None:
                for w in app.topLevelWidgets():
                    try:
                        hwnd = int(w.winId())
                        if hwnd:
                            my_hwnds.add(hwnd)
                    except Exception:
                        pass
        except Exception:
            pass
        # Explicit fallback: even if topLevelWidgets() missed something,
        # always include the three windows we care about most.
        for attr in ("_bbox_overlay", "_explanation"):
            try:
                obj = getattr(self, attr, None)
                if obj is not None:
                    my_hwnds.add(int(obj.winId()))
            except Exception:
                pass
        try:
            my_hwnds.add(int(self.winId()))
        except Exception:
            pass
        return my_hwnds

    # ── Pixel-content watcher ─────────────────────────────

    def _grab_thumbnail(self, x: int, y: int, w: int, h: int,
                        target: int = 32) -> "np.ndarray | None":
        """
        Capture a small thumbnail (≈`target`×`target` px) of a screen
        region for fast change-detection. Returns None on any failure.

        Uses our existing `ScreenCapturer.capture_region` (mss) so it
        works the same way as the original screenshot and respects the
        same DPI / monitor configuration.
        """
        try:
            if w <= 4 or h <= 4:
                return None
            from capture.screen_capture import ScreenCapturer
            img = ScreenCapturer().capture_region(int(x), int(y), int(w), int(h))
            if img is None or img.size == 0:
                return None
            # Cheap NumPy subsampling — avoids cv2 import + handles any
            # input resolution.
            step_h = max(1, img.shape[0] // target)
            step_w = max(1, img.shape[1] // target)
            return img[::step_h, ::step_w]
        except Exception as exc:
            print(f"[overlay] thumbnail grab failed: {exc}")
            return None

    def _take_pixel_reference(self) -> None:
        """Snapshot the current state of the captured region as ground truth."""
        if not self._pixel_check_region:
            return
        x, y, w, h = self._pixel_check_region
        self._pixel_reference = self._grab_thumbnail(x, y, w, h)
        if self._pixel_reference is not None:
            print(
                f"[overlay] pixel reference captured "
                f"shape={self._pixel_reference.shape} for region ({x},{y},{w}x{h})"
            )

    def _check_pixel_change(self) -> None:
        """
        Sample the captured region again and compare with the reference.
        Closes the overlays if the mean absolute difference exceeds the
        threshold — the visible content has changed (e.g. user scrolled
        to a different photo inside the same window).
        """
        if self._pixel_reference is None or not self._pixel_check_region:
            return
        x, y, w, h = self._pixel_check_region
        current = self._grab_thumbnail(x, y, w, h)
        if current is None:
            return
        if current.shape != self._pixel_reference.shape:
            # Region or DPI changed under us — treat as content change.
            self._auto_close_overlays("pixel reference shape mismatch")
            return
        try:
            diff = float(
                np.abs(
                    current.astype(np.int32)
                    - self._pixel_reference.astype(np.int32)
                ).mean()
            )
        except Exception:
            return
        # Threshold tuned for "the underlying content changed" while
        # tolerating cursor / minor UI tweaks. Lower = more sensitive.
        # 12 is sensitive enough to catch slide switches in PowerPoint /
        # Canva (where part of the screen — e.g. the thumbnail bar —
        # changes) without firing on small mouse movements.
        if diff > 12.0:
            print(f"[overlay] pixel change MAE={diff:.1f} → auto-close")
            self._auto_close_overlays(f"content changed (MAE={diff:.1f})")

    def _capture_external_foreground(self) -> tuple[int, str]:
        """
        Return `(hwnd, title)` for the user's "external" foreground app.

        First checks `GetForegroundWindow()` directly; if that returns 0
        or one of our own overlay windows, falls back to the
        continuously-polled `_last_external_hwnd`. This is robust to
        focus shifts that happen when the user clicks our floating
        button (some OS configurations briefly promote our overlay to
        the foreground).
        """
        captured = _get_foreground_hwnd()
        own = self._own_hwnds()
        if (not captured) or (captured in own):
            captured       = self._last_external_hwnd
            captured_title = self._last_external_title
        else:
            captured_title = _get_window_title(captured)
        return captured, captured_title

    def _show_no_faces_toast(self) -> None:
        """Pop the 'no faces detected' banner. Safe to call repeatedly."""
        try:
            if hasattr(self, "_no_faces_toast") and self._no_faces_toast is not None:
                self._no_faces_toast.pop()
                print("[overlay] toast → no faces detected on screen")
        except Exception as exc:
            print(f"[overlay] toast failed: {exc}")

    def keyPressEvent(self, event) -> None:
        """
        Esc on the overlay (or on any of its child widgets that don't
        consume the key first) clears the active analysis. Useful for
        the demo when the user just wants the bboxes off the screen
        without clicking the × on the explanation card.
        """
        if event.key() == Qt.Key_Escape:
            self._clear_active_analysis(reason="user pressed Esc")
            event.accept()
            return
        super().keyPressEvent(event)

    def _clear_active_analysis(self, reason: str = "") -> None:
        """
        Hide the bboxes + the AI explanation card right now, and stop all
        running auto-close timers. Used at the start of a new capture so
        the previous analysis disappears immediately (instead of
        lingering on screen for the 2-5 seconds it takes the new worker
        to finish) and also exposed to keyboard / click handlers as a
        "clear it all" shortcut.
        """
        if reason:
            print(f"[overlay] clearing previous analysis ({reason})")
        try:
            self._fg_watch_timer.stop()
            self._fg_max_timer.stop()
            self._fg_pending_close.stop()
            self._pixel_check_timer.stop()
        except Exception:
            pass
        self._pixel_reference = None
        self._pixel_check_region = None
        self._captured_hwnd  = 0
        self._captured_title = ""
        self._captured_pid   = 0
        try:
            if hasattr(self, "_bbox_overlay") and self._bbox_overlay is not None:
                self._bbox_overlay.hide_faces()
        except Exception:
            pass
        try:
            if hasattr(self, "_explanation") and self._explanation is not None:
                # Use Qt's hide() — we don't want to emit `closed`
                # (that would trigger _on_explanation_closed → another
                # bbox-hide loop).
                self._explanation.hide()
        except Exception:
            pass

    def _poll_external_foreground(self) -> None:
        """
        Polled every ~250ms. Records the OS window-handle of the user's
        current foreground app whenever it ISN'T one of our own windows.
        `_on_capture` / `_on_region` later read this if the foreground at
        click-time happens to be our overlay (which can happen on some
        OS configurations when the floating button momentarily grabs
        focus).
        """
        if _USER32 is None:
            return
        try:
            fg = _get_foreground_hwnd()
            if not fg:
                return
            if fg in self._own_hwnds():
                return
            if fg == self._last_external_hwnd:
                # Keep title fresh in case the user just switched tabs.
                self._last_external_title = _get_window_title(fg)
                return
            self._last_external_hwnd  = fg
            self._last_external_title = _get_window_title(fg)
        except Exception:
            pass

    def _check_foreground(self) -> None:
        """
        Polled every ~400ms. If the user has switched to an application
        other than the one we captured (and other than our own overlays),
        auto-hide everything so we don't litter the screen.

        Also handles the *browser tab switch* case: the OS hwnd of a
        browser window is identical across tabs, so we additionally
        compare the window title — any non-trivial title change is
        treated as a context switch and the overlays are hidden.
        """
        current = _get_foreground_hwnd()
        if not current:
            return

        my_hwnds = self._own_hwnds()

        # Focus is on our own overlay (card / bboxes / camera button) —
        # never auto-close in that case.
        if current in my_hwnds:
            return

        # If we don't have a captured hwnd to compare against (typical
        # for region-select capture, or when Qt promoted our overlay to
        # the foreground at click time), we can't tell whether the user
        # is "still on the captured app" or not. Instead of closing
        # aggressively, fall back to the max-timer + Qt focus signals.
        if not self._captured_hwnd:
            return

        # Foreground switched to a completely different window → close.
        if current != self._captured_hwnd:
            self._auto_close_overlays(
                f"foreground window changed "
                f"(captured={self._captured_hwnd}, current={current})"
            )
            return

        # Same hwnd as captured — but maybe the user switched *tabs*
        # inside a browser (same window, different document). In that
        # case the window title changes; treat that as a context switch.
        current_title = _get_window_title(current)
        if (
            self._captured_title
            and current_title
            and current_title != self._captured_title
        ):
            self._auto_close_overlays(
                f"window title changed ({self._captured_title!r} → {current_title!r})"
            )

    def _on_app_state_changed(self, state) -> None:
        """
        Qt signal: our application became (in)active. Used as an
        additional trigger so the bboxes vanish the moment focus leaves
        all of our windows, even if the OS-foreground check is slow.
        """
        if not self._fg_watch_timer.isActive() and not self._fg_max_timer.isActive():
            return
        try:
            inactive = state != Qt.ApplicationActive
        except Exception:
            inactive = False
        if inactive:
            print("[overlay] app state went inactive → scheduling close")
            self._fg_pending_close.start()
        else:
            self._fg_pending_close.stop()

    def _on_focus_window_changed(self, window) -> None:
        """
        Qt signal: focusWindow changed. If the new focus window is None
        (focus left our process) and we're showing overlays, close them
        after a short debounce.
        """
        if not self._fg_watch_timer.isActive() and not self._fg_max_timer.isActive():
            return
        if window is None:
            self._fg_pending_close.start()
        else:
            self._fg_pending_close.stop()

    def _auto_close_overlays(self, reason: str) -> None:
        """
        Hide screen-anchored analysis overlays (the bbox highlights that
        sit on top of the captured screen). Called by the foreground +
        pixel watchers when the underlying screen changes.

        The floating Explanation card is intentionally kept visible —
        once the AI has explained a face the user usually wants to keep
        reading the explanation while switching to other windows. The
        card has its own × button for manual dismissal.
        """
        # Avoid double-close spam if multiple signals fire at once.
        if (
            not self._fg_watch_timer.isActive()
            and not self._fg_max_timer.isActive()
            and not self._pixel_check_timer.isActive()
            and not self._captured_hwnd
        ):
            return
        print(f"[overlay] auto-close ({reason})")
        self._fg_watch_timer.stop()
        self._fg_max_timer.stop()
        self._fg_pending_close.stop()
        self._pixel_check_timer.stop()
        self._pixel_reference = None
        self._pixel_check_region = None
        self._captured_hwnd  = 0
        self._captured_title = ""
        self._captured_pid   = 0
        if hasattr(self, "_bbox_overlay") and self._bbox_overlay is not None:
            self._bbox_overlay.hide_faces()
        if hasattr(self, "_explanation") and self._explanation is not None:
            self._explanation.hide_card()

    def _on_bbox_clicked(self, idx: int) -> None:
        """User clicked a face's bbox on the screen."""
        target_id = f"face:{idx}"
        print(f"[overlay] bbox clicked → {target_id}")
        if hasattr(self, "_explanation") and self._explanation is not None:
            self._explanation.set_active_tab(target_id)
        # bbox already highlighted itself in its own mousePressEvent
        self.face_explanation_requested.emit(target_id)

        # The card is about to appear on top of the captured screen.
        # Without re-baselining, the pixel watcher would diff "screen
        # without card" vs "screen with card", trip its threshold, and
        # auto-close the bboxes — causing the card to flicker (open,
        # vanish for a tick, then re-open when the AI worker returns).
        # Re-take the reference shortly after the card is laid out so
        # the card itself becomes part of the new baseline.
        QTimer.singleShot(120, self._take_pixel_reference)
        QTimer.singleShot(450, self._take_pixel_reference)

    def _on_card_tab_selected(self, target_id: str) -> None:
        """User clicked a visible tab inside the card (e.g. 'Overall')."""
        # Card already updated its own active state + chip.
        # Sync the bbox highlight.
        if hasattr(self, "_bbox_overlay") and self._bbox_overlay is not None:
            if target_id == "overall":
                self._bbox_overlay.set_active_face_idx(None)
            elif target_id.startswith("face:"):
                try:
                    idx = int(target_id.split(":", 1)[1])
                    self._bbox_overlay.set_active_face_idx(idx)
                except ValueError:
                    pass
        self.face_explanation_requested.emit(target_id)

    # ── Explainable AI slots (main thread) ────────────────

    def _do_explain_prepare(self, results: dict) -> None:
        if hasattr(self, "_explanation") and self._explanation is not None:
            self._explanation.prepare(results)
        # New analysis → clear any previous bbox highlight ("Overall" view).
        if hasattr(self, "_bbox_overlay") and self._bbox_overlay is not None:
            self._bbox_overlay.set_active_face_idx(None)
        self._raise_card_above_bboxes()

    def _do_explain_loading(self, emotion: str, confidence: float) -> None:
        if hasattr(self, "_explanation") and self._explanation is not None:
            # If the caller passed an emotion, also refresh the chip — this
            # keeps the legacy signature alive for tests / direct callers.
            if emotion:
                self._explanation.show_loading(emotion, confidence)
            else:
                self._explanation.show_loading()
        self._raise_card_above_bboxes()

    def _do_explain_text(self, text: str) -> None:
        if hasattr(self, "_explanation") and self._explanation is not None:
            self._explanation.show_text(text)
        self._raise_card_above_bboxes()

    def _do_explain_error(self, reason: str) -> None:
        if hasattr(self, "_explanation") and self._explanation is not None:
            self._explanation.show_error(reason)
        self._raise_card_above_bboxes()

    def _raise_card_above_bboxes(self) -> None:
        """
        Make sure the explanation card / EmotionOverlay window sits ABOVE
        the (separate) BBoxScreenOverlay top-level window. Both windows are
        always-on-top; without this, the newer / last-raised one ends up
        on top, which can hide parts of the popup behind the bboxes.
        """
        try:
            if hasattr(self, "_bbox_overlay") and self._bbox_overlay is not None:
                # Re-stack: bboxes first, then the popup on top of them.
                self._bbox_overlay.raise_()
            self.raise_()
            if hasattr(self, "_explanation") and self._explanation is not None:
                self._explanation.raise_()
        except Exception as exc:
            print(f"[overlay] raise card failed: {exc}")

    # ── callbacks ─────────────────────────

    def _on_capture(self) -> None:
        """לחיצה על כפתור צילום מסך."""
        if self._worker and self._worker.isRunning():
            return

        # Clear any leftover bboxes / explanation from the PREVIOUS
        # analysis BEFORE we start a new capture — otherwise the user
        # sees stale rectangles drifting on the screen for several
        # seconds while the new worker is still running.
        self._clear_active_analysis(reason="new capture starting")

        # Snapshot the OS-level handle of whatever the user is looking at
        # RIGHT NOW. Recorded before we hide anything; used by the
        # foreground watcher to auto-close when she switches apps OR even
        # switches browser tabs (same hwnd, different title).
        self._fg_watch_timer.stop()
        self._fg_max_timer.stop()
        self._fg_pending_close.stop()
        captured, captured_title = self._capture_external_foreground()
        self._captured_hwnd  = captured
        self._captured_title = captured_title
        self._captured_pid   = _get_window_process_id(captured) if captured else 0
        print(
            f"[overlay] full capture — captured hwnd={self._captured_hwnd} "
            f"pid={self._captured_pid} title={self._captured_title!r}"
        )

        self._panel.hide()
        # מסתירים רק את הכפתור (לא את כל ה-overlay) כדי שלא יופיע בצילום
        self._btn.hide()
        QApplication.processEvents()
        QTimer.singleShot(150, self._start_capture_worker)

    def _start_capture_worker(self) -> None:
        cb = self._capture_callback
        if not cb:
            print("No capture callback set. Run via main.py")
            self._btn.set_processing(False)
            self._btn.show()
            return

        # מציגים את הכפתור מחדש עם spinner — הצילום כבר נלקח
        self._btn.set_processing(True)
        self._btn.show()

        self._worker = _AnalysisWorker(cb)
        self._worker.done.connect(self._on_worker_done)
        self._worker.failed.connect(self._on_worker_failed)
        self._worker.start()

    def _on_worker_done(self) -> None:
        self._btn.set_processing(False)
        self.raise_()

    def _on_worker_failed(self, err: str) -> None:
        print(f"[ERROR] capture worker: {err}")
        self._btn.set_processing(False)
        self.raise_()

    def _on_region(self) -> None:
        """
        לחיצה על בחירת אזור — רץ סינכרונית ב-main thread.
        RegionSelector יוצר חלון Qt שחייב להיות ב-main thread.
        """
        if self._worker and self._worker.isRunning():
            return

        # Clear any previous analysis from the screen first.
        self._clear_active_analysis(reason="new region capture starting")

        # Snapshot foreground (with robust fallback to the last *external*
        # window we polled) — same logic as `_on_capture`. Without this
        # the foreground-watcher saw `_captured_hwnd == 0` and triggered
        # auto-close ~400ms after the bboxes appeared.
        self._fg_watch_timer.stop()
        self._fg_max_timer.stop()
        self._fg_pending_close.stop()
        captured, captured_title = self._capture_external_foreground()
        self._captured_hwnd  = captured
        self._captured_title = captured_title
        self._captured_pid   = _get_window_process_id(captured) if captured else 0
        print(
            f"[overlay] region capture — captured hwnd={self._captured_hwnd} "
            f"pid={self._captured_pid} title={self._captured_title!r}"
        )

        self._btn.set_processing(True)
        self._panel.hide()
        QApplication.processEvents()

        cb = self._region_callback
        if not cb:
            print("No region callback set. Run via main.py")
            self._btn.set_processing(False)
            return

        try:
            cb()
        except Exception as e:
            print(f"[ERROR] region: {e}")
        finally:
            self._btn.set_processing(False)


# ──────────────────────────────────────────
# עזרים
# ──────────────────────────────────────────

def _add_glow(widget: QWidget, color: str, radius: int = 16) -> None:
    """מוסיף drop-shadow זוהר ל-widget."""
    eff = QGraphicsDropShadowEffect(widget)
    eff.setBlurRadius(radius)
    eff.setColor(QColor(color))
    eff.setOffset(0, 0)
    widget.setGraphicsEffect(eff)


def _numpy_to_pixmap(image: np.ndarray) -> QPixmap | None:
    """ממיר numpy RGB ל-QPixmap."""
    try:
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        if image.dtype != np.uint8:
            image = (image * 255).clip(0, 255).astype(np.uint8)
        h, w, ch = image.shape
        return QPixmap.fromImage(QImage(image.data, w, h, w * ch, QImage.Format_RGB888))
    except Exception:
        return None


# ══════════════════════════════════════════
# Demo / הרצה עצמאית
# ══════════════════════════════════════════

def _demo_results() -> dict:
    dummy = np.random.randint(80, 200, (64, 64, 3), dtype=np.uint8)
    return {
        "faces": [
            {"emotion": "happy",   "confidence": 0.87,
             "bbox": (30,  40, 120, 120), "face_image": dummy,
             "all_emotions": {"happy": 0.87, "neutral": 0.08, "sad": 0.05}},
            {"emotion": "sad",     "confidence": 0.74,
             "bbox": (200, 40, 110, 110), "face_image": dummy,
             "all_emotions": {"sad": 0.74, "neutral": 0.15, "fear": 0.11}},
            {"emotion": "angry",   "confidence": 0.63,
             "bbox": (360, 40, 100, 100), "face_image": dummy,
             "all_emotions": {"angry": 0.63, "disgust": 0.20, "neutral": 0.17}},
        ],
        "final_emotion": "happy",
        "confidence":    0.75,
    }


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    overlay = EmotionOverlay()
    overlay.show()

    def _run_demo():
        overlay._btn.set_processing(True)
        QTimer.singleShot(1800, lambda: (
            overlay._btn.set_processing(False),
            overlay.update_results(_demo_results()),
        ))

    QTimer.singleShot(600, _run_demo)
    sys.exit(app.exec_())
