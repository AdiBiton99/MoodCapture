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
    QPainterPath,
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
# BBoxScreenOverlay — תיבות זיהוי על המסך
# ══════════════════════════════════════════

class BBoxScreenOverlay(QWidget):
    """
    שכבת-על שקופה מלאת-מסך שמצייר תיבות זיהוי צבעוניות ישירות על המסך.

    • מופעלת אחרי ניתוח, עם offset לתמיכה ב-region captures.
    • מציגה corner-brackets, צבע לפי רגש, label עם face ID + confidence.
    • מתפוגגת בהדרגה (fade-out) ונעלמת אוטומטית.
    """

    AUTO_HIDE_MS = 10_000   # מסתנכרן עם זמן הפאנל
    FADE_STEP    = 0.05     # כמות אטימות לכל צעד (40 צעדים ≈ 1.2 שניות)
    FADE_MS      = 30       # מרווח בין צעדי fade

    def __init__(self):
        # חלון עצמאי (top-level) שקוף — מקואורדינטות מסך מוחלטות
        super().__init__(
            None,
            Qt.FramelessWindowHint |
            Qt.WindowStaysOnTopHint |
            Qt.Tool,
        )
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WA_NoSystemBackground)
        # כיסוי מסך מלא
        screen = QApplication.primaryScreen().geometry()
        self.setGeometry(screen)

        self._faces:    list  = []
        self._offset_x: int   = 0
        self._offset_y: int   = 0
        self._scale_x:  float = 1.0
        self._scale_y:  float = 1.0
        self._opacity:  float = 1.0

        self._hide_timer = QTimer(self)
        self._hide_timer.setSingleShot(True)
        self._hide_timer.timeout.connect(self._start_fade)

        self._fade_timer = QTimer(self)
        self._fade_timer.setInterval(self.FADE_MS)
        self._fade_timer.timeout.connect(self._fade_step)

        self.hide()

    # ── ממשק ציבורי ───────────────────────

    def show_faces(self, faces: list,
                   offset_x: int = 0, offset_y: int = 0,
                   scale_x: float = 1.0, scale_y: float = 1.0) -> None:
        """מציג תיבות זיהוי על הפנים שזוהו."""
        self._faces    = [f for f in faces if f.get("bbox")]
        self._offset_x = offset_x
        self._offset_y = offset_y
        self._scale_x  = scale_x
        self._scale_y  = scale_y
        self._opacity  = 1.0
        self._fade_timer.stop()
        self._hide_timer.stop()
        self.update()
        self.show()
        self.raise_()
        if self._faces:
            self._hide_timer.start(self.AUTO_HIDE_MS)

    def hide_faces(self) -> None:
        """מסתיר מיד (ללא fade)."""
        self._hide_timer.stop()
        self._fade_timer.stop()
        self._faces = []
        self.hide()

    # ── fade ──────────────────────────────

    def _start_fade(self) -> None:
        self._fade_timer.start()

    def _fade_step(self) -> None:
        self._opacity = max(0.0, self._opacity - self.FADE_STEP)
        if self._opacity <= 0.0:
            self._fade_timer.stop()
            self.hide()
            self._faces = []
        else:
            self.update()

    # ── ציור ──────────────────────────────

    def paintEvent(self, event) -> None:
        if not self._faces:
            return

        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.TextAntialiasing)
        p.setOpacity(self._opacity)

        for idx, face in enumerate(self._faces):
            bbox = face.get("bbox")
            if not bbox or len(bbox) < 4:
                continue

            # קואורדינטות bbox בשטח התמונה → שטח מסך לוגי
            sx = self._scale_x if self._scale_x > 0 else 1.0
            sy = self._scale_y if self._scale_y > 0 else 1.0
            fx = int((bbox[0] + self._offset_x) * sx)
            fy = int((bbox[1] + self._offset_y) * sy)
            fw = int(bbox[2] * sx)
            fh = int(bbox[3] * sy)
            if idx == 0:
                print(f"[PAINT] Face1 drawn at screen ({fx},{fy},{fw},{fh})"
                      f" overlay_geo={self.geometry().x()},{self.geometry().y()}"
                      f" size={self.width()}x{self.height()}")

            emotion  = (face.get("emotion") or "unknown").lower()
            conf     = face.get("confidence", 0.0)
            hex_col  = EMOTION_COLORS.get(emotion, C_PRIMARY)
            color    = QColor(hex_col)

            # ── תיבת זיהוי ───────────────
            # מילוי רקע עדין
            fill = QColor(color)
            fill.setAlpha(18)
            p.fillRect(QRectF(fx, fy, fw, fh), fill)

            # מסגרת דקה שקופה למחצה
            p.setPen(QPen(QColor(hex_col + "88"), 2.0))
            p.setBrush(Qt.NoBrush)
            p.drawRoundedRect(QRectF(fx, fy, fw, fh), 6, 6)

            # פינות L עבות (corner brackets)
            bk = max(14, min(fw, fh) // 6)
            pen = QPen(color, 3.5)
            pen.setCapStyle(Qt.RoundCap)
            p.setPen(pen)
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

            # ── תווית מעל התיבה ──────────
            face_lbl = f"Face {idx + 1}"
            conf_lbl = f"{emotion.capitalize()}  {conf:.0%}"

            fnt_b = QFont("Segoe UI", 9)
            fnt_b.setWeight(QFont.Bold)
            fnt_r = QFont("Segoe UI", 8)
            fm_b  = QFontMetrics(fnt_b)
            fm_r  = QFontMetrics(fnt_r)

            lw = max(fm_b.horizontalAdvance(face_lbl),
                     fm_r.horizontalAdvance(conf_lbl)) + 20
            lh = fm_b.height() + fm_r.height() + 12
            lx = fx
            ly = max(0, fy - lh - 6)

            # רקע תווית עם שקיפות
            bg = QColor(C_PANEL)
            bg.setAlpha(230)
            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(bg))
            p.drawRoundedRect(QRectF(lx, ly, lw, lh), 8, 8)

            # פס צבע שמאלי
            p.setBrush(QBrush(color))
            p.drawRoundedRect(QRectF(lx, ly, 5, lh), 3, 3)

            # Face N — bold, צבע הרגש
            p.setFont(fnt_b)
            p.setPen(color)
            p.drawText(
                QRectF(lx + 10, ly + 4, lw - 12, fm_b.height() + 2),
                Qt.AlignLeft | Qt.AlignVCenter, face_lbl,
            )

            # Emotion XX% — רגיל, כהה
            p.setFont(fnt_r)
            p.setPen(QColor(C_TEXT))
            p.drawText(
                QRectF(lx + 10, ly + fm_b.height() + 6, lw - 12, fm_r.height() + 2),
                Qt.AlignLeft | Qt.AlignVCenter, conf_lbl,
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

    def __init__(self):
        super().__init__()
        self._capture_callback  = None
        self._region_callback   = None
        self._last_screenshot: "np.ndarray | None" = None
        self._last_faces:  list = []
        self._last_offset: tuple = (0, 0)
        self._last_reference_size: tuple = (0, 0)
        self._worker: "_AnalysisWorker | None" = None

        self._setup_window()
        self._build_ui()
        self.setStyleSheet(GLOBAL_SS)

        # חיבור signals → slots ב-main thread
        self._sig_update_results.connect(self._do_update_results)
        self._sig_display_image.connect(self._do_display_image)
        self._sig_reset.connect(self._do_reset)

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

    # ── callbacks ─────────────────────────

    def _on_capture(self) -> None:
        """לחיצה על כפתור צילום מסך."""
        if self._worker and self._worker.isRunning():
            return

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
