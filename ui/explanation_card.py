"""
explanation_card.py — Premium Explainable-AI popup.

A modern, glassmorphism-styled card with sidebar navigation, animated
emotion ring, face avatars, emotion breakdown and detected-signals chips.

Public API (unchanged — see overlay_ui.py for callers):
    card = ExplanationCard(parent)
    card.prepare(results)
    card.show_loading(emotion=None, confidence=None)
    card.show_text("...")
    card.show_error("...")
    card.update_emotion_meta(emotion, confidence)
    card.set_active_tab(target_id)
    card.get_active_tab()
    card.hide_card()

Signals:
    card.face_selected -> "overall" or "face:N"
    card.closed        -> emitted when × is clicked
"""

from __future__ import annotations

import cv2
import numpy as np

from PyQt5.QtCore import (
    Qt, QPoint, QRectF, QTimer, QPropertyAnimation, QEasingCurve, pyqtSignal,
    pyqtProperty,
)
from PyQt5.QtGui import (
    QColor, QFont, QPainter, QBrush, QPen, QPixmap, QImage, QPainterPath,
)
from PyQt5.QtWidgets import (
    QApplication, QFrame, QGraphicsDropShadowEffect, QHBoxLayout, QLabel,
    QPushButton, QScrollArea, QSizePolicy, QVBoxLayout, QWidget, QGridLayout,
    QProgressBar,
)


# ══════════════════════════════════════════════════════════════════════
# Palette — cream + sage green + warm brown accents
# ══════════════════════════════════════════════════════════════════════
C_BG          = "#F7F2EB"
C_PANEL       = "#FDFAF7"
C_PANEL_SOFT  = "#F2EAD9"
C_CARD        = "#FBF7F1"
C_BORDER      = "#E4DBCC"
C_BORDER_SOFT = "#EDE5D7"
C_TEXT        = "#2D241B"
C_TEXT_SOFT   = "#5A4A3B"
C_DIM         = "#9A8878"
C_DIM_2       = "#B5A493"
C_PRIMARY     = "#A0806A"
C_ACCENT      = "#7A5C44"
C_GREEN       = "#7AAE82"
C_GREEN_DARK  = "#5E8E68"
C_GREEN_SOFT  = "#D6E8D8"
C_DANGER      = "#C26B5A"

EMOTION_COLORS = {
    "happy":    "#5E8E68",
    "sad":      "#6B8AB8",
    "angry":    "#C26B5A",
    "neutral":  "#9A8878",
    "surprise": "#D4A055",
    "fear":     "#8A6FB3",
    "disgust":  "#B5689A",
}

DETECTED_SIGNALS = {
    "happy":    ["Smile intensity", "Eye openness", "Mouth curvature", "Face symmetry"],
    "sad":      ["Mouth downturn", "Eye droop", "Brow tension", "Reduced energy"],
    "angry":    ["Brow furrow", "Mouth tightness", "Eye narrowing", "Jaw tension"],
    "neutral":  ["Relaxed features", "Symmetric face", "Steady gaze", "Soft mouth"],
    "surprise": ["Wide eyes", "Raised brows", "Open mouth", "Forehead lines"],
    "fear":     ["Wide eyes", "Brow elevation", "Mouth tension", "Eye strain"],
    "disgust":  ["Nose wrinkle", "Upper-lip raise", "Brow lowering", "Mouth asymmetry"],
}

ALL_EMOTIONS = ["happy", "neutral", "sad", "angry", "fear", "surprise", "disgust"]


# ══════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════

def _hex_alpha(hex_color: str, alpha_pct: int) -> str:
    """Return a `rgba(...)` QSS string from a #RRGGBB color + alpha %."""
    h = (hex_color or "").lstrip("#")
    if len(h) != 6:
        return hex_color
    try:
        r = int(h[0:2], 16); g = int(h[2:4], 16); b = int(h[4:6], 16)
    except ValueError:
        return hex_color
    a = max(0.0, min(1.0, alpha_pct / 100.0))
    return f"rgba({r}, {g}, {b}, {a:.3f})"


def _make_circular_pixmap(img: np.ndarray, size: int):
    """Crop a numpy RGB image into a circular QPixmap of the given size."""
    try:
        if img is None:
            return None
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
        result = QPixmap(size, size)
        result.fill(Qt.transparent)
        p = QPainter(result)
        p.setRenderHint(QPainter.Antialiasing)
        path = QPainterPath()
        path.addEllipse(0, 0, size, size)
        p.setClipPath(path)
        p.drawPixmap(0, 0, source)
        p.end()
        return result
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════
# _EmotionRing — animated circular progress ring
# ══════════════════════════════════════════════════════════════════════

class _EmotionRing(QWidget):
    """Animated circular progress ring with the emotion symbol at the centre."""

    SIZE      = 124
    THICKNESS = 10

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(self.SIZE, self.SIZE)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self._progress = 0.0
        self._color    = C_PRIMARY
        self._symbol   = "•"
        self._anim     = QPropertyAnimation(self, b"progress", self)
        self._anim.setDuration(1100)
        self._anim.setEasingCurve(QEasingCurve.OutCubic)

    def set_emotion(self, emotion: str, confidence: float):
        emo = (emotion or "").lower()
        self._color  = EMOTION_COLORS.get(emo, C_PRIMARY)
        self._symbol = emo[:1].upper() if emo else "•"
        try:
            target = float(confidence or 0.0)
        except (TypeError, ValueError):
            target = 0.0
        target = max(0.0, min(1.0, target))
        self._anim.stop()
        self._anim.setStartValue(self._progress)
        self._anim.setEndValue(target)
        self._anim.start()

    def get_progress(self) -> float:
        return self._progress

    def set_progress(self, value: float):
        self._progress = float(value)
        self.update()

    progress = pyqtProperty(float, get_progress, set_progress)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        s = float(self.SIZE)
        cx = cy = s / 2.0
        radius = (s - self.THICKNESS - 14.0) / 2.0
        rect = QRectF(cx - radius, cy - radius, radius * 2, radius * 2)

        # Background ring
        bg_pen = QPen(QColor(C_BORDER_SOFT))
        bg_pen.setWidthF(self.THICKNESS)
        bg_pen.setCapStyle(Qt.RoundCap)
        p.setPen(bg_pen)
        p.drawArc(rect, 90 * 16, -360 * 16)

        # Progress arc
        if self._progress > 0:
            arc_pen = QPen(QColor(self._color))
            arc_pen.setWidthF(self.THICKNESS)
            arc_pen.setCapStyle(Qt.RoundCap)
            p.setPen(arc_pen)
            span = -int(360.0 * self._progress * 16.0)
            p.drawArc(rect, 90 * 16, span)

        # Soft inner tint
        inner_r = radius - self.THICKNESS / 2.0 - 6.0
        if inner_r > 0:
            tint = QColor(self._color); tint.setAlpha(20)
            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(tint))
            p.drawEllipse(QRectF(cx - inner_r, cy - inner_r, inner_r * 2, inner_r * 2))

        # Centre symbol
        fnt = QFont("Segoe UI", 40)
        fnt.setWeight(QFont.Light)
        p.setFont(fnt)
        p.setPen(QColor(self._color))
        p.drawText(rect, Qt.AlignCenter, self._symbol)

        p.end()


# ══════════════════════════════════════════════════════════════════════
# _SidebarButton — vertical nav button (icon + label)
# ══════════════════════════════════════════════════════════════════════

class _SidebarButton(QPushButton):
    def __init__(self, icon_text: str, label: str, parent=None):
        super().__init__(parent)
        self._active = False
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedHeight(40)
        self.setText(f"  {icon_text}    {label}")
        self.setStyleSheet(self._style(False))

    def set_active(self, active: bool):
        if self._active == active:
            return
        self._active = active
        self.setStyleSheet(self._style(active))

    @staticmethod
    def _style(active: bool) -> str:
        if active:
            return (
                f"QPushButton {{"
                f"  background: {_hex_alpha(C_GREEN, 20)};"
                f"  color: {C_GREEN_DARK};"
                f"  border: none;"
                f"  border-radius: 10px;"
                f"  text-align: left;"
                f"  padding-left: 14px;"
                f"  font-size: 12px;"
                f"  font-weight: 700;"
                f"  letter-spacing: 0.3px;"
                f"}}"
                f"QPushButton:hover {{"
                f"  background: {_hex_alpha(C_GREEN, 28)};"
                f"}}"
            )
        return (
            f"QPushButton {{"
            f"  background: transparent;"
            f"  color: {C_DIM};"
            f"  border: none;"
            f"  border-radius: 10px;"
            f"  text-align: left;"
            f"  padding-left: 14px;"
            f"  font-size: 12px;"
            f"  font-weight: 500;"
            f"}}"
            f"QPushButton:hover {{"
            f"  background: {_hex_alpha(C_BORDER, 60)};"
            f"  color: {C_TEXT_SOFT};"
            f"}}"
        )


# ══════════════════════════════════════════════════════════════════════
# _FaceMiniCard — small face card (avatar + emotion + confidence)
# ══════════════════════════════════════════════════════════════════════

class _FaceMiniCard(QFrame):
    selected = pyqtSignal(int)

    CARD_W = 128
    CARD_H = 158
    AVATAR = 64

    def __init__(self, index: int, emotion: str, confidence: float,
                 face_image=None, parent=None):
        super().__init__(parent)
        self.setObjectName("FaceMiniCard")
        self._index   = index
        self._active  = False
        self._emotion = (emotion or "unknown").lower()
        self._color   = EMOTION_COLORS.get(self._emotion, C_PRIMARY)
        try:
            self._conf = float(confidence or 0.0)
        except (TypeError, ValueError):
            self._conf = 0.0
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedSize(self.CARD_W, self.CARD_H)
        self._build(face_image)
        self._apply_style()

    def _build(self, face_image):
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 12, 8, 12)
        root.setSpacing(6)
        root.setAlignment(Qt.AlignTop | Qt.AlignHCenter)

        avatar = QLabel()
        avatar.setFixedSize(self.AVATAR, self.AVATAR)
        avatar.setAlignment(Qt.AlignCenter)

        used_image = False
        if face_image is not None:
            pm = _make_circular_pixmap(face_image, self.AVATAR)
            if pm is not None:
                avatar.setPixmap(pm)
                avatar.setStyleSheet(
                    f"QLabel {{"
                    f"  border: 2px solid {_hex_alpha(self._color, 55)};"
                    f"  border-radius: {self.AVATAR // 2}px;"
                    f"  background: transparent;"
                    f"}}"
                )
                used_image = True
        if not used_image:
            avatar.setText(str(self._index + 1))
            avatar.setStyleSheet(
                f"QLabel {{"
                f"  background: {_hex_alpha(self._color, 14)};"
                f"  border: 2px solid {_hex_alpha(self._color, 45)};"
                f"  border-radius: {self.AVATAR // 2}px;"
                f"  color: {self._color};"
                f"  font-size: 22px;"
                f"  font-weight: 800;"
                f"}}"
            )

        root.addWidget(avatar, 0, Qt.AlignHCenter)

        face_lbl = QLabel(f"FACE {self._index + 1}")
        face_lbl.setAlignment(Qt.AlignCenter)
        face_lbl.setStyleSheet(
            f"font-size: 9px; font-weight: 700;"
            f"color: {C_DIM}; letter-spacing: 1.3px;"
            f"background: transparent;"
        )
        root.addWidget(face_lbl)

        emo = QLabel(self._emotion.capitalize())
        emo.setAlignment(Qt.AlignCenter)
        emo.setStyleSheet(
            f"font-size: 13px; font-weight: 800;"
            f"color: {self._color}; background: transparent;"
        )
        root.addWidget(emo)

        conf = QLabel(f"{self._conf * 100:.0f}%")
        conf.setAlignment(Qt.AlignCenter)
        conf.setStyleSheet(
            f"font-size: 11px; font-weight: 600;"
            f"color: {C_TEXT_SOFT}; background: transparent;"
        )
        root.addWidget(conf)

    def _apply_style(self):
        if self._active:
            self.setStyleSheet(
                f"QFrame#FaceMiniCard {{"
                f"  background: {C_PANEL};"
                f"  border: 2px solid {C_GREEN};"
                f"  border-radius: 16px;"
                f"}}"
            )
        else:
            self.setStyleSheet(
                f"QFrame#FaceMiniCard {{"
                f"  background: {C_PANEL};"
                f"  border: 1px solid {C_BORDER};"
                f"  border-radius: 16px;"
                f"}}"
                f"QFrame#FaceMiniCard:hover {{"
                f"  background: {C_CARD};"
                f"  border: 1.5px solid {_hex_alpha(self._color, 70)};"
                f"}}"
            )

    def set_active(self, active: bool):
        if self._active == active:
            return
        self._active = active
        self._apply_style()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.selected.emit(self._index)
            event.accept()
        else:
            super().mousePressEvent(event)


# ══════════════════════════════════════════════════════════════════════
# _SignalChip — pill-style detected signal
# ══════════════════════════════════════════════════════════════════════

class _SignalChip(QLabel):
    def __init__(self, text: str, parent=None):
        super().__init__(f"✓   {text}", parent)
        self.setFixedHeight(30)
        self.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
        self.setStyleSheet(
            f"QLabel {{"
            f"  background: {_hex_alpha(C_GREEN, 14)};"
            f"  color: {C_GREEN_DARK};"
            f"  border: 1px solid {_hex_alpha(C_GREEN, 50)};"
            f"  border-radius: 15px;"
            f"  padding: 0 16px 1px 14px;"
            f"  font-size: 12px;"
            f"  font-weight: 700;"
            f"  letter-spacing: 0.2px;"
            f"}}"
        )


# ══════════════════════════════════════════════════════════════════════
# ExplanationCard — main floating popup
# ══════════════════════════════════════════════════════════════════════

class ExplanationCard(QFrame):
    """Premium floating popup for the Explainable AI Emotion Assistant."""

    CARD_WIDTH        = 560
    CARD_MIN_HEIGHT   = 440
    MAX_CARD_HEIGHT_RATIO = 0.80
    LOADER_INTERVAL   = 320

    closed         = pyqtSignal()
    face_selected  = pyqtSignal(str)   # "overall" or "face:N"

    def __init__(self, parent: QWidget = None) -> None:
        super().__init__(parent)

        # ── Drag state ──
        self._drag_offset = QPoint()
        self._is_dragging = False

        # ── Analysis state ──
        self._faces:         list  = []
        self._final_emotion: str   = "unknown"
        self._final_conf:    float = 0.0
        self._active_tab:    str   = "overall"
        self._tab_meta:      dict  = {}   # tid -> (emotion, confidence)
        self._tab_color:     dict  = {}   # tid -> hex color

        # ── Dynamic widget refs ──
        self._face_cards:   dict = {}     # tid -> _FaceMiniCard
        self._signal_chips: list = []
        self._emotion_bars: list = []
        self._scroll_anim   = None

        # ── Loader animation ──
        self._loader_step  = 0
        self._is_loading   = False
        self._loader_timer = QTimer(self)
        self._loader_timer.setInterval(self.LOADER_INTERVAL)
        self._loader_timer.timeout.connect(self._tick_loader)

        # ── Build & style ──
        self._build()
        self._apply_style()

        self.setFixedWidth(self.CARD_WIDTH)
        self.setMinimumHeight(self.CARD_MIN_HEIGHT)
        screen_h = QApplication.primaryScreen().geometry().height()
        self.setMaximumHeight(int(screen_h * self.MAX_CARD_HEIGHT_RATIO))
        self.hide()

    # ──────────────────────────────────────────
    # Construction — high-level
    # ──────────────────────────────────────────

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(self._build_header())
        # Sidebar removed — main pane takes the full body width.
        self._nav_buttons = {}
        root.addWidget(self._build_main(), 1)
        root.addWidget(self._build_footer())

    def _apply_style(self) -> None:
        self.setStyleSheet(
            f"ExplanationCard {{"
            f"  background: {C_PANEL};"
            f"  border: 1px solid {C_BORDER};"
            f"  border-radius: 20px;"
            f"}}"
        )
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(40)
        shadow.setColor(QColor(60, 40, 25, 65))
        shadow.setOffset(0, 10)
        self.setGraphicsEffect(shadow)
        self.setCursor(Qt.SizeAllCursor)

    # ──────────────────────────────────────────
    # Construction — header / footer
    # ──────────────────────────────────────────

    def _build_header(self) -> QFrame:
        header = QFrame()
        header.setObjectName("ExpHeader")
        header.setFixedHeight(60)
        header.setStyleSheet(
            f"QFrame#ExpHeader {{"
            f"  background: qlineargradient(x1:0, y1:0, x2:1, y2:0,"
            f"    stop:0 {C_BG}, stop:1 {C_PANEL_SOFT});"
            f"  border-bottom: 1px solid {C_BORDER_SOFT};"
            f"  border-top-left-radius: 20px;"
            f"  border-top-right-radius: 20px;"
            f"}}"
        )
        h = QHBoxLayout(header)
        h.setContentsMargins(22, 0, 14, 0)
        h.setSpacing(11)

        title = QLabel("MoodCapture")
        title.setStyleSheet(
            f"font-size: 19px; font-weight: 800;"
            f"color: {C_TEXT}; letter-spacing: -0.4px;"
            f"background: transparent;"
        )

        badge = QLabel("✦  Emotion AI")
        badge.setFixedHeight(22)
        badge.setAlignment(Qt.AlignCenter)
        badge.setStyleSheet(
            f"QLabel {{"
            f"  background: {_hex_alpha(C_GREEN, 18)};"
            f"  color: {C_GREEN_DARK};"
            f"  border: 1px solid {_hex_alpha(C_GREEN, 50)};"
            f"  border-radius: 11px;"
            f"  padding: 0 10px;"
            f"  font-size: 10px;"
            f"  font-weight: 700;"
            f"  letter-spacing: 0.6px;"
            f"}}"
        )

        self._close_btn = QPushButton("×")
        self._close_btn.setFixedSize(28, 28)
        self._close_btn.setCursor(Qt.PointingHandCursor)
        self._close_btn.setToolTip("Close")
        self._close_btn.setStyleSheet(
            f"QPushButton {{"
            f"  background: transparent;"
            f"  color: {C_DIM};"
            f"  border: none;"
            f"  border-radius: 14px;"
            f"  font-size: 18px;"
            f"  font-weight: 500;"
            f"  padding-bottom: 2px;"
            f"}}"
            f"QPushButton:hover {{"
            f"  background: {_hex_alpha(C_DANGER, 20)};"
            f"  color: {C_DANGER};"
            f"}}"
        )
        self._close_btn.clicked.connect(self.hide_card)

        h.addWidget(title, 0, Qt.AlignVCenter)
        h.addWidget(badge, 0, Qt.AlignVCenter)
        h.addStretch()
        h.addWidget(self._close_btn, 0, Qt.AlignVCenter)
        return header

    def _build_footer(self) -> QFrame:
        footer = QFrame()
        footer.setObjectName("ExpFooter")
        footer.setFixedHeight(34)
        footer.setStyleSheet(
            f"QFrame#ExpFooter {{"
            f"  background: {C_PANEL_SOFT};"
            f"  border-top: 1px solid {C_BORDER_SOFT};"
            f"  border-bottom-left-radius: 20px;"
            f"  border-bottom-right-radius: 20px;"
            f"}}"
        )
        h = QHBoxLayout(footer)
        h.setContentsMargins(20, 0, 16, 0)
        h.setSpacing(8)

        brand = QLabel("✦   Explainable Emotion Assistant")
        brand.setStyleSheet(
            f"font-size: 10px; font-weight: 600;"
            f"color: {C_DIM}; letter-spacing: 0.4px;"
            f"background: transparent;"
        )

        hint = QLabel("⋮⋮   Drag to move")
        hint.setStyleSheet(
            f"font-size: 10px; color: {C_DIM_2};"
            f"background: transparent;"
        )

        h.addWidget(brand)
        h.addStretch()
        h.addWidget(hint)
        return footer

    # ──────────────────────────────────────────
    # Construction — sidebar
    # ──────────────────────────────────────────

    def _build_sidebar(self) -> QFrame:
        sidebar = QFrame()
        sidebar.setObjectName("ExpSidebar")
        sidebar.setFixedWidth(196)
        sidebar.setStyleSheet(
            f"QFrame#ExpSidebar {{"
            f"  background: {C_PANEL_SOFT};"
            f"  border-right: 1px solid {C_BORDER_SOFT};"
            f"}}"
        )
        v = QVBoxLayout(sidebar)
        v.setContentsMargins(14, 22, 14, 22)
        v.setSpacing(4)

        section_lbl = QLabel("NAVIGATION")
        section_lbl.setStyleSheet(
            f"font-size: 10px; font-weight: 800;"
            f"color: {C_DIM_2}; letter-spacing: 1.8px;"
            f"background: transparent;"
            f"padding: 0 14px 12px 14px;"
        )
        v.addWidget(section_lbl)

        self._nav_buttons = {}
        items = [
            ("overview",    "◉", "Overview"),
            ("explanation", "✦", "AI Explanation"),
        ]
        for key, icon, label in items:
            btn = _SidebarButton(icon, label)
            btn.clicked.connect(
                lambda checked=False, k=key: self._on_nav_clicked(k)
            )
            self._nav_buttons[key] = btn
            v.addWidget(btn)

        v.addStretch()

        sep = QFrame()
        sep.setFixedHeight(1)
        sep.setStyleSheet(f"background: {C_BORDER_SOFT};")
        v.addWidget(sep)

        info = QLabel("Explainable AI\nFinal Project · 2026")
        info.setStyleSheet(
            f"font-size: 9px; color: {C_DIM_2};"
            f"background: transparent;"
            f"padding: 12px 14px 0 14px;"
            f"line-height: 1.5;"
        )
        v.addWidget(info)

        self._nav_buttons["overview"].set_active(True)
        return sidebar

    # ──────────────────────────────────────────
    # Construction — main scrollable content
    # ──────────────────────────────────────────

    def _build_main(self) -> QFrame:
        main = QFrame()
        main.setObjectName("ExpMain")
        main.setStyleSheet(
            f"QFrame#ExpMain {{ background: {C_PANEL}; }}"
        )
        layout = QVBoxLayout(main)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet(
            f"QScrollArea {{ background: transparent; border: none; }}"
            f"QScrollBar:vertical {{"
            f"  background: transparent; width: 6px; border-radius: 3px; margin: 6px 0;"
            f"}}"
            f"QScrollBar::handle:vertical {{"
            f"  background: {_hex_alpha(C_PRIMARY, 45)};"
            f"  border-radius: 3px; min-height: 24px;"
            f"}}"
            f"QScrollBar::handle:vertical:hover {{"
            f"  background: {_hex_alpha(C_PRIMARY, 75)};"
            f"}}"
            f"QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}"
        )

        content = QWidget()
        content.setStyleSheet(f"QWidget {{ background: {C_PANEL}; }}")
        cv = QVBoxLayout(content)
        cv.setContentsMargins(22, 16, 22, 18)
        cv.setSpacing(16)

        self._sec_overview    = self._build_sec_overview()
        self._sec_explanation = self._build_sec_explanation()

        # NOTE: Faces grid + Emotion details sections are built but NOT shown
        # in this version (per project request). They remain instantiated so
        # the rest of the code that references their layouts keeps working.
        self._sec_faces   = self._build_sec_faces()
        self._sec_details = self._build_sec_details()
        self._sec_faces.hide()
        self._sec_details.hide()

        cv.addWidget(self._sec_overview)
        cv.addWidget(self._sec_explanation)
        cv.addStretch()

        scroll.setWidget(content)
        self._scroll_area    = scroll
        self._scroll_content = content
        layout.addWidget(scroll, 1)
        return main

    @staticmethod
    def _make_section_title(text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet(
            f"font-size: 11px; font-weight: 800;"
            f"color: {C_DIM}; letter-spacing: 2.5px;"
            f"background: transparent;"
        )
        return lbl

    # ── Section: Overview (hero card) ──

    def _build_sec_overview(self) -> QFrame:
        section = QFrame()
        sv = QVBoxLayout(section)
        sv.setContentsMargins(0, 0, 0, 0)
        sv.setSpacing(14)
        sv.addWidget(self._make_section_title("OVERVIEW"))

        hero = QFrame()
        hero.setObjectName("HeroCard")
        hero.setStyleSheet(
            f"QFrame#HeroCard {{"
            f"  background: qlineargradient(x1:0, y1:0, x2:1, y2:1,"
            f"    stop:0 {C_PANEL}, stop:1 {C_CARD});"
            f"  border: 1px solid {C_BORDER_SOFT};"
            f"  border-radius: 16px;"
            f"}}"
        )
        hh = QHBoxLayout(hero)
        hh.setContentsMargins(20, 18, 20, 18)
        hh.setSpacing(20)

        self._ring = _EmotionRing()
        hh.addWidget(self._ring, 0, Qt.AlignVCenter)

        info = QVBoxLayout()
        info.setSpacing(4)
        info.setAlignment(Qt.AlignVCenter)

        eyebrow = QLabel("DETECTED EMOTION")
        eyebrow.setStyleSheet(
            f"font-size: 9px; font-weight: 800;"
            f"color: {C_DIM}; letter-spacing: 1.8px;"
            f"background: transparent;"
        )

        self._hero_emotion = QLabel("—")
        self._hero_emotion.setStyleSheet(
            f"font-size: 30px; font-weight: 800;"
            f"color: {C_TEXT}; letter-spacing: -0.5px;"
            f"background: transparent;"
        )

        self._hero_conf = QLabel("—  Confidence")
        self._hero_conf.setStyleSheet(
            f"font-size: 15px; font-weight: 700;"
            f"color: {C_TEXT_SOFT}; background: transparent;"
        )

        self._hero_face = QLabel("")
        self._hero_face.setFixedHeight(22)
        self._hero_face.setStyleSheet(
            f"QLabel {{"
            f"  background: {_hex_alpha(C_PRIMARY, 12)};"
            f"  color: {C_ACCENT};"
            f"  border: 1px solid {_hex_alpha(C_PRIMARY, 40)};"
            f"  border-radius: 11px;"
            f"  padding: 0 11px;"
            f"  font-size: 10px;"
            f"  font-weight: 700;"
            f"  letter-spacing: 0.4px;"
            f"}}"
        )
        self._hero_face.setVisible(False)

        face_row = QHBoxLayout()
        face_row.setContentsMargins(0, 0, 0, 0)
        face_row.setSpacing(0)
        face_row.addWidget(self._hero_face)
        face_row.addStretch()

        self._hero_sub = QLabel(
            "Detected emotional state based on facial analysis."
        )
        self._hero_sub.setWordWrap(True)
        self._hero_sub.setStyleSheet(
            f"font-size: 11px; color: {C_DIM};"
            f"background: transparent; line-height: 1.45;"
        )

        info.addWidget(eyebrow)
        info.addWidget(self._hero_emotion)
        info.addWidget(self._hero_conf)
        info.addSpacing(4)
        info.addLayout(face_row)
        info.addSpacing(2)
        info.addWidget(self._hero_sub)

        hh.addLayout(info, 1)
        sv.addWidget(hero)
        return section

    # ── Section: Faces detected ──

    def _build_sec_faces(self) -> QFrame:
        section = QFrame()
        sv = QVBoxLayout(section)
        sv.setContentsMargins(0, 0, 0, 0)
        sv.setSpacing(14)
        sv.addWidget(self._make_section_title("FACES DETECTED"))

        wrapper = QFrame()
        wrapper.setStyleSheet("QFrame { background: transparent; }")
        wv = QVBoxLayout(wrapper)
        wv.setContentsMargins(0, 0, 0, 0)
        wv.setSpacing(10)

        self._faces_grid_widget = QWidget()
        self._faces_grid_widget.setStyleSheet("QWidget { background: transparent; }")
        self._faces_grid = QGridLayout(self._faces_grid_widget)
        self._faces_grid.setContentsMargins(0, 0, 0, 0)
        self._faces_grid.setSpacing(12)
        self._faces_grid.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        wv.addWidget(self._faces_grid_widget)

        self._faces_empty = QLabel("No faces detected in this analysis.")
        self._faces_empty.setAlignment(Qt.AlignCenter)
        self._faces_empty.setStyleSheet(
            f"font-size: 13px; color: {C_DIM};"
            f"background: {C_CARD}; border: 1px dashed {C_BORDER};"
            f"border-radius: 14px; padding: 30px 12px;"
        )
        wv.addWidget(self._faces_empty)
        self._faces_empty.hide()

        sv.addWidget(wrapper)
        return section

    # ── Section: Emotion details (bar list) ──

    def _build_sec_details(self) -> QFrame:
        section = QFrame()
        sv = QVBoxLayout(section)
        sv.setContentsMargins(0, 0, 0, 0)
        sv.setSpacing(14)
        sv.addWidget(self._make_section_title("EMOTION DETAILS"))

        card = QFrame()
        card.setObjectName("DetailsCard")
        card.setStyleSheet(
            f"QFrame#DetailsCard {{"
            f"  background: {C_PANEL};"
            f"  border: 1px solid {C_BORDER_SOFT};"
            f"  border-radius: 16px;"
            f"}}"
        )
        cv = QVBoxLayout(card)
        cv.setContentsMargins(24, 20, 24, 22)
        cv.setSpacing(10)
        self._details_layout = cv

        sv.addWidget(card)
        return section

    # ── Section: AI explanation + signals ──

    def _build_sec_explanation(self) -> QFrame:
        section = QFrame()
        sv = QVBoxLayout(section)
        sv.setContentsMargins(0, 0, 0, 0)
        sv.setSpacing(10)
        sv.addWidget(self._make_section_title("AI EMOTION EXPLANATION"))

        card = QFrame()
        card.setObjectName("ExplCard")
        card.setStyleSheet(
            f"QFrame#ExplCard {{"
            f"  background: {C_PANEL};"
            f"  border: 1px solid {C_BORDER_SOFT};"
            f"  border-radius: 14px;"
            f"}}"
        )
        cv = QVBoxLayout(card)
        cv.setContentsMargins(18, 16, 18, 18)
        cv.setSpacing(10)

        self._status_label = QLabel("GENERATING EXPLANATION")
        self._status_label.setStyleSheet(
            f"font-size: 9px; font-weight: 800;"
            f"color: {C_DIM}; letter-spacing: 1.7px;"
            f"background: transparent;"
        )

        self._text_label = QLabel("")
        self._text_label.setWordWrap(True)
        self._text_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self._text_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self._text_label.setStyleSheet(
            f"color: {C_TEXT};"
            f"font-size: 13px;"
            f"line-height: 1.55;"
            f"background: transparent;"
        )
        self._text_label.setFont(QFont("Segoe UI", 11))

        signals_eyebrow = QLabel("DETECTED SIGNALS")
        signals_eyebrow.setStyleSheet(
            f"font-size: 9px; font-weight: 800;"
            f"color: {C_DIM}; letter-spacing: 1.5px;"
            f"background: transparent; padding-top: 2px;"
        )

        self._signals_grid_widget = QWidget()
        self._signals_grid_widget.setStyleSheet("QWidget { background: transparent; }")
        self._signals_layout = QGridLayout(self._signals_grid_widget)
        self._signals_layout.setContentsMargins(0, 0, 0, 0)
        self._signals_layout.setHorizontalSpacing(10)
        self._signals_layout.setVerticalSpacing(8)
        self._signals_layout.setColumnStretch(0, 0)
        self._signals_layout.setColumnStretch(1, 0)
        self._signals_layout.setColumnStretch(2, 1)

        cv.addWidget(self._status_label)
        cv.addWidget(self._text_label)
        cv.addSpacing(4)
        cv.addWidget(signals_eyebrow)
        cv.addWidget(self._signals_grid_widget)

        sv.addWidget(card)
        return section

    # ──────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────

    def prepare(self, results: dict) -> None:
        self._faces = results.get("faces") or []
        self._final_emotion = results.get("final_emotion") or "unknown"
        try:
            self._final_conf = float(results.get("confidence", 0.0) or 0.0)
        except (TypeError, ValueError):
            self._final_conf = 0.0
        self._rebuild_meta()
        self._active_tab = "overall"
        self._refresh_face_cards()
        self._apply_active()
        # Reset scroll + nav highlight
        self._set_active_nav("overview")
        if hasattr(self, "_scroll_area"):
            self._scroll_area.verticalScrollBar().setValue(0)
        self._show_and_reposition()

    def set_active_tab(self, target_id: str) -> None:
        if target_id not in self._tab_meta:
            return
        if target_id == self._active_tab:
            return
        self._active_tab = target_id
        self._apply_active()

    def get_active_tab(self) -> str:
        return self._active_tab

    def show_loading(self, emotion: str = None, confidence: float = None) -> None:
        if emotion is not None:
            try:
                conf = float(confidence or 0.0)
            except (TypeError, ValueError):
                conf = 0.0
            self._set_hero(emotion, conf)
        self._is_loading = True
        self._loader_step = 0
        self._text_label.setText(self._loader_text())
        self._status_label.setText("GENERATING EXPLANATION")
        self._loader_timer.start()
        self._show_and_reposition()

    def show_text(self, text: str) -> None:
        self._stop_loader()
        cleaned = (text or "").strip() or "No explanation was generated."
        self._text_label.setText(cleaned)
        self._status_label.setText("AI EXPLANATION")
        self._show_and_reposition()

    def show_error(self, reason: str) -> None:
        self._stop_loader()
        self._status_label.setText("EXPLANATION UNAVAILABLE")
        self._text_label.setText(
            reason or "The explanation could not be generated."
        )
        self._show_and_reposition()

    def update_emotion_meta(self, emotion: str, confidence: float) -> None:
        try:
            conf = float(confidence or 0.0)
        except (TypeError, ValueError):
            conf = 0.0
        self._set_hero(emotion or "", conf)

    def hide_card(self) -> None:
        self._stop_loader()
        self.hide()
        self.closed.emit()

    # ──────────────────────────────────────────
    # Internals — state
    # ──────────────────────────────────────────

    def _rebuild_meta(self) -> None:
        self._tab_meta.clear()
        self._tab_color.clear()

        overall_color = EMOTION_COLORS.get(
            (self._final_emotion or "").lower(), C_PRIMARY
        )
        self._tab_meta["overall"]  = (self._final_emotion, self._final_conf)
        self._tab_color["overall"] = overall_color

        for i, face in enumerate(self._faces):
            tid     = f"face:{i}"
            emo_raw = face.get("emotion") or "unknown"
            color   = EMOTION_COLORS.get(emo_raw.lower(), C_PRIMARY)
            try:
                conf = float(face.get("confidence", 0.0) or 0.0)
            except (TypeError, ValueError):
                conf = 0.0
            self._tab_meta[tid]  = (emo_raw, conf)
            self._tab_color[tid] = color

    def _refresh_face_cards(self) -> None:
        # Clear old cards
        for card in self._face_cards.values():
            card.setParent(None)
            card.deleteLater()
        self._face_cards.clear()
        while self._faces_grid.count():
            item = self._faces_grid.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()

        if not self._faces:
            self._faces_empty.show()
            return
        self._faces_empty.hide()

        cols = 4
        for i, face in enumerate(self._faces):
            tid = f"face:{i}"
            card = _FaceMiniCard(
                index=i,
                emotion=face.get("emotion") or "unknown",
                confidence=face.get("confidence", 0.0),
                face_image=face.get("face_image"),
            )
            card.selected.connect(self._on_face_card_clicked)
            self._faces_grid.addWidget(card, i // cols, i % cols)
            self._face_cards[tid] = card
        # Push grid cells to the left
        self._faces_grid.setColumnStretch(cols, 1)

    def _apply_active(self) -> None:
        for tid, card in self._face_cards.items():
            card.set_active(tid == self._active_tab)
        emo, conf = self._tab_meta.get(self._active_tab, ("unknown", 0.0))
        self._set_hero(emo, conf)
        self._refresh_emotion_bars()
        self._refresh_signals()

    def _set_hero(self, emotion: str, confidence: float) -> None:
        emo = (emotion or "unknown").lower()
        color = EMOTION_COLORS.get(emo, C_PRIMARY)
        try:
            pct = float(confidence or 0.0) * 100.0
        except (TypeError, ValueError):
            pct = 0.0

        self._ring.set_emotion(emo, confidence)
        self._hero_emotion.setText(emo.upper())
        self._hero_emotion.setStyleSheet(
            f"font-size: 30px; font-weight: 800;"
            f"color: {color}; letter-spacing: -0.5px;"
            f"background: transparent;"
        )
        self._hero_conf.setText(f"{pct:.0f}%   Confidence")

        if self._active_tab == "overall":
            face_text = "OVERALL"
        elif self._active_tab.startswith("face:"):
            try:
                idx = int(self._active_tab.split(":", 1)[1])
                face_text = f"FACE  {idx + 1}"
            except ValueError:
                face_text = ""
        else:
            face_text = ""

        if face_text:
            self._hero_face.setText(face_text)
            self._hero_face.setStyleSheet(
                f"QLabel {{"
                f"  background: {_hex_alpha(color, 14)};"
                f"  color: {color};"
                f"  border: 1px solid {_hex_alpha(color, 45)};"
                f"  border-radius: 11px;"
                f"  padding: 0 11px;"
                f"  font-size: 10px;"
                f"  font-weight: 800;"
                f"  letter-spacing: 1px;"
                f"}}"
            )
            self._hero_face.setVisible(True)
        else:
            self._hero_face.setVisible(False)

    # ── Emotion details bars ──

    def _refresh_emotion_bars(self) -> None:
        for bar in self._emotion_bars:
            bar.setParent(None)
            bar.deleteLater()
        self._emotion_bars.clear()

        dist = self._current_distribution()
        items = sorted(dist.items(), key=lambda kv: kv[1], reverse=True)
        for emo, val in items:
            row = self._make_emotion_bar(emo, val)
            self._details_layout.addWidget(row)
            self._emotion_bars.append(row)

    def _current_distribution(self) -> dict:
        if self._active_tab == "overall":
            return self._aggregate_distribution()
        if self._active_tab.startswith("face:"):
            try:
                idx = int(self._active_tab.split(":", 1)[1])
                face = self._faces[idx]
                ae = face.get("all_emotions") or {}
                if ae:
                    return {
                        k.lower(): float(v)
                        for k, v in ae.items()
                        if k.lower() in ALL_EMOTIONS
                    }
            except (ValueError, IndexError):
                pass
        emo, conf = self._tab_meta.get(self._active_tab, ("unknown", 0.0))
        dist = {e: 0.0 for e in ALL_EMOTIONS}
        if (emo or "").lower() in dist:
            dist[emo.lower()] = conf
        return dist

    def _aggregate_distribution(self) -> dict:
        sums = {e: 0.0 for e in ALL_EMOTIONS}
        count = 0
        for face in self._faces:
            ae = face.get("all_emotions") or {}
            if not ae:
                continue
            for emo, prob in ae.items():
                k = emo.lower()
                if k in sums:
                    try:
                        sums[k] += float(prob or 0.0)
                    except (TypeError, ValueError):
                        pass
            count += 1
        if count > 0:
            return {e: v / count for e, v in sums.items()}
        dist = dict(sums)
        emo = (self._final_emotion or "").lower()
        if emo in dist:
            dist[emo] = self._final_conf
        return dist

    def _make_emotion_bar(self, emotion: str, value: float) -> QFrame:
        emo = (emotion or "").lower()
        color = EMOTION_COLORS.get(emo, C_PRIMARY)
        try:
            value = max(0.0, min(1.0, float(value or 0.0)))
        except (TypeError, ValueError):
            value = 0.0

        frame = QFrame()
        frame.setFixedHeight(30)
        frame.setStyleSheet("QFrame { background: transparent; }")
        h = QHBoxLayout(frame)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(14)

        name = QLabel(emo.capitalize())
        name.setFixedWidth(88)
        name.setStyleSheet(
            f"font-size: 12px; font-weight: 700;"
            f"color: {C_TEXT_SOFT}; background: transparent;"
        )

        bar = QProgressBar()
        bar.setFixedHeight(8)
        bar.setRange(0, 100)
        bar.setValue(0)
        bar.setTextVisible(False)
        bar.setStyleSheet(
            f"QProgressBar {{"
            f"  background: {C_BORDER_SOFT};"
            f"  border: none;"
            f"  border-radius: 4px;"
            f"}}"
            f"QProgressBar::chunk {{"
            f"  background: {color};"
            f"  border-radius: 4px;"
            f"}}"
        )

        anim = QPropertyAnimation(bar, b"value", frame)
        anim.setDuration(900)
        anim.setEasingCurve(QEasingCurve.OutCubic)
        anim.setEndValue(int(value * 100))
        QTimer.singleShot(120, anim.start)
        frame._bar_anim = anim  # keep reference

        val_lbl = QLabel(f"{value * 100:.0f}%")
        val_lbl.setFixedWidth(46)
        val_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        val_lbl.setStyleSheet(
            f"font-size: 12px; font-weight: 700;"
            f"color: {C_TEXT}; background: transparent;"
        )

        h.addWidget(name)
        h.addWidget(bar, 1)
        h.addWidget(val_lbl)
        return frame

    # ── Signals chips ──

    def _refresh_signals(self) -> None:
        # Clear existing
        for chip in self._signal_chips:
            chip.setParent(None)
            chip.deleteLater()
        self._signal_chips.clear()
        while self._signals_layout.count():
            item = self._signals_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()

        emo, _ = self._tab_meta.get(self._active_tab, ("unknown", 0.0))
        signals = DETECTED_SIGNALS.get((emo or "").lower(), [])

        for i, text in enumerate(signals):
            chip = _SignalChip(text)
            self._signals_layout.addWidget(chip, i // 2, i % 2,
                                           Qt.AlignLeft | Qt.AlignVCenter)
            self._signal_chips.append(chip)

    # ──────────────────────────────────────────
    # Navigation
    # ──────────────────────────────────────────

    def _set_active_nav(self, key: str) -> None:
        for k, btn in self._nav_buttons.items():
            btn.set_active(k == key)

    def _on_nav_clicked(self, key: str) -> None:
        self._set_active_nav(key)
        section_map = {
            "overview":    getattr(self, "_sec_overview",    None),
            "explanation": getattr(self, "_sec_explanation", None),
        }
        section = section_map.get(key)
        if section is None or not hasattr(self, "_scroll_area"):
            return
        bar = self._scroll_area.verticalScrollBar()
        target_y = max(0, section.y() - 12)
        if self._scroll_anim is not None:
            self._scroll_anim.stop()
        anim = QPropertyAnimation(bar, b"value", self)
        anim.setDuration(380)
        anim.setEasingCurve(QEasingCurve.OutCubic)
        anim.setStartValue(bar.value())
        anim.setEndValue(target_y)
        anim.start()
        self._scroll_anim = anim

    def _on_face_card_clicked(self, idx: int) -> None:
        tid = f"face:{idx}"
        if tid not in self._tab_meta:
            return
        if tid == self._active_tab:
            return
        self._active_tab = tid
        self._apply_active()
        self.face_selected.emit(tid)

    # ──────────────────────────────────────────
    # Loader animation
    # ──────────────────────────────────────────

    def _loader_text(self) -> str:
        dots = "." * (self._loader_step % 4)
        return f"Analyzing model output and composing an explanation{dots}"

    def _tick_loader(self) -> None:
        self._loader_step += 1
        self._text_label.setText(self._loader_text())

    def _stop_loader(self) -> None:
        if self._is_loading:
            self._loader_timer.stop()
            self._is_loading = False

    # ──────────────────────────────────────────
    # Show / position
    # ──────────────────────────────────────────

    def _show_and_reposition(self) -> None:
        self.adjustSize()
        if not self.isVisible():
            parent = self.parentWidget()
            if parent is not None:
                pr = parent.rect()
                x = max(10, (pr.width() - self.width()) // 2)
                y = max(20, (pr.height() - self.height()) // 3)
                self.move(x, y)
        self.show()
        self.raise_()

    # ──────────────────────────────────────────
    # Dragging — click anywhere on the card to drag it
    # ──────────────────────────────────────────

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drag_offset = event.globalPos() - self.mapToGlobal(QPoint(0, 0))
            self._is_dragging = True
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._is_dragging and (event.buttons() & Qt.LeftButton):
            new_global = event.globalPos() - self._drag_offset
            parent = self.parentWidget()
            if parent is not None:
                new_local = parent.mapFromGlobal(new_global)
                pr = parent.rect()
                x = max(0, min(new_local.x(), pr.width()  - self.width()))
                y = max(0, min(new_local.y(), pr.height() - self.height()))
                self.move(x, y)
            else:
                self.move(new_global)
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._is_dragging = False
            event.accept()
        else:
            super().mouseReleaseEvent(event)


# ══════════════════════════════════════════════════════════════════════
# Standalone preview (manual test)
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication as _QA

    app = _QA(sys.argv)
    host = QWidget()
    host.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
    host.setAttribute(Qt.WA_TranslucentBackground)
    host.setGeometry(_QA.primaryScreen().geometry())
    host.show()

    card = ExplanationCard(host)
    demo = {
        "final_emotion": "happy",
        "confidence":    0.94,
        "faces": [
            {"emotion": "happy",    "confidence": 0.94, "bbox": (0, 0, 80, 80),
             "all_emotions": {"happy": 0.94, "neutral": 0.04, "sad": 0.01,
                              "angry": 0.0, "fear": 0.0, "surprise": 0.01,
                              "disgust": 0.0}},
            {"emotion": "neutral",  "confidence": 0.78, "bbox": (0, 0, 80, 80),
             "all_emotions": {"happy": 0.10, "neutral": 0.78, "sad": 0.05,
                              "angry": 0.02, "fear": 0.02, "surprise": 0.02,
                              "disgust": 0.01}},
            {"emotion": "surprise", "confidence": 0.66, "bbox": (0, 0, 80, 80),
             "all_emotions": {"happy": 0.10, "neutral": 0.14, "sad": 0.02,
                              "angry": 0.02, "fear": 0.06, "surprise": 0.66,
                              "disgust": 0.0}},
        ],
    }
    card.prepare(demo)
    card.show_loading()
    QTimer.singleShot(1800, lambda: card.show_text(
        "The system classified this emotion as Happy because facial landmarks "
        "indicate elevated mouth corners, relaxed eyebrows, and positive "
        "expression patterns. The next strongest signal was Neutral, which "
        "was considered but not selected."
    ))

    sys.exit(app.exec_())
