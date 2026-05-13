"""
explanation_card.py — Floating "Explainable AI" card.

A draggable, glassmorphism-styled card that displays a natural-language
explanation of WHY a particular emotion was selected by the analysis
pipeline. The card is intentionally a standalone widget so it does not
modify or interfere with the existing `ResultsPanel` / `BBoxScreenOverlay`
in `overlay_ui.py`.

Behavior:
    * Stays visible until the user clicks the X button — no auto-close.
    * Three states: loading (animated dots), text (final explanation),
      error (graceful fallback hint).
    * Draggable by clicking anywhere on its background.

Public API (all safe to call from the main GUI thread):
    card = ExplanationCard(parent)
    card.show_loading(emotion="happy", confidence=0.82)
    card.show_text("...")
    card.show_error("OpenAI unavailable, local fallback used.")
    card.hide_card()

Section title displayed on the card:
    "Why was this emotion selected?"
"""

from __future__ import annotations

from PyQt5.QtCore import (
    Qt, QPoint, QRectF, QTimer, QPropertyAnimation, QEasingCurve, pyqtSignal,
)
from PyQt5.QtGui import (
    QColor, QFont, QPainter, QBrush, QPen,
)
from PyQt5.QtWidgets import (
    QApplication, QFrame, QGraphicsDropShadowEffect, QHBoxLayout, QLabel,
    QPushButton, QScrollArea, QSizePolicy, QVBoxLayout, QWidget,
)


# ──────────────────────────────────────────
# Palette — matches ui/overlay_ui.py
# ──────────────────────────────────────────
C_BG       = "#F4EDE4"
C_PANEL    = "#FDFAF7"
C_CARD     = "#F9F5F0"
C_BORDER   = "#DDD0BF"
C_TEXT     = "#3A2D22"
C_DIM      = "#9A8878"
C_PRIMARY  = "#A0806A"
C_ACCENT   = "#7A5C44"
C_SUCCESS  = "#7AAE82"
C_DANGER   = "#C26B5A"

EMOTION_COLORS = {
    "happy":    "#4CAE6C",
    "sad":      "#4F82C8",
    "angry":    "#D05830",
    "neutral":  "#8A8480",
    "surprise": "#C87818",
    "fear":     "#8850C8",
    "disgust":  "#C84888",
}


class ExplanationCard(QFrame):
    """
    Floating glassmorphism card for the Explainable AI Emotion Assistant.

    The card is created hidden. Call `show_loading(...)` first when an
    explanation is being generated, then either `show_text(...)` when it
    arrives, or `show_error(...)` if generation failed.

    The card never closes itself — only the X button triggers `hide_card()`.
    """

    CARD_WIDTH       = 640
    CARD_MIN_HEIGHT  = 440       # guarantees the body has room to breathe
    MAX_CARD_HEIGHT_RATIO = 0.75 # fraction of the screen height
    LOADER_INTERVAL  = 320       # ms per animated-dots tick

    closed         = pyqtSignal()
    face_selected  = pyqtSignal(str)   # "overall" or "face:N" — user clicked a tab

    def __init__(self, parent: QWidget = None) -> None:
        super().__init__(parent)
        self._drag_offset       = QPoint()
        self._is_dragging       = False
        self._loader_step       = 0
        self._is_loading        = False

        # Tab state
        self._faces:          list = []
        self._final_emotion:  str  = "unknown"
        self._final_conf:     float = 0.0
        self._active_tab:     str  = "overall"
        self._tab_buttons:    dict = {}   # target_id -> QPushButton
        self._tab_meta:       dict = {}   # target_id -> (emotion, confidence)
        self._tab_color:      dict = {}   # target_id -> hex color

        self._loader_timer = QTimer(self)
        self._loader_timer.setInterval(self.LOADER_INTERVAL)
        self._loader_timer.timeout.connect(self._tick_loader)

        self._build()
        self._style()

        self.setFixedWidth(self.CARD_WIDTH)
        self.setMinimumHeight(self.CARD_MIN_HEIGHT)
        screen_h = QApplication.primaryScreen().geometry().height()
        self.setMaximumHeight(int(screen_h * self.MAX_CARD_HEIGHT_RATIO))
        self.hide()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(self._build_header())
        root.addWidget(self._build_tabs_row())
        root.addWidget(self._build_chip_row())
        root.addWidget(self._build_body(), 1)
        root.addWidget(self._build_footer())

    def _build_tabs_row(self) -> QFrame:
        """
        Row of selectable tabs: [ Overall ] [ Face 1 ] [ Face 2 ] ...
        Empty until `prepare(results)` is called.
        """
        frame = QFrame()
        frame.setObjectName("ExpTabsRow")
        frame.setStyleSheet(f"""
            QFrame#ExpTabsRow {{
                background: #F4EDE4;
                border-bottom: 1px solid {C_BORDER};
            }}
        """)

        outer = QHBoxLayout(frame)
        outer.setContentsMargins(14, 10, 14, 10)
        outer.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setFixedHeight(40)
        scroll.setStyleSheet(f"""
            QScrollArea {{ background: transparent; border: none; }}
            QScrollBar:horizontal {{
                background: transparent; height: 3px; border-radius: 1px;
            }}
            QScrollBar::handle:horizontal {{
                background: {C_PRIMARY}66; border-radius: 1px; min-width: 20px;
            }}
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}
        """)

        container = QWidget()
        container.setStyleSheet("background: transparent;")
        self._tabs_layout = QHBoxLayout(container)
        self._tabs_layout.setContentsMargins(0, 0, 0, 0)
        self._tabs_layout.setSpacing(6)
        self._tabs_layout.addStretch()  # tail stretch so buttons left-align

        scroll.setWidget(container)
        outer.addWidget(scroll)

        # Placeholder shown when there are no faces (no analysis yet).
        self._tabs_placeholder = QLabel("No analysis yet")
        self._tabs_placeholder.setAlignment(Qt.AlignCenter)
        self._tabs_placeholder.setStyleSheet(
            f"color:{C_DIM}; font-size:10px; background:transparent;"
        )
        outer.addWidget(self._tabs_placeholder)
        self._tabs_placeholder.hide()
        scroll.hide()
        self._tabs_scroll = scroll
        return frame

    def _build_header(self) -> QFrame:
        header = QFrame()
        header.setObjectName("ExpHeader")
        header.setStyleSheet(f"""
            QFrame#ExpHeader {{
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 #F4EDE4, stop:1 #EFE6DA
                );
                border-top-left-radius: 18px;
                border-top-right-radius: 18px;
                border-bottom: 1px solid {C_BORDER};
            }}
        """)
        hbox = QHBoxLayout(header)
        hbox.setContentsMargins(22, 16, 14, 16)
        hbox.setSpacing(10)

        title_lbl = QLabel("MoodCapture")
        title_lbl.setStyleSheet(
            f"font-size:22px; font-weight:800; color:{C_TEXT}; "
            f"background:transparent; letter-spacing:-0.3px;"
        )
        hbox.addWidget(title_lbl, 1, Qt.AlignVCenter)

        # Close button — plain × (no decorative symbols)
        self._close_btn = QPushButton("×")
        self._close_btn.setFixedSize(34, 34)
        self._close_btn.setCursor(Qt.PointingHandCursor)
        self._close_btn.setToolTip("Close explanation")
        self._close_btn.setStyleSheet(f"""
            QPushButton {{
                background: transparent;
                color: {C_DIM};
                border: none;
                border-radius: 17px;
                font-size: 22px;
                font-weight: 600;
                padding-bottom: 3px;
            }}
            QPushButton:hover {{
                background: {C_DANGER}22;
                color: {C_DANGER};
            }}
        """)
        self._close_btn.clicked.connect(self.hide_card)
        hbox.addWidget(self._close_btn, 0, Qt.AlignTop)

        return header

    def _build_chip_row(self) -> QFrame:
        """Row showing the final emotion chip + confidence percentage."""
        frame = QFrame()
        frame.setObjectName("ExpChipRow")
        frame.setStyleSheet(f"""
            QFrame#ExpChipRow {{
                background: #F2E8DE;
                border-bottom: 1px solid {C_BORDER};
            }}
        """)
        hbox = QHBoxLayout(frame)
        hbox.setContentsMargins(22, 14, 22, 14)
        hbox.setSpacing(16)

        self._emotion_chip = QLabel("—")
        self._emotion_chip.setAlignment(Qt.AlignCenter)
        self._emotion_chip.setFixedHeight(38)
        self._emotion_chip.setStyleSheet(self._chip_style(C_PRIMARY))

        self._conf_label = QLabel("Confidence —")
        self._conf_label.setStyleSheet(
            f"font-size:17px; font-weight:600; color:{C_TEXT}; background:transparent;"
        )

        hbox.addWidget(self._emotion_chip, 0, Qt.AlignVCenter)
        hbox.addWidget(self._conf_label, 0, Qt.AlignVCenter)
        hbox.addStretch()

        return frame

    def _build_body(self) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet(f"""
            QScrollArea {{ background: transparent; border: none; }}
            QScrollBar:vertical {{
                background: {C_BORDER}; width: 5px; border-radius: 2px;
            }}
            QScrollBar::handle:vertical {{
                background: {C_PRIMARY}AA; border-radius: 2px; min-height: 20px;
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
        """)

        content = QWidget()
        content.setStyleSheet("background: transparent;")
        layout = QVBoxLayout(content)
        layout.setContentsMargins(22, 18, 22, 18)
        layout.setSpacing(12)

        self._status_label = QLabel("Generating explanation")
        self._status_label.setStyleSheet(
            f"font-size:13px; font-weight:700; color:{C_DIM}; "
            f"letter-spacing:1.5px; background:transparent; text-transform:uppercase;"
        )

        self._text_label = QLabel("")
        self._text_label.setWordWrap(True)
        self._text_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self._text_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        self._text_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self._text_label.setStyleSheet(f"""
            color: {C_TEXT};
            font-size: 18px;
            line-height: 1.7;
            background: transparent;
        """)
        self._text_label.setFont(QFont("Segoe UI", 13))

        layout.addWidget(self._status_label)
        layout.addWidget(self._text_label, 1)   # ← grows to fill the body

        scroll.setWidget(content)
        return scroll

    def _build_footer(self) -> QFrame:
        footer = QFrame()
        footer.setObjectName("ExpFooter")
        footer.setStyleSheet(f"""
            QFrame#ExpFooter {{
                background: #F4EDE4;
                border-top: 1px solid {C_BORDER};
                border-bottom-left-radius: 18px;
                border-bottom-right-radius: 18px;
            }}
        """)
        hbox = QHBoxLayout(footer)
        hbox.setContentsMargins(22, 12, 22, 12)

        self._source_label = QLabel("Explainable AI Emotion Assistant")
        self._source_label.setStyleSheet(
            f"font-size:12px; color:{C_DIM}; background:transparent; letter-spacing:0.5px;"
        )
        hbox.addWidget(self._source_label)
        hbox.addStretch()

        hint = QLabel("Drag to move")
        hint.setStyleSheet(
            f"font-size:11px; color:{C_DIM}; background:transparent;"
        )
        hbox.addWidget(hint)
        return footer

    def _style(self) -> None:
        # Glassmorphism: translucent panel + soft warm shadow.
        self.setStyleSheet(f"""
            ExplanationCard {{
                background: rgba(253, 250, 247, 0.96);
                border: 1px solid {C_BORDER};
                border-radius: 18px;
            }}
        """)
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(46)
        shadow.setColor(QColor(120, 85, 60, 70))
        shadow.setOffset(0, 12)
        self.setGraphicsEffect(shadow)
        self.setCursor(Qt.SizeAllCursor)

    @staticmethod
    def _chip_style(hex_color: str) -> str:
        return (
            f"background: {hex_color}1A;"
            f"color: {hex_color};"
            f"border: 1.5px solid {hex_color}55;"
            f"border-radius: 19px;"
            f"font-size: 17px;"
            f"font-weight: 800;"
            f"padding: 0 20px;"
            f"letter-spacing: 0.5px;"
        )

    # ------------------------------------------------------------------
    # Public API — called from the main thread by EmotionOverlay
    # ------------------------------------------------------------------

    def prepare(self, results: dict) -> None:
        """
        Initialize the card for a new analysis result.

        Builds the tab strip (Overall + one tab per detected face),
        defaults the active tab to 'overall', updates the emotion chip,
        and shows the card. The body is left untouched — the caller is
        expected to follow up with `show_loading()` / `show_text(...)`.
        """
        self._faces         = results.get("faces") or []
        self._final_emotion = results.get("final_emotion") or "unknown"
        try:
            self._final_conf = float(results.get("confidence", 0.0) or 0.0)
        except (TypeError, ValueError):
            self._final_conf = 0.0
        self._build_tab_buttons()
        self._active_tab = "overall"
        self._apply_active_state()
        self._show_and_reposition()

    def set_active_tab(self, target_id: str) -> None:
        """
        Programmatically activate a tab. Updates the chip + button styles.
        Does NOT emit `face_selected` (that signal is for USER clicks only).

        Works for both visible-button targets (e.g. "overall") and metadata-
        only targets (e.g. "face:N") — face selection happens via bbox
        clicks on the screen, so face tabs have no buttons.
        """
        if target_id not in self._tab_meta:
            return
        if target_id == self._active_tab:
            return
        self._active_tab = target_id
        self._apply_active_state()

    def get_active_tab(self) -> str:
        """Return the currently active tab id ('overall' or 'face:N')."""
        return self._active_tab

    def show_loading(self, emotion: str = None, confidence: float = None) -> None:
        """
        Display the loader animation in the body.

        If `emotion` is provided, the chip is also updated — kept for
        legacy callers and for tests. New code should use `prepare(...)`
        followed by `show_loading()` (no args).
        """
        if emotion is not None:
            self._set_emotion_chip(emotion, confidence or 0.0)
        self._is_loading = True
        self._loader_step = 0
        self._text_label.setText(self._loader_text())
        self._status_label.setText("Generating explanation")
        self._loader_timer.start()
        self._show_and_reposition()

    def show_text(self, text: str) -> None:
        """Display the final explanation text. Stops the loader animation."""
        self._stop_loader()
        cleaned = (text or "").strip() or "No explanation was generated."
        self._text_label.setText(cleaned)
        self._status_label.setText("AI Explanation")
        self._show_and_reposition()

    def show_error(self, reason: str) -> None:
        """Soft-error state — keeps the card open with an explanatory note."""
        self._stop_loader()
        self._status_label.setText("Explanation unavailable")
        msg = reason or "The explanation could not be generated."
        self._text_label.setText(msg)
        self._show_and_reposition()

    def update_emotion_meta(self, emotion: str, confidence: float) -> None:
        """Refresh just the emotion chip + confidence row (no text change)."""
        self._set_emotion_chip(emotion, confidence)

    def hide_card(self) -> None:
        """Hide the card and stop any animation. Emits `closed`."""
        self._stop_loader()
        self.hide()
        self.closed.emit()

    # ------------------------------------------------------------------
    # Internals — tabs
    # ------------------------------------------------------------------

    def _build_tab_buttons(self) -> None:
        """
        (Re)build the tab strip from the current `_faces` / final values.

        Only the "Overall" tab is rendered as a clickable button — per-face
        selection happens by clicking the face's bbox directly on the screen,
        not via tab buttons. Per-face metadata is still kept in `_tab_meta` /
        `_tab_color` so the chip and active state can update programmatically
        when a bbox is clicked.
        """
        # Remove existing tab buttons
        for btn in self._tab_buttons.values():
            btn.setParent(None)
            btn.deleteLater()
        self._tab_buttons.clear()
        self._tab_meta.clear()
        self._tab_color.clear()

        # Overall — the ONLY visible tab. Clicking it returns to the
        # aggregated explanation after a face was selected.
        overall_color = EMOTION_COLORS.get(
            (self._final_emotion or "").lower(), C_PRIMARY
        )
        self._add_tab_button("overall", "Overall", overall_color)
        self._tab_meta["overall"]  = (self._final_emotion, self._final_conf)
        self._tab_color["overall"] = overall_color

        # Per-face metadata only (no button — selection via bbox click).
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

        if self._tab_buttons:
            self._tabs_placeholder.hide()
            self._tabs_scroll.show()
        else:
            self._tabs_placeholder.show()
            self._tabs_scroll.hide()

    def _add_tab_button(self, target_id: str, label: str, color: str) -> None:
        btn = QPushButton(label)
        btn.setCursor(Qt.PointingHandCursor)
        btn.setToolTip(label)
        btn.setStyleSheet(self._tab_style(color, active=False))
        btn.clicked.connect(
            lambda checked=False, tid=target_id: self._on_tab_clicked(tid)
        )
        # Insert before the trailing stretch.
        idx = self._tabs_layout.count() - 1
        self._tabs_layout.insertWidget(idx, btn)
        self._tab_buttons[target_id] = btn

    def _apply_active_state(self) -> None:
        for tid, btn in self._tab_buttons.items():
            color  = self._tab_color.get(tid, C_PRIMARY)
            active = (tid == self._active_tab)
            btn.setStyleSheet(self._tab_style(color, active=active))
        emo, conf = self._tab_meta.get(self._active_tab, ("unknown", 0.0))
        self._set_emotion_chip(emo, conf)

    def _on_tab_clicked(self, target_id: str) -> None:
        if target_id == self._active_tab:
            return
        self._active_tab = target_id
        self._apply_active_state()
        self.face_selected.emit(target_id)

    @staticmethod
    def _tab_style(color: str, active: bool) -> str:
        if active:
            return (
                f"QPushButton {{"
                f"  background: {color}33;"
                f"  color: {color};"
                f"  border: 1.5px solid {color}AA;"
                f"  border-radius: 15px;"
                f"  padding: 5px 18px;"
                f"  font-size: 13px;"
                f"  font-weight: 800;"
                f"  letter-spacing: 0.3px;"
                f"}}"
            )
        return (
            f"QPushButton {{"
            f"  background: transparent;"
            f"  color: {C_DIM};"
            f"  border: 1px solid {C_BORDER};"
            f"  border-radius: 15px;"
            f"  padding: 5px 18px;"
            f"  font-size: 13px;"
            f"  font-weight: 600;"
            f"}}"
            f"QPushButton:hover {{"
            f"  color: {color};"
            f"  border-color: {color}88;"
            f"  background: {color}11;"
            f"}}"
        )

    # ------------------------------------------------------------------
    # Internals — chip / loader / show
    # ------------------------------------------------------------------

    def _set_emotion_chip(self, emotion: str, confidence: float) -> None:
        emo  = (emotion or "unknown").lower()
        color = EMOTION_COLORS.get(emo, C_PRIMARY)
        self._emotion_chip.setText(emo.capitalize())
        self._emotion_chip.setStyleSheet(self._chip_style(color))
        try:
            pct = float(confidence) * 100.0
        except (TypeError, ValueError):
            pct = 0.0
        self._conf_label.setText(f"Confidence  {pct:.0f}%")

    def _loader_text(self) -> str:
        dots = "." * (self._loader_step % 4)
        return f"Analyzing the model output and composing an explanation{dots}"

    def _tick_loader(self) -> None:
        self._loader_step += 1
        self._text_label.setText(self._loader_text())

    def _stop_loader(self) -> None:
        if self._is_loading:
            self._loader_timer.stop()
            self._is_loading = False

    def _show_and_reposition(self) -> None:
        self.adjustSize()
        # Default position: top-center of the parent (which covers the screen)
        # but only on first show — preserve any user-dragged position after that.
        if not self.isVisible():
            parent = self.parentWidget()
            if parent is not None:
                pr = parent.rect()
                x = max(10, (pr.width() - self.width()) // 2)
                y = 30
                self.move(x, y)
        self.show()
        self.raise_()

    # ------------------------------------------------------------------
    # Dragging — click anywhere on the card to drag it
    # ------------------------------------------------------------------

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


# ══════════════════════════════════════════
# Standalone preview (manual test)
# ══════════════════════════════════════════

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
    card.show_loading("happy", 0.82)

    QTimer.singleShot(2500, lambda: card.show_text(
        "The system selected \"happy\" because the detected face(s) showed signals "
        "consistent with happiness at a high confidence of 82%. The next strongest "
        "signal was \"neutral\" at 8%, which the model considered but did not select."
    ))

    sys.exit(app.exec_())
