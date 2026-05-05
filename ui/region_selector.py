"""
region_selector.py - Full-screen region selection overlay.

Shows a screenshot as background, lets the user drag to select a rectangle.
Returns the selected region as (x, y, width, height) in screen coordinates.

Usage:
    region = RegionSelector.select()
    # -> (100, 200, 640, 480)  or  None if cancelled
"""

import cv2
import numpy as np
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QVBoxLayout
from PyQt5.QtCore    import Qt, QRect, QPoint
from PyQt5.QtGui     import QPainter, QColor, QPen, QPixmap, QImage, QFont


class RegionSelector(QWidget):
    """
    Full-screen transparent overlay for region selection.

    The user clicks and drags to draw a rectangle.
    On mouse release the selection is confirmed and the widget closes.
    Press Escape to cancel.
    """

    def __init__(self, screenshot: np.ndarray):
        super().__init__()
        self._screenshot  = screenshot
        self._origin      = QPoint()
        self._current     = QPoint()
        self._selecting   = False
        self._result: tuple | None = None

        self._setup_window()
        self._bg_pixmap = self._numpy_to_pixmap(screenshot)

    def _setup_window(self) -> None:
        self.setWindowFlags(
            Qt.FramelessWindowHint |
            Qt.WindowStaysOnTopHint |
            Qt.Tool
        )
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setCursor(Qt.CrossCursor)
        # Cover all screens
        screen = QApplication.primaryScreen().geometry()
        self.setGeometry(screen)
        self.showFullScreen()

    # ============================
    # Drawing
    # ============================

    def paintEvent(self, event) -> None:
        painter = QPainter(self)

        # Draw screenshot as background (darkened)
        if self._bg_pixmap:
            painter.drawPixmap(0, 0, self._bg_pixmap)

        # Dark overlay over entire screen
        painter.fillRect(self.rect(), QColor(0, 0, 0, 100))

        # Draw selection rectangle
        if self._selecting and not self._origin.isNull():
            rect = self._get_selection_rect()

            # Clear (bright) inside the selection
            painter.fillRect(rect, QColor(0, 0, 0, 0))
            if self._bg_pixmap:
                painter.drawPixmap(rect, self._bg_pixmap, rect)

            # Border
            pen = QPen(QColor("#6C63FF"), 2, Qt.SolidLine)
            painter.setPen(pen)
            painter.drawRect(rect)

            # Corner handles
            handle = 8
            painter.fillRect(rect.left(),               rect.top(),    handle, handle, QColor("#6C63FF"))
            painter.fillRect(rect.right() - handle,     rect.top(),    handle, handle, QColor("#6C63FF"))
            painter.fillRect(rect.left(),               rect.bottom() - handle, handle, handle, QColor("#6C63FF"))
            painter.fillRect(rect.right() - handle,     rect.bottom() - handle, handle, handle, QColor("#6C63FF"))

            # Dimensions label
            w = abs(self._current.x() - self._origin.x())
            h = abs(self._current.y() - self._origin.y())
            label_text = f"{w} x {h}"
            painter.setPen(QColor("#FFFFFF"))
            painter.setFont(QFont("Segoe UI", 11, QFont.Bold))
            label_x = min(self._origin.x(), self._current.x())
            label_y = max(self._origin.y(), self._current.y()) + 20
            painter.drawText(label_x, label_y, label_text)

        # Instruction text (top center)
        painter.setPen(QColor("#FFFFFF"))
        painter.setFont(QFont("Segoe UI", 13))
        instruction = "Drag to select region  |  Esc to cancel  |  Release to confirm"
        painter.drawText(self.rect(), Qt.AlignTop | Qt.AlignHCenter, instruction)

    # ============================
    # Mouse events
    # ============================

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self._origin    = event.pos()
            self._current   = event.pos()
            self._selecting = True
            self.update()

    def mouseMoveEvent(self, event) -> None:
        if self._selecting:
            self._current = event.pos()
            self.update()

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.LeftButton and self._selecting:
            self._current   = event.pos()
            self._selecting = False
            rect = self._get_selection_rect()

            if rect.width() > 10 and rect.height() > 10:
                # Convert to screen coordinates
                screen_ratio_x = self._screenshot.shape[1] / self.width()
                screen_ratio_y = self._screenshot.shape[0] / self.height()
                x = int(rect.x()      * screen_ratio_x)
                y = int(rect.y()      * screen_ratio_y)
                w = int(rect.width()  * screen_ratio_x)
                h = int(rect.height() * screen_ratio_y)
                self._result = (x, y, w, h)

            self.close()

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key_Escape:
            self._result = None
            self.close()

    # ============================
    # Helpers
    # ============================

    def _get_selection_rect(self) -> QRect:
        x1 = min(self._origin.x(), self._current.x())
        y1 = min(self._origin.y(), self._current.y())
        x2 = max(self._origin.x(), self._current.x())
        y2 = max(self._origin.y(), self._current.y())
        return QRect(x1, y1, x2 - x1, y2 - y1)

    @staticmethod
    def _numpy_to_pixmap(image: np.ndarray) -> QPixmap | None:
        try:
            if image.ndim == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            if image.dtype != np.uint8:
                image = (image * 255).clip(0, 255).astype(np.uint8)
            h, w, ch = image.shape
            q_img = QImage(image.data, w, h, w * ch, QImage.Format_RGB888)
            return QPixmap.fromImage(q_img)
        except Exception:
            return None

    # ============================
    # Public API
    # ============================

    @staticmethod
    def select(screenshot: np.ndarray) -> tuple | None:
        """
        Show the region selector and wait for user to select.

        Parameters:
            screenshot - full screen RGB numpy array to show as background

        Returns:
            (x, y, width, height) in screen pixels, or None if cancelled
        """
        selector = RegionSelector(screenshot)
        loop = selector._wait_for_close()
        return selector._result

    def _wait_for_close(self):
        """Block until the window is closed."""
        loop = QApplication.instance()
        while self.isVisible():
            loop.processEvents()
