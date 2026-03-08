"""Main window for Qwen3-ASR desktop app.

Shows real-time transcription from microphone input with start/stop control.
"""

import time
import traceback

import numpy as np
from PySide6.QtCore import Qt, QThread, Signal, Slot
from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QTextEdit,
    QLabel,
    QStatusBar,
    QProgressBar,
)

from ..audio.capture import AudioCapture
from ..inference.engine import ASREngine, StreamingState


class InferenceThread(QThread):
    """Background thread for ASR inference.

    Receives PCM audio chunks via feed(), runs the streaming engine,
    and emits transcription updates.
    """

    text_updated = Signal(str, str)  # (language, text)
    error_occurred = Signal(str)
    chunk_processed = Signal(float)  # processing time in ms

    def __init__(self, engine: ASREngine, parent=None):
        super().__init__(parent)
        self._engine = engine
        self._state: StreamingState | None = None
        self._pending_chunks: list[np.ndarray] = []
        self._running = False

    def start_session(self):
        """Start a new ASR session."""
        self._state = self._engine.new_session()
        self._pending_chunks.clear()
        self._running = True
        if not self.isRunning():
            self.start()

    def stop_session(self):
        """Stop the current session and flush remaining audio."""
        self._running = False

    def feed(self, pcm: np.ndarray):
        """Queue a PCM chunk for processing."""
        if self._running:
            self._pending_chunks.append(pcm)

    def run(self):
        """Main inference loop."""
        while self._running or self._pending_chunks:
            if self._pending_chunks:
                chunk = self._pending_chunks.pop(0)
                try:
                    t0 = time.perf_counter()
                    self._engine.feed(chunk, self._state)
                    elapsed = (time.perf_counter() - t0) * 1000
                    self.chunk_processed.emit(elapsed)
                    self.text_updated.emit(self._state.language, self._state.text)
                except Exception as e:
                    self.error_occurred.emit(f"{type(e).__name__}: {e}\n{traceback.format_exc()}")
            else:
                self.msleep(10)

        # Flush remaining buffer
        if self._state is not None:
            try:
                self._engine.finish(self._state)
                self.text_updated.emit(self._state.language, self._state.text)
            except Exception as e:
                self.error_occurred.emit(f"{type(e).__name__}: {e}")


class MainWindow(QMainWindow):
    """Main application window."""

    def __init__(self, engine: ASREngine):
        super().__init__()
        self._engine = engine
        self._recording = False

        self.setWindowTitle("Qwen3-ASR - Streaming Speech Recognition")
        self.setMinimumSize(600, 400)

        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # Language label
        self._lang_label = QLabel("Language: --")
        self._lang_label.setStyleSheet("font-size: 14px; color: #666;")
        layout.addWidget(self._lang_label)

        # Transcription display
        self._text_edit = QTextEdit()
        self._text_edit.setReadOnly(True)
        self._text_edit.setStyleSheet("font-size: 18px; padding: 10px;")
        self._text_edit.setPlaceholderText("Press Start to begin recording...")
        layout.addWidget(self._text_edit)

        # Volume indicator
        self._volume_bar = QProgressBar()
        self._volume_bar.setRange(0, 100)
        self._volume_bar.setValue(0)
        self._volume_bar.setTextVisible(False)
        self._volume_bar.setMaximumHeight(8)
        layout.addWidget(self._volume_bar)

        # Control buttons
        btn_layout = QHBoxLayout()

        self._start_btn = QPushButton("Start")
        self._start_btn.setStyleSheet("font-size: 16px; padding: 8px 24px;")
        self._start_btn.clicked.connect(self._on_start)
        btn_layout.addWidget(self._start_btn)

        self._stop_btn = QPushButton("Stop")
        self._stop_btn.setStyleSheet("font-size: 16px; padding: 8px 24px;")
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self._on_stop)
        btn_layout.addWidget(self._stop_btn)

        self._clear_btn = QPushButton("Clear")
        self._clear_btn.setStyleSheet("font-size: 16px; padding: 8px 24px;")
        self._clear_btn.clicked.connect(self._on_clear)
        btn_layout.addWidget(self._clear_btn)

        layout.addLayout(btn_layout)

        # Status bar
        self._status = QStatusBar()
        self.setStatusBar(self._status)
        self._status.showMessage("Ready. Models loaded.")

        # Audio capture
        self._capture = AudioCapture(self)

        # Inference thread
        self._infer_thread = InferenceThread(self._engine, self)
        self._infer_thread.text_updated.connect(self._on_text_updated)
        self._infer_thread.error_occurred.connect(self._on_error)
        self._infer_thread.chunk_processed.connect(self._on_chunk_processed)

        # Connect audio to inference
        self._capture.chunk_ready.connect(self._on_audio_chunk)

    @Slot()
    def _on_start(self):
        """Start recording and recognition."""
        self._recording = True
        self._start_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._text_edit.clear()
        self._lang_label.setText("Language: --")
        self._status.showMessage("Recording...")

        self._infer_thread.start_session()
        self._capture.start()

    @Slot()
    def _on_stop(self):
        """Stop recording."""
        self._recording = False
        self._capture.stop()
        self._infer_thread.stop_session()
        self._start_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._volume_bar.setValue(0)
        self._status.showMessage("Stopped. Flushing remaining audio...")

    @Slot()
    def _on_clear(self):
        """Clear transcription display."""
        self._text_edit.clear()
        self._lang_label.setText("Language: --")

    @Slot(np.ndarray)
    def _on_audio_chunk(self, pcm: np.ndarray):
        """Handle incoming audio from microphone."""
        # Update volume indicator
        rms = float(np.sqrt(np.mean(pcm ** 2)))
        db = max(0, min(100, int(rms * 500)))  # rough scaling
        self._volume_bar.setValue(db)

        # Forward to inference thread
        self._infer_thread.feed(pcm)

    @Slot(str, str)
    def _on_text_updated(self, language: str, text: str):
        """Update display with new transcription."""
        if language:
            self._lang_label.setText(f"Language: {language}")
        if text:
            self._text_edit.setPlainText(text)
            # Scroll to bottom
            cursor = self._text_edit.textCursor()
            cursor.movePosition(cursor.MoveOperation.End)
            self._text_edit.setTextCursor(cursor)

    @Slot(str)
    def _on_error(self, msg: str):
        """Display inference error."""
        self._status.showMessage(f"Error: {msg[:100]}")

    @Slot(float)
    def _on_chunk_processed(self, ms: float):
        """Update status with processing time."""
        self._status.showMessage(f"Recording... (last chunk: {ms:.0f}ms)")

    def closeEvent(self, event):
        """Clean up on window close."""
        self._capture.stop()
        self._infer_thread.stop_session()
        self._infer_thread.wait(3000)
        super().closeEvent(event)
