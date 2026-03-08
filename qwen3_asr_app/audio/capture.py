"""Microphone audio capture using QAudioSource (PySide6).

Captures 16kHz 16-bit mono PCM from the default microphone and emits
chunks via a Qt signal for the inference thread to consume.
"""

import numpy as np
from PySide6.QtCore import QObject, Signal, QByteArray, QIODevice
from PySide6.QtMultimedia import QAudioFormat, QAudioSource, QMediaDevices


class AudioCapture(QObject):
    """Captures PCM audio from the microphone.

    Signals:
        chunk_ready(np.ndarray): Emitted with float32 PCM samples.
    """

    chunk_ready = Signal(np.ndarray)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._source: QAudioSource | None = None
        self._io: QIODevice | None = None

    def start(self):
        """Start audio capture from default input device."""
        fmt = QAudioFormat()
        fmt.setSampleRate(16000)
        fmt.setChannelCount(1)
        fmt.setSampleFormat(QAudioFormat.SampleFormat.Int16)

        device = QMediaDevices.defaultAudioInput()
        if device.isNull():
            raise RuntimeError("No audio input device found")

        self._source = QAudioSource(device, fmt, self)
        self._source.setBufferSize(16000 * 2)  # 1 second buffer

        self._io = self._source.start()
        if self._io is None:
            raise RuntimeError("Failed to start audio source")

        self._io.readyRead.connect(self._on_ready_read)

    def stop(self):
        """Stop audio capture."""
        if self._source is not None:
            self._source.stop()
            self._source = None
            self._io = None

    def _on_ready_read(self):
        """Read available audio data and emit as float32 PCM."""
        if self._io is None:
            return

        data = self._io.readAll()
        if data.isEmpty():
            return

        raw = bytes(data)
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        self.chunk_ready.emit(samples)
