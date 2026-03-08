"""Entry point for Qwen3-ASR desktop application.

Usage:
    powershell.exe -Command '$env:Path = "C:\\Users\\taowen\\.local\\bin;$env:Path"; $env:PYTHONIOENCODING = "utf-8"; cd C:\\Apps\\Qwen3-ASR; uv run python -m qwen3_asr_app.main'
"""

import sys


def main():
    from PySide6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    app.setApplicationName("Qwen3-ASR")

    # Show splash/loading status
    print("Loading models... (this may take a moment)")

    from .inference.engine import ASREngine
    from .ui.main_window import MainWindow

    engine = ASREngine(encoder_device="NPU", language="Chinese")
    print("Models loaded. Starting UI...")

    window = MainWindow(engine)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
