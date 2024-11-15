import sys

from PyQt5.QtWidgets import QApplication

from src.application.not_used.app import FallDetectionApp


def main():
    app = QApplication(sys.argv)
    window = FallDetectionApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()