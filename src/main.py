import sys
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout, QGraphicsView
from PyQt5.QtCore import Qt
from capture_thread import ThreadCapture
from process_thread import ThreadProcessing
from stream_thread import ThreadStream

class VideoApp(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

        # Khởi tạo các luồng
        self.capture_thread = ThreadCapture(video_path="data/video/XLBM.CAM.06.avi")
        self.process_thread = ThreadProcessing()
        self.stream_thread = ThreadStream(self.video_view, self.fps_label)

        # Kết nối tín hiệu
        self.capture_thread.new_frame.connect(self.process_thread.process_frame)
        self.process_thread.processed_frame.connect(self.stream_thread.display_frame)

        # Bắt đầu các luồng
        self.capture_thread.start()
        self.process_thread.start()
        self.stream_thread.start()

    def init_ui(self):
        """Khởi tạo giao diện PyQt5"""
        self.setWindowTitle("Multi-threaded Video Processing")
        self.setGeometry(100, 100, 1200, 800)

        self.video_view = QGraphicsView(self)
        self.fps_label = QLabel("FPS: 0", self)

        layout = QVBoxLayout()
        layout.addWidget(self.video_view)
        layout.addWidget(self.fps_label)
        self.setLayout(layout)

    def closeEvent(self, event):
        """Dừng các luồng khi đóng ứng dụng"""
        self.capture_thread.stop()
        self.process_thread.stop()
        self.stream_thread.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoApp()
    window.show()
    sys.exit(app.exec_())
