from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QGraphicsScene, QGraphicsPixmapItem
import cv2

class ThreadStream(QThread):
    """Luồng hiển thị frame nhận từ ThreadProcessing."""
    updateFrame = pyqtSignal(QImage, float)  # Nhận frame + FPS

    def __init__(self, video_view, fps_label):
        super().__init__()
        self.video_view = video_view  
        self.fps_label = fps_label
        self.running = True

        # Khởi tạo giao diện hiển thị
        self.scene = QGraphicsScene()
        self.video_view.setScene(self.scene)
        self.pixmap_item = QGraphicsPixmapItem()
        self.scene.addItem(self.pixmap_item)

    def display_frame(self, frame, fps):
        """Nhận frame từ ThreadProcessing và hiển thị lên giao diện."""
        if not self.running:
            return

        # Chuyển đổi frame từ OpenCV (BGR) sang PyQt (RGB)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Hiển thị FPS lên frame
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        h, w, ch = frame.shape
        bytes_per_line = ch * w
        qimg = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)

        # Hiển thị hình ảnh lên giao diện
        self.pixmap_item.setPixmap(QPixmap.fromImage(qimg))
        self.video_view.setScene(self.scene)

        # Cập nhật nhãn FPS
        self.fps_label.setText(f"FPS: {fps:.2f}")

    def stop(self):
        """Dừng luồng hiển thị."""
        self.running = False
        self.quit()
        self.wait()
