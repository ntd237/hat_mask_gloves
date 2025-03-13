import sys
import cv2
import time
import torch
import queue
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem
from ultralytics import YOLO

# Đường dẫn model và video
MODEL_YOLO_PERSON = "models/yolov8m.pt"
MODEL_YOLO_SEGMENT = "models/best_model.pt"
VIDEO_PATH = "data/video/XLBM.CAM.06.avi"


CLASS_NAMES = ["gloves", "hat", "mask"]
COLORS = {"gloves": (0, 255, 255), "hat": (0, 255, 0), "mask": (255, 0, 0)}

# ======================== THREAD CAPTURE ========================
class ThreadCapture(QThread):
    new_frame = pyqtSignal(object)  # Gửi frame đi

    def __init__(self, video_path):
        super().__init__()
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.running = True

        if not self.cap.isOpened():
            print(f"❌ LỖI: Không thể mở video {self.video_path}")

    def run(self):
        while self.running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                print("📢 Đã đọc xong video!")
                break

            print("📸 Đọc frame thành công!")  # Debug kiểm tra
            self.new_frame.emit(frame)  # Gửi frame đi

    def stop(self):
        self.running = False
        self.cap.release()
        self.quit()
        self.wait()

# ======================== THREAD PROCESSING ========================
class ThreadProcessing(QThread):
    processed_frame = pyqtSignal(object, float)  # Gửi frame đã xử lý + FPS

    def __init__(self):
        super().__init__()
        self.running = True
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model_person = YOLO(MODEL_YOLO_PERSON).to(self.device)
        self.model_segment = YOLO(MODEL_YOLO_SEGMENT).to(self.device)

        self.frame_queue = queue.Queue(maxsize=5)

    def process_frame(self, frame):
        if not self.running:
            return
        
        if frame is None or frame.shape[0] == 0 or frame.shape[1] == 0:
            print("❌ LỖI: Frame nhận vào bị rỗng!")
            return

        print("✅ Frame đã đến process_thread!")  # Debug kiểm tra
        if not self.frame_queue.full():
            self.frame_queue.put(frame)

    def run(self):
        while self.running:
            if not self.frame_queue.empty():
                frame = self.frame_queue.get()
                start_time = time.time()

                # ==========================
                # 🔹 BƯỚC 1: PHÁT HIỆN NGƯỜI 🔹
                # ==========================
                with torch.inference_mode():
                    results = self.model_person(frame, conf=0.5, device=self.device)
                
                person_boxes = []
                for box in results[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    person_boxes.append((x1, y1, x2, y2))

                if len(person_boxes) == 0:
                    print("❌ LỖI: Không phát hiện được người trong frame!")
                    fps = 1.0 / (time.time() - start_time)
                    self.processed_frame.emit(frame, fps)
                    continue

                # ==========================
                # 🔹 BƯỚC 2: SEGMENTATION TRONG VÙNG NGƯỜI 🔹
                # ==========================
                for (x1, y1, x2, y2) in person_boxes:
                    person_crop = frame[y1:y2, x1:x2].copy()

                    if person_crop.shape[0] == 0 or person_crop.shape[1] == 0:
                        print(f"❌ LỖI: Bounding Box rỗng! ({x1}, {y1}) → ({x2}, {y2})")
                        continue

                    with torch.inference_mode():
                        seg_results = self.model_segment(person_crop, conf=0.4)

                    seg_result = seg_results[0]

                    if seg_result.masks is None:
                        print(f"❌ Không phát hiện segmentation trong vùng người tại ({x1}, {y1})")
                        continue

                    for j in range(len(seg_result.boxes)):
                        class_id = int(seg_result.boxes.cls[j])
                        confidence = float(seg_result.boxes.conf[j])
                        mask = seg_result.masks.data[j].cpu().numpy()

                        class_name = CLASS_NAMES[class_id]
                        color = COLORS[class_name]

                        mask_resized = cv2.resize(mask, (x2 - x1, y2 - y1))
                        mask_binary = (mask_resized > 0.5).astype(np.uint8) * 255
                        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        cv2.drawContours(frame[y1:y2, x1:x2], contours, -1, color, thickness=cv2.FILLED)

                        x, y, w, h = cv2.boundingRect(mask_binary)
                        cv2.putText(frame, f"{class_name} {confidence:.2f}", (x1 + x, y1 + y - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                fps = 1.0 / (time.time() - start_time)
                self.processed_frame.emit(frame, fps)

    def stop(self):
        self.running = False
        self.quit()
        self.wait()

# ======================== THREAD STREAM ========================
class ThreadStream(QThread):
    updateFrame = pyqtSignal(QImage, float)

    def __init__(self, video_view, fps_label):
        super().__init__()
        self.video_view = video_view  
        self.fps_label = fps_label
        self.running = True

        self.scene = QGraphicsScene()
        self.video_view.setScene(self.scene)
        self.pixmap_item = QGraphicsPixmapItem()
        self.scene.addItem(self.pixmap_item)

    def display_frame(self, frame, fps):
        if not self.running:
            return

        print("✅ Frame đã đến stream_thread!")  # Debug kiểm tra
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        qimg = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)

        self.pixmap_item.setPixmap(QPixmap.fromImage(qimg))
        self.video_view.setScene(self.scene)
        self.fps_label.setText(f"FPS: {fps:.2f}")

    def stop(self):
        self.running = False

# ======================== CHẠY ỨNG DỤNG ========================
if __name__ == "__main__":
    app = QApplication(sys.argv)

    video_view = QGraphicsView()
    fps_label = QLabel("FPS: 0")

    capture_thread = ThreadCapture(VIDEO_PATH)
    process_thread = ThreadProcessing()
    stream_thread = ThreadStream(video_view, fps_label)

    capture_thread.new_frame.connect(process_thread.process_frame)
    process_thread.processed_frame.connect(stream_thread.display_frame)

    capture_thread.start()
    process_thread.start()

    video_view.show()
    sys.exit(app.exec_())
