import cv2
import time
import torch
import queue
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal
from ultralytics import YOLO

# Load mô hình
MODEL_YOLO_PERSON = "models/yolov8m.pt"
MODEL_YOLO_SEGMENT = "models/best_model.pt"
CLASS_NAMES = ["gloves", "hat", "mask"]
COLORS = {"gloves": (0, 255, 255), "hat": (0, 255, 0), "mask": (255, 0, 0)}

class ThreadProcessing(QThread):
    processed_frame = pyqtSignal(object, float)  # Gửi frame đã xử lý + FPS

    def __init__(self):
        super().__init__()
        self.running = True
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load model YOLO
        self.model_person = YOLO(MODEL_YOLO_PERSON).to(self.device)  # Phát hiện người
        self.model_segment = YOLO(MODEL_YOLO_SEGMENT).to(self.device)  # Nhận diện gloves, hat, mask

        self.frame_queue = queue.Queue(maxsize=10)  # Hàng đợi tránh tràn bộ nhớ

    def process_frame(self, frame):
        """Nhận frame từ `capture_thread` và đẩy vào queue để xử lý."""
        if not self.running:
            return
        
        if not self.frame_queue.full():
            self.frame_queue.put(frame)

    def run(self):
        """Xử lý frame: phát hiện người, nhận diện gloves/hat/mask và vẽ lên ảnh gốc."""
        while self.running:
            if not self.frame_queue.empty():
                frame = self.frame_queue.get()
                start_time = time.time()

                # ===============
                # PHÁT HIỆN NGƯỜI
                # ===============
                with torch.inference_mode():
                    results = self.model_person(frame, device=self.device)
                
                # Lấy bounding box người
                person_boxes = []
                for box in results[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    person_boxes.append((x1, y1, x2, y2))

                # Nếu không phát hiện được người, gửi frame gốc đi luôn
                if len(person_boxes) == 0:
                    print("❌ LỖI: Không phát hiện được người trong frame!")
                    fps = 1.0 / (time.time() - start_time)
                    self.processed_frame.emit(frame, fps)
                    continue

                # ==================================
                # CHẠY SEGMENTATION TRONG VÙNG NGƯỜI
                # ==================================
                for (x1, y1, x2, y2) in person_boxes:
                    person_crop = frame[y1:y2, x1:x2].copy()  # Cắt vùng người từ ảnh gốc

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

                        # Resize mask về kích thước vùng người
                        mask_resized = cv2.resize(mask, (x2 - x1, y2 - y1))

                        # Tạo lớp mask màu
                        mask_binary = (mask_resized > 0.5).astype(np.uint8) * 255
                        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                        # Vẽ mask lên ảnh gốc chỉ trong vùng bounding box người
                        cv2.drawContours(frame[y1:y2, x1:x2], contours, -1, color, thickness=cv2.FILLED)

                        # Vẽ nhãn class
                        x, y, w, h = cv2.boundingRect(mask_binary)
                        cv2.putText(frame, f"{class_name} {confidence:.2f}", (x1 + x, y1 + y - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                        print(f"✅ {class_name} phát hiện với độ tin cậy: {confidence:.2f} tại ({x1}, {y1})")

                # ================
                # HIỂN THỊ KẾT QUẢ
                # ================
                # for (x1, y1, x2, y2) in person_boxes:
                #     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Vẽ bounding box người
                #     cv2.putText(frame, "Person", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                fps = 1.0 / (time.time() - start_time)
                self.processed_frame.emit(frame, fps)

    def stop(self):
        self.running = False
        self.quit()
        self.wait()
