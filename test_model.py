import torch
import cv2
import os
import numpy as np
from ultralytics import YOLO

# Định nghĩa đường dẫn
MODEL_PATH = "models/best_model.pt"
IMAGE_DIR = "data/test/images"
OUTPUT_DIR = "output/"

# Đảm bảo thư mục output tồn tại
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load mô hình YOLOv8 Segment
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = YOLO(MODEL_PATH)  # Load trực tiếp mô hình YOLO

# Danh sách các class (hat, mask, gloves)
CLASS_NAMES = ["gloves", "hat", "mask"]
COLORS = {
    "hat": (0, 255, 0),      # Màu xanh lá
    "mask": (255, 0, 0),     # Màu đỏ
    "gloves": (0, 255, 255)  # Màu vàng
}

# Duyệt qua tất cả ảnh trong thư mục test
for image_name in os.listdir(IMAGE_DIR):
    image_path = os.path.join(IMAGE_DIR, image_name)
    img = cv2.imread(image_path)
    
    if img is None:
        print(f"⚠️ Không thể đọc ảnh: {image_name}")
        continue

    # Chạy mô hình dự đoán
    results = model(image_path, conf=0.4)  # Chỉ nhận dự đoán có độ tin cậy > 40%

    # Lấy kết quả dự đoán đầu tiên
    result = results[0]
    masks = result.masks  # Lấy segmentation masks
    names = result.names  # Tên các class

    if masks is None:
        print(f"❌ Không tìm thấy object nào trong ảnh {image_name}")
        continue

    # Vẽ segmentation lên ảnh
    for i in range(len(result.boxes)):
        class_id = int(result.boxes.cls[i])  # Lấy class ID
        confidence = float(result.boxes.conf[i])  # Lấy độ tin cậy
        mask = masks.data[i].cpu().numpy()  # Lấy mask của object

        class_name = CLASS_NAMES[class_id]
        color = COLORS[class_name]

        # Vẽ vùng segmentation
        mask = (mask > 0.5).astype(np.uint8) * 255  # Chuyển mask về nhị phân
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, contours, -1, color, thickness=cv2.FILLED)  # Tô màu mask

        # Vẽ bounding box
        x1, y1, x2, y2 = map(int, result.boxes.xyxy[i])
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # Viết tên class và độ tin cậy
        label = f"{class_name} {confidence:.2f}"
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Lưu ảnh kết quả
    output_path = os.path.join(OUTPUT_DIR, image_name)
    cv2.imwrite(output_path, img)
    print(f"✅ Xử lý xong: {image_name}, lưu tại {output_path}")

print("🎯 Hoàn thành nhận diện tất cả ảnh trong thư mục test!")
