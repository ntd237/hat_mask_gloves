### Cài đặt thư viện

!pip install ultralytics opencv-python matplotlib

from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2
import os
import torch

# Kiểm tra GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)


### Cấu hình đường dẫn dataset
DATASET_PATH = "/kaggle/input/hat-mask-glovess/hat_mask_glovess"
DATA_YAML = os.path.join(DATASET_PATH, "data.yaml")

# Kiểm tra file dataset
os.listdir(DATASET_PATH)



### Khởi tạo và train model
MODEL_PATH = "/kaggle/input/yolo11n-seg/pytorch/default/1/yolo11n-seg.pt"  
model = YOLO(MODEL_PATH, task="segment").to(device)

# Train model
model.train(
    data=DATA_YAML,
    epochs=50,  # Train từ 150-200 epochs
    batch=16,
    imgsz=640,
    device=device
)


### Đánh giá mô hình trên tập validation
# Đánh giá mô hình trên tập validation
metrics = model.val()

# 📦 Lấy thông tin về Bounding Box (BB)
mAP50_95_bb = metrics.box.map  # mAP 50-95 cho BB
precision_bb = metrics.box.mp  # Precision BB
recall_bb = metrics.box.mr  # Recall BB

# 🎭 Lấy thông tin về Mask Segmentation (thay metrics.mask → metrics.seg)
mAP50_95_mask = metrics.seg.map  # mAP 50-95 cho Mask Segmentation
precision_mask = metrics.seg.mp  # Precision Mask
recall_mask = metrics.seg.mr  # Recall Mask

# ✅ In kết quả
print(f"📦 Bounding Box Metrics:")
print(f" - mAP 50-95 (BB): {mAP50_95_bb:.4f}")
print(f" - Precision (BB): {precision_bb:.4f}")
print(f" - Recall (BB): {recall_bb:.4f}\n")

print(f"🎭 Mask Segmentation Metrics:")
print(f" - mAP 50-95 (Mask): {mAP50_95_mask:.4f}")
print(f" - Precision (Mask): {precision_mask:.4f}")
print(f" - Recall (Mask): {recall_mask:.4f}")



### Vẽ biểu đồ đánh giá
import matplotlib.pyplot as plt
import numpy as np

# Nhãn trục x
labels = ["mAP 50-95", "Precision", "Recall"]

# Giá trị cho Bounding Box
bb_values = [mAP50_95_bb, precision_bb, recall_bb]

# Giá trị cho Mask Segmentation
mask_values = [mAP50_95_mask, precision_mask, recall_mask]

# Tạo vị trí cho cột
x = np.arange(len(labels))  # Vị trí cột (0, 1, 2)

# Độ rộng của mỗi cột
width = 0.4  

# Vẽ biểu đồ
plt.figure(figsize=(10, 5))
plt.bar(x - width/2, bb_values, width=width, label="Bounding Box", color='blue')
plt.bar(x + width/2, mask_values, width=width, label="Mask Segmentation", color='purple')

# Thêm nhãn
plt.xticks(x, labels)
plt.title("Comparison of Bounding Box and Mask Segmentation Metrics")
plt.ylabel("Score")
plt.ylim(0, 1)  # Giới hạn trục y từ 0 đến 1
plt.legend()  # Hiển thị chú thích

# Hiển thị biểu đồ
plt.show()



### Lưu model tốt nhất
import os
import shutil
from ultralytics import YOLO

# Định nghĩa đường dẫn
model_path = "runs/segment/train/weights/best.pt"
exported_model_path = "runs/segment/train/weights/best.onnx"
save_dir = "/kaggle/working/saved_models"

# Tạo thư mục lưu mô hình nếu chưa có
os.makedirs(save_dir, exist_ok=True)

# Kiểm tra xem mô hình đã được export sang ONNX chưa
if not os.path.exists(exported_model_path):
    if os.path.exists(model_path):
        print("🔄 Đang xuất mô hình sang ONNX...")
        model = YOLO(model_path)
        model.export(format="onnx")
    else:
        print(f"❌ LỖI: Không tìm thấy mô hình {model_path}. Kiểm tra lại quá trình train!")
        exit()

# Kiểm tra lại xem tệp ONNX có tồn tại không sau khi export
if os.path.exists(exported_model_path):
    shutil.move(exported_model_path, os.path.join(save_dir, "best_model.onnx"))
    print(f"✅ Mô hình ONNX đã được lưu tại: {os.path.join(save_dir, 'best_model.onnx')}")
else:
    print("❌ LỖI: Xuất ONNX thất bại, kiểm tra lại quá trình export!")


### Test trên tập test
TEST_PATH = os.path.join(DATASET_PATH, "test/images")
results = model.predict(source=TEST_PATH, save=True, save_txt=True)
print("Testing complete. Check the output folder for results.")