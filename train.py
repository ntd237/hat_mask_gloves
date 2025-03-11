import torch
import os

# Kiểm tra thiết bị (GPU/CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Chạy lệnh train YOLOv11
os.system(f"python yolov11/train.py --data data.yaml --weights yolov11.pt --epochs 2 --batch-size 4 --img 640 --device {device}")
