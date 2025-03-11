import os

# Đánh giá mô hình
os.system("python yolov11/val.py --data data.yaml --weights saved_models/yolov11_best.pt --img 640 --device 0")
