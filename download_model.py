import os
import subprocess

def download_yolov11(destination="yolov11"):
    """
    Tải mã nguồn YOLOv11 từ repository và lưu vào thư mục chỉ định.
    """
    yolov11_repo = "https://github.com/ultralytics/ultralytics.git"  # Cập nhật đường dẫn đúng
    
    if not os.path.exists(destination):
        print(f"Đang tải YOLOv11 vào thư mục {destination}...")
        subprocess.run(["git", "clone", yolov11_repo, destination], check=True)
        print("Tải xuống thành công!")
    else:
        print("Thư mục YOLOv11 đã tồn tại. Không cần tải lại.")

if __name__ == "__main__":
    download_yolov11()