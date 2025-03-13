import torch

model_path = "results/saved_models/best_model.pt"  # Đường dẫn file .pt
output_file = "results/saved_models/best_weights.pth"  # Nơi lưu trọng số

# Tải toàn bộ checkpoint (chứa cả model)
checkpoint = torch.load(model_path, map_location=torch.device("cpu"), weights_only=False)

# Kiểm tra model có chứa key "model" hay không
if "model" in checkpoint:
    weights = checkpoint["model"]
else:
    weights = checkpoint  # Trường hợp file chỉ chứa trọng số

# Lưu trọng số thành file .pth
torch.save(weights, output_file)

print(f"✅ Trọng số đã được lưu tại {output_file}")
