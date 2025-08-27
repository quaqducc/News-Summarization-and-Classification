# 1. Chọn base image
# Nếu chỉ CPU:
# FROM python:3.10-slim
# Nếu muốn dùng GPU với PyTorch:
FROM pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime

# 2. Đặt thư mục làm việc trong container
WORKDIR /app

# 3. Copy file requirements trước để tận dụng cache layer
COPY requirements.txt .

# 4. Cài các thư viện Python
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy toàn bộ code vào container
COPY . .

# 6. Đặt entrypoint mặc định (có thể override bằng docker run)
ENTRYPOINT ["python", "infer.py"]
