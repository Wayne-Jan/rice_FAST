FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# 修改啟動命令，先下載模型再啟動服務
CMD python download_models.py && uvicorn main:app --host 0.0.0.0 --port 8000