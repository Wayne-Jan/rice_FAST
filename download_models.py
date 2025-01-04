import os
import requests
from pathlib import Path

def download_from_drive(file_id: str, destination: str):
    """從 Google Drive 下載檔案"""
    url = f"https://drive.google.com/uc?id={file_id}"
    response = requests.get(url)
    with open(destination, 'wb') as f:
        f.write(response.content)

def setup_models():
    """下載所需的模型檔案"""
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # 從共用連結中提取檔案 ID
    # 例如：從 https://drive.google.com/file/d/YOUR_FILE_ID/view
    # 提取 YOUR_FILE_ID 部分
    models = {
        "YOLO.onnx": "1MHkvBLx6fyc2poldZDxFkG7Ensttvdcm",
        "u2net.onnx": "1hqeMJw_fao7ivAMPPAG35OFy7ydhKVuT",
        "model.onnx": "1hgf4t3keKON27E79CscjGYT6LMhnUx-l/",
        "scaler.pkl": "1iXVfXxwiQtbNoCARapcO4uFgo1aW2KwR"
    }
    
    for filename, file_id in models.items():
        dest_path = models_dir / filename
        if not dest_path.exists():
            print(f"下載 {filename}...")
            download_from_drive(file_id, str(dest_path))
            print(f"{filename} 下載完成")

if __name__ == "__main__":
    setup_models()