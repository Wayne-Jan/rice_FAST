import os
import asyncio
import aiohttp
from pathlib import Path
import time
from tqdm import tqdm

async def download_from_drive(session, file_id: str, destination: Path, filename: str):
    """從 Google Drive 異步下載檔案"""
    try:
        url = f"https://drive.google.com/uc?id={file_id}"
        
        # 建立進度條
        progress = None
        
        async with session.get(url) as response:
            if response.status == 200:
                total_size = int(response.headers.get('content-length', 0))
                
                # 初始化進度條
                progress = tqdm(
                    total=total_size,
                    unit='iB',
                    unit_scale=True,
                    desc=f'下載 {filename}'
                )
                
                # 開啟檔案並寫入
                with open(destination, 'wb') as f:
                    async for chunk in response.content.iter_chunked(8192):
                        f.write(chunk)
                        progress.update(len(chunk))
                
                print(f"{filename} 下載完成")
                return True
            else:
                print(f"{filename} 下載失敗: HTTP {response.status}")
                return False
                
    except Exception as e:
        print(f"{filename} 下載時發生錯誤: {str(e)}")
        return False
    finally:
        if progress:
            progress.close()

async def setup_models():
    """下載所需的模型檔案"""
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # 模型資訊
    models = {
        "YOLO.onnx": "1MHkvBLx6fyc2poldZDxFkG7Ensttvdcm",
        "u2net.onnx": "1hqeMJw_fao7ivAMPPAG35OFy7ydhKVuT",
        "model.onnx": "1hgf4t3keKON27E79CscjGYT6LMhnUx-l",
        "scaler.pkl": "1iXVfXxwiQtbNoCARapcO4uFgo1aW2KwR"
    }
    
    async with aiohttp.ClientSession() as session:
        tasks = []
        for filename, file_id in models.items():
            dest_path = models_dir / filename
            if not dest_path.exists():
                task = download_from_drive(session, file_id, dest_path, filename)
                tasks.append(task)
            else:
                print(f"{filename} 已存在，跳過下載")
        
        if tasks:
            results = await asyncio.gather(*tasks)
            if all(results):
                print("所有模型下載完成")
            else:
                print("部分模型下載失敗，請檢查錯誤訊息")
        else:
            print("所有模型都已存在")

def verify_models():
    """驗證所需模型是否都存在"""
    models_dir = Path("models")
    required_models = ["YOLO.onnx", "u2net.onnx", "model.onnx", "scaler.pkl"]
    
    missing_models = []
    for model in required_models:
        if not (models_dir / model).exists():
            missing_models.append(model)
    
    return missing_models

if __name__ == "__main__":
    # 檢查模型
    missing = verify_models()
    if missing:
        print(f"需要下載以下模型: {', '.join(missing)}")
        # 執行異步下載
        asyncio.run(setup_models())
    else:
        print("所有模型已就緒")