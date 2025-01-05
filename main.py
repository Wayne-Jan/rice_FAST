import os
import asyncio
import gc
from datetime import datetime, timedelta
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path
import shutil
import uvicorn
from inference.pipeline import InferencePipeline
from PIL import Image, ExifTags
import io
import piexif
from typing import Optional, Dict, Tuple
import onnxruntime as ort
from download_models import verify_models, setup_models

# 設定 ONNX Runtime 全域配置
ort.set_default_logger_severity(3)  # 只顯示錯誤
sess_options = ort.SessionOptions()
sess_options.intra_op_num_threads = 1
sess_options.inter_op_num_threads = 1
sess_options.enable_mem_pattern = False
sess_options.enable_cpu_mem_arena = False

# 定義所有需要的目錄
MODEL_DIR = Path("/tmp/models")
OUTPUT_DIR = Path("/tmp/outputs")
UPLOAD_DIR = Path("/tmp/uploads")
ORIGINAL_OUTPUT_DIR = OUTPUT_DIR / "original"
U2NET_OUTPUT_DIR = OUTPUT_DIR / "u2net"

# 設定檔案大小限制（5MB）
MAX_FILE_SIZE = 5 * 1024 * 1024

# 獲取專案根目錄的絕對路徑
BASE_DIR = Path(__file__).resolve().parent

def ensure_temp_directories():
    """確保臨時目錄存在並可寫入"""
    directories = [
        MODEL_DIR, 
        OUTPUT_DIR, 
        UPLOAD_DIR, 
        ORIGINAL_OUTPUT_DIR, 
        U2NET_OUTPUT_DIR,
        BASE_DIR / "templates"
    ]
    for directory in directories:
        try:
            directory.mkdir(parents=True, exist_ok=True)
            # 測試目錄是否可寫入
            test_file = directory / ".test"
            test_file.touch()
            test_file.unlink()
        except Exception as e:
            raise RuntimeError(f"無法創建或寫入目錄 {directory}: {str(e)}")

# 確保所有目錄存在
ensure_temp_directories()

# FastAPI 應用程式
app = FastAPI(title="稻米分析系統")

# 設定靜態文件和模板
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}

def initialize_models():
    """確保模型檔案存在並初始化"""
    missing = verify_models()
    if missing:
        print(f"需要下載以下模型: {', '.join(missing)}")
        # 執行異步下載
        asyncio.run(setup_models())
    else:
        print("所有模型已就緒")

def get_pipeline():
    """取得 pipeline 實例，使用後立即釋放"""
    return InferencePipeline(
        model_dir=str(MODEL_DIR),
        output_dir=str(OUTPUT_DIR),
        device="cpu"
    )

async def cleanup_temp_files():
    """定期清理臨時檔案"""
    while True:
        try:
            # 清理超過1小時的檔案
            cutoff = datetime.now() - timedelta(hours=1)
            for directory in [UPLOAD_DIR, ORIGINAL_OUTPUT_DIR, U2NET_OUTPUT_DIR]:
                if directory.exists():
                    for file_path in directory.glob("*"):
                        try:
                            if file_path.stat().st_mtime < cutoff.timestamp():
                                file_path.unlink(missing_ok=True)
                        except Exception as e:
                            print(f"清理檔案 {file_path} 時發生錯誤: {str(e)}")
            await asyncio.sleep(600)  # 每10分鐘執行一次
        except Exception as e:
            print(f"清理臨時檔案時發生錯誤: {str(e)}")
            await asyncio.sleep(60)

@app.on_event("startup")
async def startup_event():
    """應用程式啟動時執行"""
    try:
        ensure_temp_directories()
        initialize_models()
        # 啟動清理任務
        asyncio.create_task(cleanup_temp_files())
    except Exception as e:
        print(f"應用程式啟動時發生錯誤: {str(e)}")
        raise e

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """首頁路由"""
    try:
        return templates.TemplateResponse(
            "index.html",
            {"request": request}
        )
    except Exception as e:
        print(f"讀取模板時發生錯誤: {str(e)}")
        return HTMLResponse(content="<h1>系統錯誤</h1><p>無法載入首頁模板</p>")

def fix_image_orientation(img: Image.Image, orientation: Optional[Dict]) -> Image.Image:
    """根據 EXIF 方向資訊修正圖片方向"""
    if not orientation:
        return img

    ORIENTATION_DICT = {
        1: None,
        2: (Image.FLIP_LEFT_RIGHT,),
        3: (Image.ROTATE_180,),
        4: (Image.FLIP_TOP_BOTTOM,),
        5: (Image.FLIP_LEFT_RIGHT, Image.ROTATE_90),
        6: (Image.ROTATE_270,),
        7: (Image.FLIP_LEFT_RIGHT, Image.ROTATE_270),
        8: (Image.ROTATE_90,)
    }

    orientation_value = orientation.get('Orientation')
    if orientation_value in ORIENTATION_DICT:
        operations = ORIENTATION_DICT[orientation_value]
        if operations:
            for operation in operations:
                img = img.transpose(operation)
    
    return img

def get_exif_data(img: Image.Image) -> Tuple[Optional[Dict], Optional[Dict]]:
    """獲取圖片的 EXIF 資料"""
    try:
        exif = img._getexif()
        if not exif:
            return None, None

        gps_info = {}
        orientation = None

        for tag_id in exif:
            tag = ExifTags.TAGS.get(tag_id, tag_id)
            if tag == 'Orientation':
                orientation = exif[tag_id]

        return None, {'Orientation': orientation} if orientation else None

    except Exception as e:
        print(f"讀取 EXIF 資料時發生錯誤: {str(e)}")
        return None, None

def compress_image(img: Image.Image, max_size=500) -> bytes:
    """壓縮圖片
    
    Args:
        img: PIL Image 物件
        max_size: 最大檔案大小（KB）
    
    Returns:
        bytes: 壓縮後的圖片數據
    """
    # 初始品質
    quality = 95
    while True:
        # 保存到內存
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=quality)
        size_kb = buffer.getbuffer().nbytes / 1024
        
        if size_kb <= max_size or quality <= 10:
            buffer.seek(0)
            return buffer.getvalue()
            
        quality -= 5

@app.post("/analyze", response_class=JSONResponse)
async def analyze_image(file: UploadFile = File(...)):
    """處理上傳的圖片"""
    try:
        # 檢查檔案類型
        file_extension = file.filename.split(".")[-1].lower()
        if file_extension not in ALLOWED_EXTENSIONS:
            raise HTTPException(status_code=400, detail="不支持的檔案類型")

        # 分批讀取檔案
        file_size = 0
        with io.BytesIO() as file_bytes:
            chunk_size = 8192  # 8KB chunks
            while chunk := await file.read(chunk_size):
                file_size += len(chunk)
                if file_size > MAX_FILE_SIZE:
                    raise HTTPException(status_code=400, detail="檔案太大")
                file_bytes.write(chunk)
            
            file_bytes.seek(0)
            
            # 建立暫存檔案路徑
            temp_file_path = UPLOAD_DIR / file.filename
            
            try:
                # 處理圖片方向和壓縮
                img = Image.open(file_bytes)
                _, orientation_info = get_exif_data(img)
                img = fix_image_orientation(img, orientation_info)
                
                # 儲存處理後的圖片
                compressed_data = compress_image(img)
                with open(temp_file_path, "wb") as f:
                    f.write(compressed_data)
                
                # 儲存壓縮後的原始圖片
                compressed_original_path = ORIGINAL_OUTPUT_DIR / file.filename
                with open(compressed_original_path, "wb") as f:
                    f.write(compressed_data)

                try:
                    # 取得 pipeline 並處理圖片
                    pipeline = get_pipeline()
                    result = pipeline.process_image(str(temp_file_path))
                    
                    # 立即清理
                    del pipeline
                    gc.collect()
                    
                    if result["status"] == "error":
                        raise HTTPException(status_code=500, detail=result["error"])
                    
                    # 構建回應
                    response_data = {
                        "message": "分析完成",
                        "filename": file.filename,
                        "prediction": round(float(result["prediction"]), 2),
                        "processing_time": round(float(result["processing_time"]), 2),
                        "masked_image_url": f"/outputs/u2net/{Path(result['masked_image_path']).name}",
                        "mask_image_url": f"/outputs/u2net/{Path(result['mask_path']).name}",
                        "original_image_url": f"/outputs/original/{file.filename}"
                    }
                    
                    return JSONResponse(response_data)
                    
                finally:
                    # 清理暫存檔案
                    temp_file_path.unlink(missing_ok=True)
                    gc.collect()

            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"圖片處理失敗: {str(e)}")
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"處理過程中發生錯誤: {str(e)}")

@app.get("/health")
async def health_check():
    """健康檢查端點"""
    return {"status": "healthy"}

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True
    )