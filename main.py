import os
import asyncio
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

from download_models import verify_models, setup_models

# 使用臨時目錄來存儲模型
MODEL_DIR = Path("/tmp/models")
OUTPUT_DIR = Path("/tmp/outputs")
ORIGINAL_OUTPUT_DIR = OUTPUT_DIR / "original"
U2NET_OUTPUT_DIR = OUTPUT_DIR / "u2net"

# 獲取專案根目錄的絕對路徑
BASE_DIR = Path(__file__).resolve().parent

# FastAPI 應用程式
app = FastAPI(title="稻米分析系統")

# 設定靜態文件和模板，使用絕對路徑
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# 建立必要的目錄
UPLOAD_DIR = Path("/tmp/uploads")

# 定義所需的目錄列表
REQUIRED_DIRS = [
    Path("/tmp/static"),
    BASE_DIR / "templates",
    UPLOAD_DIR,
    OUTPUT_DIR,
    MODEL_DIR,
    ORIGINAL_OUTPUT_DIR,
    U2NET_OUTPUT_DIR
]

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}

# 初始化推論管線為 None，稍後通過 get_pipeline() 初始化
pipeline = None

def initialize_models():
    """確保模型檔案存在並初始化"""
    missing = verify_models()
    if missing:
        print(f"需要下載以下模型: {', '.join(missing)}")
        # 執行異步下載
        asyncio.run(setup_models())
    else:
        print("所有模型已就緒")

@app.on_event("startup")
async def startup_event():
    """應用程式啟動時執行"""
    try:
        # 確保所有必要的目錄都存在
        for dir_path in REQUIRED_DIRS:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # 初始化模型
        initialize_models()
        
    except Exception as e:
        print(f"應用程式啟動時發生錯誤: {str(e)}")
        raise e

def get_pipeline():
    """取得 pipeline 實例，如果不存在則初始化"""
    global pipeline
    if pipeline is None:
        pipeline = InferencePipeline(
            model_dir=str(MODEL_DIR),
            output_dir=str(OUTPUT_DIR),
            device="cpu"
        )
    return pipeline

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

def convert_to_serializable(value):
    """將 EXIF 值轉換為可序列化的格式"""
    if isinstance(value, tuple) and len(value) == 2:  # IFDRational
        return float(value[0]) / float(value[1]) if value[1] != 0 else 0
    elif isinstance(value, bytes):
        return value.decode('utf-8', errors='ignore')
    elif isinstance(value, (int, float, str)):
        return value
    elif isinstance(value, (tuple, list)):
        return [convert_to_serializable(x) for x in value]
    elif isinstance(value, dict):
        return {k: convert_to_serializable(v) for k, v in value.items()}
    return str(value)

def get_exif_data(img: Image.Image) -> Tuple[Optional[Dict], Optional[Dict]]:
    """獲取圖片的 EXIF 資料，包含 GPS 資訊和方向資訊
    
    Args:
        img: PIL Image 物件
    
    Returns:
        Tuple[Optional[Dict], Optional[Dict]]: GPS 資訊和方向資訊
    """
    try:
        exif = img._getexif()
        if not exif:
            return None, None

        gps_info = {}
        orientation = None

        for tag_id in exif:
            tag = ExifTags.TAGS.get(tag_id, tag_id)
            if tag == 'GPSInfo':
                gps_data = exif[tag_id]
                for t in gps_data:
                    sub_tag = ExifTags.GPSTAGS.get(t, t)
                    gps_info[sub_tag] = convert_to_serializable(gps_data[t])
            elif tag == 'Orientation':
                orientation = exif[tag_id]

        return gps_info, {'Orientation': orientation} if orientation else None

    except Exception as e:
        print(f"讀取 EXIF 資料時發生錯誤: {str(e)}")
        return None, None

def format_gps_data(gps_info: Optional[Dict]) -> Optional[Dict]:
    """格式化 GPS 資訊為易讀格式
    
    Args:
        gps_info: 原始 GPS 資訊字典
        
    Returns:
        Optional[Dict]: 格式化後的 GPS 資訊
    """
    if not gps_info:
        return None
        
    try:
        formatted_gps = {}
        
        # 處理緯度
        if 'GPSLatitude' in gps_info and 'GPSLatitudeRef' in gps_info:
            lat = gps_info['GPSLatitude']
            lat_ref = gps_info['GPSLatitudeRef']
            formatted_gps['latitude'] = round(lat * (-1 if lat_ref.upper() == 'S' else 1), 6)
            
        # 處理經度
        if 'GPSLongitude' in gps_info and 'GPSLongitudeRef' in gps_info:
            lon = gps_info['GPSLongitude']
            lon_ref = gps_info['GPSLongitudeRef']
            formatted_gps['longitude'] = round(lon * (-1 if lon_ref.upper() == 'W' else 1), 6)
            
        # 處理海拔
        if 'GPSAltitude' in gps_info:
            formatted_gps['altitude'] = round(float(gps_info['GPSAltitude']), 2)
            
        return formatted_gps if formatted_gps else None
        
    except Exception as e:
        print(f"格式化 GPS 資料時發生錯誤: {str(e)}")
        return None

def fix_image_orientation(img: Image.Image, orientation: Optional[Dict]) -> Image.Image:
    """根據 EXIF 方向資訊修正圖片方向
    
    Args:
        img: PIL Image 物件
        orientation: 方向資訊字典
    
    Returns:
        Image.Image: 修正方向後的圖片
    """
    if not orientation:
        return img

    ORIENTATION_DICT = {
        1: None,  # 正常
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

def compress_image(input_path: Path, output_path: Path, max_size=500):
    """壓縮圖片並保留 EXIF 資訊
    
    Args:
        input_path (Path): 原始圖片路徑
        output_path (Path): 壓縮後圖片的保存路徑
        max_size (int): 最大檔案大小（KB）
    """
    img = Image.open(input_path)
    
    # 讀取 EXIF 資料
    try:
        exif_dict = piexif.load(img.info['exif'])
    except:
        exif_dict = None
    
    # 獲取圖片格式
    img_format = img.format

    # 初始品質
    quality = 95
    while True:
        # 保存到內存
        buffer = io.BytesIO()
        if exif_dict:
            exif_bytes = piexif.dump(exif_dict)
            img.save(buffer, format=img_format, quality=quality, exif=exif_bytes)
        else:
            img.save(buffer, format=img_format, quality=quality)
            
        size_kb = buffer.getbuffer().nbytes / 1024
        if size_kb <= max_size or quality <= 10:
            break
        quality -= 5

    # 保存到檔案
    with open(output_path, "wb") as f:
        f.write(buffer.getvalue())

@app.post("/analyze", response_class=JSONResponse)
async def analyze_image(file: UploadFile = File(...)):
    """處理上傳的圖片"""
    try:
        # 檢查檔案類型
        file_extension = file.filename.split(".")[-1].lower()
        if file_extension not in ALLOWED_EXTENSIONS:
            raise HTTPException(status_code=400, detail="不支持的檔案類型")

        # 儲存上傳的檔案
        file_path = UPLOAD_DIR / file.filename
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 讀取圖片並獲取 EXIF 資料
        img = Image.open(file_path)
        gps_info, orientation_info = get_exif_data(img)
        
        # 格式化 GPS 資料
        formatted_gps = format_gps_data(gps_info)
        
        # 修正圖片方向
        img = fix_image_orientation(img, orientation_info)
        
        # 重新保存修正方向後的圖片
        img.save(file_path)

        # 壓縮原圖並保存到 outputs/original/
        compressed_original_path = ORIGINAL_OUTPUT_DIR / file.filename
        compress_image(file_path, compressed_original_path, max_size=500)

        # 使用 get_pipeline() 替代直接使用 pipeline
        result = get_pipeline().process_image(str(file_path))

        if result["status"] == "error":
            raise HTTPException(status_code=500, detail=result["error"])
        
        # 構建輸出圖片的URL
        masked_image_name = Path(result['masked_image_path']).name
        mask_image_name = Path(result['mask_path']).name
        masked_image_url = f"/outputs/u2net/{masked_image_name}"
        mask_image_url = f"/outputs/u2net/{mask_image_name}"
        original_image_url = f"/outputs/original/{file.filename}"
        
        # 在推論完成後刪除原始上傳的圖片
        file_path.unlink(missing_ok=True)
        
        return JSONResponse({
            "message": "分析完成",
            "filename": file.filename,
            "prediction": round(float(result["prediction"]), 2),
            "processing_time": round(float(result["processing_time"]), 2),
            "original_image_url": original_image_url,
            "masked_image_url": masked_image_url,
            "mask_image_url": mask_image_url,
            "gps_info": formatted_gps,
            "orientation_fixed": orientation_info is not None
        })

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"處理過程中發生錯誤: {str(e)}"
        )

@app.get("/health")
async def health_check():
    """健康檢查端點"""
    return {"status": "healthy"}

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))  # 從環境變數讀取 PORT，若無則使用 8000
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True
    )
