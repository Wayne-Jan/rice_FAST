import os
import requests
from pathlib import Path
import onnx

def download_file(url: str, destination: Path, filename: str) -> bool:
    """
    從指定的 URL 下載文件。

    :param url: 文件的下載 URL
    :param destination: 下載後的文件路徑
    :param filename: 文件名（用於打印信息）
    :return: 成功返回 True，否則返回 False
    """
    try:
        print(f"開始下載 {filename}...")
        with requests.get(url, stream=True) as response:
            response.raise_for_status()  # 確保請求成功
            with open(destination, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:  # 避免寫入空的塊
                        f.write(chunk)
        print(f"{filename} 下載完成")
        return True
    except Exception as e:
        print(f"{filename} 下載時發生錯誤: {e}")
        return False

def verify_onnx_model(model_path: Path) -> bool:
    """
    驗證 ONNX 模型文件的完整性。

    :param model_path: 模型文件路徑
    :return: 如果模型有效，返回 True，否則返回 False
    """
    try:
        onnx_model = onnx.load(str(model_path))
        onnx.checker.check_model(onnx_model)
        print(f"{model_path.name} 驗證成功")
        return True
    except Exception as e:
        print(f"{model_path.name} 驗證失敗: {e}")
        return False

def setup_models():
    """
    下載並驗證所需的模型文件。
    """
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)

    # 定義模型及其對應的 GitHub Releases 下載 URL
    base_url = "https://github.com/Wayne-Jan/rice_FAST/releases/download/v1.0.0"
    models = {
        "YOLO.onnx": f"{base_url}/YOLO.onnx",
        "u2net.onnx": f"{base_url}/u2net.onnx",
        "model.onnx": f"{base_url}/model.onnx",
        "scaler.pkl": f"{base_url}/scaler.pkl"
    }

    for filename, url in models.items():
        dest_path = models_dir / filename
        if dest_path.exists():
            print(f"{filename} 已存在，進行驗證...")
            if filename.endswith('.onnx'):
                if verify_onnx_model(dest_path):
                    print(f"{filename} 已經是有效的模型，跳過下載")
                    continue
                else:
                    print(f"{filename} 驗證失敗，重新下載")
            else:
                print(f"{filename} 存在且不需要驗證，跳過下載")
                continue

        # 下載文件
        success = download_file(url, dest_path, filename)
        if not success:
            print(f"{filename} 下載失敗，請手動檢查或重新嘗試")
            continue

        # 驗證下載的模型（如果是 ONNX 模型）
        if filename.endswith('.onnx'):
            if not verify_onnx_model(dest_path):
                print(f"{filename} 下載後驗證失敗，請重新下載或檢查文件")
        else:
            print(f"{filename} 下載完成")

    print("所有模型下載與驗證完成")

def verify_models() -> list:
    """
    檢查所有必要的模型文件是否存在且有效。

    :return: 缺失或無效的模型文件列表
    """
    models_dir = Path("models")
    required_models = ["YOLO.onnx", "u2net.onnx", "model.onnx", "scaler.pkl"]

    missing_models = []
    for model in required_models:
        model_path = models_dir / model
        if not model_path.exists():
            missing_models.append(model)
        elif model.endswith('.onnx'):
            try:
                onnx_model = onnx.load(str(model_path))
                onnx.checker.check_model(onnx_model)
            except Exception:
                missing_models.append(model)

    return missing_models

if __name__ == "__main__":
    # 檢查缺失的模型文件
    missing = verify_models()
    if missing:
        print(f"需要下載以下模型: {', '.join(missing)}")
        setup_models()
    else:
        print("所有模型已就緒且驗證成功")
