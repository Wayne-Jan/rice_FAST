# inference/pipeline.py

import os
from pathlib import Path
import time
import numpy as np
import onnxruntime
import joblib
from typing import Dict, Tuple, Optional

# 從相應模組導入所需的函數和類別
from .yolo import run_yolo_inference
from .u2net import U2NetInference
from .analyzer import RiceAnalyzer

class InferencePipeline:
    """整合式推論管線，包含 YOLO、U2NET 和稻米分析"""
    
    def __init__(
        self,
        model_dir: str,
        output_dir: str = "./outputs",
        device: str = "cpu"
    ):
        """
        初始化推論管線
        
        Args:
            model_dir: 模型目錄路徑
            output_dir: 輸出目錄路徑
            device: 使用的設備 ("cpu" 或 "cuda")
        """
        self.model_dir = Path(model_dir)
        self.output_dir = Path(output_dir)
        self.device = device
        
        # 建立必要的目錄
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "u2net").mkdir(exist_ok=True)
        (self.output_dir / "results").mkdir(exist_ok=True)
        
        # 載入所有需要的模型
        self._load_models()
        
        # 設定推論提供者
        self.providers = ['CUDAExecutionProvider'] if device == "cuda" and 'CUDAExecutionProvider' in onnxruntime.get_available_providers() else ['CPUExecutionProvider']

    def _load_models(self):
        """載入所有需要的模型和 scaler"""
        try:
            # 載入 YOLO 模型
            self.yolo_onnx_path = str(self.model_dir / "YOLO.onnx")
            
            # 載入 U2NET 模型
            self.u2net_onnx_path = str(self.model_dir / "u2net.onnx")
            self.u2net_inference = U2NetInference(self.u2net_onnx_path)
            
            # 載入預測模型
            self.predictor_onnx_path = str(self.model_dir / "model.onnx")
            
            # 載入 scaler
            self.scaler_path = str(self.model_dir / "scaler.pkl")
            self.scaler = joblib.load(self.scaler_path)
            
            print("所有模型載入完成")
            
        except Exception as e:
            raise RuntimeError(f"載入模型時發生錯誤: {str(e)}")

    def process_image(self, image_path: str) -> Dict:
        """
        處理單張圖片的完整流程
        
        Args:
            image_path: 輸入圖片路徑
            
        Returns:
            Dict: 包含處理結果的字典
        """
        start_time = time.time()
        
        try:
            # 1. YOLO 偵測
            print("執行 YOLO 偵測...")
            checker_centers = run_yolo_inference(self.yolo_onnx_path, image_path)
            
            # 2. U2NET 分割
            print("執行 U2NET 分割...")
            mask_path, masked_image_path = self.u2net_inference(
                image_path=image_path,
                output_dir=str(self.output_dir / "u2net"),
                target_size=512
            )
            
            # 3. 特徵提取
            print("執行特徵提取...")
            analyzer = RiceAnalyzer(output_dir=str(self.output_dir / "results"))
            image_list = [
                {
                    'image_path': image_path,
                    'mask_path': masked_image_path,  # 使用去背後的影像進行分析
                    'checker_centers': checker_centers
                }
            ]
            analyzer.analyze_batch(image_list)
            analyzer.save_results()
            
            # 4. 預測
            print("執行最終預測...")
            prediction = self._predict(analyzer)
            
            process_time = time.time() - start_time
            
            return {
                "status": "success",
                "prediction": float(prediction),
                "processing_time": process_time,
                "masked_image_path": masked_image_path,
                "mask_path": mask_path
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }

    def _predict(self, analyzer: RiceAnalyzer) -> float:
        """執行最終預測
        
        Args:
            analyzer (RiceAnalyzer): 已完成分析的 RiceAnalyzer 實例
        
        Returns:
            float: 預測結果
        """
        # 假設只分析一張圖片，取得特徵
        if not analyzer.results:
            raise ValueError("沒有分析結果可供預測")
        
        features = analyzer.results[0]['features']
        features_array = np.array(features).reshape(1, -1)
        
        # 標準化特徵
        scaled_features = self.scaler.transform(features_array).astype(np.float32)
        
        # 建立 ONNX Runtime session
        session = onnxruntime.InferenceSession(self.predictor_onnx_path, providers=self.providers)
        
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        
        # 推論
        prediction = session.run([output_name], {input_name: scaled_features})[0][0]
        return prediction
