import os
from pathlib import Path
import time
import numpy as np
import onnxruntime
import joblib
from typing import Dict, Tuple, Optional
import gc

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
        
        # 設定推論提供者
        self.providers = ['CUDAExecutionProvider'] if device == "cuda" and 'CUDAExecutionProvider' in onnxruntime.get_available_providers() else ['CPUExecutionProvider']
        
        # 初始化延遲載入的模型變數
        self._yolo_session = None
        self._u2net_inference = None
        self._predictor_session = None
        self._scaler = None
        
        # 設定模型路徑
        self.yolo_onnx_path = str(self.model_dir / "YOLO.onnx")
        self.u2net_onnx_path = str(self.model_dir / "u2net.onnx")
        self.predictor_onnx_path = str(self.model_dir / "model.onnx")
        self.scaler_path = str(self.model_dir / "scaler.pkl")

    def _get_providers(self):
        """獲取推論提供者列表"""
        return self.providers

    def _load_yolo(self):
        """延遲載入 YOLO 模型"""
        if self._yolo_session is None:
            self._yolo_session = onnxruntime.InferenceSession(
                self.yolo_onnx_path,
                providers=self._get_providers(),
                sess_options=self._get_session_options()
            )

    def _load_u2net(self):
        """延遲載入 U2NET 模型"""
        if self._u2net_inference is None:
            self._u2net_inference = U2NetInference(self.u2net_onnx_path)

    def _load_predictor(self):
        """延遲載入預測模型和 scaler"""
        if self._predictor_session is None:
            self._predictor_session = onnxruntime.InferenceSession(
                self.predictor_onnx_path,
                providers=self._get_providers(),
                sess_options=self._get_session_options()
            )
        if self._scaler is None:
            self._scaler = joblib.load(self.scaler_path)

    def _get_session_options(self):
        """獲取 ONNX Runtime 會話選項"""
        sess_options = onnxruntime.SessionOptions()
        sess_options.intra_op_num_threads = 1
        sess_options.inter_op_num_threads = 1
        sess_options.enable_mem_pattern = False
        sess_options.enable_cpu_mem_arena = False
        return sess_options

    def _unload_models(self):
        """卸載所有模型釋放記憶體"""
        if self._yolo_session:
            del self._yolo_session
            self._yolo_session = None
        
        if self._u2net_inference:
            del self._u2net_inference
            self._u2net_inference = None
            
        if self._predictor_session:
            del self._predictor_session
            self._predictor_session = None
            
        if self._scaler:
            del self._scaler
            self._scaler = None
        
        gc.collect()

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
            self._load_yolo()
            checker_centers = run_yolo_inference(self.yolo_onnx_path, image_path, providers=self.providers)
            del self._yolo_session
            self._yolo_session = None
            gc.collect()
            
            # 2. U2NET 分割
            print("執行 U2NET 分割...")
            self._load_u2net()
            mask_path, masked_image_path = self._u2net_inference(
                image_path=image_path,
                output_dir=str(self.output_dir / "u2net"),
                target_size=512
            )
            del self._u2net_inference
            self._u2net_inference = None
            gc.collect()
            
            # 3. 特徵提取
            print("執行特徵提取...")
            analyzer = RiceAnalyzer(output_dir=str(self.output_dir / "results"))
            image_list = [
                {
                    'image_path': image_path,
                    'mask_path': masked_image_path,
                    'checker_centers': checker_centers
                }
            ]
            analyzer.analyze_batch(image_list)
            
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
        finally:
            # 確保所有模型都被卸載
            self._unload_models()
            gc.collect()

    def _predict(self, analyzer: RiceAnalyzer) -> float:
        """執行最終預測
        
        Args:
            analyzer (RiceAnalyzer): 已完成分析的 RiceAnalyzer 實例
        
        Returns:
            float: 預測結果
        """
        try:
            # 載入預測模型和 scaler
            self._load_predictor()
            
            # 假設只分析一張圖片，取得特徵
            if not analyzer.results:
                raise ValueError("沒有分析結果可供預測")
            
            features = analyzer.results[0]['features']
            features_array = np.array(features).reshape(1, -1)
            
            # 標準化特徵
            scaled_features = self._scaler.transform(features_array).astype(np.float32)
            
            # 取得輸入輸出名稱
            input_name = self._predictor_session.get_inputs()[0].name
            output_name = self._predictor_session.get_outputs()[0].name
            
            # 推論
            prediction = self._predictor_session.run(
                [output_name], 
                {input_name: scaled_features}
            )[0][0]
            
            return prediction
            
        finally:
            # 釋放預測模型和 scaler
            if self._predictor_session:
                del self._predictor_session
                self._predictor_session = None
            if self._scaler:
                del self._scaler
                self._scaler = None
            gc.collect()

    def __del__(self):
        """解構函數，確保資源被釋放"""
        self._unload_models()