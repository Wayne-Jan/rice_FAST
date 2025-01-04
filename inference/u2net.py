# inference/u2net.py

import numpy as np
import onnxruntime
import cv2
from pathlib import Path
from typing import Union, Tuple

class U2NetInference:
    def __init__(self, model_path: str):
        """初始化 U2NET 推理類

        Args:
            model_path (str): ONNX 模型路徑
        """
        # 初始化 ONNX Runtime session
        providers = ['CUDAExecutionProvider'] if 'CUDAExecutionProvider' in onnxruntime.get_available_providers() else ['CPUExecutionProvider']
        self.session = onnxruntime.InferenceSession(model_path, providers=providers)
        
        # 緩存輸入輸出名稱
        self.output_names = [o.name for o in self.session.get_outputs()]
        self.input_name = self.session.get_inputs()[0].name
        
        # 預設參數
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    @staticmethod
    def load_image(image_path: Union[str, Path]) -> np.ndarray:
        """載入並預處理圖像

        Args:
            image_path: 圖像路徑

        Returns:
            np.ndarray: BGR 格式的圖像數組
        """
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"無法讀取圖片: {image_path}")
        return img

    @staticmethod
    def resize_maintain_aspect(image: np.ndarray, target_size: int = 512) -> np.ndarray:
        """保持長寬比的圖像縮放

        Args:
            image: 輸入圖像
            target_size: 目標尺寸

        Returns:
            np.ndarray: 縮放後的圖像
        """
        height, width = image.shape[:2]
        
        if height > width:
            new_height = target_size
            new_width = int(target_size * width / height)
        else:
            new_height = int(target_size * height / width)
            new_width = target_size
            
        # 確保寬高為偶數
        new_height += (new_height % 2)
        new_width += (new_width % 2)
        
        return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """圖像預處理

        Args:
            image: BGR 格式輸入圖像

        Returns:
            np.ndarray: 預處理後的圖像數組
        """
        # 轉換為 RGB 並標準化到 [0, 1]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        if image.max() > 1e-6:
            image /= 255.0
            
        # 調整維度順序並進行標準化
        image = image.transpose(2, 0, 1)
        image = (image - self.mean[:, None, None]) / self.std[:, None, None]
        
        return np.expand_dims(image, axis=0)

    @staticmethod
    def normalize_mask(mask: np.ndarray) -> np.ndarray:
        """最小最大值正規化

        Args:
            mask: 輸入遮罩

        Returns:
            np.ndarray: 正規化後的遮罩
        """
        min_val = mask.min()
        max_val = mask.max()
        
        if max_val - min_val > 1e-6:
            mask = (mask - min_val) / (max_val - min_val)
        
        return mask

    def process_mask(self, mask: np.ndarray, original_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """處理預測遮罩

        Args:
            mask: 預測的遮罩
            original_image: 原始圖像

        Returns:
            Tuple[np.ndarray, np.ndarray]: 調整大小的遮罩和遮罩後的圖片
        """
        # 轉換為 uint8 並調整大小
        mask_uint8 = (mask * 255).astype(np.uint8)
        oh, ow = original_image.shape[:2]
        mask_resized = cv2.resize(mask_uint8, (ow, oh), interpolation=cv2.INTER_LINEAR)
        
        # 應用高斯模糊
        mask_blurred = cv2.GaussianBlur(mask_resized, (5, 5), 0)
        
        # 創建三通道遮罩並應用到原始圖像
        mask_3ch = np.repeat(mask_blurred[:, :, np.newaxis], 3, axis=2).astype(np.float32) / 255.0
        masked_image = (original_image * mask_3ch).astype(np.uint8)
        
        return mask_resized, masked_image

    def __call__(self, 
                 image_path: Union[str, Path], 
                 output_dir: Union[str, Path] = "./u2net_output", 
                 target_size: int = 512) -> Tuple[str, str]:
        """執行 U2NET 推理

        Args:
            image_path: 輸入圖像路徑
            output_dir: 輸出目錄路徑
            target_size: 目標尺寸

        Returns:
            Tuple[str, str]: 遮罩圖像路徑和遮罩後的圖像路徑
        """
        # 確保輸出目錄存在
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 載入和預處理圖像
        original_image = self.load_image(image_path)
        resized_image = self.resize_maintain_aspect(original_image, target_size)
        input_tensor = self.preprocess(resized_image)
        
        # 執行推理
        results = self.session.run(self.output_names, {self.input_name: input_tensor})
        pred_mask = self.normalize_mask(results[0][0, 0])
        
        # 處理遮罩和生成結果
        mask_resized, masked_image = self.process_mask(pred_mask, original_image)
        
        # 保存結果
        mask_path = output_dir / "resized_mask.png"
        masked_image_path = output_dir / "masked_image.png"
        
        cv2.imwrite(str(mask_path), mask_resized)
        cv2.imwrite(str(masked_image_path), masked_image)
        
        return str(mask_path), str(masked_image_path)
