# inference/analyzer.py

from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import cv2
import pandas as pd

class RiceAnalyzer:
    def __init__(self, output_dir="./results", clean_params=None):
        """初始化RiceAnalyzer類別
        
        Args:
            output_dir (str): 輸出目錄路徑
            clean_params (dict): 清理參數字典，包含:
                - gray_threshold: 灰階閾值
                - rgb_gap_threshold: RGB差異閾值
                - exg_threshold: ExG指數閾值
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 預設的清理參數
        self.clean_params = {
            'gray_threshold': 80,
            'rgb_gap_threshold': 20,
            'exg_threshold': 0
        } if clean_params is None else clean_params
        
        # 特徵名稱列表
        self.feature_names = [
            # 基本顏色特徵
            'R', 'G', 'B',
            'H1', 'S1', 'V1',
            'L2', 'S2',
            'L', 'a', 'b',
            'Y', 'Cr', 'Cb',
            'NDI', 'GI', 'RGRI',
            
            # 對比度校正後的顏色特徵
            'RC', 'GC', 'BC',
            'H1C', 'S1C', 'V1C',
            'L2C', 'S2C',
            'LC', 'aC', 'bC',
            'YC', 'CrC', 'CbC',
            'NDIC', 'GIC', 'RGRIC',
            
            # 色板特徵
            'B_CB', 'G_CB', 'W_CB',
            'B_RGB_B', 'B_RGB_G', 'B_RGB_R',
            'G_RGB_B', 'G_RGB_G', 'G_RGB_R',
            'W_RGB_B', 'W_RGB_G', 'W_RGB_R'
        ]
        
        self.results = []
        self._setup_lookup_tables()

    def _setup_lookup_tables(self):
        """設置查找表以加速處理"""
        # Gamma校正查找表
        self.gamma_lut = np.array([
            np.clip(255 * (i/255) ** 1.2, 0, 255) 
            for i in range(256)
        ], dtype=np.uint8)
        
        # 色彩空間轉換常數
        self.rgb_to_float = np.array([1/255], dtype=np.float32)

    def analyze_batch(self, image_list, max_workers=None):
        """批次處理圖片

        Args:
            image_list (list): 包含圖片資訊的字典列表，每個字典需包含:
                - image_path: 原始圖片路徑
                - mask_path: 遮罩圖片路徑
                - checker_centers: 色板中心點位置
            max_workers (int, optional): 最大工作執行緒數
        """
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for image_info in image_list:
                future = executor.submit(
                    self.analyze_image,
                    image_info['image_path'],
                    image_info['mask_path'],
                    image_info['checker_centers']
                )
                futures.append((image_info['image_path'], future))
            
            for image_path, future in futures:
                try:
                    result = future.result()
                    self.results.append({
                        'image_name': Path(image_path).stem,
                        'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
                        'features': result
                    })
                except Exception as e:
                    print(f"處理圖片時發生錯誤 {image_path}: {str(e)}")

    def _extract_checker(self, image, centers, size=25):
        """提取色板區域
        
        Args:
            image (np.ndarray): 輸入圖片
            centers (dict): 色板中心點座標
            size (int): 色板大小的一半
            
        Returns:
            np.ndarray: 提取的色板影像
        """
        checker = np.zeros((size*2, size*6, 3), np.uint8)
        colors = ['black', 'gray', 'white']
        default_values = {'black': 40, 'gray': 120, 'white': 230}
        
        for i, color in enumerate(colors):
            start_col = i * size * 2
            end_col = (i + 1) * size * 2
            if color in centers:
                y, x = centers[color]
                checker[:, start_col:end_col] = image[y-size:y+size, x-size:x+size]
            else:
                checker[:, start_col:end_col] = default_values[color]
                
        return checker

    def _clean_panicle_image(self, image):
        """進行稻穗影像的深度清理
        
        Args:
            image (np.ndarray): 輸入圖片
            
        Returns:
            np.ndarray: 清理後的圖片
        """
        # 轉換為灰階並建立基礎遮罩
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        base_mask = (gray >= self.clean_params['gray_threshold'])
        blue_mask = (image[:,:,0] <= image[:,:,2])
        combined_mask = base_mask & blue_mask
        
        # 計算各通道差異
        bgr_float = image.astype(np.float32)
        B = bgr_float[:,:,0]
        G = bgr_float[:,:,1]
        R = bgr_float[:,:,2]
        
        # 計算各種植被指數
        BRGAP = B - R
        BGGAP = B - G
        ExB = 2 * B - G - R
        ExG = 2 * G - R - B
        RGB_GAP = (np.abs(B-G) + np.abs(B-R) + np.abs(G-R)) / 3
        
        # 合併所有條件
        advanced_mask = (
            (BRGAP < 0) &
            (BGGAP < 0) &
            (ExB < 0) &
            (RGB_GAP > self.clean_params['rgb_gap_threshold']) &
            (ExG > self.clean_params['exg_threshold'])
        )
        
        # 應用最終遮罩
        final_mask = combined_mask & advanced_mask
        return cv2.bitwise_and(image, image, mask=final_mask.astype(np.uint8))

    def _gamma_correction(self, image, gray_checker):
        """使用查找表進行 Gamma 校正
        
        Args:
            image (np.ndarray): 輸入圖片
            gray_checker (np.ndarray): 灰色色板區域
            
        Returns:
            np.ndarray: Gamma校正後的圖片
        """
        return cv2.LUT(image, self.gamma_lut)

    def _calculate_color_features(self, pixels):
        """計算顏色特徵
        
        Args:
            pixels (np.ndarray): 像素值陣列
            
        Returns:
            list: 計算出的顏色特徵列表
        """
        # 計算RGB平均值
        pixels_float = pixels.astype(np.float32)
        rgb_means = pixels_float.mean(axis=0)[::-1]  # BGR->RGB
        
        # 批量轉換顏色空間
        pixels_2d = pixels.reshape(1, -1, 3)
        hsv = cv2.cvtColor(pixels_2d, cv2.COLOR_BGR2HSV)[0].mean(axis=0)
        hls = cv2.cvtColor(pixels_2d, cv2.COLOR_BGR2HLS)[0].mean(axis=0)
        lab = cv2.cvtColor(pixels_2d, cv2.COLOR_BGR2Lab)[0].mean(axis=0)
        ycc = cv2.cvtColor(pixels_2d, cv2.COLOR_BGR2YCrCb)[0].mean(axis=0)
        
        # 計算植被指數
        R = pixels_float[:,2] * self.rgb_to_float
        G = pixels_float[:,1] * self.rgb_to_float
        B = pixels_float[:,0] * self.rgb_to_float
        
        ndi = ((G-R)/(G+R+1e-10)).mean()
        gi = (G/(R+1e-10)).mean()
        rgri = (R/(G+1e-10)).mean()
        
        return [
            *rgb_means,
            hsv[0], hsv[1], hsv[2],
            hls[1], hls[2],
            lab[0], lab[1], lab[2],
            ycc[0], ycc[1], ycc[2],
            ndi, gi, rgri
        ]

    def _calculate_features(self, image, checker):
        """計算所有特徵
        
        Args:
            image (np.ndarray): 處理後的圖片
            checker (np.ndarray): 色板圖片
            
        Returns:
            list: 所有計算出的特徵列表
        """
        # 找出非背景像素
        mask = np.any(image != 0, axis=2)
        pixels = image[mask]
        
        if len(pixels) == 0:
            return [0] * len(self.feature_names)
        
        # 計算基本特徵
        features = []
        features.extend(self._calculate_color_features(pixels))
        
        # 計算對比度校正後的特徵
        black_ref = checker[:,:50].mean(axis=(0,1))
        white_ref = checker[:,100:].mean(axis=(0,1))
        corrected = np.clip(
            255 * (pixels.astype(np.float32) - black_ref) /
            (white_ref - black_ref + 1e-10),
            0, 255
        ).astype(np.uint8)
        
        features.extend(self._calculate_color_features(corrected))
        features.extend(self._calculate_checker_features(checker))
        
        return features

    def _calculate_checker_features(self, checker):
        """計算色板特徵
        
        Args:
            checker (np.ndarray): 色板圖片
            
        Returns:
            list: 色板特徵列表
        """
        # 分割色板區域
        black = checker[:,:50].mean(axis=(0,1))   # [B,G,R]
        gray = checker[:,50:100].mean(axis=(0,1)) # [B,G,R]
        white = checker[:,100:].mean(axis=(0,1))  # [B,G,R]
        
        # 計算平均值
        avg_black = black.mean()
        avg_gray = gray.mean()
        avg_white = white.mean()
        
        # 按照feature_names的順序返回特徵
        return [
            avg_black, avg_gray, avg_white,        # B_CB, G_CB, W_CB
            black[0], black[1], black[2],          # B_RGB_B, B_RGB_G, B_RGB_R
            gray[0], gray[1], gray[2],            # G_RGB_B, G_RGB_G, G_RGB_R
            white[0], white[1], white[2]          # W_RGB_B, W_RGB_G, W_RGB_R
        ]

    def analyze_image(self, image_path, mask_path, checker_centers):
        """分析單張圖片並返回特徵值
        
        Args:
            image_path (str): 原始圖片路徑
            mask_path (str): 遮罩圖片路徑
            checker_centers (dict): 色板中心點座標
            
        Returns:
            list: 計算出的所有特徵值
        """
        # 讀取圖片
        original = cv2.imread(image_path)
        if original is None:
            raise ValueError(f"無法讀取圖片: {image_path}")
            
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"無法讀取遮罩: {mask_path}")
        
        # 提取色板
        checker = self._extract_checker(original, checker_centers)
        
        # 處理圖片
        masked = cv2.bitwise_and(original, original, mask=mask)
        gamma_corrected = self._gamma_correction(masked, checker[:,50:100])
        cleaned = self._clean_panicle_image(gamma_corrected)
        
        # 計算特徵
        return self._calculate_features(cleaned, checker)

    def save_results(self, filename=None):
        """儲存分析結果
        
        Args:
            filename (str, optional): 輸出檔案名稱
        """
        if not self.results:
            print("沒有可儲存的結果")
            return
            
        if filename is None:
            filename = f"rice_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            
        # 建立DataFrame
        df = pd.DataFrame([
            {
                '圖片名稱': result['image_name'],
                '分析時間': result['timestamp'],
                **dict(zip(self.feature_names, result['features']))
            }
            for result in self.results
        ])
        
        # 儲存結果
        output_path = self.output_dir / filename
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"結果已儲存至: {output_path}")
