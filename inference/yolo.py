# inference/yolo.py

import numpy as np
import onnxruntime
import cv2
from typing import Dict
from pathlib import Path

def letterbox(im, new_shape=640, color=(114, 114, 114)):
    shape = im.shape[:2]  # 原始長寬
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, r, (dw, dh)

def xywh2xyxy(x):
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y

def scale_coords(img1_shape, coords, img0_shape):
    gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
    pad = ((img1_shape[1] - img0_shape[1] * gain) / 2,
           (img1_shape[0] - img0_shape[0] * gain) / 2)  # x, y
    coords[:, [0, 2]] -= pad[0]
    coords[:, [1, 3]] -= pad[1]
    coords[:, :4] /= gain
    coords[:, 0] = np.clip(coords[:, 0], 0, img0_shape[1])
    coords[:, 1] = np.clip(coords[:, 1], 0, img0_shape[0])
    coords[:, 2] = np.clip(coords[:, 2], 0, img0_shape[1])
    coords[:, 3] = np.clip(coords[:, 3], 0, img0_shape[0])
    return coords

def bbox_iou_np(box1, box2):
    x1 = np.maximum(box1[0], box2[:, 0])
    y1 = np.maximum(box1[1], box2[:, 1])
    x2 = np.minimum(box1[2], box2[:, 2])
    y2 = np.minimum(box1[3], box2[:, 3])
    inter = np.maximum((x2 - x1), 0) * np.maximum((y2 - y1), 0)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    union = area1 + area2 - inter
    return inter / (union + 1e-16)

def nms_numpy(boxes, scores, iou_thres=0.45):
    idxs = scores.argsort()[::-1]
    selected = []
    while len(idxs) > 0:
        i = idxs[0]
        selected.append(i)
        if len(idxs) == 1:
            break
        ious = bbox_iou_np(boxes[i], boxes[idxs[1:]])
        idxs = idxs[1:][ious < iou_thres]
    return np.array(selected)

def get_region(img, center, box_size=25):
    y, x = center
    y1 = max(y - box_size, 0)
    y2 = min(y + box_size, img.shape[0])
    x1 = max(x - box_size, 0)
    x2 = min(x + box_size, img.shape[1])
    return img[y1:y2, x1:x2]

def get_centers(pred, im0):
    center_position = {}
    if len(pred):
        boxes = pred[:, :4].copy()
        scores = pred[:, 4].copy()
        classes = pred[:, 5].copy()
        for i in range(len(boxes)):
            conf = scores[i]
            cls = classes[i]
            if conf > 0.7:  # 信心度門檻
                xyxy = boxes[i]
                center = [
                    int((xyxy[1] + xyxy[3]) / 2),
                    int((xyxy[0] + xyxy[2]) / 2)
                ]
                c = int(cls)
                # 假設 0->black, 1->gray, 2->white
                if c == 0:  # black
                    if 'black' not in center_position:
                        center_position['black'] = center
                    else:
                        region1 = get_region(im0, center_position['black'])
                        region2 = get_region(im0, center)
                        if np.mean(region1) > np.mean(region2):
                            center_position['black'] = center
                elif c == 1:  # gray
                    if 'gray' not in center_position:
                        center_position['gray'] = center
                elif c == 2:  # white
                    if 'white' not in center_position:
                        center_position['white'] = center
                    else:
                        region1 = get_region(im0, center_position['white'])
                        region2 = get_region(im0, center)
                        if np.mean(region1) < np.mean(region2):
                            center_position['white'] = center
    return center_position

def run_yolo_inference(yolo_onnx_path: str, image_path: str) -> Dict:
    providers = ['CUDAExecutionProvider'] if 'CUDAExecutionProvider' in onnxruntime.get_available_providers() else ['CPUExecutionProvider']
    session = onnxruntime.InferenceSession(yolo_onnx_path, providers=providers)

    im0 = cv2.imread(image_path)
    if im0 is None:
        raise FileNotFoundError(f"無法讀取影像: {image_path}")
    # 前處理
    img, r, (dw, dh) = letterbox(im0, new_shape=640)
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img, dtype=np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    results = session.run([output_name], {input_name: img})[0]  # (1, 25200, 8)
    pred = results[0]  # => (25200, 8)

    # 後處理
    conf_thres = 0.25
    iou_thres = 0.45
    mask = pred[:, 4] > conf_thres
    pred = pred[mask]
    if pred.shape[0] == 0:
        return {}

    cls_conf = pred[:, 5:]
    cls_conf_value = np.max(cls_conf, axis=1)
    cls_id = np.argmax(cls_conf, axis=1)
    scores = pred[:, 4] * cls_conf_value
    boxes = xywh2xyxy(pred[:, :4])
    idxs = nms_numpy(boxes, scores, iou_thres=iou_thres)
    boxes = boxes[idxs]
    scores = scores[idxs]
    cls_id = cls_id[idxs]
    boxes = scale_coords((640, 640), boxes.copy(), im0.shape).round()
    merged_pred = np.hstack((boxes, scores[:, np.newaxis], cls_id[:, np.newaxis]))
    centers = get_centers(merged_pred, im0)
    return centers
