"""
FullImageValidator: 基于切片元数据评估 YOLOv8 在 4K 原图上的真实性能。

读取已有切片和元数据 JSON，推理后通过坐标映射还原到原图，计算 mAP。
"""

import json
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import torch
import torchvision
from tqdm import tqdm
from ultralytics import YOLO
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


class FullImageValidator:
    def __init__(
        self,
        model_path: str,
        index_dir: str,
        images_dir: str,
        gt_path: str,
        conf_thresh: float = 0.001,
        iou_thresh: float = 0.5,
    ):
        self.model = YOLO(model_path)
        self.index_dir = Path(index_dir)
        self.images_dir = Path(images_dir)
        self.gt_path = gt_path
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        
        self.coco_gt = COCO(gt_path)
        # 建立 filename -> image_id 映射
        self.filename_to_id = {
            img["file_name"]: img["id"] for img in self.coco_gt.imgs.values()
        }

    def _load_index(self, image_id: str) -> Dict[str, Any]:
        """加载单个 image_id 的索引 JSON"""
        index_file = self.index_dir / f"{image_id}.json"
        with open(index_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def _map_slice_to_original(
        self, boxes: np.ndarray, roi: List[int]
    ) -> np.ndarray:
        """将切片坐标映射回原图坐标 (xyxy 格式)"""
        x1, y1, _, _ = roi
        mapped = boxes.copy()
        mapped[:, 0] += x1
        mapped[:, 1] += y1
        mapped[:, 2] += x1
        mapped[:, 3] += y1
        return mapped

    def _map_resized_to_original(
        self, boxes: np.ndarray, scale: float, padding: List[float], original_shape: List[int]
    ) -> np.ndarray:
        """将 letterbox resized 图的坐标映射回原图坐标 (xyxy 格式)"""
        pad_x, pad_y = padding
        mapped = boxes.copy()
        # 去掉 padding
        mapped[:, 0] -= pad_x
        mapped[:, 1] -= pad_y
        mapped[:, 2] -= pad_x
        mapped[:, 3] -= pad_y
        # 缩放回原图
        mapped /= scale
        # 裁剪到原图范围
        h, w = original_shape
        mapped[:, 0] = np.clip(mapped[:, 0], 0, w)
        mapped[:, 1] = np.clip(mapped[:, 1], 0, h)
        mapped[:, 2] = np.clip(mapped[:, 2], 0, w)
        mapped[:, 3] = np.clip(mapped[:, 3], 0, h)
        return mapped

    def _global_nms(
        self, boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray
    ) -> tuple:
        """按类别执行全局 NMS"""
        if len(boxes) == 0:
            return boxes, scores, classes

        keep_boxes, keep_scores, keep_classes = [], [], []
        
        for cls_id in np.unique(classes):
            mask = classes == cls_id
            cls_boxes = torch.from_numpy(boxes[mask]).float()
            cls_scores = torch.from_numpy(scores[mask]).float()
            
            keep_idx = torchvision.ops.nms(cls_boxes, cls_scores, self.iou_thresh)
            
            keep_boxes.append(cls_boxes[keep_idx].numpy())
            keep_scores.append(cls_scores[keep_idx].numpy())
            keep_classes.append(np.full(len(keep_idx), cls_id))

        return (
            np.concatenate(keep_boxes) if keep_boxes else np.array([]),
            np.concatenate(keep_scores) if keep_scores else np.array([]),
            np.concatenate(keep_classes) if keep_classes else np.array([]),
        )

    def _process_single_image(self, image_id: str) -> List[Dict[str, Any]]:
        """处理单张原图：推理所有切片 -> 坐标映射 -> 全局 NMS -> 返回 COCO 格式结果"""
        index_data = self._load_index(image_id)
        original_shape = index_data["original_shape"]  # [h, w]
        
        all_boxes, all_scores, all_classes = [], [], []

        # 处理所有切片
        for slice_info in index_data["slices"]:
            img_path = self.images_dir / slice_info["filename"]
            roi = slice_info["roi"]
            
            results = self.model.predict(str(img_path), conf=self.conf_thresh, verbose=False)
            result = results[0]
            
            if len(result.boxes) == 0:
                continue
                
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            
            # 映射到原图坐标
            boxes = self._map_slice_to_original(boxes, roi)
            
            all_boxes.append(boxes)
            all_scores.append(scores)
            all_classes.append(classes)

        # 处理 full_resized 图
        full_info = index_data["full_resized"]
        img_path = self.images_dir / full_info["filename"]
        
        results = self.model.predict(str(img_path), conf=self.conf_thresh, verbose=False)
        result = results[0]
        
        if len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            
            boxes = self._map_resized_to_original(
                boxes, full_info["scale_factor"], full_info["padding"], original_shape
            )
            
            all_boxes.append(boxes)
            all_scores.append(scores)
            all_classes.append(classes)

        # 合并并执行全局 NMS
        if not all_boxes:
            return []
            
        all_boxes = np.concatenate(all_boxes)
        all_scores = np.concatenate(all_scores)
        all_classes = np.concatenate(all_classes)
        
        final_boxes, final_scores, final_classes = self._global_nms(
            all_boxes, all_scores, all_classes
        )

        # 转换为 COCO 结果格式
        coco_image_id = self.filename_to_id.get(f"{image_id}.jpg")
        if coco_image_id is None:
            return []

        coco_results = []
        for box, score, cls_id in zip(final_boxes, final_scores, final_classes):
            x1, y1, x2, y2 = box
            coco_results.append({
                "image_id": coco_image_id,
                "category_id": int(cls_id),
                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],  # xywh
                "score": float(score),
            })
        
        return coco_results

    def run(self) -> Dict[str, float]:
        """执行完整验证流程，返回 mAP 指标"""
        # 获取所有 image_id
        index_files = list(self.index_dir.glob("*.json"))
        image_ids = [f.stem for f in index_files]
        
        print(f"Found {len(image_ids)} images to validate")
        
        # 收集所有结果
        all_results = []
        for image_id in tqdm(image_ids, desc="Processing"):
            results = self._process_single_image(image_id)
            all_results.extend(results)
        
        print(f"Total detections: {len(all_results)}")
        
        if len(all_results) == 0:
            print("No detections, cannot evaluate")
            return {"mAP50": 0.0, "mAP50-95": 0.0}

        # COCO 评估
        coco_dt = self.coco_gt.loadRes(all_results)
        coco_eval = COCOeval(self.coco_gt, coco_dt, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        
        return {
            "mAP50-95": coco_eval.stats[0],
            "mAP50": coco_eval.stats[1],
        }


if __name__ == "__main__":
    validator = FullImageValidator(
        model_path="path/to/best.pt",
        index_dir="/Users/weixianfu/Documents/Datas/mtsd-resized/val/index",
        images_dir="/Users/weixianfu/Documents/Datas/mtsd-resized/val/images",
        gt_path="path/to/val_annotations.json",
    )
    metrics = validator.run()
    print(f"\nmAP50: {metrics['mAP50']:.4f}")
    print(f"mAP50-95: {metrics['mAP50-95']:.4f}")

