"""
FullImageValidator: 基于切片元数据评估 YOLOv8 在 4K 原图上的真实性能。

读取已有切片和元数据 JSON，推理后通过坐标映射还原到原图。
使用 YOLO 格式的 GT labels 计算 mAP，无需 COCO 格式。
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import torch
import torchvision
from tqdm import tqdm
from ultralytics import YOLO


def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """计算两个 xyxy 格式 box 的 IoU"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    
    return inter / union if union > 0 else 0


def compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """计算 AP (Average Precision) 使用 11 点插值法"""
    ap = 0.0
    for t in np.arange(0, 1.1, 0.1):
        mask = recalls >= t
        if mask.any():
            ap += precisions[mask].max()
    return ap / 11


class FullImageValidator:
    def __init__(
        self,
        model,  # Can be YOLO object or path string
        index_dir: str,
        images_dir: str,
        gt_labels_dir: str,  # YOLO 格式 labels 目录
        conf_thresh: float = 0.25,  # 0.001太低导致计算极慢
        iou_thresh: float = 0.5,
        device: str = None,
    ):
        # Support both YOLO object and path string
        if isinstance(model, str):
            self.model = YOLO(model)
        else:
            self.model = model
        
        self.index_dir = Path(index_dir)
        self.images_dir = Path(images_dir)
        self.gt_labels_dir = Path(gt_labels_dir)
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.device = device

    def _load_index(self, image_id: str) -> Dict[str, Any]:
        """加载单个 image_id 的索引 JSON"""
        index_file = self.index_dir / f"{image_id}.json"
        with open(index_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_yolo_gt(self, image_id: str, img_w: int, img_h: int) -> List[Tuple[int, np.ndarray]]:
        """加载 YOLO 格式 GT labels，返回 [(cls_id, xyxy_box), ...]"""
        label_path = self.gt_labels_dir / f"{image_id}.txt"
        if not label_path.exists():
            return []
        
        gt_boxes = []
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls_id = int(parts[0])
                cx, cy, w, h = map(float, parts[1:5])
                # 归一化 -> 绝对坐标 (xyxy)
                x1 = (cx - w / 2) * img_w
                y1 = (cy - h / 2) * img_h
                x2 = (cx + w / 2) * img_w
                y2 = (cy + h / 2) * img_h
                gt_boxes.append((cls_id, np.array([x1, y1, x2, y2])))
        
        return gt_boxes

    def _map_slice_to_original(self, boxes: np.ndarray, roi: List[int]) -> np.ndarray:
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
        mapped[:, 0] -= pad_x
        mapped[:, 1] -= pad_y
        mapped[:, 2] -= pad_x
        mapped[:, 3] -= pad_y
        mapped /= scale
        h, w = original_shape
        mapped[:, 0] = np.clip(mapped[:, 0], 0, w)
        mapped[:, 1] = np.clip(mapped[:, 1], 0, h)
        mapped[:, 2] = np.clip(mapped[:, 2], 0, w)
        mapped[:, 3] = np.clip(mapped[:, 3], 0, h)
        return mapped

    def _global_nms(self, boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray) -> tuple:
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

    def _process_single_image(self, image_id: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List]:
        """处理单张原图，返回 (pred_boxes, pred_scores, pred_classes, gt_boxes_list)"""
        index_data = self._load_index(image_id)
        original_shape = index_data["original_shape"]  # [h, w]
        h, w = original_shape
        
        all_boxes, all_scores, all_classes = [], [], []

        # 处理所有切片
        for slice_info in index_data["slices"]:
            img_path = self.images_dir / slice_info["filename"]
            roi = slice_info["roi"]
            
            results = self.model.predict(str(img_path), conf=self.conf_thresh, verbose=False, device=self.device)
            result = results[0]
            
            if len(result.boxes) == 0:
                continue
                
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            
            boxes = self._map_slice_to_original(boxes, roi)
            
            all_boxes.append(boxes)
            all_scores.append(scores)
            all_classes.append(classes)

        # 处理 full_resized 图
        full_info = index_data["full_resized"]
        img_path = self.images_dir / full_info["filename"]
        
        results = self.model.predict(str(img_path), conf=self.conf_thresh, verbose=False, device=self.device)
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
            pred_boxes = np.array([]).reshape(0, 4)
            pred_scores = np.array([])
            pred_classes = np.array([])
        else:
            all_boxes = np.concatenate(all_boxes)
            all_scores = np.concatenate(all_scores)
            all_classes = np.concatenate(all_classes)
            pred_boxes, pred_scores, pred_classes = self._global_nms(all_boxes, all_scores, all_classes)

        # 加载 GT
        gt_boxes = self._load_yolo_gt(image_id, w, h)
        
        return pred_boxes, pred_scores, pred_classes, gt_boxes

    def _compute_metrics(
        self, 
        all_preds: List[Tuple[np.ndarray, np.ndarray, np.ndarray]], 
        all_gts: List[List[Tuple[int, np.ndarray]]],
        iou_thresh: float
    ) -> Tuple[float, float, float]:
        """计算 Precision, Recall, AP@iou_thresh（按图像分组优化版）"""
        # 按图像分组收集预测和 GT
        pred_list = []  # [(score, cls, box, img_idx), ...]
        gt_by_image = {}  # {img_idx: [(gt_idx, cls, box, matched), ...]}
        total_gt = 0
        gt_global_idx = 0
        
        for img_idx, (preds, gts) in enumerate(zip(all_preds, all_gts)):
            boxes, scores, classes = preds
            for i in range(len(boxes)):
                pred_list.append((scores[i], int(classes[i]), boxes[i], img_idx))
            
            # 按图像分组 GT
            gt_by_image[img_idx] = []
            for cls_id, box in gts:
                gt_by_image[img_idx].append([gt_global_idx, cls_id, box, False])
                gt_global_idx += 1
                total_gt += 1
        
        if total_gt == 0:
            return 0.0, 0.0, 0.0
        
        # 按置信度排序
        pred_list.sort(key=lambda x: x[0], reverse=True)
        
        # 匹配（只在对应图像的 GT 中查找）
        tp = np.zeros(len(pred_list))
        fp = np.zeros(len(pred_list))
        
        for pred_idx, (score, pred_cls, pred_box, img_idx) in enumerate(pred_list):
            best_iou = 0
            best_gt_local_idx = -1
            
            # 只遍历当前图像的 GT
            for local_idx, (gt_global_idx, gt_cls, gt_box, matched) in enumerate(gt_by_image[img_idx]):
                if gt_cls != pred_cls or matched:
                    continue
                iou = compute_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_local_idx = local_idx
            
            if best_iou >= iou_thresh and best_gt_local_idx >= 0:
                tp[pred_idx] = 1
                gt_by_image[img_idx][best_gt_local_idx][3] = True  # mark as matched
            else:
                fp[pred_idx] = 1
        
        # 计算累积 TP/FP
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        recalls = tp_cumsum / total_gt
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-16)
        
        # 计算 AP
        ap = compute_ap(recalls, precisions)
        
        # 最终 precision/recall
        final_precision = tp.sum() / (tp.sum() + fp.sum() + 1e-16)
        final_recall = tp.sum() / total_gt
        
        return final_precision, final_recall, ap

    def run(self) -> Dict[str, float]:
        """执行完整验证流程，返回 mAP 指标"""
        index_files = list(self.index_dir.glob("*.json"))
        image_ids = [f.stem for f in index_files]
        
        print(f"Found {len(image_ids)} images to validate")
        
        all_preds = []
        all_gts = []
        
        for image_id in tqdm(image_ids, desc="Processing"):
            pred_boxes, pred_scores, pred_classes, gt_boxes = self._process_single_image(image_id)
            all_preds.append((pred_boxes, pred_scores, pred_classes))
            all_gts.append(gt_boxes)
        
        total_preds = sum(len(p[0]) for p in all_preds)
        total_gts = sum(len(g) for g in all_gts)
        print(f"Total predictions: {total_preds}, Total GT: {total_gts}")
        
        if total_gts == 0:
            print("No GT labels found")
            return {"mAP50": 0.0, "mAP50-95": 0.0, "precision": 0.0, "recall": 0.0}

        # 计算 mAP50
        p50, r50, ap50 = self._compute_metrics(all_preds, all_gts, iou_thresh=0.5)
        
        # 计算 mAP50-95 (平均 IoU 从 0.5 到 0.95，步长 0.05)
        aps = []
        for iou in np.arange(0.5, 1.0, 0.05):
            # 重置 GT matched 状态
            for gt in all_gts:
                for item in gt:
                    if len(item) > 3:
                        item[3] = False
            _, _, ap = self._compute_metrics(all_preds, all_gts, iou_thresh=iou)
            aps.append(ap)
        map50_95 = np.mean(aps)
        
        print(f"\nResults:")
        print(f"  Precision@50: {p50:.4f}")
        print(f"  Recall@50:    {r50:.4f}")
        print(f"  AP@50:        {ap50:.4f}")
        print(f"  mAP@50-95:    {map50_95:.4f}")
        
        return {
            "mAP50": ap50,
            "mAP50-95": map50_95,
            "precision": p50,
            "recall": r50,
        }


if __name__ == "__main__":
    # Server paths (update if running locally)
    validator = FullImageValidator(
        model="runs/mtsd/yolov8m_production/weights/best.pt",
        index_dir="/root/autodl-tmp/MTSD_download/mtsd-resized/val/index",
        images_dir="/root/autodl-tmp/MTSD_download/mtsd-resized/val/images",
        gt_labels_dir="/root/autodl-tmp/MTSD_download/mtsd-resized/valfull/labels",
    )
    metrics = validator.run()
    print(f"\nmAP50: {metrics['mAP50']:.4f}")
    print(f"mAP50-95: {metrics['mAP50-95']:.4f}")
