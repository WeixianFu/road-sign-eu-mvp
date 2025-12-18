"""
Training script with periodic full-image validation.

Supports both MacBook (MPS) and CUDA server.
Runs FullImageValidator every N epochs to compute metrics on original 4K images.

Usage:
    # MacBook debug (with full-val)
    python src/detect/train.py --config configs/train_debug.yaml --data configs/data.yaml --full-val-interval 5
    
    # CUDA server (with full-val)
    python src/detect/train.py --config configs/train.yaml --data configs/data.yaml --full-val-interval 10
    
    # Without full-val (faster, only YOLO default validation)
    python src/detect/train.py --config configs/train_debug.yaml --data configs/data.yaml --no-full-val
"""

import argparse
from pathlib import Path

import yaml
from ultralytics import YOLO

# Import after setting up path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.detect.validator_full import FullImageValidator


class TrainerWithFullVal:
    """YOLOv8 trainer with periodic full-image validation."""
    
    def __init__(
        self,
        config_path: str,
        data_path: str,
        index_dir: str,
        images_dir: str,
        gt_labels_dir: str = None,  # YOLO 格式 labels 目录
        full_val_interval: int = 10,
    ):
        # Load configs
        with open(config_path, "r") as f:
            self.train_cfg = yaml.safe_load(f)
        
        self.data_path = data_path
        self.index_dir = index_dir
        self.images_dir = images_dir
        self.gt_labels_dir = gt_labels_dir
        self.full_val_interval = full_val_interval
        
        # Extract key params
        self.model_name = self.train_cfg.pop("model", "yolov8s.pt")
        self.device = self.train_cfg.get("device", 0)
        self.epochs = self.train_cfg.get("epochs", 100)
        
        # Initialize model
        self.model = YOLO(self.model_name)
        
        # Track metrics
        self.full_val_metrics = []
        self.current_epoch = 0
    
    def _on_train_epoch_end(self, trainer):
        """Callback after each epoch."""
        self.current_epoch = trainer.epoch + 1
        
        # Run full-image validation at specified intervals
        if self.gt_labels_dir and self.current_epoch % self.full_val_interval == 0:
            print(f"\n[Epoch {self.current_epoch}] Running full-image validation...")
            
            # Get current best weights path
            weights_path = trainer.best if trainer.best.exists() else trainer.last
            
            validator = FullImageValidator(
                model=str(weights_path),
                index_dir=self.index_dir,
                images_dir=self.images_dir,
                gt_labels_dir=self.gt_labels_dir,
                device=self.device if isinstance(self.device, str) else self.device,
            )
            
            metrics = validator.run()
            self.full_val_metrics.append({
                "epoch": self.current_epoch,
                **metrics
            })
            
            print(f"[Full-Image] mAP50: {metrics['mAP50']:.4f}, mAP50-95: {metrics['mAP50-95']:.4f}\n")
    
    def _on_train_end(self, trainer):
        """Callback at training end."""
        # Final full-image validation
        if self.gt_labels_dir:
            print("\n" + "=" * 60)
            print("Final full-image validation on best weights...")
            print("=" * 60)
            
            validator = FullImageValidator(
                model=str(trainer.best),
                index_dir=self.index_dir,
                images_dir=self.images_dir,
                gt_labels_dir=self.gt_labels_dir,
                device=self.device if isinstance(self.device, str) else self.device,
            )
            
            metrics = validator.run()
            
            print("\n" + "=" * 60)
            print(f"FINAL RESULTS (Full-Image)")
            print(f"  mAP50:    {metrics['mAP50']:.4f}")
            print(f"  mAP50-95: {metrics['mAP50-95']:.4f}")
            print("=" * 60)
    
    def train(self):
        """Run training with callbacks."""
        # Add custom callbacks
        self.model.add_callback("on_train_epoch_end", self._on_train_epoch_end)
        self.model.add_callback("on_train_end", self._on_train_end)
        
        # Start training
        results = self.model.train(
            data=self.data_path,
            **self.train_cfg
        )
        
        return results


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 with full-image validation")
    parser.add_argument("--config", type=str, required=True, help="Training config YAML path")
    parser.add_argument("--data", type=str, required=True, help="Data config YAML path")
    parser.add_argument("--index-dir", type=str, default=None, help="Val index JSON directory")
    parser.add_argument("--images-dir", type=str, default=None, help="Val sliced images directory")
    parser.add_argument("--gt-labels-dir", type=str, default=None, help="YOLO format GT labels directory")
    parser.add_argument("--full-val-interval", type=int, default=10, help="Run full-val every N epochs")
    parser.add_argument("--no-full-val", action="store_true", help="Disable full-image validation")
    args = parser.parse_args()
    
    # Load data config to get default paths
    with open(args.data, "r") as f:
        data_cfg = yaml.safe_load(f)
    
    # Use provided paths or defaults from data config
    original_root = data_cfg.get("path", "")
    sliced_root = data_cfg.get("path_sliced_resized", "")
    
    index_dir = args.index_dir or f"{sliced_root}/val/index"
    images_dir = args.images_dir or f"{sliced_root}/val/images"
    
    # 默认使用原始 val labels 目录，除非 --no-full-val
    if args.no_full_val:
        gt_labels_dir = None
    else:
        gt_labels_dir = args.gt_labels_dir or f"{original_root}/val/labels"
    
    trainer = TrainerWithFullVal(
        config_path=args.config,
        data_path=args.data,
        index_dir=index_dir,
        images_dir=images_dir,
        gt_labels_dir=gt_labels_dir,
        full_val_interval=args.full_val_interval,
    )
    
    trainer.train()


if __name__ == "__main__":
    main()

