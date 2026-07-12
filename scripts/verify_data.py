#!/usr/bin/env python3
"""mtsd_core146 数据完整性校验。

检查项:
  1. 目录结构 (train_full/val 的 images+labels)
  2. 图片数量与预期完全一致 (train 131,429 / val 54,518)
  3. 图片↔标签配对 (孤儿标签=错误; 无标签图片=背景负样本, 仅统计)
  4. 标签内容合法性 (全量扫描: 5列、类别id∈[0,nc)、坐标∈[0,1]、宽高>0)
  5. 类别覆盖 (train 中缺失的类别id会警告)
  6. 图片抽样解码 (均匀抽样 N 张, 检测截断/损坏的 JPEG)

Usage:
    python scripts/verify_data.py                          # 服务器默认路径
    python scripts/verify_data.py --root /path/to/mtsd_core146 --sample 2000
"""

import argparse
import sys
from pathlib import Path

EXPECTED_COUNTS = {"train_full": 131429, "val": 54518}


def fail(msg, errors):
    print(f"  [ERROR] {msg}")
    errors.append(msg)


def check_labels(labels, nc, errors):
    """全量扫描标签文件，返回 (出现过的类别id集合, 空标签数)。"""
    seen_classes = set()
    n_empty = 0
    for i, lb in enumerate(labels):
        if i % 20000 == 0:
            print(f"    ... {i}/{len(labels)}")
        try:
            text = lb.read_text()
        except Exception as e:
            fail(f"标签不可读 {lb.name}: {e}", errors)
            continue
        lines = [l for l in text.strip().splitlines() if l.strip()]
        if not lines:
            n_empty += 1
            continue
        for ln, line in enumerate(lines, 1):
            parts = line.split()
            if len(parts) != 5:
                fail(f"{lb.name}:{ln} 列数={len(parts)} (应为5)", errors)
                break
            try:
                cls = int(parts[0])
                x, y, w, h = map(float, parts[1:])
            except ValueError:
                fail(f"{lb.name}:{ln} 数值解析失败: {line!r}", errors)
                break
            if not 0 <= cls < nc:
                fail(f"{lb.name}:{ln} 类别id {cls} 超出 [0,{nc})", errors)
                break
            if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                fail(f"{lb.name}:{ln} 坐标越界: {line!r}", errors)
                break
            seen_classes.add(cls)
    return seen_classes, n_empty


def check_images_decode(images, sample, errors):
    """均匀抽样解码图片，检测损坏/截断的 JPEG。"""
    try:
        from PIL import Image
    except ImportError:
        print("  [SKIP] 未安装 Pillow, 跳过图片解码抽检 (先跑 setup_server.sh)")
        return
    step = max(1, len(images) // sample)
    picked = images[::step][:sample]
    bad = 0
    for i, im in enumerate(picked):
        if i % 500 == 0:
            print(f"    ... {i}/{len(picked)}")
        try:
            with Image.open(im) as img:
                img.verify()
        except Exception as e:
            fail(f"图片损坏 {im.name}: {e}", errors)
            bad += 1
    print(f"  抽检 {len(picked)} 张, 损坏 {bad} 张")


def check_split(root, split, nc, sample, errors):
    print(f"\n== 检查 {split} ==")
    img_dir = root / split / "images"
    lbl_dir = root / split / "labels"
    for d in (img_dir, lbl_dir):
        if not d.is_dir():
            fail(f"目录缺失: {d}", errors)
            return

    images = sorted(img_dir.glob("*.jpg"))
    labels = sorted(lbl_dir.glob("*.txt"))
    expected = EXPECTED_COUNTS[split]
    print(f"  images: {len(images)} (预期 {expected}) | labels: {len(labels)}")
    if len(images) != expected:
        fail(f"{split} 图片数 {len(images)} != 预期 {expected} (解压不完整?)", errors)

    img_stems = {p.stem for p in images}
    lbl_stems = {p.stem for p in labels}
    orphan_labels = lbl_stems - img_stems
    bg_images = img_stems - lbl_stems
    if orphan_labels:
        fail(f"{len(orphan_labels)} 个标签没有对应图片, 如: {sorted(orphan_labels)[:3]}", errors)
    print(f"  无标签图片(背景负样本): {len(bg_images)}")

    print("  扫描标签内容...")
    seen, n_empty = check_labels(labels, nc, errors)
    print(f"  空标签文件(背景负样本): {n_empty} | 出现类别数: {len(seen)}/{nc}")
    if split == "train_full":
        missing = sorted(set(range(nc)) - seen)
        if missing:
            print(f"  [WARN] train 中未出现的类别id: {missing}")

    print("  抽样解码图片...")
    check_images_decode(images, sample, errors)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="/root/autodl-tmp/mtsd_core146")
    ap.add_argument("--nc", type=int, default=146)
    ap.add_argument("--sample", type=int, default=2000, help="每个 split 解码抽检张数")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        print(f"[ERROR] 数据根目录不存在: {root}")
        sys.exit(1)

    errors = []
    for split in EXPECTED_COUNTS:
        check_split(root, split, args.nc, args.sample, errors)

    print("\n" + "=" * 50)
    if errors:
        print(f"校验失败: {len(errors)} 个错误 (前10条):")
        for e in errors[:10]:
            print(f"  - {e}")
        sys.exit(1)
    print("校验通过: 结构/数量/配对/标签/抽样解码 全部正常, 可以开始训练。")


if __name__ == "__main__":
    main()
