#!/bin/bash
# AutoDL 服务器环境一键配置（开机后运行一次）
# 前提: 基础镜像已带 PyTorch 2.x + CUDA（选镜像时已含），数据已在 /root/autodl-tmp/mtsd_core146
# Usage: bash scripts/setup_server.sh [数据目录]
set -e

DATA_ROOT="${1:-/root/autodl-tmp/mtsd_core146}"

echo "== [1/5] 开启 AutoDL 学术加速（加速 pip / GitHub / 权重下载）=="
if [ -f /etc/network_turbo ]; then
    source /etc/network_turbo
else
    echo "  (非 AutoDL 环境或无学术加速，跳过)"
fi

echo "== [2/5] 安装 Python 依赖 =="
pip install -q -U pip
pip install -q ultralytics pycocotools tensorboard

echo "== [3/5] GPU 检查 =="
python - <<'EOF'
import torch
n = torch.cuda.device_count()
print(f"  PyTorch {torch.__version__} | CUDA {torch.version.cuda} | GPU 数量: {n}")
for i in range(n):
    p = torch.cuda.get_device_properties(i)
    print(f"  [{i}] {p.name}  {p.total_memory/1024**3:.0f}GB")
assert n >= 1, "未检测到 CUDA GPU"
EOF

echo "== [4/5] 数据完整性校验: $DATA_ROOT（全量标签扫描+抽样解码，约2-3分钟）=="
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python "$SCRIPT_DIR/verify_data.py" --root "$DATA_ROOT"

echo "== [5/5] 预下载模型权重 yolov8m.pt =="
python -c "from ultralytics import YOLO; YOLO('yolov8m.pt')"

echo ""
echo "环境就绪。启动训练:  ./scripts/train.sh"
echo "训练不依赖外网；新开的 shell 无需再执行学术加速。"
