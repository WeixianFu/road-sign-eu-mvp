# Ubuntu 训练手册

所有真实数据处理和训练在 Ubuntu 台式机或 Linux 服务器运行。建议 Python 3.12；轻量核心保持 Python 3.9 兼容。没有在本次重构中执行真实训练。

## 安装

```bash
git clone -b rebase2 https://github.com/WeixianFu/road-sign-eu-mvp.git
cd road-sign-eu-mvp
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e '.[train,evaluate,dev]' -c requirements/constraints.txt
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))'
python tools/check.py
```

CUDA 12.8 构建是为包括 RTX 50 系列在内的 Blackwell 环境选择的。官方 [PyTorch 2.7 发布说明](https://pytorch.org/blog/pytorch-2-7/)介绍了该支持，[安装矩阵](https://pytorch.org/get-started/previous-versions/)列出了 PyTorch 2.9.1 与 torchvision 0.24.1 的 CUDA 12.8 安装组合。NVIDIA 驱动需支持所选 CUDA 运行时。

`requirements/constraints.txt` 固定核心依赖和训练接口；运行记录保存完整已安装版本。它不是包含所有间接依赖的跨平台锁文件。

## 准备图片

准备 [设计文档](design.md) 中的原始 401 类 YOLO 图片和标签。编辑 `configs/data.yaml` 的 `source_root`、`output_root`，然后：

```bash
python tools/run_logged.py --log /data/logs/prepare-v1.log rs-prepare --config configs/data.yaml
rs-analyze dataset --manifest /data/mtsd-europe-v1/manifest.json --output /data/reports/dataset-v1
```

先看生成的 `class_counts.csv`、待复核图片清单、训练／验证原图数量及整图视图数量。数据准备会在目标机器检查配对、格式和原图跨集合重复，并生成数据指纹。

抽查图片与标签，例如：

```bash
rs-preview --image /data/mtsd-europe-v1/images/train/真实图片名__tile_000.jpg --labels /data/mtsd-europe-v1/labels/train/真实图片名__tile_000.txt --data /data/mtsd-europe-v1/data.yaml --output /data/reports/sample.png
```

## 训练

桌面配置从 `yolov8m`、1280 输入、batch 2、4 个数据加载进程、关闭 AMP 开始。这是便于首次测量的起点，不是已经在 5070 Ti 实测过的最佳参数。根据实际峰值显存和吞吐调整 `profiles.desktop.batch`；不通过降到 640 来省显存。

```bash
python tools/run_logged.py --log /data/logs/baseline.log rs-train --data /data/mtsd-europe-v1/data.yaml --config configs/train.yaml --profile desktop --output /data/runs/baseline
```

建议在 `tmux` 会话中执行，断开 SSH 后继续运行。服务器使用 `--profile server`，先设置该 profile 的 `device`、`batch`、`workers`。多 GPU 可把 `device` 写成 `[0, 1]`；总 batch 应能被 GPU 数整除。适配器测试检查框架接口，但多 GPU 的实际训练仍需服务器验证。

输出：

```text
/data/runs/baseline/
  run.json                   本体、数据、代码、环境、初始权重、GPU 信息
  operations.jsonl           启动、epoch、恢复和完成记录
  train.log                  项目阶段日志
  model/
    args.yaml                框架实际采用的参数
    results.csv              每轮训练与切片验证指标
    weights/{best,last}.pt
```

完整控制台输出在 `--log` 指定的文件。不要把这个文件放在 `model/` 中；框架可能清理内部目录。loss 出现 NaN/Inf 时会保存 `model/nonfinite-rank*.json` 和相应增强后图片张量，然后停止；查看证据，不凭一次曲线判断根因。

## 中断恢复

只恢复本项目生成、尚未完成的运行，使用同一数据路径、本体和训练参数。允许修改设备与 batch；其他配方改动应创建新实验，不能以为普通 `resume` 会采用修改后的学习率或 AMP。

```bash
python tools/run_logged.py --log /data/logs/baseline.log rs-train --data /data/mtsd-europe-v1/data.yaml --config configs/train.yaml --profile desktop --output /data/runs/baseline --resume /data/runs/baseline/model/weights/last.pt
```

训练完成或已出现非有限参数的 checkpoint 不作为继续训练的起点。旧 core146 模型不是此处的恢复对象。初始 `model` 可换成对应 YOLO11 预训练 `.pt` 文件进行独立实验，其余数据、切片及评价保持一致。

## 在 Mac 查看结果

将 `results.csv`、`run.json`、评价报告和精选错误图片复制到 Mac，运行：

```bash
rs-analyze training --csv /本地报告目录/results.csv --output /本地报告目录/training-summary
```

不要在 Mac 运行完整 `rs-prepare` 或 `rs-train`。`tools/check.py` 只构建少量合成样本；安装了训练依赖时还会检查真实数据加载器，但不会启动训练或下载模型权重。
