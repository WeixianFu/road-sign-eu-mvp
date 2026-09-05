# 欧洲交通标志图片训练

从 MTSD 原图准备数据，训练 YOLOv8，在原图上切片预测，再做 COCO 评价和误报／漏检分析。当前范围是图片；视频、跟踪、OCR 和半监督留到后续。

欧洲范围包含瑞士、英国。当前本体有 **153 个目标类别**，地域来源单独记录；这并不表示 MTSD 已经完成欧洲地域筛选。

```text
configs/                 数据、训练、预测配置和唯一类别映射表
src/roadsigns/
  image_tools/           标签、切片、数据构建、图片预览
  training/              70/30 抽样、小目标 Mosaic、YOLO 训练
  evaluation/            原图切片预测、标准 COCO 指标
  analysis/              类别统计、训练曲线摘要、FP/FN 图片
tests/                   合成图片与适配器测试
tools/                   完整命令日志、轻量检查
docs/refactor/           本次审查、操作日志、阶段报告和验证记录
```

| 设备 | 工作 |
|---|---|
| MacBook | 查看代码、轻量测试、读取台式机导出的报告 |
| Ubuntu / RTX 5070 Ti | 准备数据、训练、原图预测与评价 |
| Linux 远程服务器 | 同一套命令，调整数据路径、设备和 batch |

Mac 上的轻量环境不需要 PyTorch：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,evaluate]' -c requirements/constraints.txt
python tools/check.py
```

Ubuntu 的完整安装和训练步骤见 [训练手册](docs/training.md)。原始输入仍使用旧项目的 **401 类 YOLO 原图格式**；新的类别拆分不能从已合并的 core146 标签中还原。

依次使用以下入口，各入口都支持 `--help`：

1. `rs-prepare`：原图、标签 → 1280 切片、合适的整图视图、原图 GT 和数据指纹。
2. `rs-preview`：核对原图或准备后的标签、切片布局。
3. `rs-train`：在 Ubuntu／服务器上训练。
4. `rs-predict`：对原始验证图片切片预测，可从已完成的图片继续。
5. `rs-evaluate`：COCO AP、逐类指标、小目标召回和 FP/FN 清单。
6. `rs-analyze`：数据统计、训练摘要、错误样本图。

[设计与数据约定](docs/design.md) · [类别调整](docs/ontology.md) · [评价与分析](docs/evaluation.md) · [重构审查](docs/refactor/audit.md)

本项目代码检查使用合成数据。真实 MTSD、RTX 5070 Ti 显存／吞吐、多 GPU 训练和新模型精度需要在目标设备验证；合成测试通过不等于模型已经训练完成。
