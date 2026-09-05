# 原图评价与分析

训练中的 `results.csv` 是切片级验证。判断新模型是否改善，需要另跑原图评价，使用同一批原图、GT、本体及预测配置。

```bash
python tools/run_logged.py --log /data/logs/predict-baseline.log rs-predict --coco /data/mtsd-europe-v1/val.coco.json --weights /data/runs/baseline/model/weights/best.pt --config configs/predict.yaml --output /data/predictions/baseline
rs-evaluate --coco /data/mtsd-europe-v1/val.coco.json --predictions /data/predictions/baseline --output /data/reports/baseline --confidence 0.25
rs-analyze errors --coco /data/mtsd-europe-v1/val.coco.json --errors /data/reports/baseline/errors.json --output /data/reports/baseline/gallery --limit 20
```

`rs-predict` 对原图做 1280／20% 切片，批量预测，还原坐标后做同类别 NMS。不会把整张 4K 原图缩到 640。补边内的预测会移除，最终框使用原图像素坐标，保存产生该框的切片位置。

默认预测置信度下限为 0.001，用于保存评价候选；0.25 是另行计算的工作点，不是经过标定的部署阈值。可以根据逐类 precision/recall 进一步选择阈值，但不能用较高下限的预测缓存冒充完整召回范围。

逐图结果保存在 `parts/`，包括空预测图片。中断后在相同命令加 `--resume`，已经提交的图片不会再写入。模型、GT 或配置改变则使用新输出目录。只有原图清单完整完成才允许 `rs-evaluate`。

评价调用 [官方 pycocotools COCOeval](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py)：

- 逐类别 AP，再对可评价类别平均；101 个召回采样点，IoU 0.50 到 0.95。
- 标准 `maxDets=[1,10,100]`；AP_small 等使用 COCO 的原图框面积定义。
- 另行报告工作点 precision/recall、TP/FP/FN，以及**原图框短边小于 32 像素**目标的 recall@IoU50。该指标与 COCO AP_small 的定义不同。
- 验证集没有 GT 的类别，AP 为 `null`；不会当作 AP=0 拖低平均值。
- 完整空预测是合法结果，对有 GT 的类别记零召回。漏掉图片的未完成运行不能进入评价。

结果为 `metrics.json`、`per_class.csv`、`errors.json`。误差图库用绿色显示 GT，红色突出当前 FP/FN 框。数据统计使用原图实例数，而不是重复切片后的框数。

本次类别调整后，一些歧义原图被留待复核。所有报告都应带数据与本体指纹，不能直接把新 AP 与旧 core146 的 11 点 AP，或旧切片 mAP 数值对比。当前尚无新模型的真实精度结果。
