# mtsd_core146 服务器训练手册（AutoDL · 2×魔改4090-48G）

从零开机到拿回权重的完整流程。所有命令均在服务器上执行（除最后下载权重）。

## 0. 本次训练概览

| 项 | 值 |
| --- | --- |
| 数据 | `/root/autodl-tmp/mtsd_core146`（146 类，train 131,429 / val 54,518 张 1280 切片，59GB） |
| 模型 | yolov8m, imgsz 1280 |
| 硬件 | 2× 魔改 RTX 4090 48GB，DDP，总 batch 48（每卡 24，约 40G/卡） |
| 配置 | `configs/train_core146.yaml` + `configs/data_core146.yaml`（改参数只改这两个文件） |
| 输出 | `runs/detect/runs/mtsd_core146/yolov8m_core146/`（权重在其 `weights/` 下） |
| 验证 | YOLO 切片级 val 每 epoch 跑；全图 full-val 已禁用（见文末"已知限制"） |

## 1. 租机（AutoDL 控制台）

1. 机型：**RTX 4090 48GB（魔改）× 2 卡**，按量计费。
2. 镜像：基础镜像 → **PyTorch 2.8.0 / Python 3.12 / CUDA 12.x**。
3. **数据盘扩容 ≥100GB**（数据 59GB；若 tar 分卷也要放数据盘上解压，则 ≥150GB，解压完删 tar 可缩回）。
4. vCPU ≥24 核为佳（DDP 两卡各开 10 个 dataloader 进程）；不足则见文末"故障排查"。

## 2. 准备数据

数据最终必须长这样（`setup_server.sh` 会自动校验数量）：

```text
/root/autodl-tmp/mtsd_core146/
├── train_full/{images,labels}   # 131,429 张
└── val/{images,labels}          #  54,518 张
```

### 2.1 校验 tar 分卷（解压前）

传输 60GB 最容易出的问题是分卷缺失或字节不全。解压前先对一下**每卷的字节数**（免费、秒出）：

```bash
ls -l /root/autodl-tmp/mtsd_core146.tar*
```

| 卷 | 字节数 |
| --- | --- |
| tar00 ~ tar04 | 每卷 `10737418240` |
| tar05 | `9867438080` |

任何一卷对不上 = 传输不完整，重传该卷。如需更强校验（可选，两端各读一遍 60GB，约几分钟）：

```bash
# 本地 Mac:
cat /Users/weixianfu/Documents/Datas/mtsd_core146.tar* | md5
# 服务器:
cat /root/autodl-tmp/mtsd_core146.tar* | md5sum
# 两个值一致即完好
```

### 2.2 解压

分卷名是 `mtsd_core146.tar00` ~ `tar05`（无点号）：

```bash
cd /root/autodl-tmp
cat mtsd_core146.tar* | tar -xf - --warning=no-unknown-keyword   # 退出码非 0 = 数据流损坏
rm mtsd_core146.tar*                   # 2.3 校验通过后再删，省 60GB
```

会刷屏大量 `Ignoring unknown extended header keyword 'LIBARCHIVE.xattr.com.apple.provenance'`
属**正常现象**：包是 macOS 打的，Linux tar 不认识苹果的文件元数据字段，忽略的只是元数据，
文件内容完整解出。加上面的 `--warning=no-unknown-keyword` 可静音；忘加也无需重解压。

**解压后必须清理 macOS 垃圾文件**——Mac 打包会给每个文件塞一个 `._` 开头的 AppleDouble
元数据孪生文件（所以解压后文件数正好翻倍），不删的话 YOLO 会把它们当图片/标签读：

```bash
find /root/autodl-tmp/mtsd_core146 \( -name '._*' -o -name '.DS_Store' \) -delete
```

（`verify_data.py` 会检测到这些文件并报错提醒。根治办法：以后在 Mac 上打包时加环境变量
`COPYFILE_DISABLE=1 tar -cf - mtsd_core146 | split -b 10g -d - mtsd_core146.tar`，就不会产生 `._` 文件。）

### 2.3 数据完整性校验（解压后）

完整校验由 `scripts/verify_data.py` 完成（在第 3 步 clone 代码后运行，`setup_server.sh` 会自动调用）。它检查：

- 目录结构、图片数量与预期**完全一致**（train 131,429 / val 54,518）
- 图片↔标签一一配对（孤儿标签报错；无标签图片计为背景负样本）
- **全量**标签内容合法性（5 列、类别 id ∈ [0,146)、坐标 ∈ [0,1]）
- 146 个类别在 train 中全部出现
- 均匀抽样 2000 张图片解码，检测截断/损坏的 JPEG

本地数据的基准值（服务器上应完全一致）：train 背景空标签 47,174，val 背景空标签 42,298，train 类别 146/146，val 类别 145/146。

等不及 clone 代码想先快速看一眼，可以手动数一下：

```bash
ls /root/autodl-tmp/mtsd_core146/train_full/images | wc -l   # 应为 131429
ls /root/autodl-tmp/mtsd_core146/val/images | wc -l          # 应为 54518
```

## 3. 首次部署（开机后跑一次）

```bash
source /etc/network_turbo                                        # AutoDL 学术加速（加速 GitHub）
git clone -b re-phase https://github.com/WeixianFu/road-sign-eu-mvp.git /root/road-sign-eu-mvp
# 注意用 HTTPS：服务器上没有你的 SSH key，git@github.com: 会报 Permission denied (publickey)
cd /root/road-sign-eu-mvp
bash scripts/setup_server.sh                                     # 装依赖 + 查 GPU + 验数据 + 下载预训练权重
# 脚本内部自己管理代理开关：pip 走国内源（不代理），权重下载走 GitHub（开代理）
```

脚本任何一步报错都会停下并说明原因（最常见：数据没解压完整）。

## 4. 启动训练

```bash
cd /root/road-sign-eu-mvp
./scripts/train.sh
```

- nohup 后台运行，**关掉 SSH / 断网训练不中断**。
- 全部输出写入 `scripts/train.log`，进程号存 `scripts/train.pid`。
- 启动后先 `tail -f scripts/train.log` 盯 2 分钟：确认 DDP 起来（日志出现 `DDP: debug command ... --device 0,1`）、数据集扫描通过、第一个 epoch 开始出进度条，再放心断开。
- **实际输出目录以日志里 `Logging results to` 那一行为准**。DDP 子进程重建参数时会把相对的
  `project` 拼到默认目录下，实测为 `runs/detect/runs/mtsd_core146/yolov8m_core146/`（比配置里多一层
  `runs/detect/`），本手册后文路径均按此实测值。

（可选）挂一个"训练完自动关机"的看门狗，防止跑完空烧钱：

```bash
nohup bash -c 'while kill -0 $(cat /root/road-sign-eu-mvp/scripts/train.pid) 2>/dev/null; do sleep 300; done; sleep 60; shutdown' >/dev/null 2>&1 &
```

## 5. 随时连回来看进度

重新 SSH 上去后，任选：

```bash
# 实时日志（最常用）
tail -f /root/road-sign-eu-mvp/scripts/train.log

# 是否还在跑 / 停止训练
source /root/road-sign-eu-mvp/scripts/train_utils.sh
train_status        # 或 train_stop

# GPU 占用（两张卡都应 ~40G 显存、高利用率）
watch -n 2 nvidia-smi

# 每个 epoch 的指标表（mAP50 / mAP50-95 / loss）
column -s, -t /root/road-sign-eu-mvp/runs/detect/runs/mtsd_core146/yolov8m_core146/results.csv | less -S
```

**网页可视化（不用 SSH）**：AutoDL 控制台的 TensorBoard 面板读 `/root/tf-logs`，启动训练后做一次软链即可：

```bash
rm -rf /root/tf-logs && ln -s /root/road-sign-eu-mvp/runs/detect/runs/mtsd_core146 /root/tf-logs
```

之后在 AutoDL 实例页面点「TensorBoard」就能在浏览器看损失曲线和 mAP。

## 6. 中断后恢复（断电 / 手滑 kill / 主动停过）

checkpoint 每 5 个 epoch 存一次（`save_period: 5`），且 `last.pt` 每个 epoch 都更新。恢复：

```bash
cd /root/road-sign-eu-mvp
nohup yolo train resume model=runs/detect/runs/mtsd_core146/yolov8m_core146/weights/last.pt \
    >> scripts/train.log 2>&1 &
echo $! > scripts/train.pid
```

ultralytics 会从 `last.pt` 连同优化器状态、epoch 数、DDP 配置一起恢复，不需要重新传参。

## 7. 训练结束后

1. 看最终指标：`tail -n 50 scripts/train.log`，以及 `runs/.../results.png`、`confusion_matrix.png`。
2. 下载权重到本地（在**本地 Mac** 执行，端口/地址按 AutoDL 实例 SSH 信息替换）：

   ```bash
   scp -P <端口> root@<地址>:/root/road-sign-eu-mvp/runs/detect/runs/mtsd_core146/yolov8m_core146/weights/best.pt ~/Downloads/
   # 想要完整训练产物（曲线图、csv、args）：
   ssh -p <端口> root@<地址> "cd /root/road-sign-eu-mvp/runs/detect/runs/mtsd_core146 && tar czf /root/autodl-tmp/yolov8m_core146_run.tar.gz yolov8m_core146 --exclude='*.pt' "
   scp -P <端口> root@<地址>:/root/autodl-tmp/yolov8m_core146_run.tar.gz ~/Downloads/
   ```

3. **确认文件都拿到后再关机/释放实例**（数据盘内容在"关机"状态保留，"释放"则全部清空）。

## 8. 费用控制

- 单价约 ¥6/h（2×¥3）。**前 2–3 个 epoch 记一下耗时**，×150 就是总时长上界（有 `patience: 50` 早停，通常跑不满）。
- 中途觉得太慢/太贵，可以 `train_stop` 后换单卡：把 `configs/train_core146.yaml` 改 `device: 0`、`batch: 24`，用第 6 节的方式 resume（换 batch 后建议直接重新 `./scripts/train.sh` 并接受从头训，或接受 resume 时 lr 调度按原计划走）。
- 账户余额耗尽实例会被强制关机——长训练前确认余额够撑到结束。

## 9. 故障排查

| 症状 | 处理 |
| --- | --- |
| CUDA out of memory | `train_core146.yaml` 里 `batch` 降到 40 → 32 试；魔改卡偶发驱动怪异，也可换实例 |
| CPU 打满 / dataloader 卡慢 | `workers` 从 10 降到 6–8 |
| 启动即退，日志报数据路径 | 确认 `configs/data_core146.yaml` 的 `path` 与实际解压位置一致 |
| 权重下载超时（首次） | 先 `source /etc/network_turbo` 再跑 `setup_server.sh` |
| DDP 卡在启动 | 日志搜 `DDP`；确认 `device: [0, 1]` 且 `nvidia-smi` 能看到两张卡 |
| 日志很久不动 | `train_status` 看进程在不在；在的话看 `nvidia-smi` 利用率，val 阶段（5.4 万张）单卡跑、耗时较长属正常 |

## 10. 已知限制

- **全图 full-val 已禁用**（`--no-full-val`）：core146 数据集没有 `val/index` 和 `valfull` 原图 GT，且旧 GT 是 401 类 id、与 146 类模型不匹配。当前 mAP 是**切片级**指标。若要恢复原图级 mAP，需在 `notebooks/data_analysis/mtsd_classes.ipynb` 中用 401→146 映射重新生成 valfull labels + val index 并上传服务器，再把 `scripts/train.sh` 里 `--no-full-val` 换回 `--gt-labels-dir <路径>`。
- ultralytics DDP 模式下自定义 callback 不生效——full-val 禁用后无影响，但恢复 full-val 时若仍是双卡，需要改为训练后单独跑 `validator_full.py`。
- 数据集自带的 `mtsd_core146/data.yaml` 里是本地 Mac 路径，训练**不使用它**（用的是仓库里的 `configs/data_core146.yaml`），无需修改。
