# Training Scripts

用于在 Linux 服务器上运行正式训练的脚本。

## 文件说明

- `train.sh`: 主训练脚本，启动后台训练
- `train_utils.sh`: 训练管理工具（查看日志、停止训练等）
- `train.log`: 训练日志文件（自动生成）
- `train.pid`: 训练进程 ID 文件（自动生成）

## 使用方法

### 1. 启动训练

```bash
# 在项目根目录下运行
./scripts/train.sh
```

训练会在后台启动，所有输出会实时写入 `scripts/train.log`。

**特点：**
- ✅ 使用 `nohup`，SSH 断开后训练继续运行
- ✅ Python 使用 `-u` 参数，输出无缓冲，日志实时更新
- ✅ 所有输出（stdout + stderr）都写入日志文件
- ✅ 自动处理相对路径问题

### 2. 查看训练日志

**实时查看（推荐）：**
```bash
tail -f scripts/train.log
```

**查看最后 50 行：**
```bash
tail -n 50 scripts/train.log
```

**使用工具脚本：**
```bash
# 加载工具函数
source scripts/train_utils.sh

# 实时查看日志
train_log

# 查看最后 N 行
train_log_last 100
```

### 3. 检查训练状态

```bash
# 使用工具脚本
source scripts/train_utils.sh
train_status

# 或手动检查
ps -p $(cat scripts/train.pid)
```

### 4. 停止训练

```bash
# 使用工具脚本
source scripts/train_utils.sh
train_stop

# 或手动停止
kill $(cat scripts/train.pid)
```

## 配置说明

训练脚本使用以下配置文件：
- **训练配置**: `configs/train.yaml`
- **数据配置**: `configs/data.yaml`

如需修改配置，请编辑对应的 YAML 文件，然后重新启动训练。

## 注意事项

1. **日志文件大小**: 长时间训练会产生大量日志，建议定期检查磁盘空间
2. **进程管理**: 训练进程 ID 保存在 `scripts/train.pid`，可用于进程管理
3. **路径问题**: 脚本会自动处理相对路径，确保在项目根目录运行即可
4. **Python 环境**: 确保已激活正确的 conda/virtualenv 环境

## 故障排查

**训练没有启动？**
- 检查 `scripts/train.log` 查看错误信息
- 确认配置文件路径正确
- 确认 Python 环境已激活

**日志没有更新？**
- 检查进程是否还在运行：`ps -p $(cat scripts/train.pid)`
- 检查日志文件权限：`ls -l scripts/train.log`

**如何重新启动训练？**
- 先停止当前训练：`train_stop` 或 `kill $(cat scripts/train.pid)`
- 删除旧的日志文件（可选）：`rm scripts/train.log`
- 重新运行：`./scripts/train.sh`

