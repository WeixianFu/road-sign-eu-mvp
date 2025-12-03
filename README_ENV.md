# 环境配置

## 安装

```bash
conda env create -f environment.yml
conda activate road-sign-eu-mvp
```

## 服务器（GPU）

如需CUDA版本PyTorch，安装后运行：
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
