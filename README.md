# 医学图像分类项目（MONAI）

基于 [MONAI](https://monai.io/) 和迁移学习的医学图像分类系统，支持 TIFF / PNG / JPEG 格式图像。

MONAI（Medical Open Network for Artificial Intelligence）是专为医学影像 AI 研发打造的 PyTorch 框架，
提供医学图像专用的数据增强、网络架构和训练工具。

## 项目结构

```
├── config.py          # 项目配置（路径、超参数、wandb 设置等）
├── dataset.py         # MONAI CacheDataset + 数据加载
├── transforms.py      # MONAI 字典式医学图像变换（nnU-Net 风格增强）
├── model.py           # MONAI 网络架构（DenseNet / EfficientNet / SE-ResNet）
├── train.py           # 训练脚本（支持 wandb / AMP / 梯度裁剪 / 断点续训）
├── evaluate.py        # 评估脚本（指标 + 可视化 + wandb 日志）
├── predict.py         # 单张/批量推理脚本
├── requirements.txt   # Python 依赖
├── data/              # 数据目录（不纳入版本控制）
│   ├── class_1/       # 第1类图像文件夹
│   ├── class_2/       # 第2类图像文件夹
│   └── ...
└── outputs/           # 输出目录
    ├── checkpoints/   # 模型权重
    └── logs/          # 训练日志
```

## 环境准备

```bash
# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 安装依赖
pip install -r requirements.txt
```

> **提示**：如需处理特殊 TIFF 格式（如 float32、多页 TIFF），建议额外安装
> `pip install SimpleITK`，MONAI 会自动使用 ITK Reader 获得更好的兼容性。

## 数据准备

将医学图像按类别放入 `data/` 目录：

```
data/
├── abdomen/           # 腹部图像
│   ├── img001.tif
│   ├── img002.tiff
│   └── ...
├── chest/             # 胸部图像
│   └── ...
├── hand/              # 手部图像
│   └── ...
├── head/              # 头部图像
│   └── ...
└── pelvis/            # 骨盆图像
    └── ...
```

> **注意**：文件夹名称即为类别名称，会自动按字母顺序排列。支持 `.tif`, `.tiff`, `.png`, `.jpg`, `.jpeg` 格式。

## 使用方法

### 1. 训练模型

```bash
# 使用默认配置训练
python train.py

# 自定义训练参数
python train.py --epochs 100 --batch_size 32 --lr 0.0001 --model densenet169

# 使用 EfficientNet
python train.py --model efficientnet-b0

# 冻结骨干网络（仅训练分类头）
python train.py --freeze_backbone

# 禁用 wandb 日志
python train.py --no_wandb

# 禁用混合精度训练
python train.py --no_amp

# 使用 StepLR 调度器
python train.py --scheduler_type step

# 从断点恢复训练
python train.py --resume outputs/checkpoints/last_checkpoint.pth
```

**主要参数：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--epochs` | 50 | 最大训练轮数 |
| `--batch_size` | 16 | 批次大小 |
| `--lr` | 1e-4 | 学习率 |
| `--model` | efficientnet-b0 | 模型架构 |
| `--freeze_backbone` | False | 是否冻结预训练骨干 |
| `--patience` | 10 | 早停耐心值 |
| `--scheduler_type` | cosine | 学习率调度器（`cosine` / `step`） |
| `--grad_clip` | 1.0 | 梯度裁剪最大范数（0 禁用） |
| `--use_amp` | True | 启用自动混合精度 |
| `--no_wandb` | - | 禁用 wandb 日志 |
| `--resume` | None | 断点续训检查点路径 |

**支持的模型架构：**

| 模型 | 说明 |
|------|------|
| `densenet121` | MONAI 医学预训练 DenseNet-121 |
| `densenet169` | DenseNet-169 |
| `densenet201` | DenseNet-201 |
| `efficientnet-b0` ~ `b7` | EfficientNet（预训练） |
| `se_resnet50` | SE-ResNet-50 |

### 2. 评估模型

训练完成后，运行评估脚本生成详细报告和可视化：

```bash
python evaluate.py

# 指定检查点
python evaluate.py --checkpoint outputs/checkpoints/best_model.pth

# 同时记录到 wandb
python evaluate.py --use_wandb
```

输出：
- 分类报告（精确率、召回率、F1 值）
- 混淆矩阵图 (`outputs/confusion_matrix.png`)
- 训练曲线图 (`outputs/training_curves.png`)

### 3. 预测新图像

```bash
# 预测单张图像
python predict.py path/to/image.tif

# 批量预测整个文件夹
python predict.py path/to/image_dir/
```

## Wandb 实验追踪

项目集成了 [Weights & Biases (wandb)](https://wandb.ai/) 进行实验管理和可视化。

### 配置

在 `config.py` 中设置 wandb 参数：

```python
WANDB_ENABLED = True                    # 启用/禁用
WANDB_PROJECT = "DSA-image-classification"  # 项目名
WANDB_ENTITY = None                     # 团队/用户名（None 使用默认）
WANDB_LOG_FREQ = 1                      # 每 N 个 epoch 记录一次
```

### 功能

- **训练指标**：自动记录每个 epoch 的 loss、accuracy、learning rate
- **模型监控**：记录模型梯度和参数分布
- **超参数管理**：自动保存所有训练配置
- **评估结果**：混淆矩阵、分类报告、训练曲线
- **实验对比**：在 wandb 面板中对比不同实验

### 首次使用

```bash
# 登录 wandb（仅需一次）
wandb login

# 或者设置环境变量
export WANDB_API_KEY=your_api_key

# 离线模式（不上传到云端）
export WANDB_MODE=offline
```

## 技术细节

### MONAI 数据管线

项目使用 MONAI 的字典式（dictionary-based）变换管线：

1. **`LoadImaged`** — 加载图像（自动处理 TIFF、PNG、JPEG 等格式）
2. **`EnsureChannelFirstd`** — 保证通道维度在前 `(C, H, W)`
3. **`EnsureSingleChanneld`** — 将多通道图像转换为单通道灰度图
4. **`ScaleIntensityd`** — 强度归一化到 `[0, 1]`
5. **`Resized`** — 统一尺寸到 `224 × 224`
6. 训练时额外应用 nnU-Net 风格的随机增强

### 数据增强策略（nnU-Net 风格）

参考 nnU-Net 的数据增强策略，训练时采用丰富的增强管线：

**空间变换：**
- `RandFlipd` — 随机水平/垂直翻转
- `RandRotated` — 随机旋转（±30°）
- `RandZoomd` — 随机缩放（0.7×–1.4×）
- `RandAffined` — 随机仿射变换（旋转 + 缩放 + 剪切）

**强度变换：**
- `RandGaussianNoised` — 随机高斯噪声
- `RandGaussianSmoothd` — 随机高斯模糊
- `RandAdjustContrastd` — 随机 Gamma 对比度调整
- `RandScaleIntensityd` — 随机强度缩放
- `RandShiftIntensityd` — 随机强度偏移

**正则化：**
- `RandCoarseDropoutd` — 随机粗粒度 Dropout（类似 CutOut）

### CacheDataset 加速

使用 MONAI 的 `CacheDataset` 缓存确定性变换结果（如图像加载、归一化），
随机增强变换每个 epoch 重新计算，兼顾训练速度与数据多样性。

### 训练优化

- **自动混合精度 (AMP)**：使用 `torch.cuda.amp` 加速训练并减少显存占用
- **梯度裁剪**：防止梯度爆炸，默认 max_norm=1.0
- **余弦退火学习率**：nnU-Net 风格的学习率调度，训练更稳定
- **断点续训**：保存完整训练状态（模型 + 优化器 + 调度器），支持中断后继续训练

### 模型架构

- **默认模型**：MONAI EfficientNet-B0（支持预训练权重）
- **输入通道**：1（灰度，适合医学图像）
- **损失函数**：CrossEntropyLoss
- **优化器**：Adam (lr=1e-4, weight_decay=1e-4)
- **学习率调度**：CosineAnnealingLR（默认）或 StepLR

### 防止过拟合

- 迁移学习（MONAI 医学预训练权重）
- nnU-Net 风格数据增强（空间 + 强度 + 正则化）
- Dropout（DenseNet 内置）
- 早停机制
- L2 权重衰减
- 梯度裁剪

## 修改配置

所有超参数集中在 `config.py` 中，可以直接修改或通过命令行参数覆盖。
