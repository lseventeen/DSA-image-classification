# 医学图像分类项目（MONAI）

基于 [MONAI](https://monai.io/) 和迁移学习的医学图像分类系统，支持 TIFF / PNG / JPEG 格式图像。

MONAI（Medical Open Network for Artificial Intelligence）是专为医学影像 AI 研发打造的 PyTorch 框架，
提供医学图像专用的数据增强、网络架构和训练工具。

## 项目结构

```
├── config.py          # 项目配置（路径、超参数等）
├── dataset.py         # MONAI CacheDataset + 数据加载
├── transforms.py      # MONAI 字典式医学图像变换
├── model.py           # MONAI 网络架构（DenseNet / EfficientNet / SE-ResNet）
├── train.py           # 训练脚本
├── evaluate.py        # 评估脚本（指标 + 可视化）
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
# 使用默认配置训练（DenseNet121）
python train.py

# 自定义训练参数
python train.py --epochs 100 --batch_size 32 --lr 0.0001 --model densenet169

# 使用 EfficientNet
python train.py --model efficientnet-b0

# 冻结骨干网络（仅训练分类头）
python train.py --freeze_backbone
```

**主要参数：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--epochs` | 50 | 最大训练轮数 |
| `--batch_size` | 16 | 批次大小 |
| `--lr` | 1e-4 | 学习率 |
| `--model` | densenet121 | 模型架构 |
| `--freeze_backbone` | False | 是否冻结预训练骨干 |
| `--patience` | 10 | 早停耐心值 |

**支持的模型架构：**

| 模型 | 说明 |
|------|------|
| `densenet121` | MONAI 医学预训练 DenseNet-121（默认） |
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

## 技术细节

### MONAI 数据管线

项目使用 MONAI 的字典式（dictionary-based）变换管线：

1. **`LoadImaged`** — 加载图像（自动处理 TIFF、PNG、JPEG 等格式）
2. **`EnsureChannelFirstd`** — 保证通道维度在前 `(C, H, W)`
3. **`EnsureSingleChanneld`** — 将多通道图像转换为单通道灰度图
4. **`ScaleIntensityd`** — 强度归一化到 `[0, 1]`
5. **`Resized`** — 统一尺寸到 `224 × 224`
6. 训练时额外应用随机增强（翻转、旋转、缩放、噪声、对比度调整）

### 数据增强策略

由于医学数据量通常较小，训练时采用丰富的数据增强（MONAI 提供）：
- `RandFlipd` — 随机水平/垂直翻转
- `RandRotated` — 随机旋转（±15°）
- `RandZoomd` — 随机缩放（0.9×–1.1×）
- `RandGaussianNoised` — 随机高斯噪声
- `RandAdjustContrastd` — 随机对比度调整

### CacheDataset 加速

使用 MONAI 的 `CacheDataset` 缓存确定性变换结果（如图像加载、归一化），
随机增强变换每个 epoch 重新计算，兼顾训练速度与数据多样性。

### 模型架构

- **默认模型**：MONAI DenseNet-121（支持医学图像预训练权重）
- **输入通道**：1（灰度，适合医学图像）
- **损失函数**：CrossEntropyLoss
- **优化器**：Adam (lr=1e-4, weight_decay=1e-4)
- **学习率调度**：StepLR（每 10 轮衰减 0.5）

### 防止过拟合

- 迁移学习（MONAI 医学预训练权重）
- MONAI 医学专用数据增强
- Dropout（DenseNet 内置）
- 早停机制
- L2 权重衰减

## 修改配置

所有超参数集中在 `config.py` 中，可以直接修改或通过命令行参数覆盖。
