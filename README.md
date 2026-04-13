# X-ray 图像分类项目

基于 PyTorch 和迁移学习 (ResNet) 的 X-ray 医学图像分类系统，支持 TIFF 格式图像。

## 项目结构

```
├── config.py          # 项目配置（路径、超参数等）
├── dataset.py         # 数据集加载与预处理（支持 TIFF）
├── transforms.py      # 数据增强策略
├── model.py           # 模型定义（ResNet 迁移学习）
├── train.py           # 训练脚本
├── evaluate.py        # 评估脚本（指标 + 可视化）
├── predict.py         # 单张/批量推理脚本
├── requirements.txt   # Python 依赖
├── data/              # 数据目录（不纳入版本控制）
│   ├── class_1/       # 第1类图像文件夹
│   ├── class_2/       # 第2类图像文件夹
│   ├── class_3/       # 第3类图像文件夹
│   ├── class_4/       # 第4类图像文件夹
│   └── class_5/       # 第5类图像文件夹
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

## 数据准备

将 X-ray TIFF 图像按类别放入 `data/` 目录：

```
data/
├── abdomen/           # 腹部 X-ray
│   ├── img001.tif
│   ├── img002.tiff
│   └── ...
├── chest/             # 胸部 X-ray
│   └── ...
├── hand/              # 手部 X-ray
│   └── ...
├── head/              # 头部 X-ray
│   └── ...
└── pelvis/            # 骨盆 X-ray
    └── ...
```

> **注意**：文件夹名称即为类别名称，会自动按字母顺序排列。支持 `.tif`, `.tiff`, `.png`, `.jpg`, `.jpeg` 格式。

## 使用方法

### 1. 训练模型

```bash
# 使用默认配置训练
python train.py

# 自定义训练参数
python train.py --epochs 100 --batch_size 32 --lr 0.0001 --model resnet50

# 冻结骨干网络（仅训练分类头）
python train.py --freeze_backbone
```

**主要参数：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--epochs` | 50 | 最大训练轮数 |
| `--batch_size` | 16 | 批次大小 |
| `--lr` | 1e-4 | 学习率 |
| `--model` | resnet18 | 模型架构 (resnet18/34/50) |
| `--freeze_backbone` | False | 是否冻结预训练骨干 |
| `--patience` | 10 | 早停耐心值 |

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

### 数据增强策略

由于数据量较小（每类约 100 张），训练时采用较强的数据增强：
- 随机水平/垂直翻转
- 随机旋转（±15°）
- 随机仿射变换（平移 + 缩放）
- 亮度和对比度抖动
- 随机灰度化

### 模型架构

- **骨干网络**：ImageNet 预训练的 ResNet
- **分类头**：Dropout(0.5) → FC(256) → ReLU → Dropout(0.3) → FC(num_classes)
- **损失函数**：CrossEntropyLoss
- **优化器**：Adam (lr=1e-4, weight_decay=1e-4)
- **学习率调度**：StepLR（每 10 轮衰减 0.5）

### TIFF 图像处理

项目使用 `tifffile` 库读取 TIFF 格式图像，自动处理：
- 不同位深（8-bit、16-bit、float）
- 多页 TIFF
- 灰度到 RGB 的转换

### 防止过拟合

- 迁移学习（利用 ImageNet 预训练知识）
- 数据增强
- Dropout (0.3–0.5)
- 早停机制
- L2 权重衰减

## 修改配置

所有超参数集中在 `config.py` 中，可以直接修改或通过命令行参数覆盖。
