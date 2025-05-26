# Multi-architecture Image Classification Framework 🚀

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9%2B-orange)](https://pytorch.org/)

> This project is modified from the MaskedAutoEncoder (MAE) project and is licensed under CC BY-NC 4.0.

This framework provides a unified interface for training various classification models, ranging from CNNs to Vision Transformers, on image datasets.

## 📋 Table of Contents

- [Supported Architectures](#-supported-architectures)
- [Usage](#-usage)
- [Important Parameters](#-important-parameters)
- [Examples](#-examples)
- [License Information](#-license-information)

## 🏗️ Supported Architectures

The framework currently supports the following model architectures:

<table>
  <tr>
    <th>Model Family</th>
    <th>Available Variants</th>
  </tr>
  <tr>
    <td>Vision Transformers (ViT)</td>
    <td>
      <code>vit_base_patch16</code> - ViT-Base (768 dim, 12 layers, 12 heads)<br>
      <code>vit_large_patch16</code> - ViT-Large (1024 dim, 24 layers, 16 heads)<br>
      <code>vit_huge_patch14</code> - ViT-Huge (1280 dim, 32 layers, 16 heads)
    </td>
  </tr>
  <tr>
    <td>ResNet</td>
    <td>
      <code>resnet18</code>, <code>resnet34</code>, <code>resnet50</code>, <code>resnet101</code>, <code>resnet152</code>
    </td>
  </tr>
  <tr>
    <td>VGG</td>
    <td>
      <code>vgg16</code>, <code>vgg19</code>
    </td>
  </tr>
  <tr>
    <td>MobileNet</td>
    <td>
      <code>mobilenet_v3_small</code>, <code>mobilenet_v3_large</code>
    </td>
  </tr>
  <tr>
    <td>EfficientNet</td>
    <td>
      <code>efficientnet_b0</code>, <code>efficientnet_b1</code>, <code>efficientnet_b2</code>, <code>efficientnet_b3</code>
    </td>
  </tr>
  <tr>
    <td>RegNet</td>
    <td>
      <code>regnet_y_400mf</code>, <code>regnet_y_8gf</code>
    </td>
  </tr>
  <tr>
    <td>DenseNet</td>
    <td>
      <code>densenet121</code>, <code>densenet169</code>, <code>densenet201</code>
    </td>
  </tr>
</table>

## 🚀 Usage

### Basic Usage

```bash
python main_train.py \
  --model_type resnet \
  --model resnet50 \
  --batch_size 128 \
  --epochs 100 \
  --data_path /path/to/dataset \
  --output_dir ./output_resnet50
```

### Using Pre-trained Models

Use the `--pretrained` flag to initialize the model with pre-trained weights from timm:

```bash
python main_train.py \
  --model_type resnet \
  --model resnet50 \
  --pretrained \
  --batch_size 128 \
  --epochs 50 \
  --data_path /path/to/dataset \
  --output_dir ./output_resnet50_pretrained
```

This loads ImageNet pre-trained weights for the backbone while initializing a new classification head for your specific task.

### Training with Multiple GPUs

```bash
python -m torch.distributed.launch --nproc_per_node=8 main_train.py \
  --model_type vit \
  --model vit_base_patch16 \
  --batch_size 64 \
  --epochs 100 \
  --data_path /path/to/dataset \
  --output_dir ./output_vit_base
```

### Using Data Augmentation with Mixup and CutMix

```bash
python main_train.py \
  --model_type efficientnet \
  --model efficientnet_b0 \
  --batch_size 128 \
  --epochs 100 \
  --mixup 0.8 \
  --cutmix 1.0 \
  --data_path /path/to/dataset \
  --output_dir ./output_efficientnet_b0
```

### Resuming Training

```bash
python main_train.py \
  --model_type resnet \
  --model resnet50 \
  --batch_size 128 \
  --resume /path/to/checkpoint.pth \
  --data_path /path/to/dataset \
  --output_dir ./output_resnet50
```

## ⚙️ Important Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--model_type` | Type of model architecture | - |
| `--model` | Specific model variant | - |
| `--pretrained` | Use pre-trained weights from timm | `False` |
| `--batch_size` | Batch size per GPU | - |
| `--epochs` | Number of training epochs | - |
| `--lr` | Learning rate | Computed from base LR |
| `--blr` | Base learning rate | `1e-3` |
| `--weight_decay` | Weight decay | `0.05` |
| `--input_size` | Input image size | `224` |
| `--mixup` | Mixup alpha | `0` (disabled) |
| `--cutmix` | CutMix alpha | `0` (disabled) |
| `--data_path` | Path to dataset | - |
| `--output_dir` | Directory for saving outputs | - |
| `--resume` | Resume from a checkpoint | - |

## 📊 Examples

### Fine-tuning a Pre-trained ResNet-50 on a Custom Dataset

```bash
python main_train.py \
  --model_type resnet \
  --model resnet50 \
  --pretrained \
  --batch_size 128 \
  --epochs 30 \
  --warmup_epochs 3 \
  --blr 5e-4 \
  --weight_decay 1e-5 \
  --nb_classes 10 \
  --data_path /path/to/custom_dataset \
  --output_dir ./output_resnet50_finetune
```

### Training ViT-Base from Scratch on ImageNet

```bash
python -m torch.distributed.launch --nproc_per_node=8 main_train.py \
  --model_type vit \
  --model vit_base_patch16 \
  --batch_size 64 \
  --epochs 300 \
  --warmup_epochs 20 \
  --blr 1e-3 \
  --weight_decay 0.05 \
  --mixup 0.8 \
  --cutmix 1.0 \
  --data_path /path/to/imagenet \
  --output_dir ./output_vit_base
```

### Training ResNet-50 from Scratch on ImageNet

```bash
python -m torch.distributed.launch --nproc_per_node=8 main_train.py \
  --model_type resnet \
  --model resnet50 \
  --batch_size 256 \
  --epochs 100 \
  --warmup_epochs 5 \
  --blr 1e-3 \
  --weight_decay 1e-4 \
  --mixup 0.2 \
  --cutmix 0.0 \
  --data_path /path/to/imagenet \
  --output_dir ./output_resnet50
```

### Transfer Learning with Pre-trained EfficientNet-B0 on a Small Dataset

```bash
python main_train.py \
  --model_type efficientnet \
  --model efficientnet_b0 \
  --pretrained \
  --batch_size 64 \
  --epochs 20 \
  --warmup_epochs 2 \
  --blr 3e-4 \
  --weight_decay 1e-5 \
  --nb_classes 5 \
  --data_path /path/to/small_dataset \
  --output_dir ./output_efficientnet_b0_transfer
```

## 📜 License Information

This project is licensed under the [Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/).

### License Requirements:
- **Attribution**: You must give appropriate credit, provide a link to the license, and indicate if changes were made.
- **NonCommercial**: You may not use the material for commercial purposes.
- **No additional restrictions**: You may not apply legal terms or technological measures that legally restrict others from doing anything the license permits.

### Original Project
This project is modified from [MaskedAutoEncoder](https://github.com/facebookresearch/mae). 