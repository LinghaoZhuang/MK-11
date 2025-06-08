# MK-11: An Open Bone-Marrow Megakaryocyte Dataset for Automated Morphologic Studies 🩸

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9%2B-orange)](https://pytorch.org/)

> **Authors**: Linghao Zhuang¹, Ying Zhang²³⁴, Xingyue Zhao⁵, and Zhiping Jiang²³⁴  
> ¹School of Software Engineering, Xinjiang University  
> ²Department of Hematology, Xiangya Hospital, Central South University  
> ³National Clinical Research Center for Geriatric Diseases, Xiangya Hospital  
> ⁴Hunan Hematology Oncology Clinical Medical Research Center  
> ⁵School of Software Engineering, Xi'an Jiaotong University

## 🔬 Abstract

Precise classification of megakaryocyte subtypes is not only critical for the diagnosis, stratification, and prognostic assessment of **Myelodysplastic Syndromes (MDS)**, but also significant for research into various platelet-production related disorders; however, high-quality image resources for megakaryocytes with open licensing remain extremely scarce. 

Here we present the **MK-11 dataset**, comprising **7,204 Wright-Giemsa stained single-cell images** covering **11 clinically relevant megakaryocyte subtypes**. The class distribution partially reflects the real-world "long-tail" distribution, making it suitable as a benchmark for classification, severe class imbalance, and few-shot learning tasks.

As the first publicly available megakaryocyte subtype image dataset, MK-11 establishes the foundation for hematopathology and computer-aided diagnosis research in MDS (and related platelet disorders), while also creating opportunities for advanced research topics such as rare subtype detection, transfer learning, and domain adaptation.

## 📋 Table of Contents

- [Dataset Overview](#-dataset-overview)
- [Megakaryocyte Subtypes](#-megakaryocyte-subtypes)
- [Supported Architectures](#-supported-architectures)
- [Installation](#-installation)
- [Usage](#-usage)
- [Baseline Results](#-baseline-results)
- [Data Format](#-data-format)
- [Citation](#-citation)
- [License Information](#-license-information)

## 📊 Dataset Overview

### Key Statistics
- **Total Images**: 7,204 Wright-Giemsa stained single-cell images
- **Subtypes**: 11 clinically relevant megakaryocyte subtypes
- **Source**: 70 MDS patients diagnosed according to WHO-2022 criteria
- **Resolution**: 0.253 μm/pixel (40× magnification)
- **Format**: PNG (24-bit RGB color depth)
- **Annotation**: Expert-verified by certified hematopathologists

### Clinical Significance
- **~90%** of newly diagnosed MDS cases exhibit megakaryocyte dysplasia
- **≥10%** megakaryocyte abnormality proportion constitutes a significant criterion for confirming MDS
- Different subtypes correlate closely with disease progression and prognosis
- Enables standardized automated recognition to improve diagnostic accuracy and consistency

### Research Applications
- ✅ **Classification** tasks with severe class imbalance
- ✅ **Few-shot learning** for rare subtypes (some classes have <200 samples)
- ✅ **Transfer learning** experiments
- ✅ **Domain adaptation** studies
- ✅ **Computer-aided diagnosis** development for MDS

## 🔬 Megakaryocyte Subtypes

The dataset includes 11 mutually exclusive megakaryocyte subtypes based on the International Working Group on Morphology of MDS standards:

| Code | Subtype | Count | Key Features | Clinical Significance |
|------|---------|-------|--------------|----------------------|
| **A** | Megakaryoblast (MK-blast) | 692 | Small cell, primitive chromatin; often CD34⁺ | If total marrow blasts >5% → consider MDS-EB-1/-2 or AML |
| **B** | Promegakaryocyte (Pro-MK) | 506 | Intermediate size; fine chromatin; absent/shallow nuclear lobulation | Reflects ineffective thrombopoiesis |
| **C** | Granular megakaryocyte (G-MK) | 1,954 | Excessive clustering of α/δ granules or hypo-/agranular cytoplasm | Signals megakaryocytic activation or dysfunction |
| **D** | Proplatelet-forming megakaryocyte (PP-MK) | 660 | Visible primitive or budding proplatelet extensions | Typical of reactive thrombocytosis or ITP |
| **E** | Bare (naked-nucleus) megakaryocyte (NN-MK) | 623 | Residual naked nucleus after cytoplasm shed | When >10% fulfils WHO criterion for dysplasia |
| **F** | Normal-sized unilobated megakaryocyte (N-MK) | 548 | Normal diameter; nucleus with <2 lobes | Classic clue to del(5q) syndrome |
| **G** | Small unilobated megakaryocyte (S-MK) | 513 | 10–18 µm; round or single-lobed nucleus | Common in early/low-grade MDS |
| **H** | Micromegakaryocyte (MMK) | 573 | <12 µm, lymphocyte-sized | **Highest specificity for MDS**; ≥25% predicts poor survival |
| **I** | Multinucleated megakaryocyte (MN-MK) | 647 | ≥3 round, non-confluent nuclei | Correlates with high IPSS-R category |
| **J** | Large hyperlobulated megakaryocyte (HL-MK) | 141 | ≥40 µm diameter, >8 lobes or wreath-like nucleus | Associated with clonal expansion |
| **K** | Cytoplasmic abnormalities megakaryocyte (CA-MK) | 347 | Cytoplasmic vacuoles, granule depletion, persistent basophilia | Indicates organelle biogenesis defect |

## 🏗️ Supported Architectures

The framework supports various state-of-the-art architectures with baseline implementations:

<table>
  <tr>
    <th>Model Family</th>
    <th>Available Variants</th>
    <th>Best Performance</th>
  </tr>
  <tr>
    <td>Vision Transformers (ViT)</td>
    <td><code>vit_base_patch16</code></td>
    <td>78.76% ± 0.69% accuracy</td>
  </tr>
  <tr>
    <td>ConvNeXt</td>
    <td><code>convnext_tiny</code>, <code>convnext_small</code>, <code>convnext_base</code></td>
    <td>79.23% ± 0.79% accuracy</td>
  </tr>
  <tr>
    <td>VGG</td>
    <td><code>vgg16</code></td>
    <td>78.48% ± 0.48% accuracy</td>
  </tr>
  <tr>
    <td>EfficientNet</td>
    <td><code>efficientnet_b0</code></td>
    <td>77.73% ± 0.41% accuracy</td>
  </tr>
  <tr>
    <td>ResNet</td>
    <td><code>resnet50</code></td>
    <td>76.18% ± 0.49% accuracy</td>
  </tr>
</table>


## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/LinghaoZhuang/MK-11.git
cd MK-11

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Usage

### Basic Training with 5-Fold Cross-Validation

```bash
# ConvNeXt (Best performing model)
bash script/convnext/fold1.sh
bash script/convnext/fold2.sh
bash script/convnext/fold3.sh
bash script/convnext/fold4.sh
bash script/convnext/fold5.sh

# Vision Transformer
bash script/vit/fold1.sh
bash script/vit/fold2.sh
bash script/vit/fold3.sh
bash script/vit/fold4.sh
bash script/vit/fold5.sh

# ResNet-50
bash script/resnet50/fold1.sh
bash script/resnet50/fold2.sh
bash script/resnet50/fold3.sh
bash script/resnet50/fold4.sh
bash script/resnet50/fold5.sh
```

### Custom Training

```bash
python main_train.py \
  --model_type cnn \
  --model convnext_tiny \
  --batch_size 64 \
  --epochs 50 \
  --data_path ./MK-11-CV5/fold_0 \
  --output_dir ./output_convnext \
  --nb_classes 11 \
  --input_size 224 \
  --drop_path 0.1 \
  --weight_decay 0.05 \
  --blr 1e-3 \
  --min_lr 1e-6 \
  --warmup_epochs 5 \
  --smoothing 0.1 \
  --reprob 0.25
```

## 📈 Baseline Results

### 5-Fold Cross-Validation Performance

| Model | Accuracy (%) | F1-Score (%) | PR-AUC (%) |
|-------|-------------|-------------|------------|
| **ConvNeXt** | **79.23 ± 0.79** | **78.92 ± 0.90** | **83.89 ± 0.69** |
| ViT | 78.76 ± 0.69 | 78.64 ± 0.68 | 84.24 ± 0.90 |
| VGG-16 | 78.48 ± 0.48 | 78.36 ± 0.50 | 82.61 ± 0.55 |
| EfficientNet-B0 | 77.73 ± 0.41 | 77.39 ± 0.42 | 81.70 ± 0.87 |
| ResNet-50 | 76.18 ± 0.49 | 75.77 ± 0.59 | 79.75 ± 1.02 |

*All results obtained using standardized preprocessing (224×224 pixels, ImageNet normalization) with comprehensive data augmentation including AutoAugment, Random Erasing, and Label Smoothing.*

### Training Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| `batch_size` | 64 | Batch size per GPU |
| `epochs` | 50 | Total training epochs |
| `input_size` | 224 | Input image size (pixels) |
| `drop_path` | 0.1 | Drop path rate |
| `weight_decay` | 0.05 | Weight decay |
| `blr` | 1e-3 | Base learning rate |
| `min_lr` | 1e-6 | Minimum learning rate |
| `warmup_epochs` | 5 | Warmup epochs |
| `smoothing` | 0.1 | Label smoothing |
| `reprob` | 0.25 | Random erase probability |

## 📁 Data Format

```
MK-11/
├── MK-blast/           # 692 images
├── Pro-MK/             # 506 images  
├── G-MK/               # 1,954 images
├── PP-MK/              # 660 images
├── NN-MK/              # 623 images
├── N-MK/               # 548 images
├── S-MK/               # 513 images
├── MMK/                # 573 images
├── MN-MK/              # 647 images
├── HL-MK/              # 141 images
└── CA-MK/              # 347 images

MK-11-CV5/              # 5-fold cross-validation splits
├── fold_0/
│   ├── train/
│   ├── val/
│   └── test/
├── fold_1/
├── fold_2/
├── fold_3/
└── fold_4/
```

## 🤝 Contributing

We welcome contributions to improve the dataset and codebase:

- 🐛 **Bug reports** and feature requests via GitHub Issues
- 📝 **Documentation** improvements
- 🔬 **New baseline models** and evaluation metrics
- 🎯 **Clinical validation** studies

## 📜 License Information

This dataset is licensed under the [Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).

### Dataset Availability
The MK-11 dataset is publicly archived on the **Figshare platform** under CC-BY 4.0 license:
- **DOI**: 10.6084/m9.figshare.29264819
- **Format**: Standard compressed format with ImageNet-style organization
- **Access**: Open access for research and educational purposes

### Ethics Statement
All procedures complied with the Declaration of Helsinki and Good Clinical Practice guidelines. The research protocol was approved by the Clinical Research Ethics Committee of Xiangya Hospital, Central South University (Approval No. 2024091075, September 23, 2024).

## 🏥 Clinical Impact

This dataset addresses critical challenges in MDS diagnosis:

- **Standardization**: Reduces inter-observer variability in megakaryocyte classification
- **Efficiency**: Enables automated assessment of bone marrow samples
- **Accuracy**: Improves diagnostic consistency across hospitals and physicians
- **Research**: Facilitates development of computer-aided diagnostic systems

**⚠️ Important**: This dataset is intended for research purposes only and should not be used for clinical diagnosis without proper validation.

---

**📧 Contact**: 

- Linghao Zhuang: [20222501513@xju.edu.cn]
- Zhiping Jiang: [jiangzhp@csu.edu.cn]

**🏥 Institutional Affiliations**:
- Xinjiang University, School of Software Engineering
- Xiangya Hospital, Central South University
- Xi'an Jiaotong University, School of Software Engineering