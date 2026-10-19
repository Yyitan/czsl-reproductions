# AR 生成式分类器 复现笔记（ar-classifier）

> **文档来源**：官方仓库 [iasudakov/ar-classifier](https://github.com/iasudakov/ar-classifier) README 存档。本合集不收录代码本体（含 DiT / SiT / RandAR 子模块），请从官方仓库 `git clone --recursive` 获取；原文中的示意图 `assets/main_scheme.png` 见上游仓库。
>
> 论文：**Revisiting Autoregressive Models for Generative Image Classification**（ECCV 2026, [arXiv:2603.19122](https://arxiv.org/abs/2603.19122)）

# [ECCV 2026] Revisiting Autoregressive Models for Generative Image Classification
<a href='https://arxiv.org/pdf/2603.19122'><img src='https://img.shields.io/badge/ArXiv-PDF-red'></a> &nbsp;

This is the official implementation of [Revisiting Autoregressive Models for Generative Image Classification](https://arxiv.org/pdf/2603.19122).

## News & Updates
  - [x] **[2026/06/18]** Our paper is accepted to ECCV 2026!
  - [x] **[2026/03/19]** Our code and paper is now available! The paper is released [here](https://arxiv.org/pdf/2603.19122).

## Abstract 

Class-conditional generative models have emerged as accurate and robust classifiers, with diffusion models demonstrating clear advantages over other visual generative paradigms, including autoregressive (AR) models. In this work, we revisit visual AR-based generative classifiers and identify an important limitation of prior approaches: their reliance on a fixed token order, which imposes a restrictive inductive bias for image understanding. We observe that single-order predictions rely more on partial discriminative cues, while averaging over multiple token orders provides a more comprehensive signal. Based on this insight, we leverage recent any-order AR models to estimate order-marginalized predictions, unlocking the high classification potential of AR models. Our approach consistently outperforms diffusion-based classifiers across diverse image classification benchmarks, while being up to 25x more efficient. Compared to state-of-the-art self-supervised discriminative models, our method delivers competitive classification performance - a notable achievement for generative classifiers.

## Installation

1) Clone with submodules:
   ```bash
   git clone https://github.com/iasudakov/ar-classifier.git --recursive
   ```

3) Create the conda environment:
   ```bash
   conda env create -f environment.yml
   ```

   环境要点（`environment.yml`，conda env 名 `genclass`）：`python>=3.8,<3.10`、`pytorch=1.12.1`、torchvision，以及 pip 侧 `timm / accelerate / einops / diffusers / omegaconf` 等。

## Pretrained models

To reproduce the paper results, we provide our RandAR and DiT checkpoints that were pretrained with random crop augmentation for improved classification performance: 

[![HuggingFace](https://img.shields.io/badge/HuggingFace-Model-blue?style=flat&logo=HuggingFace&logoColor=blue)](https://huggingface.co/iasudakov/ar_classifier)

## Order-marginalized AR classifier

### RandAR

```bash
torchrun  --nproc_per_node=N scripts/RandAR_classifier.py  --batch_size=125  --n_trials=20  --n_samples=2  --dataset=X  --gpt_ckpt=/path/to/gpt_ckpt  --vq_ckpt=/path/to/vq_ckpt --config=model_repositories/RandAR/configs/randar/randar_l_0.3b_llamagen.yaml  --imagenet_val_path=/path/to/imagenet/val --imagenet_X_path=/path/to/imagenet-X
```

## Diffusion Classifiers

### DiT
```bash
torchrun --nproc_per_node=N scripts/DiT_classifier.py --batch_size=125  --step_size=4  --n_samples=2  --dataset=X  --dit_ckpt=/path/to/dit_ckpt  --imagenet_val_path=/path/to/imagenet/val --imagenet_X_path=/path/to/imagenet-X
```

### SiT
```bash
torchrun  --nproc_per_node=N scripts/SiT_classifier.py  --batch_size=125  --step_size=4  --n_samples=2  --dataset=X  --sit_ckpt=/path/to/sit_ckpt  --imagenet_val_path=/path/to/imagenet/val --imagenet_X_path=/path/to/imagenet-X
```

### If evaluation takes too long, consider the following options:

1. Parallelize evaluation across multiple devices ( `--nproc_per_node`).

2. Play with the evaluation strategy (`--n_trials` and `--step_size` parameters).

3. Evaluate on a smaller dataset subset. Use `--n_samples` to specify the number of images per class to classify.

## Citation

```bibtex
@article{sudakov2026revisiting,
      title={Revisiting Autoregressive Models for Generative Image Classification}, 
      author={Ilia Sudakov and Artem Babenko and Dmitry Baranchuk},
      year={2026},
      eprint={2603.19122},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2603.19122}, 
}
```
