[English](README.md) | 简体中文

# CZSL 方法复现合集（Compositional Zero-Shot Learning Reproductions）

作者：**Yyitan**（GitHub: [@Yyitan](https://github.com/Yyitan)）

这是研一阶段对**组合零样本学习（Compositional Zero-Shot Learning, CZSL）**多个主流方法的系统性复现合集，目标是为后续研究建立**可靠、可对照的实验基线**：同一套数据（MIT-States / UT-Zappos / C-GQA）、同一套闭世界与开世界评测协议，逐个方法对齐论文报告指标后再在此之上做改进实验。

**本仓库的性质**：这里**只放说明文档、复现笔记与上游协议文本**，不收录任何第三方代码本体——所有代码均指向各方法官方仓库，复现时在本地独立目录中进行。各方法的完整说明见 [`notes/`](notes/) 目录，上游协议文本见 [`licenses/`](licenses/) 目录。

## 复现一览表

| 方法 | 论文 / 会议 | 协议 | 复现状态 | 备注 |
|---|---|---|---|---|
| Troika | [Troika: Multi-Path Cross-Modal Traction for CZSL](https://github.com/bighuang624/troika)（CVPR 2024） | MIT | 已复现：MIT-States / UT-Zappos / C-GQA 三数据集训练 + 闭世界/开世界评测 | 本合集的核心基线；ViT-L/14，后续两项自研复现均挂载其上 |
| CAILA | [CAILA: Concept-Aware Intra-Layer Adapters for CZSL](https://arxiv.org/abs/2305.16681)（WACV 2024） | MIT | 已复现：按官方配置多卡训练与评测，并与官方 checkpoint 对齐 | CLIP ViT-B/32 与 ViT-L/14 两档；含 VAW-CZSL 配置 |
| PLID | [Prompting Language-Informed Distribution for CZSL](https://arxiv.org/abs/2305.14428)（ECCV 2024） | BSD-3-Clause | 调研 / 文档阶段（尚未跑通） | LLM 生成文本 + 高斯分布提示；依赖上游 Gdrive 的文本特征与 GloVe 可行性得分 |
| FlowComposer（挂载 Troika） | [FlowComposer: Composable Flows for CZSL](https://arxiv.org/abs/2603.16641)（CVPR 2026） | MIT（随 Troika 基线） | 已实现并训练：基于 Troika 官方代码叠加整流流模块的自研复现 | 论文未提供官方代码；笔记中完整记录了论文未明确的实现决策 |
| H2EM（挂载 Troika） | [H2EM: Learning Hierarchical Hyperbolic Embeddings for CZSL](https://arxiv.org/abs/2512.20029)（arXiv） | MIT（随 Troika 基线） | 已实现：闭世界完整可用，UT-Zappos 开放世界可跑；MIT/CGQA 开放世界待完善 | 双曲（Lorentz）改造 Troika；语义树用离线规则替代论文的 LLM 生成以保证可复现 |
| ar-classifier | [Revisiting Autoregressive Models for Generative Image Classification](https://arxiv.org/abs/2603.19122)（ECCV 2026） | 未附 LICENSE（见上游） | 已复现：按官方流程搭建环境并跑通分类评测 | 生成式图像分类（AR / 扩散），非 CZSL，作为相邻方向复现收录 |

> "复现状态"以实际跑通程度为准；标注"挂载 Troika"的两项为论文官方未开源（或未完全开源）时按论文公式在 Troika 基线上的自研实现。

## 各方法说明

### 1. Troika（CVPR 2024）— 核心基线

- **要点**：在 CLIP ViT-L/14 上做多路径跨模态牵引——属性/物体/组合三路解耦 + CMT 跨模态交互 + 特征增强，开世界评测用 DFSP 式 feasibility calibration。
- **复现要点**：Python 3.7 + PyTorch（cu113）；CLIP 权重 `ViT-L-14.pt` 需手动下载；数据 `mit-states / ut-zappos / cgqa`；`config/troika/<dataset>.yml` 提供全套超参；闭/开世界分别用 `<dataset>.yml` 与 `<dataset>-ow.yml` 评测。
- **上游**：<https://github.com/bighuang624/troika>（MIT，见 [`licenses/troika-LICENSE`](licenses/troika-LICENSE)）
- **笔记**：[`notes/troika.md`](notes/troika.md)
- 致谢引用（BibTeX）：

```bibtex
@inproceedings{Huang2024Troika,
    title={Troika: Multi-Path Cross-Modal Traction for Compositional Zero-Shot Learning},
    author={Siteng Huang and Biao Gong and Yutong Feng and Min Zhang and Yiliang Lv and Donglin Wang},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    year={2024}
}
```

### 2. CAILA（WACV 2024）

- **要点**：概念感知的层内适配器（Concept-Aware Intra-Layer Adapters），把组合概念先验注入 CLIP 的 adapter 层；官方提供 MIT-States / C-GQA / UT-Zappos / VAW-CZSL 四套 checkpoint（AUC Base/Large）。
- **复现要点**：Python 3.8 + PyTorch 1.13，`torchrun` 多卡训练；CLIP 权重从 HuggingFace 取 ViT-B/32 与 ViT-L/14；数据目录结构与 `flags.py` 中的 `DATA_FOLDER` 需按官方约定配置。
- **上游**：<https://github.com/zhaohengz/CAILA>（MIT，见 [`licenses/caila-LICENSE`](licenses/caila-LICENSE)）
- **笔记**：[`notes/caila.md`](notes/caila.md)
- 致谢引用（BibTeX）：

```bibtex
@InProceedings{Zheng_2024_WACV,
    author    = {Zheng, Zhaoheng and Zhu, Haidong and Nevatia, Ram},
    title     = {CAILA: Concept-Aware Intra-Layer Adapters for Compositional Zero-Shot Learning},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month     = {January},
    year      = {2024},
    pages     = {1721-1731}
}
```

### 3. PLID（ECCV 2024）

- **要点**：基于 CLIP，用 LLM（默认 OPT-1.5B）生成语言信息丰富的描述，以高斯分布形式构造多样提示（language-informed distribution），支持闭世界与开世界。
- **复现要点**：Python 3.7 + CLIP；除原始数据外还需上游 Gdrive 提供的 LLM 文本特征（`opt_xxx.pkl`）与 GloVe feasibility 得分；训练/评测脚本在 `exp/<dataset>/` 下。
- **上游**：<https://github.com/Cogito2012/PLID>（BSD-3-Clause，见 [`licenses/plid-LICENSE`](licenses/plid-LICENSE)）
- **笔记**：[`notes/plid.md`](notes/plid.md)
- 致谢引用（BibTeX）：

```bibtex
@InProceedings{bao2023eccv24,
  title={Prompting Language-Informed Distribution for Compositional Zero-Shot Learning},
  author={Bao, Wentao and Chen, Lichang and Huang, Heng and Kong, Yu},
  booktitle = {Proceedings of the European Conference on Computer Vision (ECCV)},
  year={2024}
}
```

### 4. FlowComposer + Troika（CVPR 2026，自研复现）

- **要点**：把整流流（rectified flow）引入 CZSL——属性/物体两路 primitive flow（MAR/DiT 风格 adaLN 残差 MLP）、Composer 预测组合系数（最小二乘闭式目标）、泄漏引导增强，一步传输 `x_hat_c1 = x0^c + h * v_hat_c`；挂载在 Troika 三路分支之上。
- **复现要点**：与 Troika 同环境；`flow_depth: 24` 时每路 flow 约 +70M 参数（合计 +150~170M），OOM 时可降 `flow_depth` / `flow_hidden_dim`。笔记中记录了论文未明确的三处实现决策（x0/x1 特征来源、融合打分公式、Composer 输入 detach）。
- **上游**：论文 [arXiv:2603.16641](https://arxiv.org/abs/2603.16641)（未见官方代码）；基线 Troika [bighuang624/troika](https://github.com/bighuang624/troika)（MIT）
- **笔记**：[`notes/flowcomposer-troika.md`](notes/flowcomposer-troika.md)

### 5. H2EM（arXiv，基于 Troika 的自研复现）

- **要点**：层级双曲嵌入——把 Troika 的欧氏 CMT 替换为双曲跨模态注意力 HCA（Lorentz 流形、可学习曲率），文本分支扩到 5 类（含 LLM 生成的父类语义树），损失 = 判别对齐 L_DA + 分类蕴含 L_TE + 原语辅助 L_aux。
- **复现要点**：与 Troika 同环境；可训练参数约 14.6M；语义树由离线规则（`build_taxonomy.py`）生成，替代论文的 Gemini-2.5 Pro，零外部依赖；闭世界完整可用，UT-Zappos 开放世界可跑，MIT/CGQA 开放世界需 chunked text-first 路径（见笔记"已知限制"）。
- **上游**：论文 [arXiv:2512.20029](https://arxiv.org/abs/2512.20029)（未见官方代码）；基线 Troika [bighuang624/troika](https://github.com/bighuang624/troika)（MIT）
- **笔记**：[`notes/h2em.md`](notes/h2em.md)

### 6. ar-classifier（ECCV 2026，相邻方向）

- **要点**：重访视觉自回归生成式分类器——用 any-order AR（RandAR）对多个 token 序做多序边缘化预测，超过扩散分类器（DiT/SiT）且最高省 25 倍算力。
- **复现要点**：conda env `genclass`（`python<3.10`、`pytorch=1.12.1`、timm/accelerate/diffusers 等）；需 `git clone --recursive` 拉取 DiT / SiT / RandAR 子模块；官方 checkpoint 在 HuggingFace [iasudakov/ar_classifier](https://huggingface.co/iasudakov/ar_classifier)；评测需 ImageNet val 与 imagenet-X，可用 `--n_trials` / `--step_size` / `--n_samples` 控制开销。
- **上游**：<https://github.com/iasudakov/ar-classifier>（论文 [arXiv:2603.19122](https://arxiv.org/abs/2603.19122)）
- **笔记**：[`notes/ar-classifier.md`](notes/ar-classifier.md)
- 致谢引用（BibTeX）：

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

## 目录结构

```
czsl-reproductions/
├── README.md            # 本文件：总览 + 一览表 + 分方法说明
├── notes/               # 各方法的说明文档与复现笔记（含官方 README 存档）
│   ├── troika.md
│   ├── caila.md
│   ├── plid.md
│   ├── flowcomposer-troika.md
│   ├── h2em.md
│   └── ar-classifier.md
└── licenses/            # 各上游项目的协议文本
    ├── troika-LICENSE   # MIT
    ├── caila-LICENSE    # MIT
    └── plid-LICENSE     # BSD 3-Clause（取自上游仓库）
```

## 致谢

感谢各方法作者开源代码与实验细节：Troika（[bighuang624/troika](https://github.com/bighuang624/troika)）、CAILA（[zhaohengz/CAILA](https://github.com/zhaohengz/CAILA)）、PLID（[Cogito2012/PLID](https://github.com/Cogito2012/PLID)）、ar-classifier（[iasudakov/ar-classifier](https://github.com/iasudakov/ar-classifier)），以及 FlowComposer 与 H2EM 的论文作者。本合集的复现工作分别遵循各上游项目的开源协议（见 [`licenses/`](licenses/)）。
