# H2EM 复现（基于 Troika）

> **文档来源**：复现者（Yyitan）自写的复现说明（原 `H2EM/README_H2EM.md`）。该复现以 Troika (CVPR'24) 官方代码为基线改造，代码未随本合集发布；Troika 基线见官方仓库 [bighuang624/troika](https://github.com/bighuang624/troika)。
>
> 复现对象：**H2EM: Learning Hierarchical Hyperbolic Embeddings for Compositional Zero-Shot Learning**（[arXiv:2512.20029](https://arxiv.org/abs/2512.20029)）

## 方法 → 代码 对照

| 论文模块 | 实现位置 |
|---|---|
| 双曲几何（Lorentz model, exp map, 测地距离, 蕴含锥, 双曲线性层/激活, Lorentz 残差加法） | `model/hyperbolic.py` |
| H2EM 主模型（三路视觉 + 5 类文本 + 投影 + 分类 + 三损失） | `model/h2em.py` |
| HCA 双曲跨模态注意力（Eq.5-7，双曲线性注意力 + FFN + Lorentz 残差） | `model/h2em.py: HyperbolicCrossModalAttention` |
| 三层语义树（LLM 生成的父类 hypernym） | `taxonomy/build_taxonomy.py` → `taxonomy/taxonomy_<dataset>.json` |
| L_DA 判别对齐（带难负样本挖掘的加权 InfoNCE，Eq.11） | `h2em.py: _discriminative_alignment` |
| L_TE 分类蕴含损失（双层级 6 条边，Eq.9-10） | `h2em.py: _taxonomic_entailment` + `hyperbolic.entailment_cone_loss` |
| L_aux 原语辅助对比损失（Eq.12） | `h2em.py: loss_calu` |
| 推理融合 ĉ=argmax p(c)+p(s)p(o)（Eq.8） | `h2em.py: logit_infer` |

### 相对 Troika 的关键改动
- 文本分支由 3 类（comp/attr/obj）扩展到 **5 类**，新增 parent-state、parent-obj。
- Troika 的欧氏 CMT 跨模态模块替换为 **HCA**（双曲）。
- 所有特征经 **exp map 投影到 Lorentz 流形**（可学习曲率 κ，exp 参数化保证 >0）。
  欧氏特征先 L2 归一化再乘可学习半径 `entry_scale`，保证数值稳定。
- 分类 logits = **−测地距离 × dist_scale**（替代 cosine 相似度）。
- 损失：`L = β1·L_DA + β2·L_TE + β3·L_aux`，默认 β1=1.0, β2=0.1, β3=0.5, 难负样本权重 w=3.0。

## 运行

训练（以 UT-Zappos 为例；MIT-States / CGQA 把 ut-zappos 换成对应名即可）：
```bash
CUDA_VISIBLE_DEVICES=0 python -u train.py \
  --yml_path config/h2em/ut-zappos.yml \
  --dataset_path <DATASET_ROOT>/ut-zappos \
  --clip_arch <CLIP_MODEL_ROOT>/ViT-L-14.pt \
  --save_path output/ut --num_workers 8 --seed 0
```

闭世界评测：
```bash
python -u test.py --yml_path config/h2em/ut-zappos.yml \
  --dataset_path <DATASET_ROOT>/ut-zappos \
  --clip_arch <CLIP_MODEL_ROOT>/ViT-L-14.pt \
  --save_path output/ut --load_model output/ut/val_best.pt --num_workers 8 --seed 0
```

数据集路径：`mit-states` / `ut-zappos` / `cgqa` 均置于 `<DATASET_ROOT>/<name>`。

## 超参（论文 §A）
ViT-L/14，768 维，Adam，lr=1e-4，batch=64，context length=8，HCA 12 头，
κ 初始 1.0（可学习），γ=0.1，τ 用 CLIP logit scale。

## taxonomy 说明
父类用离线规则（关键词桶 + 显式覆盖，`build_taxonomy.py`）生成，替代论文中的
Gemini-2.5 Pro，保证可复现、零外部依赖。MIT-States / UT-Zappos 覆盖完整；
CGQA 词汇极杂，部分原语归入通用父节点（β2=0.1，影响有限）。如需更精细的
父类，替换 `taxonomy/taxonomy_<dataset>.json` 即可（格式：
`{"attr_parent": {state: parent}, "obj_parent": {object: parent}}`）。

## 已知限制
- 开放世界（open-world）：HCA 使文本特征依赖图像，超大 pair 集合（MIT/CGQA 全空间
  ~28k/278k）需要按 chunk 缓存文本特征。当前配置 `text_first: False`，闭世界完整可用，
  UT-Zappos 开放世界可跑；MIT/CGQA 的开放世界大规模评测需后续实现 chunked text-first 路径。
- 可训练参数约 14.6M（与 Troika 同量级；与论文 21.8M 的差异源于 HCA 与 Troika CMT
  的模块大小不同，不影响方法忠实度）。
