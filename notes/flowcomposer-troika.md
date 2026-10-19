# FlowComposer + Troika（复现笔记）

> **文档来源**：复现者（Yyitan）自写的复现说明（原 `FlowComposer-Troika/README.md`）。该复现目录是在 `Troika-master/code` 副本上叠加 FlowComposer 模块，代码未随本合集发布；Troika 基线代码见官方仓库 [bighuang624/troika](https://github.com/bighuang624/troika)。
>
> 复现对象：**FlowComposer: Composable Flows for Compositional Zero-Shot Learning**（CVPR 2026, [arXiv:2603.16641](https://arxiv.org/abs/2603.16641)），挂载于 **Troika**（CVPR 2024）基线之上。

Reproduction of **FlowComposer: Composable Flows for Compositional
Zero-Shot Learning** (CVPR2026, arXiv:2603.16641) plugged into the
**Troika** (CVPR2024) baseline. The reproduction directory was a copy of
`Troika-master/code` with the FlowComposer module added on top; nothing in
the original Troika code path was removed, so `--model_name troika` still
runs the vanilla baseline from this folder too.

## What's new vs. Troika

- `model/flow_modules.py` — `FlowNet` (adaLN-modulated residual-MLP flow
  network, MAR/DiT-style, with timestep conditioning), `ComposerNet`
  (LayerNorm+GELU residual MLP predicting combination coefficients), and
  `solve_composer_target` (closed-form least-squares target for the
  Composer).
- `model/troika_flowcomposer.py` — `TroikaFlowComposer(Troika)`: adds two
  primitive flows (attribute/object), the Composer, and leakage-guided
  augmentation on top of Troika's existing three branches, following
  Sec. 4 of the paper.
- `model/model_factory.py` — registers `model_name: troika_flowcomposer`.
- `parameters.py` — new `--flow_*`, `--composer_*`, `--comp_step_h`,
  `--*_loss_weight` args (see file for defaults).
- `config/troika_flowcomposer/{mit-states,ut-zappos,cgqa}.yml` — same
  training/eval hyperparameters as the original Troika configs, plus the
  FlowComposer block.
- `smoke_test.py` — a tiny end-to-end sanity check (toy vocab, real CLIP
  weights) that exercises forward/backward/eval; useful after touching the
  model code.

## Design choices not pinned down by the paper

The paper (Sec. 4, Supp. Sec. 7) specifies the flow/Composer/leakage
formulas precisely, but leaves two things implicit that had to be decided
to get a runnable implementation:

1. **x0/x1 feature source.** We reuse exactly the visual/text features
   Troika already computes per branch (no new encoder calls):
   `x0^a/x0^o` = attribute/object disentangler outputs, `x0^c` = raw CLIP
   image feature; `x1^a/x1^o/x1^c` = the corresponding per-class text
   embeddings *before* Troika's CMT cross-attention refinement (i.e. the
   "encode_text" output Troika computes anyway inside `forward`).
2. **Fusion of the transported composition embedding into the final
   score.** The paper states FlowComposer "tightens the distance between
   composed text and image embeddings" but gives no fusion equation. We
   add the flow-based cosine score as an auxiliary term on Troika's native
   composition logits:
   `comp_logits = comp_logits_troika + flow_logit_weight * cos(x_hat_c1, text_c1) * logit_scale`,
   mirroring Troika's own `pc + pa*po` score-fusion philosophy
   (`logit_infer` in `model/troika.py` is untouched and runs on top of
   this already-fused `comp_logits`).
3. Composer's primitive-velocity inputs are taken at `t=0` (matching the
   paper's one-step inference scheme) and detached before the Composer
   loss, so the Composer's regression target doesn't backprop into the
   flow nets — not stated explicitly in the paper, added for training
   stability.

Everything else (the rectified-flow MSE+CE loss per branch, the
leakage-guided augmentation sampling `j != i` from the other two branches,
the least-squares Composer target, the one-step transport
`x_hat_c1 = x0^c + h * v_hat_c`) follows the paper's equations directly.

## Compute note

`flow_depth: 24` (the paper's MAR-style 24-ResBlock flow net) adds ~70M
trainable params *per* primitive flow (two flows + Composer ≈ +150-170M
on top of Troika's adapters). If you hit OOM or training is too slow,
lower `flow_depth` (e.g. 6-8) and/or `flow_hidden_dim` in the yml configs
— this only trades off fidelity to the paper's reported model size
(Table 7: 74M/flow, 26M Composer), not correctness of the method.

## Running

Same CLI as Troika, just point `--yml_path` / `model_name` at the new
config:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
    --yml_path config/troika_flowcomposer/ut-zappos.yml \
    --dataset_path <DATASET_ROOT>/ut-zap50k \
    --clip_arch <CLIP_MODEL_ROOT>/ViT-L-14.pt \
    --save_path ./output/ut-zappos-flowcomposer \
    --num_workers 10 \
    --seed 0
```

Evaluation is unchanged from Troika (`test.py`), since
`TroikaFlowComposer.forward` / `forward_for_open` return the same
`[comp_logits, attr_logits, obj_logits]` list shape that `test.py` and
`logit_infer` expect.

## Leakage-guided augmentation only applies to multi-path baselines

Per the paper (Sec. 5.1), leakage-guided augmentation requires
branch-wise visual disentanglement, so it is only meaningful for
multi-path baselines like Troika (not single-path ones like CSP). This
reproduction only targets Troika, consistent with that.
