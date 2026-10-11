# GLASS-j-Reevaluation

Independent re-evaluation of **officially released GLASS-j checkpoints on MVTec AD**, with unified metrics and an optional matched-field-of-view comparison.

This repository contains code only: no datasets, model weights, predictions, training logs, or experimental result reports. The original MVTec AD entry evaluates released checkpoints; the independent VisA entry trains official GLASS-j and evaluates the fixed final epoch. Published MVTec checkpoints are not claimed to be fixed-final-epoch models.

## What is included

- Pinned official GLASS implementation (MIT), commit `f788d567db552820d744ca2d93e989265aad8c45`.
- Explicit official checkpoint download identities and strict loading of discriminator, projection and BatchNorm state.
- Independent `download`, `check`, `predict`, `evaluate`, and optional `compare` commands.
- Official RGB preprocessing, including the toothbrush/wood resize rule; nearest-neighbor binary GT in the same 288×288 field of view.
- Image AUROC/AP from the original GLASS image scores; Pixel AUROC/AP, fast CUDA AUPRO@0.3, and fixed-FPR coverage.
- Per-image lossless float32 predictions, original image scores, binary GT, geometry, configuration and resource records when users run the code.

The retained `DINOv3/` namespace contains only the small shared data/metric/I/O support and tests needed by this evaluator. It does not load a DINO model, retrieve a memory bank, or include the wider experimental workspace. Shared helper function bodies were exported unchanged; see [CODE_PROVENANCE.json](CODE_PROVENANCE.json).

## VisA official training with fixed-final reevaluation

The independent `external_baselines/glass_visa/` entry supports `check`, `smoke`, `train`, `evaluate` and `summarize`. It reuses the official network, foreground-based synthesis, optimizers and epoch loop. It reads the raw VisA `split_csv/1cls.csv`, uses all normal training images, preserves `fg=1`, applies the official distribution workbook, and saves the fixed 640th epoch instead of choosing a checkpoint from test metrics.

It preserves official image scores, saves native FP32 maps and maps them to the existing full-image H//4 × W//4 protocol with explicit crop-boundary extension. Metrics use the existing fast CUDA implementations. This is **GLASS-j 官方实现的 VisA 固定末轮复评**, not the paper's best-checkpoint protocol.

See [the VisA guide](external_baselines/glass_visa/README.md) and [the complete Kaggle setup script](tools/kaggle_glass_visa.py). The script downloads resources on Kaggle only, extracts only the VisA foreground masks, checks all 12 classes, runs small integration tests and a real single-class smoke. Formal training is disabled by default; `--formal` enables the selected independent GPU groups (default: candle on GPU0, capsules on GPU1). It makes no unverified session-duration claim.

```bash
python tools/kaggle_glass_visa.py
# After successful resource checks and smoke, explicitly select a formal batch:
python tools/kaggle_glass_visa.py --formal --gpu0 candle --gpu1 capsules
```

The script targets a Kaggle Notebook with Internet enabled and T4 ×2 GPUs. It creates an independent Python3.10 environment. Already prepared DTD/foreground resources can be passed explicitly with `--dtd-root` and `--foreground-root`; foreground masks are separate from the real defect annotations.

## MVTec AD released-checkpoint setup and usage

Use Python 3.10 and a compatible PyTorch environment. For example:

```bash
python -m pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt
python -m pytest DINOv3/tests/test_glass_mvtec_pretrained.py -q
python -m external_baselines.glass_mvtec_pretrained.runner --help
```

See the [detailed Chinese usage guide](external_baselines/glass_mvtec_pretrained/README.md) for offline checkpoint loading, independent single-category and 15-category commands, evaluation conventions, optional comparison, and validation limits. Run commands from this repository's root; configure every data, weight and output path explicitly.

## Protocol and attribution

[Official GLASS](https://github.com/cqylunlun/GLASS) publishes GLASS-j checkpoints for MVTec AD. Its training code includes best-checkpoint selection using test Image AUROC + Pixel AUROC. The exact selection history of the released files is not independently established here. This evaluator uses the specified released files without retraining or selecting checkpoints by local test results.

Pixel evaluation uses the official crop field of view, not a fabricated full-frame GLASS prediction. Reference full-frame predictions must be mapped into that same crop before comparison. Earlier full-frame pixel metrics cannot be directly subtracted as a matched-view comparison.

Official GLASS retains its [MIT license](external_baselines/GLASS/LICENSE). This repository is an independent re-evaluation project, not the authors' official repository. No new license is asserted for the original evaluation adapters and shared support code.

## Validation

Standalone export: 12 focused tests passed (9.85 seconds), including the small official-module CUDA check. Shared helper definitions were checked for AST equivalence to the development workspace. Python compilation and seven Bash command blocks passed static checks. These checks do not constitute a full-data MVTec evaluation.

The preceding validation describes the original MVTec export revision. For this VisA extension, six Python-only tests and syntax/CLI checks pass; four runtime integration tests are skipped locally because the scientific environment is unavailable. The MVTec regression suite could not run locally without pytest/PyTorch. VisA resource layout, real WideResNet training, strict final reload, T4 memory/time and full-data evaluation remain to be verified on Kaggle. Added shared helper definitions are AST-identical to the development workspace; no experimental reports or outputs were exported.
