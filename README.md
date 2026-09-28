# GLASS-j-Reevaluation

Independent re-evaluation of **officially released GLASS-j checkpoints on MVTec AD**, with unified metrics and an optional matched-field-of-view comparison.

This repository contains code only: no datasets, model weights, predictions, training logs, or experimental result reports. It does not train GLASS or claim that the published checkpoints are fixed-final-epoch models.

## What is included

- Pinned official GLASS implementation (MIT), commit `f788d567db552820d744ca2d93e989265aad8c45`.
- Explicit official checkpoint download identities and strict loading of discriminator, projection and BatchNorm state.
- Independent `download`, `check`, `predict`, `evaluate`, and optional `compare` commands.
- Official RGB preprocessing, including the toothbrush/wood resize rule; nearest-neighbor binary GT in the same 288×288 field of view.
- Image AUROC/AP from the original GLASS image scores; Pixel AUROC/AP, fast CUDA AUPRO@0.3, and fixed-FPR coverage.
- Per-image lossless float32 predictions, original image scores, binary GT, geometry, configuration and resource records when users run the code.

The retained `DINOv3/` namespace contains only the small shared data/metric/I/O support and tests needed by this evaluator. It does not load a DINO model, retrieve a memory bank, or include the wider experimental workspace. Shared helper function bodies were exported unchanged; see [CODE_PROVENANCE.json](CODE_PROVENANCE.json).

## Setup and usage

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
