# Exported shared helpers; bodies unchanged. See CODE_PROVENANCE.json.
"""Shared category and evaluation-grid constants, without PatchCore model code."""

CATEGORIES = {
    'btad': ('01', '02', '03'),
    'mvtec': ('bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather',
              'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper'),
    'visa': ('candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1',
             'macaroni2', 'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum'),
}


BTAD_COUNTS = {'01': (400, 21, 49), '02': (399, 30, 200), '03': (1000, 400, 41)}


def evaluation_hw(original_hw):
    h, w = (int(v) // 4 for v in original_hw)
    if min(h, w) < 1:
        raise ValueError('Original image must be at least 4x4')
    return h, w
