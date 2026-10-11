# Exported shared helpers; bodies unchanged. See CODE_PROVENANCE.json.
import torch
from .protocol import CATEGORIES

def calculate(rows, maps, masks, device, allow_cpu=False):
    from DINOv3.MADEqual.mvtec_broad6_compose2.metrics import evaluate_fast
    from DINOv3.MADEqual.mvtec_task_decoupled.runner import image_metrics
    from DINOv3.MADEqual.normal_reconstruction.scoring import fixed_fpr_diagnostics
    result = evaluate_fast([r['label'] for r in rows], masks, maps,
                           device=torch.device(device), allow_cpu_fallback=allow_cpu)
    # MUST override evaluate_fast's map-max image metrics with official scores.
    result.update(image_metrics([r['label'] for r in rows], [r['image_score'] for r in rows]))
    # Existing AUPR columns contain average precision, not trapezoidal PR area.
    result['image_AP'] = result.pop('image_AUPR')
    result['pixel_AP'] = result.pop('pixel_AUPR')
    return result, fixed_fpr_diagnostics(maps, masks)


def summarize(metrics, operating, selected, dataset):
    from DINOv3.MADEqual.guided_validation.summary import mean_available
    if len(metrics) != len(selected) or {r['category'] for r in metrics} != set(selected):
        raise ValueError('Missing/duplicate category metrics')
    scope = 'dataset_macro' if set(selected) == set(CATEGORIES[dataset]) else 'selected_categories_macro_NOT_full_dataset'
    five = ('image_AUROC', 'image_AP', 'pixel_AUROC', 'pixel_AP', 'AUPRO_at_0p3')
    macro = [dict(dataset=dataset, scope=scope, category_count=len(selected),
                  **{k: mean_available([r[k] for r in metrics])[0] for k in five})]
    fixed = []
    for cap in (.01, .05):
        group = [r for r in operating if r['fpr_cap'] == cap]
        if len(group) != len(selected) or {r['category'] for r in group} != set(selected):
            raise ValueError('Missing/duplicate fixed FPR rows')
        row = dict(dataset=dataset, scope=scope, fpr_cap=cap, category_count=len(selected),
                   averaging='category_unweighted_excluding_NA')
        for key in ('actual_fpr', 'defect_pixel_recall', 'region_mean_coverage', 'small_region_mean_coverage'):
            row[key], row[key+'_valid_category_count'] = mean_available([r[key] for r in group])
        for key in ('region_count', 'small_region_count', 'small_region_image_count'):
            row[key+'_sum'] = sum(r[key] for r in group)
        fixed.append(row)
    return macro, fixed
