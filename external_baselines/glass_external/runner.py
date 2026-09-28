# Exported shared helpers; function bodies unchanged. See CODE_PROVENANCE.json.
import numpy as np
from DINOv3.MADEqual.mvtec_broad6_compose2.metrics import METRIC_NAMES
from DINOv3.MADEqual.guided_validation.summary import mean_available

def summarize(metrics, operating, expected):
    if len(metrics)!=len(expected) or {r['category'] for r in metrics}!=set(expected):
        raise ValueError('missing or duplicate BTAD category metrics')
    macro=[dict(method='GLASS_J_FINAL',category_count=len(expected),
        **{k:float(np.mean([r[k] for r in metrics])) for k in METRIC_NAMES})]
    fmacro=[]
    for cap in (.01,.05):
        group=[r for r in operating if r['fpr_cap']==cap]
        if len(group)!=len(expected) or {r['category'] for r in group}!=set(expected): raise ValueError('incomplete fixed FPR')
        row=dict(fpr_cap=cap,averaging='category_unweighted_excluding_NA',category_count=len(expected))
        for k in ('actual_fpr','defect_pixel_recall','region_mean_coverage','small_region_mean_coverage'):
            row[k],row[k+'_valid_category_count']=mean_available([r[k] for r in group])
        for k in ('region_count','small_region_count','small_region_image_count'): row[k+'_sum']=sum(r[k] for r in group)
        fmacro.append(row)
    return macro,fmacro
