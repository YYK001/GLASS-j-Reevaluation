# Exported shared helpers; function bodies unchanged. See CODE_PROVENANCE.json.
import numpy as np
from scipy.ndimage import label

def fixed_fpr_diagnostics(maps,masks,caps=(.01,.05)):
    """Posthoc dataset-global operating points; whole ties, no deployment thresholds."""
    if len(maps)!=len(masks) or not maps:raise ValueError('maps/masks required')
    for a,b in zip(maps,masks):
        if a.shape!=b.shape or not np.isfinite(a).all():raise ValueError('invalid map/mask')
    truth=[np.asarray(m,bool) for m in masks]
    negatives=np.concatenate([np.asarray(p,np.float32)[~m] for p,m in zip(maps,truth)])
    if not len(negatives):raise ValueError('no nondefect test pixels')
    components=[label(m) for m in truth]  # scipy default: 4-connected, same host connectivity.
    sizes=[np.bincount(c.ravel(),minlength=n+1)[1:] for c,n in components]
    small=[s<=.001*m.size for s,m in zip(sizes,truth)]
    total_pos=sum(int(m.sum()) for m in truth)
    rows=[]
    for cap in caps:
        if not 0<=cap<1:raise ValueError('FPR cap must be in [0,1)')
        budget=int(np.floor(cap*len(negatives)))
        # Exclude the entire group containing the (budget+1)-th descending negative.
        boundary=np.partition(negatives,len(negatives)-1-budget)[len(negatives)-1-budget]
        tp=fp=0; cover=[]; small_cover=[]
        for p,m,(component,n),size,sm in zip(maps,truth,components,sizes,small):
            detected=np.asarray(p)>boundary
            tp+=int((detected&m).sum());fp+=int((detected&~m).sum())
            if n:
                hit=np.bincount(component[detected].ravel(),minlength=n+1)[1:]
                fractions=hit/size
                cover.extend(fractions.tolist());small_cover.extend(fractions[sm].tolist())
        rows.append(dict(fpr_cap=cap,actual_fpr=fp/len(negatives),false_positive_pixels=fp,
            nondefect_pixels=len(negatives),defect_pixels=total_pos,
            defect_pixel_recall=tp/total_pos if total_pos else 'N/A',region_count=len(cover),
            region_mean_coverage=float(np.mean(cover)) if cover else 'N/A',
            small_region_count=len(small_cover),small_region_image_count=sum(bool(s.any()) for s in small),
            small_region_mean_coverage=float(np.mean(small_cover)) if small_cover else 'N/A'))
    return rows
