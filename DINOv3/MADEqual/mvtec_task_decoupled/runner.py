# Exported shared helpers; function bodies unchanged. See CODE_PROVENANCE.json.
from typing import Sequence
import numpy as np
from sklearn.metrics import average_precision_score,roc_auc_score

def image_metrics(labels: Sequence[int], scores: Sequence[float]) -> dict[str, float]:
    target = np.asarray(labels, dtype=np.int64)
    values = np.asarray(scores, dtype=np.float64)
    if target.shape != values.shape or set(target.tolist()) != {0, 1}:
        raise ValueError("image labels/scores must be aligned with normal and anomaly support")
    if not np.isfinite(values).all():
        raise FloatingPointError("image scores contain NaN/Inf")
    return {
        "image_AUROC": float(roc_auc_score(target, values)),
        "image_AUPR": float(average_precision_score(target, values)),
    }
