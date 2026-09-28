# Exported shared helpers; function bodies unchanged. See CODE_PROVENANCE.json.
import numpy as np
from sklearn.metrics import average_precision_score,roc_auc_score

def safe_auroc(labels, scores) -> float:
    labels = np.asarray(labels).reshape(-1)
    scores = np.asarray(scores).reshape(-1)
    if labels.size == 0 or np.unique(labels).size < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def pixel_aupr(masks, maps) -> float:
    labels = np.asarray(masks, dtype=bool).reshape(-1)
    scores = np.asarray(maps, dtype=np.float64).reshape(-1)
    if labels.size == 0 or np.unique(labels).size < 2:
        return float("nan")
    return float(average_precision_score(labels, scores))
