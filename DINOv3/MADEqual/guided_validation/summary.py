# Exported shared helpers; function bodies unchanged. See CODE_PROVENANCE.json.
import numpy as np

def is_na(value):
    return value is None or str(value) == 'N/A'


def mean_available(values):
    valid = [float(v) for v in values if not is_na(v)]
    if not np.isfinite(valid).all():
        raise ValueError('nonfinite summary input')
    return (float(np.mean(valid)) if valid else 'N/A'), len(valid)
