"""Convert NumPy/pandas values to JSON-safe Python values (NaN/inf -> None)."""
from __future__ import annotations

import math


def clean(o):
    if isinstance(o, float):
        return None if math.isnan(o) or math.isinf(o) else o
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if hasattr(o, "item") and not isinstance(o, (str, bytes)):
        return clean(o.item())
    return o
