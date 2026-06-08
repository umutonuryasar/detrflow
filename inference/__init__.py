try:
    from .predictor import RTDetrPredictor
    from .visualizer import draw_detections
    _HF_AVAILABLE = True
except ImportError:
    _HF_AVAILABLE = False

from .matcher import HungarianMatcher, giou
from .criterion import SetCriterion

__all__ = [
    "HungarianMatcher",
    "SetCriterion",
    "giou",
    *(["RTDetrPredictor", "draw_detections"] if _HF_AVAILABLE else []),
]
