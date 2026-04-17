from .inspire import InspireModel
from .wuji import WujiModel
from .shadow import ShadowModel

HAND_MODELS: dict = {
    "inspire": InspireModel,
    "wuji": WujiModel,
    "shadow": ShadowModel,
}

__all__ = ["HAND_MODELS", "InspireModel", "WujiModel", "ShadowModel"]
