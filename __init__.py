import os

os.environ['FIFTYONE_ALLOW_LEGACY_ORCHESTRATORS'] = 'true'

import fiftyone as fo
import fiftyone.operators as foo
from fiftyone.operators import types    

from .grayscale import GrayscaleAugment
from .invert_colors import InvertColorsAugment
from .color_blind_sim import ColorblindSimAugment
from .task_description_augment import TaskDescriptionAugment
from .resizer import ResizeOperator


def register(plugin):
    """Register operators with the plugin."""
    # Register individual task operators
    plugin.register(GrayscaleAugment)
    plugin.register(InvertColorsAugment)
    plugin.register(ColorblindSimAugment)
    plugin.register(TaskDescriptionAugment)
    plugin.register(ResizeOperator)
    