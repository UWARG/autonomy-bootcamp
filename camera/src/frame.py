"""CameraFrame: the object every camera gives back.

Pixels, plus enough to tell captures apart. Cameras hand out copies, so
holding on to a frame or drawing on it can't affect the camera.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class CameraFrame:
    """One captured frame.

    Attributes:
        rgb: The pixels, as a ``(height, width, 3)`` ``uint8`` numpy array.
            This copy is yours, so change it however you like.
        timestamp: When it was captured, in seconds, from a clock that only
            counts up. Every capture from the same camera has a bigger
            timestamp than the last.
        index: Where this frame sits in the camera's sequence, counting up
            from 0. Starts over when the camera is turned back on.
    """

    rgb: np.ndarray
    timestamp: float
    index: int
