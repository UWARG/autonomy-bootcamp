"""SimCamera: YOUR Part 2 assignment.

A camera that makes up its own frames. Read ``src/fixed.py`` and its tests
first, then make ``tests/test_sim_camera.py`` pass:

    warg run camera test

``SimCamera(width=64, height=48)`` hands back a ``(height, width, 3)``
``uint8`` frame every time you ask, for as long as it's on, with ``index``
counting up from 0. Same rules as every camera
(``src/abstract_camera.py``), plus one:

A frame's pixels depend only on its index. Frame 2 always looks the same,
here or in any SimCamera built with the same size, and frames with different
indexes look different. Fill values, gradients, and
``numpy.random.default_rng(index)`` all work.
"""

from .abstract_camera import AbstractCamera
from .frame import CameraFrame

import time

import numpy as np

class SimCamera(AbstractCamera):
    """Fake camera that makes up its own frames.

    The docstring at the top of this file says what it has to do, and
    ``tests/test_sim_camera.py`` checks all of it.
    """

    def __init__(self, width: int = 64, height: int = 48) -> None:
        self.width = width
        self.height = height
        self._initialized = False
        self._captures = 0
        self._last_timestamp = float("-inf")
        """Save the settings and set up whatever state you need.

        Args:
            width: Frame width in pixels.
            height: Frame height in pixels.
        """

    def initialize_camera(self) -> bool:
        """Turn the fake camera on and start counting from index 0."""
        self._initialized = True
        self._captures = 0
        return True

    def capture_frame(self) -> CameraFrame:
        """Make up the next frame."""
        if not self._initialized:
            raise RuntimeError
        
        rng = np.random.default_rng(self._captures)
        rgb_array = rng.integers(0, 256, size=(self.height, self.width, 3), dtype=np.uint8)
        frame = CameraFrame(rgb=rgb_array, timestamp=self._next_timestamp(), index=self._captures,)
        self._captures += 1
        return frame

        # TODO(bootcamper): implement. Don't forget: RuntimeError if the
        # camera isn't on, the same pixels every time for a given index,
        # timestamps that always go up, and returning a copy.

    def stop(self) -> None:
        """Turn the fake camera off. Safe to call more than once."""
        # TODO(bootcamper): implement.
        self._initialized = False