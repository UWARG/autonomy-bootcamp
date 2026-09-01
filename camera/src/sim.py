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

import numpy
import time
from .abstract_camera import AbstractCamera
from .frame import CameraFrame


class SimCamera(AbstractCamera):
    """Fake camera that makes up its own frames.

    The docstring at the top of this file says what it has to do, and
    ``tests/test_sim_camera.py`` checks all of it.
    """

    def __init__(self, width: int = 64, height: int = 48, channels: int = 3) -> None:
        """Save the settings and set up whatever state you need.

        Args:
            width: Frame width in pixels.
            height: Frame height in pixels.
            channels: Number of color channels in the frame.
        """

        self.width = width 
        self.height = height 
        self.channels = channels
        self._last_timestamp = time.monotonic()
        self._initialized = False 

    def initialize_camera(self) -> bool:
        """Turn the fake camera on and start counting from index 0."""
        self._initialized = True
        self._captures = 0
        return True  

    def capture_frame(self) -> CameraFrame:
        """Make up the next frame."""

        if not self._initialized:
            raise RuntimeError("Camera is not initialized")

        rng = numpy.random.default_rng(self._captures)

        frame =  CameraFrame(
            rgb=rng.integers(0, 256, size=(self.height, self.width, self.channels), dtype=numpy.uint8),
            timestamp=self._next_timestamp(), 
            index=self._captures,
        )
        self._captures += 1 
        return frame

    def stop(self) -> None:
        """Turn the fake camera off. Safe to call more than once."""
        self._initialized = False 

    def _next_timestamp(self) -> float:
        """Read the clock, making sure the number beats the last one.

        ``time.monotonic()`` can return the same value twice if you call it
        twice fast enough, which would break ordering, so nudge it up.
        """
        timestamp = time.monotonic()
        if timestamp <= self._last_timestamp:
            timestamp = self._last_timestamp + 1e-6
        self._last_timestamp = timestamp
        return timestamp
