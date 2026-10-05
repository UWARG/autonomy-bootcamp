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
import time

import numpy as np

from .abstract_camera import AbstractCamera
from .frame import CameraFrame


class SimCamera(AbstractCamera):
    """Fake camera that makes up its own frames.

    The docstring at the top of this file says what it has to do, and
    ``tests/test_sim_camera.py`` checks all of it.
    """
    
    def __init__(self, width: int = 64, height: int = 48) -> None:
        """Save the settings and set up whatever state you need.

        Args:
            width: Frame width in pixels.
            height: Frame height in pixels.
        """
        if (not isinstance(width, int)) or (not isinstance(height, int)):
            raise TypeError("width and height must be int")
        if width <= 0 or height <= 0:
            raise ValueError("width and height must be positive integer")
        self._width = width
        self._height = height
        self._initialized = False
        self._captures = 0
        self._last_timestamp = float("-inf")

        # TODO(bootcamper): save the arguments and set up your state
        # (FixedCamera.__init__ shows you what that looks like).
        #raise NotImplementedError

    def initialize_camera(self) -> bool:
        """Turn the fake camera on and start counting from index 0."""
        # TODO(bootcamper): implement.
        self._initialized = True
        self._captures = 0
        return True
        #raise NotImplementedError

    def capture_frame(self) -> CameraFrame:
        """Make up the next frame."""
        if not self._initialized:
            raise RuntimeError("capture_frame called on uninitialized camera, call initialize_camera() first")
        frames = np.full((self._height, self._width, 3), fill_value=0, dtype=np.uint8)
        for h in range(self._height):
            for w in range(self._width):
                frames[h][w][0] = (self._captures + ((h - self._height//2)**2 + (w - self._width//2) ** 2)**0.5 * 10) % 256
                frames[h][w][1] = (255-((h - self._height//2)**2 + (w - self._width//2) ** 2)**0.5 * 10) % 256
                frames[h][w][2] = (h*self._captures) % 256
        frame = CameraFrame(
            rgb=frames.copy(),
            timestamp=self._next_timestamp(),
            index=self._captures
        )
        self._captures += 1
        return frame

        # TODO(bootcamper): implement. Don't forget: RuntimeError if the
        # camera isn't on, the same pixels every time for a given index,
        # timestamps that always go up, and returning a copy.
        #raise NotImplementedError

    def stop(self) -> None:
        """Turn the fake camera off. Safe to call more than once."""
        self._initialized = False
        # TODO(bootcamper): implement.
        #raise NotImplementedError

    def _next_timestamp(self) -> float:
        """get timestamp and increment ensuring increasing
        """
        timestamp = time.monotonic()
        if timestamp <= self._last_timestamp:
            timestamp = self._last_timestamp + 1e-6
        self._last_timestamp = timestamp
        return timestamp