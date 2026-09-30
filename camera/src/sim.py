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
        # TODO(bootcamper): save the arguments and set up your state
        # (FixedCamera.__init__ shows you what that looks like).
        #để set môi trường và loại bỏ case obvious sai

        if (not isinstance(width, int) or isinstance(width, bool)):
            raise TypeError("frames' width should be an integer")

        if (not isinstance(height, int) or isinstance(height, bool)):
            raise TypeError("frames' height should be an integer")

        if (width <= 0):
            raise ValueError("frames' width should be a positive number")

        if (height <= 0):
            raise ValueError("frames' height should be a positive number")
        self._width = width
        self._height = height
        self._initialized = False
        self._captures = 0
        self._last_timestamp = float("-inf")

    def initialize_camera(self) -> bool:
        """Turn the fake camera on and start counting from index 0."""
        # TODO(bootcamper): implement.
        self._initialized = True
        self._captures = 0
        return True
    #mục đích là turn camera on

    def capture_frame(self) -> CameraFrame:
        #pixel là 1 điểm ảnh, height là hàng, width là cột => height * width là số ô
        # tuy nhiên mỗi ô là có 3 RGB => tổng là 3 * height * width
        # mỗi ô lại có lựa chọn đậm nhạt từ 0 => 255 => tổng là 256 ^ (3 * width * height)
        # nhiệm vụ là sử dụng cú pháp random để generate ra ảnh mới dựa theo index đó
        """Make up the next frame."""
        # TODO(bootcamper): implement. Don't forget: RuntimeError if the
        # camera isn't on, the same pixels every time for a given index,
        # timestamps that always go up, and returning a copy.
        if not self._initialized:
            raise RuntimeError(
                "capture_frame() called on a camera that is not initialized; "
                "call initialize_camera() first"
            )
        #check xem nó đã initialize cam ch

        rng = np.random.default_rng(self._captures)
        #np.random.default_rng(seed) => tạo ra 1 generator random số nhưng có công thức => sau này đúng index sẽ đúng ảnh đó
        rgb = rng.integers(0, 256, size=(self._height, self._width, 3), dtype=np.uint8)
        #rng.integers(...) => để generate random số nguyên
        #size=(self._height, self._width, 3) => bảo nó đừng generate 1 số mà generate luôn 1 mảng 3 chiều
        #dtype=np.uint8 ép kiểu dữ liệu về uint8
        #==> rng để chuẩn bị công thức, rgb để random

        frame = CameraFrame(
            rgb=rgb,
            # Copy going out: whoever gets this can do what they want with it.
            timestamp=self._next_timestamp(),
            index=self._captures,
            #gói chung hết mọi thứ để trả object
            )
        self._captures += 1 #index để nó bt lần sau generate ảnh khác ảnh đầu
        #nó dùng index (seed) để generate ảnh dựa theo calculation
        return frame

    def stop(self) -> None:
        """Turn the fake camera off. Safe to call more than once."""
        # TODO(bootcamper): implement.
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
    