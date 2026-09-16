"""Tests for FixedCamera, a finished example.

Read them next to ``src/fixed.py``. Between them they cover everything your
``SimCamera`` has to do.
"""

from itertools import pairwise

import numpy as np
import pytest

from src.fixed import FixedCamera


def make_images(count: int, width: int = 6, height: int = 4) -> list[np.ndarray]:
    """Make `count` images that are easy to tell apart.

    Each one is a single solid color based on its position, so a test can
    tell right away which image a frame came from.
    """
    return [
        np.full((height, width, 3), fill_value=(i + 1) * 10, dtype=np.uint8)
        for i in range(count)
    ]


# ---------------------------------------------------------------------------
# Creating the camera, and rejecting bad input
# ---------------------------------------------------------------------------


def test_rejects_non_list_input() -> None:
    with pytest.raises(TypeError):
        FixedCamera(np.zeros((4, 6, 3), dtype=np.uint8))  # an array, not a list


def test_rejects_empty_list() -> None:
    with pytest.raises(ValueError):
        FixedCamera([])


def test_rejects_wrong_shape() -> None:
    # A 2-D grayscale image, missing the color channel at the end.
    with pytest.raises(ValueError):
        FixedCamera([np.zeros((4, 6), dtype=np.uint8)])


def test_rejects_wrong_channel_count() -> None:
    # RGBA has 4 channels. We want RGB, which has 3.
    with pytest.raises(ValueError):
        FixedCamera([np.zeros((4, 6, 4), dtype=np.uint8)])


def test_rejects_wrong_dtype() -> None:
    with pytest.raises(ValueError):
        FixedCamera([np.zeros((4, 6, 3), dtype=np.float32)])


def test_rejects_non_array_element() -> None:
    with pytest.raises(TypeError):
        FixedCamera([[[0, 0, 0]]])  # a list of lists, not a numpy array


# ---------------------------------------------------------------------------
# Using the camera before turning it on is an error
# ---------------------------------------------------------------------------


def test_capture_before_initialize_raises() -> None:
    camera = FixedCamera(make_images(2))
    with pytest.raises(RuntimeError):
        camera.capture_frame()


# ---------------------------------------------------------------------------
# Normal use, when nothing goes wrong
# ---------------------------------------------------------------------------


def test_initialize_returns_true() -> None:
    camera = FixedCamera(make_images(2))
    assert camera.initialize_camera() is True


def test_frames_come_back_in_order_with_correct_content() -> None:
    images = make_images(3)
    camera = FixedCamera(images)
    camera.initialize_camera()

    for expected_index, expected_image in enumerate(images):
        frame = camera.capture_frame()
        assert frame.index == expected_index
        assert frame.rgb.shape == expected_image.shape
        assert frame.rgb.dtype == np.uint8
        assert np.array_equal(frame.rgb, expected_image)


def test_timestamps_strictly_increase() -> None:
    camera = FixedCamera(make_images(3))
    camera.initialize_camera()

    timestamps = [camera.capture_frame().timestamp for _ in range(3)]
    # Check every neighboring pair: each capture is later than the one before.
    assert all(a < b for a, b in pairwise(timestamps))


# ---------------------------------------------------------------------------
# Reaching the end of the list, and starting over
# ---------------------------------------------------------------------------


def test_images_loop_once_the_list_runs_out() -> None:
    # Two images, four captures: the images repeat but the index keeps going.
    images = make_images(2)
    camera = FixedCamera(images)
    camera.initialize_camera()

    frames = [camera.capture_frame() for _ in range(4)]

    assert [frame.index for frame in frames] == [0, 1, 2, 3]
    for frame, expected in zip(frames, images + images):
        assert np.array_equal(frame.rgb, expected)


def test_reinitialize_starts_the_index_over() -> None:
    camera = FixedCamera(make_images(2))
    camera.initialize_camera()

    first = camera.capture_frame()
    camera.capture_frame()

    camera.initialize_camera()
    again = camera.capture_frame()
    assert again.index == 0
    assert np.array_equal(again.rgb, first.rgb)


def test_timestamps_keep_increasing_across_reinitialize() -> None:
    # Re-initializing rewinds the images, not the clock.
    camera = FixedCamera(make_images(2))
    camera.initialize_camera()

    before = camera.capture_frame().timestamp
    camera.initialize_camera()
    after = camera.capture_frame().timestamp
    assert after > before


# ---------------------------------------------------------------------------
# Stop
# ---------------------------------------------------------------------------


def test_capture_after_stop_raises() -> None:
    camera = FixedCamera(make_images(2))
    camera.initialize_camera()
    camera.stop()
    with pytest.raises(RuntimeError):
        camera.capture_frame()


def test_stop_works_anytime_and_more_than_once() -> None:
    camera = FixedCamera(make_images(1))
    camera.stop()  # never turned on, still fine
    camera.initialize_camera()
    camera.stop()
    camera.stop()  # stopping twice, still fine


def test_reinitialize_after_stop_brings_the_camera_back() -> None:
    camera = FixedCamera(make_images(2))
    camera.initialize_camera()
    camera.capture_frame()
    camera.stop()

    assert camera.initialize_camera() is True
    assert camera.capture_frame().index == 0  # counting starts over


# ---------------------------------------------------------------------------
# Copying, both on the way in and on the way out
# ---------------------------------------------------------------------------


def test_mutating_a_returned_frame_does_not_corrupt_the_camera() -> None:
    camera = FixedCamera(make_images(1))
    camera.initialize_camera()

    frame = camera.capture_frame()
    pristine = frame.rgb.copy()  # save what it looked like first
    frame.rgb[:] = 255  # now wreck the pixels we were handed

    # Rewind and grab the same frame again. It should be unchanged.
    camera.initialize_camera()
    fresh = camera.capture_frame()
    assert np.array_equal(fresh.rgb, pristine)


def test_mutating_the_input_images_does_not_corrupt_the_camera() -> None:
    images = make_images(1)
    pristine = images[0].copy()
    camera = FixedCamera(images)
    images[0][:] = 0  # wreck our own array after handing it over

    camera.initialize_camera()
    frame = camera.capture_frame()
    assert np.array_equal(frame.rgb, pristine)
