"""These tests ARE your Part 2 assignment.

Write ``src/sim.py`` until they all pass:

    warg run camera test

Do not change this file. Reviewers diff it against the original.
"""

from itertools import pairwise

import numpy as np
import pytest

from src.sim import SimCamera


def capture_many(camera: SimCamera, count: int) -> list:
    """Capture `count` frames in a row."""
    return [camera.capture_frame() for _ in range(count)]


# ---------------------------------------------------------------------------
# The arguments __init__ takes, and the size of the frames
# ---------------------------------------------------------------------------


def test_default_constructor_produces_48x64_frames() -> None:
    # Your __init__ has to look like this:
    # SimCamera(width=64, height=48).
    camera = SimCamera()
    camera.initialize_camera()

    for frame in capture_many(camera, 3):
        # numpy puts images in this order: (height, width, channels).
        assert frame.rgb.shape == (48, 64, 3)
        assert frame.rgb.dtype == np.uint8


def test_constructor_accepts_custom_parameters() -> None:
    camera = SimCamera(width=32, height=20)
    camera.initialize_camera()

    for frame in capture_many(camera, 3):
        assert frame.rgb.shape == (20, 32, 3)
        assert frame.rgb.dtype == np.uint8


def test_constructor_accepts_positional_parameters() -> None:
    # The order of the arguments matters too: (width, height).
    camera = SimCamera(8, 6)
    camera.initialize_camera()

    for frame in capture_many(camera, 2):
        assert frame.rgb.shape == (6, 8, 3)


# ---------------------------------------------------------------------------
# Using the camera before turning it on is an error
# ---------------------------------------------------------------------------


def test_capture_before_initialize_raises() -> None:
    camera = SimCamera()
    with pytest.raises(RuntimeError):
        camera.capture_frame()


def test_initialize_returns_true() -> None:
    camera = SimCamera()
    assert camera.initialize_camera() is True


# ---------------------------------------------------------------------------
# Frames keep coming, in order, for as long as the camera is on
# ---------------------------------------------------------------------------


def test_indexes_are_sequential_from_zero() -> None:
    camera = SimCamera()
    camera.initialize_camera()

    frames = capture_many(camera, 4)
    assert [frame.index for frame in frames] == [0, 1, 2, 3]


def test_distinct_indexes_have_distinct_content() -> None:
    camera = SimCamera()
    camera.initialize_camera()

    frames = capture_many(camera, 3)
    for i in range(len(frames)):
        for j in range(i + 1, len(frames)):
            assert not np.array_equal(
                frames[i].rgb, frames[j].rgb
            ), f"frames {i} and {j} have identical pixels"


def test_same_index_has_identical_content_across_instances() -> None:
    # The pixels can only depend on (index, width, height), so two cameras
    # built with the same numbers give you the same pictures.
    camera_a = SimCamera(width=16, height=12)
    camera_b = SimCamera(width=16, height=12)
    camera_a.initialize_camera()
    camera_b.initialize_camera()

    for frame_a, frame_b in zip(capture_many(camera_a, 3), capture_many(camera_b, 3)):
        assert np.array_equal(frame_a.rgb, frame_b.rgb)


# ---------------------------------------------------------------------------
# Timestamps
# ---------------------------------------------------------------------------


def test_timestamps_strictly_increase() -> None:
    camera = SimCamera()
    camera.initialize_camera()

    timestamps = [frame.timestamp for frame in capture_many(camera, 5)]
    assert all(a < b for a, b in pairwise(timestamps))


# ---------------------------------------------------------------------------
# Stop
# ---------------------------------------------------------------------------


def test_capture_after_stop_raises() -> None:
    camera = SimCamera()
    camera.initialize_camera()
    camera.stop()
    with pytest.raises(RuntimeError):
        camera.capture_frame()


def test_stop_works_anytime_and_more_than_once() -> None:
    camera = SimCamera()
    camera.stop()  # never turned on, still fine
    camera.initialize_camera()
    camera.stop()
    camera.stop()  # stopping twice, still fine


def test_reinitialize_after_stop_brings_the_camera_back() -> None:
    camera = SimCamera()
    camera.initialize_camera()
    camera.capture_frame()
    camera.stop()

    assert camera.initialize_camera() is True
    assert camera.capture_frame().index == 0  # counting starts over


# ---------------------------------------------------------------------------
# Changing a frame you were given must not affect the camera
# ---------------------------------------------------------------------------


def test_mutating_a_returned_frame_does_not_corrupt_the_camera() -> None:
    camera = SimCamera()
    camera.initialize_camera()

    frame = camera.capture_frame()
    pristine = frame.rgb.copy()  # save what it looked like first
    frame.rgb[:] = 255  # now wreck the pixels we were handed

    # Turn it off and on to get index 0 again. It should be unchanged.
    camera.stop()
    camera.initialize_camera()
    fresh = camera.capture_frame()
    assert np.array_equal(fresh.rgb, pristine)
