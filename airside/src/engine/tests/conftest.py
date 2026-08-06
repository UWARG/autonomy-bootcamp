"""
Fixtures and fakes shared by the airside tests.

Nothing here touches ROS, which is what lets the whole suite run on your
laptop with `uv run pytest`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import py_trees
import pytest

from engine import blackboard_keys

_ALL_KEYS = (
    blackboard_keys.VEHICLE_CONNECTED,
    blackboard_keys.VEHICLE_ARMED,
    blackboard_keys.VEHICLE_MODE,
    blackboard_keys.LATEST_FRAME,
    blackboard_keys.WAYPOINTS,
    blackboard_keys.WAYPOINT_INDEX,
)


# A tiny stand-in for camera.src.frame.CameraFrame. The behaviors only ever
# look at `.index`, so anything with that attribute works, and keeping a
# copy here means these tests don't need the camera project installed. The
# real CameraFrame goes through this same code in the Part 5 container.
@dataclass
class FrameStub:
    """Same three attributes as CameraFrame: pixels, timestamp, index."""

    rgb: np.ndarray
    timestamp: float
    index: int


def make_camera_frame(index: int) -> FrameStub:
    """Make a small frame with the index you asked for.

    Only the index matters to the behaviors, so the pixels stay tiny.
    """
    return FrameStub(
        rgb=np.zeros((4, 4, 3), dtype=np.uint8),
        timestamp=float(index),
        index=index,
    )


class FakePublisher:
    """
    A fake publisher that just writes down what it was given.

    The real one (engine/ros/perception_publisher.py) sends ROS messages.
    This one keeps a list of every call so the tests can check them.
    """

    def __init__(self) -> None:
        self.images: list = []
        self.statuses: list[dict] = []

    def publish_image(self, frame) -> None:
        self.images.append(frame)

    def publish_status(self, status: dict) -> None:
        self.statuses.append(status)


@pytest.fixture(autouse=True)
def clean_blackboard():
    """Wipe the blackboard around every test.

    It's global to the process, so without this one test's leftover keys
    would decide another test's result.
    """
    py_trees.blackboard.Blackboard.clear()
    yield
    py_trees.blackboard.Blackboard.clear()


@pytest.fixture
def board() -> py_trees.blackboard.Client:
    """Lets a test write any key the engine uses.

    Stands in for whatever would normally be filling them in.
    """
    client = py_trees.blackboard.Client(name="TestWriter")
    for key in _ALL_KEYS:
        client.register_key(key=key, access=py_trees.common.Access.WRITE)
    return client


@pytest.fixture
def publisher() -> FakePublisher:
    return FakePublisher()
