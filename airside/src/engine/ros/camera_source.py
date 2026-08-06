"""
Runs your SimCamera and puts its frames on the blackboard.

A timer writes the newest frame to ``camera/latest_frame``, the key your
CaptureForPerception reads, for as long as the mission runs.
"""

from __future__ import annotations

import py_trees
import rclpy.node
from camera.src.sim import SimCamera

from engine import blackboard_keys

_POLL_PERIOD_S = 0.5


class CameraSource:
    """Takes a picture on a timer and puts it on the blackboard.

    On a timer rather than on demand, so a behavior asking for a frame never
    has to wait on the camera.
    """

    def __init__(
        self,
        node: rclpy.node.Node,
        width: int = 64,
        height: int = 48,
        poll_period_s: float = _POLL_PERIOD_S,
    ) -> None:
        self._node = node
        self._camera = SimCamera(width=width, height=height)
        if not self._camera.initialize_camera():
            raise RuntimeError("SimCamera failed to initialize")

        self._board = py_trees.blackboard.Client(name="CameraSource")
        self._board.register_key(
            key=blackboard_keys.LATEST_FRAME, access=py_trees.common.Access.WRITE
        )
        self._board.set(blackboard_keys.LATEST_FRAME, None)

        self._timer = node.create_timer(poll_period_s, self._poll)

    def _poll(self) -> None:
        self._board.set(blackboard_keys.LATEST_FRAME, self._camera.capture_frame())

    def stop(self) -> None:
        """Stop the timer and turn the camera off.

        Called on shutdown so the container can exit instead of sitting on a
        live timer.
        """
        self._timer.cancel()
        self._camera.stop()
