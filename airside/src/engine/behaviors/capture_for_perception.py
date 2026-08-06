"""
``CaptureForPerception``: YOU write this one. Details in the class below.

Read ``wait_for_ready.py`` first, then make ``tests/test_capture.py`` pass:

    warg run airside test
"""

from __future__ import annotations

import py_trees

from engine import blackboard_keys


class CaptureForPerception(py_trees.behaviour.Behaviour):
    """
    Takes one *new* camera picture after the drone gets to a waypoint.

    Something else keeps writing the newest ``CameraFrame`` to the
    blackboard under ``blackboard_keys.LATEST_FRAME``. The value there is
    either a frame, which has an ``index``, or ``None``.

    Here's the catch. Whatever frame is sitting there when this behavior
    starts was probably taken during the flight over, before the drone got
    to the waypoint. It's an old picture of the wrong place. So this
    behavior ignores it and waits for a frame with a different ``index``,
    sends that one out, and returns SUCCESS.

    What it has to do (``tests/test_capture.py`` checks all of this):

    ``initialise()``
        Write down the ``index`` of whatever frame is on ``LATEST_FRAME``
        right now, or ``None`` if there isn't one yet. That's what you'll
        compare against. Also reset your tick counter. py_trees calls this
        every time the behavior starts again, so the capture at the next
        waypoint gets its own starting point.

    ``update()``
        - If ``LATEST_FRAME`` has a frame whose ``index`` isn't the one you
          wrote down, that's a new picture. Call
          ``publisher.publish_image(frame)``, then
          ``publisher.publish_status({"phase": "capture",
          "frame_index": frame.index})``, and return SUCCESS.
        - Otherwise add one to your tick counter and return RUNNING. Once
          ``timeout_ticks`` calls have gone by with no new frame, give up
          and return FAILURE, since the camera is probably dead.

    You get the publisher handed to you instead of making one. In the
    container it's the real ROS one (``engine/ros/perception_publisher.py``)
    and in the tests it's a fake that just records what it was given. Only
    use its two methods, ``publish_image(frame)`` and
    ``publish_status(status_dict)``.

    Args:
        name: The name shown for this behavior when the tree gets printed.
        publisher: Something with ``publish_image(frame)`` and
            ``publish_status(status: dict)`` methods.
        timeout_ticks: How many ``update()`` calls to wait for a new frame
            before giving up and returning FAILURE.
    """

    def __init__(self, name: str, publisher, timeout_ticks: int = 20) -> None:
        super().__init__(name=name)
        self._publisher = publisher
        self._timeout_ticks = timeout_ticks

        # We already set up the blackboard for you here, with permission to
        # read the latest camera frame.
        self.blackboard = self.attach_blackboard_client(name=self.name)
        self.blackboard.register_key(
            key=blackboard_keys.LATEST_FRAME, access=py_trees.common.Access.READ
        )

    def _latest_frame(self):
        """
        The frame on the blackboard right now, or ``None``.

        Collapses "key never written" and "key holds None" into one answer,
        so callers don't have to catch ``KeyError`` everywhere.
        """
        try:
            return self.blackboard.get(blackboard_keys.LATEST_FRAME)
        except KeyError:
            return None

    def initialise(self) -> None:
        """
        Note the index to compare against, and reset the tick counter.

        py_trees calls this on every fresh attempt, which is what lets the
        capture at the next waypoint ignore the picture from this one.
        """
        # TODO(bootcamper): implement.
        raise NotImplementedError

    def update(self) -> py_trees.common.Status:
        """
        Send out the first new frame and succeed, or give up after a while.

        Returns fast every tick, so the rest of the tree keeps running while
        we wait. The class docstring says exactly what to do.
        """
        # TODO(bootcamper): implement.
        raise NotImplementedError
