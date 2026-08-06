"""``WaitForReady``: a finished behavior example

Shows all five methods py_trees calls for you, and how to read the
blackboard without crashing on keys nobody has written yet. The airside
README has the table of what each method is for.

Read it with ``tests/test_wait_for_ready.py``.
"""

from __future__ import annotations

import py_trees

from engine import blackboard_keys

# ArduPilot flight mode in which the engine is allowed to command the drone.
GUIDED_MODE = "GUIDED"


class WaitForReady(py_trees.behaviour.Behaviour):
    """
    Holds up the mission until the drone can actually be commanded.

    SUCCESS only when the drone is connected, in ``"GUIDED"``, and armed.
    RUNNING otherwise, including before any status arrives.

    Never returns FAILURE. "Not ready yet" happens on every startup, so
    giving up is the mission timeout's job, not this behavior's.
    """

    def __init__(self, name: str = "WaitForReady") -> None:
        super().__init__(name=name)

        # Connect to the blackboard and say exactly which keys we use and
        # whether we read or write them. py_trees actually checks this:
        # reading a key you never registered raises AttributeError, so a
        # behavior quietly using data it never mentioned can't happen.
        self.blackboard = self.attach_blackboard_client(name=self.name)
        for key in (
            blackboard_keys.VEHICLE_CONNECTED,
            blackboard_keys.VEHICLE_MODE,
            blackboard_keys.VEHICLE_ARMED,
        ):
            self.blackboard.register_key(key=key, access=py_trees.common.Access.READ)

    def setup(self, **kwargs: object) -> None:
        """
        Nothing to grab, since this behavior owns nothing outside itself.

        The container passes ``node=<rclpy Node>`` here and the code in
        ``engine/ros/`` takes it. We ignore it on purpose, which is what
        keeps this behavior runnable on a laptop with no ROS.
        """

    def initialise(self) -> None:
        """Nothing to reset. Every tick re-reads the blackboard from
        scratch, so there is no per-attempt state to clear."""

    def update(self) -> py_trees.common.Status:
        """
        Check the three things on the blackboard.

        A missing key raises ``KeyError``, which we treat as "not ready"
        rather than an error: at startup it just means no status has
        arrived yet, and that resolves on its own.
        """
        try:
            connected = self.blackboard.get(blackboard_keys.VEHICLE_CONNECTED)
            mode = self.blackboard.get(blackboard_keys.VEHICLE_MODE)
            armed = self.blackboard.get(blackboard_keys.VEHICLE_ARMED)
        except KeyError:
            self.feedback_message = "waiting for vehicle telemetry"
            return py_trees.common.Status.RUNNING

        if connected is True and mode == GUIDED_MODE and armed is True:
            self.feedback_message = "vehicle ready"
            return py_trees.common.Status.SUCCESS

        self.feedback_message = (
            f"waiting (connected={connected}, mode={mode}, armed={armed})"
        )
        return py_trees.common.Status.RUNNING

    def terminate(self, new_status: py_trees.common.Status) -> None:
        """Nothing to clean up. We keep no state and start no work, so
        being interrupted costs nothing."""
