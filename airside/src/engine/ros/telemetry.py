"""
Copies the drone's status from MAVROS onto the blackboard.

Writes the ``vehicle/*`` keys that behaviors like WaitForReady read. Test
fixtures write the same keys, which is why those behaviors need no ROS.
"""

from __future__ import annotations

import py_trees
import rclpy.node
from mavros_msgs.msg import State

from engine import blackboard_keys

_STATE_TOPIC = "mavros/state"


class TelemetryBridge:
    """
    Listens to ``mavros/state`` and mirrors it onto the blackboard.

    Nothing polls MAVROS: updates arrive as they happen, so behaviors only
    ever read the blackboard and stay fast.
    """

    def __init__(self, node: rclpy.node.Node) -> None:
        self._board = py_trees.blackboard.Client(name="TelemetryBridge")
        for key in (
            blackboard_keys.VEHICLE_CONNECTED,
            blackboard_keys.VEHICLE_MODE,
            blackboard_keys.VEHICLE_ARMED,
        ):
            self._board.register_key(key=key, access=py_trees.common.Access.WRITE)

        self._sub = node.create_subscription(
            msg_type=State,
            topic=_STATE_TOPIC,
            callback=self._on_state,
            qos_profile=10,
        )

    def _on_state(self, msg: State) -> None:
        self._board.set(blackboard_keys.VEHICLE_CONNECTED, msg.connected)
        self._board.set(blackboard_keys.VEHICLE_MODE, msg.mode)
        self._board.set(blackboard_keys.VEHICLE_ARMED, msg.armed)
