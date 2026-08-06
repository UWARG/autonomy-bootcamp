"""
The behaviors that actually fly the drone, using MAVROS.

Taken from the WARG monorepo and trimmed. SetModeGuided and Arm are new:
a pilot normally does those on a controller, and the simulator has no pilot.
"""

from __future__ import annotations

import math

import py_trees
import rclpy.node
from mavros_msgs.msg import GlobalPositionTarget, State
from mavros_msgs.srv import CommandBool, CommandTOL, SetMode
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import NavSatFix
from std_msgs.msg import Float64
from utils.src.types import Coordinate
from utils.src.waypoint_utils import east_north_coordinate_offset_m

from engine import blackboard_keys

GUIDED_MODE = "GUIDED"

# Relative altitude (meters) to climb to on takeoff.
TAKEOFF_ALTITUDE_M = 10.0
# Takeoff tolerance from target altitude.
TAKEOFF_ALTITUDE_TOLERANCE_M = 1.0
# Relative altitude (meters) above which the drone counts as already flying.
TAKEOFF_AIRBORNE_THRESHOLD_M = 2.0

# How close (meters) counts as having reached a waypoint.
WAYPOINT_ACCEPTANCE_RADIUS_M = 1.0
# Give up on a waypoint if not reached within this many seconds.
WAYPOINT_NAV_TIMEOUT_S = 60.0

# Re-send a rejected arming request after this many seconds (ArduPilot
# rejects arming until its pre-arm checks pass).
ARM_RETRY_PERIOD_S = 2.0

_STATE_TOPIC = "mavros/state"
_GLOBAL_POSITION_TOPIC = "mavros/global_position/global"
_REL_ALT_TOPIC = "mavros/global_position/rel_alt"
_SETPOINT_TOPIC = "mavros/setpoint_raw/global"
_SET_MODE_SERVICE = "mavros/set_mode"
_ARMING_SERVICE = "mavros/cmd/arming"
_TAKEOFF_SERVICE = "mavros/cmd/takeoff"
_LAND_SERVICE = "mavros/cmd/land"

# Position-only setpoint: ignores velocity, acceleration and yaw fields.
_TYPE_MASK = (
    GlobalPositionTarget.IGNORE_VX
    | GlobalPositionTarget.IGNORE_VY
    | GlobalPositionTarget.IGNORE_VZ
    | GlobalPositionTarget.IGNORE_AFX
    | GlobalPositionTarget.IGNORE_AFY
    | GlobalPositionTarget.IGNORE_AFZ
    | GlobalPositionTarget.IGNORE_YAW
    | GlobalPositionTarget.IGNORE_YAW_RATE
)


class _MavrosBehaviour(py_trees.behaviour.Behaviour):
    """Shared setup: the ROS node, the latest drone status, and the clock.

    Every flight behavior needs the drone's state, so it gets subscribed
    once here instead of five times.
    """

    def setup(self, **kwargs: rclpy.node.Node) -> None:
        self._node = kwargs["node"]
        self._latest_state: State | None = None
        self._state_sub = self._node.create_subscription(
            msg_type=State,
            topic=_STATE_TOPIC,
            callback=self._state_callback,
            qos_profile=10,
        )
        self._extra_setup()

    def _extra_setup(self) -> None:
        """Subclasses make their own ROS objects here.

        Called from ``setup()``, so the node exists by the time it runs.
        """

    def _state_callback(self, msg: State) -> None:
        self._latest_state = msg

    def _now_s(self) -> float:
        return self._node.get_clock().now().nanoseconds / 1e9


class SetModeGuided(_MavrosBehaviour):
    """
    Puts the drone into GUIDED mode, the mode where we can command it.

    Returns SUCCESS once ``mavros/state`` says it's in GUIDED. Until then it
    keeps asking. Nothing here gives up, since the mission timeout is what
    stops us if this never works.
    """

    def __init__(self, name: str = "SetModeGuided") -> None:
        super().__init__(name=name)

    def _extra_setup(self) -> None:
        self._client = self._node.create_client(
            srv_type=SetMode, srv_name=_SET_MODE_SERVICE
        )
        self._future = None

    def initialise(self) -> None:
        self._future = None

    def update(self) -> py_trees.common.Status:
        if self._latest_state is None:
            self._node.get_logger().warning(
                f"{self.name}: waiting for '{_STATE_TOPIC}'",
                throttle_duration_sec=5.0,
            )
            return py_trees.common.Status.RUNNING

        if self._latest_state.mode == GUIDED_MODE:
            self._node.get_logger().info(f"{self.name}: flight mode is GUIDED")
            return py_trees.common.Status.SUCCESS

        if self._future is None:
            if not self._client.service_is_ready():
                self._node.get_logger().warning(
                    f"{self.name}: waiting for '{_SET_MODE_SERVICE}' service",
                    throttle_duration_sec=5.0,
                )
                return py_trees.common.Status.RUNNING

            request = SetMode.Request()
            request.custom_mode = GUIDED_MODE
            self._future = self._client.call_async(request)
            self._node.get_logger().info(f"{self.name}: requesting GUIDED mode")
            return py_trees.common.Status.RUNNING

        if self._future.done():
            response = self._future.result()
            self._future = None
            if response is None or not response.mode_sent:
                self._node.get_logger().warning(
                    f"{self.name}: mode change rejected, retrying"
                )

        return py_trees.common.Status.RUNNING

    def terminate(self, new_status: py_trees.common.Status) -> None:
        if new_status != py_trees.common.Status.SUCCESS:
            self._future = None


class Arm(_MavrosBehaviour):
    """
    Arms the drone, meaning the motors are allowed to spin.

    Returns SUCCESS once ``mavros/state`` says it's armed. The drone often
    says no at first, because it runs its own checks before arming and
    something like the position estimate may not be ready yet. So we ask
    again every ``ARM_RETRY_PERIOD_S``. The mission timeout is what stops us
    if it never says yes.
    """

    def __init__(self, name: str = "Arm") -> None:
        super().__init__(name=name)

    def _extra_setup(self) -> None:
        self._client = self._node.create_client(
            srv_type=CommandBool, srv_name=_ARMING_SERVICE
        )
        self._future = None
        self._next_attempt_s = 0.0

    def initialise(self) -> None:
        self._future = None
        self._next_attempt_s = 0.0

    def update(self) -> py_trees.common.Status:
        if self._latest_state is None:
            self._node.get_logger().warning(
                f"{self.name}: waiting for '{_STATE_TOPIC}'",
                throttle_duration_sec=5.0,
            )
            return py_trees.common.Status.RUNNING

        if self._latest_state.armed:
            self._node.get_logger().info(f"{self.name}: armed")
            return py_trees.common.Status.SUCCESS

        if self._future is None:
            if self._now_s() < self._next_attempt_s:
                return py_trees.common.Status.RUNNING
            if not self._client.service_is_ready():
                self._node.get_logger().warning(
                    f"{self.name}: waiting for '{_ARMING_SERVICE}' service",
                    throttle_duration_sec=5.0,
                )
                return py_trees.common.Status.RUNNING

            request = CommandBool.Request()
            request.value = True
            self._future = self._client.call_async(request)
            self._node.get_logger().info(f"{self.name}: requesting arm")
            return py_trees.common.Status.RUNNING

        if self._future.done():
            response = self._future.result()
            self._future = None
            if response is None or not response.success:
                self._node.get_logger().warning(
                    f"{self.name}: arming rejected (pre-arm checks?), "
                    f"retrying in {ARM_RETRY_PERIOD_S:.0f}s"
                )
                self._next_attempt_s = self._now_s() + ARM_RETRY_PERIOD_S

        return py_trees.common.Status.RUNNING

    def terminate(self, new_status: py_trees.common.Status) -> None:
        if new_status != py_trees.common.Status.SUCCESS:
            self._future = None


class Takeoff(_MavrosBehaviour):
    """
    Tells the drone to take off and climb to ``TAKEOFF_ALTITUDE_M``.

    First it waits until the drone is in GUIDED and armed, which the two
    behaviors before this one already took care of. Then it sends the takeoff
    command and returns RUNNING until the drone's height is within
    ``TAKEOFF_ALTITUDE_TOLERANCE_M`` of the target. Returns FAILURE if the
    drone refuses the command, and SUCCESS right away if it's already up.
    """

    def __init__(self, name: str = "Takeoff") -> None:
        super().__init__(name=name)

    def _extra_setup(self) -> None:
        self._latest_rel_alt_m: float | None = None
        self._rel_alt_sub = self._node.create_subscription(
            msg_type=Float64,
            topic=_REL_ALT_TOPIC,
            callback=self._rel_alt_callback,
            qos_profile=qos_profile_sensor_data,
        )
        self._client = self._node.create_client(
            srv_type=CommandTOL, srv_name=_TAKEOFF_SERVICE
        )
        self._future = None
        self._accepted = False

    def _rel_alt_callback(self, msg: Float64) -> None:
        self._latest_rel_alt_m = msg.data

    def initialise(self) -> None:
        self._future = None
        self._accepted = False

    def update(self) -> py_trees.common.Status:
        if self._latest_state is None or self._latest_rel_alt_m is None:
            self._node.get_logger().warning(
                f"{self.name}: waiting for '{_STATE_TOPIC}' and '{_REL_ALT_TOPIC}'",
                throttle_duration_sec=5.0,
            )
            return py_trees.common.Status.RUNNING

        if (
            not self._accepted
            and self._latest_state.armed
            and self._latest_rel_alt_m >= TAKEOFF_AIRBORNE_THRESHOLD_M
        ):
            self._node.get_logger().info(
                f"{self.name}: already flying at {self._latest_rel_alt_m:.1f}m, "
                "skipping takeoff"
            )
            return py_trees.common.Status.SUCCESS

        if self._latest_state.mode != GUIDED_MODE or not self._latest_state.armed:
            self._node.get_logger().warning(
                f"{self.name}: waiting for GUIDED + armed "
                f"(mode={self._latest_state.mode}, armed={self._latest_state.armed})",
                throttle_duration_sec=5.0,
            )
            return py_trees.common.Status.RUNNING

        if not self._accepted:
            return self._command_takeoff()

        if self._latest_rel_alt_m >= TAKEOFF_ALTITUDE_M - TAKEOFF_ALTITUDE_TOLERANCE_M:
            self._node.get_logger().info(
                f"{self.name}: reached {self._latest_rel_alt_m:.1f}m"
            )
            return py_trees.common.Status.SUCCESS

        self._node.get_logger().info(
            f"{self.name}: climbing, {self._latest_rel_alt_m:.1f}m / "
            f"{TAKEOFF_ALTITUDE_M:.1f}m",
            throttle_duration_sec=2.0,
        )
        return py_trees.common.Status.RUNNING

    def _command_takeoff(self) -> py_trees.common.Status:
        """Send the takeoff command and check the answer.

        Async, so the tick that sends it doesn't block waiting for a reply.
        """
        if self._future is None:
            if not self._client.service_is_ready():
                self._node.get_logger().warning(
                    f"{self.name}: waiting for '{_TAKEOFF_SERVICE}' service",
                    throttle_duration_sec=5.0,
                )
                return py_trees.common.Status.RUNNING

            request = CommandTOL.Request()
            request.altitude = TAKEOFF_ALTITUDE_M
            self._future = self._client.call_async(request)
            self._node.get_logger().info(
                f"{self.name}: commanding takeoff to {TAKEOFF_ALTITUDE_M:.1f}m"
            )
            return py_trees.common.Status.RUNNING

        if not self._future.done():
            return py_trees.common.Status.RUNNING

        response = self._future.result()
        self._future = None
        if response is None or not response.success:
            self._node.get_logger().error(
                f"{self.name}: takeoff command rejected: {response}"
            )
            return py_trees.common.Status.FAILURE

        self._accepted = True
        return py_trees.common.Status.RUNNING

    def terminate(self, new_status: py_trees.common.Status) -> None:
        if new_status != py_trees.common.Status.SUCCESS:
            self._future = None
            self._accepted = False


class FlyToWaypoint(_MavrosBehaviour):
    """
    Flies to one waypoint by repeatedly telling MAVROS where to go.

    Returns RUNNING while the drone is on its way, SUCCESS once it gets
    within ``WAYPOINT_ACCEPTANCE_RADIUS_M`` of the waypoint, which also adds
    one to the ``mission/waypoint_index`` blackboard key, and FAILURE if it
    hasn't arrived after ``WAYPOINT_NAV_TIMEOUT_S`` seconds.
    """

    def __init__(self, name: str, waypoint: Coordinate, index: int) -> None:
        super().__init__(name=name)
        self._waypoint = waypoint
        self._index = index

        self.blackboard = self.attach_blackboard_client(name=self.name)
        self.blackboard.register_key(
            key=blackboard_keys.WAYPOINT_INDEX, access=py_trees.common.Access.WRITE
        )

    def _extra_setup(self) -> None:
        self._latest_fix: NavSatFix | None = None
        self._latest_rel_alt_m: float | None = None
        self._setpoint: GlobalPositionTarget | None = None
        self._start_time_s = 0.0

        self._setpoint_pub = self._node.create_publisher(
            msg_type=GlobalPositionTarget,
            topic=_SETPOINT_TOPIC,
            qos_profile=10,
        )
        self._fix_sub = self._node.create_subscription(
            msg_type=NavSatFix,
            topic=_GLOBAL_POSITION_TOPIC,
            callback=self._fix_callback,
            qos_profile=qos_profile_sensor_data,
        )
        self._rel_alt_sub = self._node.create_subscription(
            msg_type=Float64,
            topic=_REL_ALT_TOPIC,
            callback=self._rel_alt_callback,
            qos_profile=qos_profile_sensor_data,
        )

    def _fix_callback(self, msg: NavSatFix) -> None:
        self._latest_fix = msg

    def _rel_alt_callback(self, msg: Float64) -> None:
        self._latest_rel_alt_m = msg.data

    def initialise(self) -> None:
        self._setpoint = GlobalPositionTarget()
        self._setpoint.coordinate_frame = GlobalPositionTarget.FRAME_GLOBAL_REL_ALT
        self._setpoint.type_mask = _TYPE_MASK
        self._setpoint.latitude = self._waypoint.lat
        self._setpoint.longitude = self._waypoint.lon
        self._setpoint.altitude = self._waypoint.alt

        self._start_time_s = self._now_s()
        self._node.get_logger().info(f"{self.name}: flying to {self._waypoint}")

    def update(self) -> py_trees.common.Status:
        if self._setpoint is None:
            self._node.get_logger().error(f"{self.name}: no setpoint")
            return py_trees.common.Status.FAILURE

        if self._now_s() - self._start_time_s > WAYPOINT_NAV_TIMEOUT_S:
            self._node.get_logger().error(
                f"{self.name}: waypoint not reached within {WAYPOINT_NAV_TIMEOUT_S}s"
            )
            return py_trees.common.Status.FAILURE

        if (
            self._latest_state is None
            or self._latest_fix is None
            or self._latest_rel_alt_m is None
        ):
            self._node.get_logger().warning(
                f"{self.name}: waiting for '{_STATE_TOPIC}', "
                f"'{_GLOBAL_POSITION_TOPIC}' and '{_REL_ALT_TOPIC}'",
                throttle_duration_sec=5.0,
            )
            return py_trees.common.Status.RUNNING

        if self._latest_state.mode != GUIDED_MODE:
            self._node.get_logger().warning(
                f"{self.name}: flight controller in '{self._latest_state.mode}' "
                f"mode, not '{GUIDED_MODE}' - holding off on setpoints",
                throttle_duration_sec=5.0,
            )
            return py_trees.common.Status.RUNNING

        self._setpoint.header.stamp = self._node.get_clock().now().to_msg()
        self._setpoint_pub.publish(self._setpoint)

        east_m, north_m = east_north_coordinate_offset_m(
            self._latest_fix.latitude,
            self._latest_fix.longitude,
            self._setpoint.latitude,
            self._setpoint.longitude,
        )
        up_m = self._setpoint.altitude - self._latest_rel_alt_m
        distance = math.sqrt(east_m**2 + north_m**2 + up_m**2)

        if distance <= WAYPOINT_ACCEPTANCE_RADIUS_M:
            self._node.get_logger().info(
                f"{self.name}: reached waypoint ({distance:.2f}m away)"
            )
            self.blackboard.set(blackboard_keys.WAYPOINT_INDEX, self._index + 1)
            return py_trees.common.Status.SUCCESS

        self._node.get_logger().info(
            f"{self.name}: {distance:.2f}m to waypoint",
            throttle_duration_sec=2.0,
        )
        return py_trees.common.Status.RUNNING

    def terminate(self, new_status: py_trees.common.Status) -> None:
        if new_status != py_trees.common.Status.SUCCESS:
            self._setpoint = None


class Land(_MavrosBehaviour):
    """
    Lands the drone right where it is.

    Sends the land command once, then returns RUNNING until the drone
    disarms on its own after touching down. Returns FAILURE if the drone
    refuses the command.
    """

    def __init__(self, name: str = "Land") -> None:
        super().__init__(name=name)

    def _extra_setup(self) -> None:
        self._client = self._node.create_client(
            srv_type=CommandTOL, srv_name=_LAND_SERVICE
        )
        self._future = None
        self._accepted = False

    def initialise(self) -> None:
        self._future = None
        self._accepted = False

    def update(self) -> py_trees.common.Status:
        if self._latest_state is None:
            self._node.get_logger().warning(
                f"{self.name}: waiting for '{_STATE_TOPIC}'",
                throttle_duration_sec=5.0,
            )
            return py_trees.common.Status.RUNNING

        if not self._accepted:
            return self._command_land()

        if not self._latest_state.armed:
            self._node.get_logger().info(f"{self.name}: landed and disarmed")
            return py_trees.common.Status.SUCCESS

        self._node.get_logger().info(
            f"{self.name}: descending",
            throttle_duration_sec=2.0,
        )
        return py_trees.common.Status.RUNNING

    def _command_land(self) -> py_trees.common.Status:
        """Send the land command and check the answer.

        Async, so the tick that sends it doesn't block waiting for a reply.
        """
        if self._future is None:
            if not self._client.service_is_ready():
                self._node.get_logger().warning(
                    f"{self.name}: waiting for '{_LAND_SERVICE}' service",
                    throttle_duration_sec=5.0,
                )
                return py_trees.common.Status.RUNNING

            self._future = self._client.call_async(CommandTOL.Request())
            self._node.get_logger().info(f"{self.name}: commanding land")
            return py_trees.common.Status.RUNNING

        if not self._future.done():
            return py_trees.common.Status.RUNNING

        response = self._future.result()
        self._future = None
        if response is None or not response.success:
            self._node.get_logger().error(
                f"{self.name}: land command rejected: {response}"
            )
            return py_trees.common.Status.FAILURE

        self._accepted = True
        return py_trees.common.Status.RUNNING

    def terminate(self, new_status: py_trees.common.Status) -> None:
        if new_status != py_trees.common.Status.SUCCESS:
            self._future = None
            self._accepted = False
