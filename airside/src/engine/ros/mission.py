"""
Where the mission starts. Container only.

Builds the tree around your behaviors, ticks it, and writes
``mission_result.json`` for the Part 5 tests to grade.

Configured by environment so compose can drive it: ``MISSION_MODE``
(``"smoke"`` or ``"perception"``), ``MISSION_TIMEOUT_S``,
``WAYPOINTS_FILE``, ``MISSION_RESULT_PATH``.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import py_trees
import rclpy
from utils.src.waypoint_utils import parse_waypoints_file, sort_clockwise_sweep

from engine import blackboard_keys
from engine.behaviors.capture_for_perception import CaptureForPerception
from engine.behaviors.wait_for_ready import WaitForReady
from engine.ros.camera_source import CameraSource
from engine.ros.flight import Arm, FlyToWaypoint, Land, SetModeGuided, Takeoff
from engine.ros.perception_publisher import PerceptionPublisher
from engine.ros.telemetry import TelemetryBridge
from engine.subtrees.perception import create_perception_sweep

# How often to tick the tree, in seconds.
TICK_PERIOD_S = 0.5

# How many ticks each CaptureForPerception gets (20 ticks is 10 seconds).
CAPTURE_TIMEOUT_TICKS = 20

_DEFAULT_TIMEOUT_S = 150.0
_DEFAULT_WAYPOINTS_FILE = Path(__file__).resolve().parents[1] / "config" / "waypoints.yaml"
_DEFAULT_RESULT_PATH = "/results/mission_result.json"

_SUCCESS = py_trees.common.Status.SUCCESS
_FAILURE = py_trees.common.Status.FAILURE


def _build_tree(mode, waypoints, publisher):
    """
    Build the mission tree.

    Also returns the handles the result file needs: ``phases`` maps a step
    name to the behavior whose SUCCESS means it's done, and
    ``fly_behaviours`` is every FlyToWaypoint, for counting waypoints.
    """
    set_mode = SetModeGuided()
    arm = Arm()
    ready = WaitForReady()
    takeoff = Takeoff()
    land = Land()

    phases = {
        "set_mode_guided": set_mode,
        "arm": arm,
        "wait_for_ready": ready,
        "takeoff": takeoff,
        "land": land,
    }
    children = [set_mode, arm, ready, takeoff]
    fly_behaviours = []

    if mode == "perception":

        def fly_factory(index: int) -> py_trees.behaviour.Behaviour:
            behaviour = FlyToWaypoint(
                name=f"FlyToWaypoint{index}",
                waypoint=waypoints[index],
                index=index,
            )
            fly_behaviours.append(behaviour)
            return behaviour

        def capture_factory(index: int) -> py_trees.behaviour.Behaviour:
            return CaptureForPerception(
                name=f"Capture{index}",
                publisher=publisher,
                timeout_ticks=CAPTURE_TIMEOUT_TICKS,
            )

        sweep = create_perception_sweep(len(waypoints), fly_factory, capture_factory)
        phases["sweep"] = sweep
        children.append(sweep)

    children.append(land)
    root = py_trees.composites.Sequence(name="Mission", memory=True, children=children)
    return root, phases, fly_behaviours


def main() -> int:
    mode = os.environ.get("MISSION_MODE", "perception")
    if mode not in ("smoke", "perception"):
        print(f"unknown MISSION_MODE {mode!r}, falling back to 'perception'")
        mode = "perception"
    timeout_s = float(os.environ.get("MISSION_TIMEOUT_S", str(_DEFAULT_TIMEOUT_S)))
    waypoints_file = os.environ.get("WAYPOINTS_FILE", str(_DEFAULT_WAYPOINTS_FILE))
    result_path = Path(os.environ.get("MISSION_RESULT_PATH", _DEFAULT_RESULT_PATH))

    home, raw_waypoints = parse_waypoints_file(waypoints_file)
    waypoints = sort_clockwise_sweep(raw_waypoints, home)

    rclpy.init()
    node = rclpy.create_node("perception_mission")

    # Hold on to these. They own subscriptions and timers on the node and
    # write to the blackboard from their callbacks, so if we let them get
    # garbage collected the mission would go blind.
    _telemetry = TelemetryBridge(node)
    camera = CameraSource(node)
    publisher = PerceptionPublisher(node)

    # Fill in the mission keys on the blackboard before we start.
    board = py_trees.blackboard.Client(name="MissionInit")
    board.register_key(
        key=blackboard_keys.WAYPOINTS, access=py_trees.common.Access.WRITE
    )
    board.register_key(
        key=blackboard_keys.WAYPOINT_INDEX, access=py_trees.common.Access.WRITE
    )
    board.set(blackboard_keys.WAYPOINTS, waypoints)
    board.set(blackboard_keys.WAYPOINT_INDEX, 0)

    root, phases, fly_behaviours = _build_tree(mode, waypoints, publisher)
    for behaviour in root.iterate():
        behaviour.setup(node=node)

    completed = {name: False for name in phases}
    flown = {behaviour.name: False for behaviour in fly_behaviours}

    start = time.monotonic()
    deadline = start + timeout_s
    next_tick_at = start
    detail = "mission did not complete"

    node.get_logger().info(
        f"mission starting: mode={mode}, timeout={timeout_s:.0f}s, "
        f"{len(waypoints)} waypoints"
    )

    try:
        while rclpy.ok():
            if time.monotonic() >= deadline:
                detail = f"timed out after {timeout_s:.0f}s"
                break

            rclpy.spin_once(node, timeout_sec=0.05)
            if time.monotonic() < next_tick_at:
                continue
            next_tick_at = time.monotonic() + TICK_PERIOD_S

            root.tick_once()

            # Once a step reports SUCCESS we remember it as done, even if
            # the tree resets its status later on.
            for name, behaviour in phases.items():
                if behaviour.status == _SUCCESS:
                    completed[name] = True
            for behaviour in fly_behaviours:
                if behaviour.status == _SUCCESS:
                    flown[behaviour.name] = True

            if root.status == _SUCCESS:
                detail = "mission completed"
                break
            if root.status == _FAILURE:
                tip = root.tip()
                detail = f"tree returned FAILURE (tip: {tip.name if tip else 'unknown'})"
                break
    except KeyboardInterrupt:
        detail = "interrupted"

    success = root.status == _SUCCESS
    result = {
        "result": "success" if success else "failure",
        "mode": mode,
        "phases": completed,
        "captures": publisher.image_count,
        "waypoints_visited": sum(flown.values()),
        "duration_s": round(time.monotonic() - start, 1),
        "detail": detail,
    }

    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    node.get_logger().info(f"mission result: {json.dumps(result)}")

    camera.stop()
    node.destroy_node()
    rclpy.try_shutdown()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
