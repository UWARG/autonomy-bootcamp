"""
Tests for the perception sweep.

Your ``engine/subtrees/perception.py`` has to pass all of these, unchanged.

The factories hand back fakes that return scripted statuses and log every
call, so the tests can check the exact order without flying anything.
"""

from __future__ import annotations

import py_trees

from engine.subtrees.perception import create_perception_sweep

RUNNING = py_trees.common.Status.RUNNING
SUCCESS = py_trees.common.Status.SUCCESS
FAILURE = py_trees.common.Status.FAILURE


class ScriptedBehaviour(py_trees.behaviour.Behaviour):
    """
    Returns statuses from a list, one per update(), repeating the last one
    forever. Logs every call, which is how the tests check ordering.
    """

    def __init__(self, name: str, statuses: list, log: list) -> None:
        super().__init__(name=name)
        self._statuses = list(statuses)
        self._log = log
        self._calls = 0

    def update(self) -> py_trees.common.Status:
        self._log.append(self.name)
        status = self._statuses[min(self._calls, len(self._statuses) - 1)]
        self._calls += 1
        return status


def _tick_until_done(
    root: py_trees.behaviour.Behaviour, limit: int = 25
) -> py_trees.common.Status:
    """Tick until the root stops being RUNNING.

    The limit stops a tree that never finishes from hanging pytest.
    """
    for _ in range(limit):
        root.tick_once()
        if root.status != RUNNING:
            break
    return root.status


def _ordered_unique(log: list) -> list:
    """The names from the log, in the order each one first showed up."""
    seen: dict = {}
    for name in log:
        seen.setdefault(name, None)
    return list(seen)


def test_visits_and_captures_every_waypoint_in_order():
    log: list = []
    fly_indices: list = []
    capture_indices: list = []

    def fly_factory(index: int) -> py_trees.behaviour.Behaviour:
        fly_indices.append(index)
        # Flying somewhere takes time, so RUNNING once, then SUCCESS.
        return ScriptedBehaviour(f"fly{index}", [RUNNING, SUCCESS], log)

    def capture_factory(index: int) -> py_trees.behaviour.Behaviour:
        capture_indices.append(index)
        return ScriptedBehaviour(f"capture{index}", [SUCCESS], log)

    root = create_perception_sweep(3, fly_factory, capture_factory)

    assert _tick_until_done(root) == SUCCESS

    # Both factories got called once per waypoint, in order.
    assert fly_indices == [0, 1, 2]
    assert capture_indices == [0, 1, 2]

    # Fly, then capture, one waypoint at a time, in exactly this order.
    assert _ordered_unique(log) == [
        "fly0",
        "capture0",
        "fly1",
        "capture1",
        "fly2",
        "capture2",
    ]

    # A child that already finished must not run again later, since a
    # capture running twice would send the same picture out twice. So each
    # fly runs exactly twice (RUNNING then SUCCESS) and each capture once.
    for index in range(3):
        assert log.count(f"fly{index}") == 2
        assert log.count(f"capture{index}") == 1


def test_middle_fly_failure_fails_the_sweep():
    log: list = []

    def fly_factory(index: int) -> py_trees.behaviour.Behaviour:
        statuses = [FAILURE] if index == 1 else [SUCCESS]
        return ScriptedBehaviour(f"fly{index}", statuses, log)

    def capture_factory(index: int) -> py_trees.behaviour.Behaviour:
        return ScriptedBehaviour(f"capture{index}", [SUCCESS], log)

    root = create_perception_sweep(3, fly_factory, capture_factory)

    assert _tick_until_done(root) == FAILURE

    # The failure ends the sweep, so nothing after fly1 ever runs.
    assert "capture1" not in log
    assert "fly2" not in log
    assert "capture2" not in log
    # The first waypoint still finished normally though.
    assert _ordered_unique(log) == ["fly0", "capture0", "fly1"]


def test_capture_failure_fails_the_sweep():
    log: list = []

    def fly_factory(index: int) -> py_trees.behaviour.Behaviour:
        return ScriptedBehaviour(f"fly{index}", [SUCCESS], log)

    def capture_factory(index: int) -> py_trees.behaviour.Behaviour:
        statuses = [FAILURE] if index == 0 else [SUCCESS]
        return ScriptedBehaviour(f"capture{index}", statuses, log)

    root = create_perception_sweep(2, fly_factory, capture_factory)

    assert _tick_until_done(root) == FAILURE
    assert _ordered_unique(log) == ["fly0", "capture0"]


def test_single_waypoint_sweep():
    log: list = []

    root = create_perception_sweep(
        1,
        lambda i: ScriptedBehaviour(f"fly{i}", [SUCCESS], log),
        lambda i: ScriptedBehaviour(f"capture{i}", [SUCCESS], log),
    )

    assert _tick_until_done(root) == SUCCESS
    assert _ordered_unique(log) == ["fly0", "capture0"]
