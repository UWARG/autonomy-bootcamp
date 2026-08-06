"""
Tests for the WaitForReady example.

Finished, and the pattern to copy: write the world to the blackboard, tick
once, check the status.
"""

from __future__ import annotations

import py_trees

from engine import blackboard_keys
from engine.behaviors.wait_for_ready import WaitForReady

RUNNING = py_trees.common.Status.RUNNING
SUCCESS = py_trees.common.Status.SUCCESS


def _tick(behaviour: py_trees.behaviour.Behaviour) -> py_trees.common.Status:
    """Ticks once and gives you back the status."""
    behaviour.tick_once()
    return behaviour.status


def _publish_state(board, connected: bool, mode: str, armed: bool) -> None:
    board.set(blackboard_keys.VEHICLE_CONNECTED, connected)
    board.set(blackboard_keys.VEHICLE_MODE, mode)
    board.set(blackboard_keys.VEHICLE_ARMED, armed)


def test_runs_when_no_telemetry_has_arrived():
    # Nothing on the blackboard at all, so the drone's status hasn't come
    # through yet. That's normal at startup, so it stays RUNNING.
    behaviour = WaitForReady()
    assert _tick(behaviour) == RUNNING


def test_runs_when_only_some_keys_are_set(board):
    # Only some of the keys are set. That must not crash the behavior.
    board.set(blackboard_keys.VEHICLE_CONNECTED, True)
    behaviour = WaitForReady()
    assert _tick(behaviour) == RUNNING


def test_runs_until_connected(board):
    _publish_state(board, connected=False, mode="GUIDED", armed=True)
    behaviour = WaitForReady()
    assert _tick(behaviour) == RUNNING


def test_runs_until_guided(board):
    _publish_state(board, connected=True, mode="STABILIZE", armed=True)
    behaviour = WaitForReady()
    assert _tick(behaviour) == RUNNING


def test_runs_until_armed(board):
    _publish_state(board, connected=True, mode="GUIDED", armed=False)
    behaviour = WaitForReady()
    assert _tick(behaviour) == RUNNING


def test_succeeds_when_fully_ready(board):
    _publish_state(board, connected=True, mode="GUIDED", armed=True)
    behaviour = WaitForReady()
    assert _tick(behaviour) == SUCCESS


def test_transitions_to_success_as_telemetry_updates(board):
    # What normally happens: RUNNING for a while, then the drone becomes
    # ready and the next tick succeeds.
    behaviour = WaitForReady()
    _publish_state(board, connected=True, mode="STABILIZE", armed=False)
    assert _tick(behaviour) == RUNNING

    _publish_state(board, connected=True, mode="GUIDED", armed=False)
    assert _tick(behaviour) == RUNNING

    _publish_state(board, connected=True, mode="GUIDED", armed=True)
    assert _tick(behaviour) == SUCCESS
