"""
Tests for CaptureForPerception.

Your ``engine/behaviors/capture_for_perception.py`` has to pass all of
these, unchanged.
"""

from __future__ import annotations

import py_trees
from conftest import make_camera_frame

from engine import blackboard_keys
from engine.behaviors.capture_for_perception import CaptureForPerception

RUNNING = py_trees.common.Status.RUNNING
SUCCESS = py_trees.common.Status.SUCCESS
FAILURE = py_trees.common.Status.FAILURE


def _tick(behaviour: py_trees.behaviour.Behaviour) -> py_trees.common.Status:
    behaviour.tick_once()
    return behaviour.status


def test_fresh_frame_is_published_and_succeeds(board, publisher):
    stale = make_camera_frame(index=4)
    fresh = make_camera_frame(index=5)
    board.set(blackboard_keys.LATEST_FRAME, stale)

    behaviour = CaptureForPerception("Capture", publisher, timeout_ticks=10)

    # The first tick writes down index 4 to compare against. That frame was
    # already there before we arrived, so it's the old one.
    assert _tick(behaviour) == RUNNING
    assert publisher.images == []
    assert publisher.statuses == []

    # A new frame shows up, so send it out and succeed.
    board.set(blackboard_keys.LATEST_FRAME, fresh)
    assert _tick(behaviour) == SUCCESS

    # Exactly one image, the new frame, and one status dict shaped like
    # this and nothing else.
    assert publisher.images == [fresh]
    assert publisher.statuses == [{"phase": "capture", "frame_index": 5}]


def test_no_frame_source_times_out(publisher):
    # Nothing ever writes LATEST_FRAME. That's RUNNING for the first
    # timeout_ticks - 1 calls, then FAILURE on the last one.
    behaviour = CaptureForPerception("Capture", publisher, timeout_ticks=3)

    assert _tick(behaviour) == RUNNING
    assert _tick(behaviour) == RUNNING
    assert _tick(behaviour) == FAILURE

    assert publisher.images == []
    assert publisher.statuses == []


def test_stale_frame_times_out(board, publisher):
    # The same frame sits there the whole time, so nothing is ever new.
    board.set(blackboard_keys.LATEST_FRAME, make_camera_frame(index=7))
    behaviour = CaptureForPerception("Capture", publisher, timeout_ticks=2)

    assert _tick(behaviour) == RUNNING
    assert _tick(behaviour) == FAILURE

    assert publisher.images == []
    assert publisher.statuses == []


def test_frame_arriving_after_start_is_fresh(board, publisher):
    # There's no frame at all when the behavior starts, so there's nothing
    # to compare against. The first frame to show up counts as new, even
    # though its index is 0.
    behaviour = CaptureForPerception("Capture", publisher, timeout_ticks=10)
    assert _tick(behaviour) == RUNNING

    first = make_camera_frame(index=0)
    board.set(blackboard_keys.LATEST_FRAME, first)
    assert _tick(behaviour) == SUCCESS
    assert publisher.images == [first]
    assert publisher.statuses == [{"phase": "capture", "frame_index": 0}]


def test_baseline_resets_across_attempts(board, publisher):
    # After a capture succeeds, the next attempt starts over and has to
    # write down a new index. The frame it just took is now the old one.
    behaviour = CaptureForPerception("Capture", publisher, timeout_ticks=10)

    assert _tick(behaviour) == RUNNING  # nothing to compare to, no frame yet

    frame_1 = make_camera_frame(index=1)
    board.set(blackboard_keys.LATEST_FRAME, frame_1)
    assert _tick(behaviour) == SUCCESS  # index 1 is different from nothing
    assert publisher.images == [frame_1]

    # Second attempt. Since the last status was SUCCESS, py_trees calls
    # initialise() again. Frame 1 is still sitting on the blackboard, and
    # this time it counts as the old one.
    assert _tick(behaviour) == RUNNING
    assert publisher.images == [frame_1]

    frame_2 = make_camera_frame(index=2)
    board.set(blackboard_keys.LATEST_FRAME, frame_2)
    assert _tick(behaviour) == SUCCESS
    assert publisher.images == [frame_1, frame_2]
    assert publisher.statuses == [
        {"phase": "capture", "frame_index": 1},
        {"phase": "capture", "frame_index": 2},
    ]


def test_timeout_counter_resets_across_attempts(board, publisher):
    # The tick counter belongs to one attempt. After an attempt fails, the
    # next one gets the full count again.
    behaviour = CaptureForPerception("Capture", publisher, timeout_ticks=2)

    assert _tick(behaviour) == RUNNING
    assert _tick(behaviour) == FAILURE

    # New attempt, full count again, so the first tick is RUNNING instead
    # of failing right away.
    assert _tick(behaviour) == RUNNING
    board.set(blackboard_keys.LATEST_FRAME, make_camera_frame(index=9))
    assert _tick(behaviour) == SUCCESS
