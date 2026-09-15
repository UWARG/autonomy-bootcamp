from __future__ import annotations

from collections.abc import Callable

import py_trees


def create_perception_sweep(
    waypoint_count: int,
    fly_factory: Callable[[int], py_trees.behaviour.Behaviour],
    capture_factory: Callable[[int], py_trees.behaviour.Behaviour],
) -> py_trees.behaviour.Behaviour:
    sequence = py_trees.composites.Sequence(
        name="Perception Sweep", memory=True
    )

    for i in range(waypoint_count):
        sequence.add_child(fly_factory(i))
        sequence.add_child(capture_factory(i))

    return sequence