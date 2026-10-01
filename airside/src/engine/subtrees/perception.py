"""
The perception sweep: YOU put this together. Details in the function below.

No new behavior classes. You arrange existing ones under a composite. Make
``tests/test_perception.py`` pass:

    warg run airside test
"""

from __future__ import annotations

from collections.abc import Callable

import py_trees


def create_perception_sweep(
    waypoint_count: int,
    fly_factory: Callable[[int], py_trees.behaviour.Behaviour],
    capture_factory: Callable[[int], py_trees.behaviour.Behaviour],
) -> py_trees.behaviour.Behaviour:

    children = []

    for i in range(waypoint_count):
        children.append(fly_factory(i))
        children.append(capture_factory(i))

    sweep = py_trees.composites.Sequence(
        name="PerceptionSweep",
        memory=True,
        children=children,
    )

    return sweep