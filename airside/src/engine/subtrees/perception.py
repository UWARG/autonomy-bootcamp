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

    perception_sweep_tasks = []

    for i in range(waypoint_count):
        perception_sweep_tasks.append(fly_factory(i))
        perception_sweep_tasks.append(capture_factory(i))

    #What do I put as the name???
    return py_trees.composites.Sequence(name="perception sweep", memory=True, children = perception_sweep_tasks)