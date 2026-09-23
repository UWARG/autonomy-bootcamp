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
    """
    Builds the part of the tree that visits every waypoint in order.

    For each waypoint ``i`` from ``0`` to ``waypoint_count - 1``, the drone
    first flies there (that's the behavior ``fly_factory(i)`` gives you) and
    then takes a picture (that's ``capture_factory(i)``), in exactly this
    order:

        fly(0), capture(0), fly(1), capture(1), ..., fly(n-1), capture(n-1)

    What it has to do (``tests/test_perception.py`` checks all of this):

    - Children run in exactly the order above. A capture never runs before
      its fly succeeds, and waypoint ``i + 1`` never starts before waypoint
      ``i`` is completely finished.
    - If any child returns FAILURE, the whole thing returns FAILURE, and the
      mission gives up on the sweep and goes to land.
    - It returns SUCCESS only once *every* waypoint has been flown to and
      captured, in order.
    - A child that already succeeded must not run again on a later tick.
      Look at the ``memory`` argument on py_trees composites for this. If a
      finished capture ran again it would send the same picture out twice.

    You get the two factories handed to you rather than making the behaviors
    yourself, so this same code works with the real flying and capturing
    behaviors in the container and with fake ones in the tests.

    Args:
        waypoint_count: How many waypoints to visit.
        fly_factory: Call it with a waypoint index, get back the behavior
            that flies there.
        capture_factory: Call it with a waypoint index, get back the
            behavior that takes the picture there.

    Returns:
        The behavior at the top of the sweep.
    """
    root = py_trees.composites.Sequence(
        #cretes a var root and assigns it to a sequence composite, a sequence is a behaviour
        # that runs its children from left to right
        #if child returns sucess the sequence moves one, if its RUNNING the sequence waits, 
        #if its FAILURE the sequence returns FAILURE
        name="PerceptionSweep",
        #Give this behaviour a readable name, so if the behaviour tree gets printed you will see
        #PerceptionSweep
        memory=True,
        #"Remember which child I was working on between ticks"
        #e.g fly0 -> Sucess
        #capture0 -> Sucess
        #fly1 -> Running
        #becus of memory = true the sequence cnotinues at fly1
    )
    for index in range(waypoint_count):
        fly_behaviour = fly_factory(index)
        #for index0 fly_factory(0) creates something like fly0 and we assign flybehaviour to that
        capture_behaviour = capture_factory(index)
        #same thing has fly_factory except now its with capture

        root.add_children( # simply creating them isnt enough as they HAVE TO become children of the Sequence
            [
                fly_behaviour,
                capture_behaviour,
            ]
        )
        #we put it into a list, with flybehave coming first then capture

    return root# the function return type is py_trees.behaviour.Behaviour and the sequence itself
# is a behaviour so returning root satisfies that requirement
