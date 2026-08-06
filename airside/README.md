# Part 4: Build a Behavior Tree

The airside engine flies missions using a **behavior tree**: a bunch of small classes, each doing one job, that get run a few times a second. Every one of them answers the same question every time it runs: am I still working (`RUNNING`), did I finish (`SUCCESS`), or did I give up (`FAILURE`)?

In this part you write one of those small classes and put a few of them together. The result makes the drone fly to each waypoint and take a picture there.

Everything you write is plain Python and py_trees. No ROS, no Docker. The same code runs inside the Part 5 container, where code we already wrote hooks it up to the real drone and to your Part 2 `SimCamera`.

## Behavior trees in five minutes

A behavior is a class. py_trees calls these methods for you, so you never call them yourself:

| Method                  | When py_trees calls it                            | What you put in it                                                                                   |
| ----------------------- | ------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| `__init__`              | Once, when the tree is built                      | Save settings, say which blackboard keys you'll use                                                  |
| `setup(**kwargs)`       | Once, during `tree.setup()`                       | Grab outside resources (ROS handles, only in the container)                                          |
| `initialise()`          | Every time this behavior starts running again     | Reset anything you count or compare against                                                          |
| `update()`              | Every tick, while this behavior is the active one | Check what's going on and return `RUNNING`, `SUCCESS`, or `FAILURE`. Return quickly, never wait here |
| `terminate(new_status)` | Whenever the behavior stops                       | Clean up                                                                                             |

One run through the tree is called a **tick**. Behaviors are grouped under **composites**, which decide what runs next:

| Type       | What it does                                                                                            |
| ---------- | ------------------------------------------------------------------------------------------------------- |
| `Sequence` | Runs its children left to right. Fails as soon as one child fails. Succeeds only if all of them succeed |
| `Selector` | Runs its children left to right. Succeeds as soon as one child succeeds. Fails only if all of them fail |
| `Parallel` | Runs all its children every tick and combines the results                                               |

One setting matters a lot here. `Sequence(memory=True)` picks up on the next tick where it left off. `Sequence(memory=False)` starts over from the first child every tick, which means it redoes work that already succeeded.

## The blackboard

Behaviors never call each other directly. They pass data through the **blackboard**, which is basically one big shared dictionary:

```python
self.blackboard = self.attach_blackboard_client(name=self.name)
self.blackboard.register_key(key="vehicle/mode", access=py_trees.common.Access.READ)

mode = self.blackboard.get("vehicle/mode")   # KeyError if nobody has written it yet
```

Every behavior has to say up front which keys it reads and which it writes (`READ`, `WRITE`, `EXCLUSIVE_WRITE`). The key names all live in `src/engine/blackboard_keys.py`. Import them from there instead of typing the strings yourself, makes things a lot easier to debug.

Reading a key nobody has written yet throws a `KeyError`. That happens all the time at startup and it isn't a bug, so your behaviors have to handle it (usually by returning `RUNNING` and trying again next tick).

Who writes what: inside the container, `ros/telemetry.py` fills in the `vehicle/*` keys from the drone and `ros/camera_source.py` fills in `camera/latest_frame` from your `SimCamera`. In the tests you run on your laptop, the fixtures in `tests/conftest.py` write those same keys. Your behaviors can't tell the two apart, which is the whole point.

## Layout

```text
airside/src/engine/
├── blackboard_keys.py                  # The key names everything agrees on
├── behaviors/
│   ├── wait_for_ready.py               # FINISHED example, read this first
│   └── capture_for_perception.py       # YOU write this
├── subtrees/
│   └── perception.py                   # YOU put this together
├── tests/
│   ├── conftest.py                     # Test setup: blackboard writer, fake publisher
│   ├── test_wait_for_ready.py          # Tests for the finished example, read second
│   ├── test_capture.py                 # Tests your behavior has to pass
│   └── test_perception.py              # Tests your subtree has to pass
├── ros/                                # WE wrote this, container only (read it, don't change it)
├── launch/perception.launch.py         # WE wrote this: starts mavros + the mission
└── config/waypoints.yaml               # Home + three waypoints to visit
```

## The assignment

```bash
warg up airside
warg run airside setup
```

1. **Read the finished example.** `behaviors/wait_for_ready.py` and `tests/test_wait_for_ready.py` show you the whole pattern: read the blackboard in `update()`, deal with keys that aren't there yet, return a status.

2. **Write `CaptureForPerception`** (`behaviors/capture_for_perception.py`). The full description is in the comment at the top of the class, and `tests/test_capture.py` says exactly what it has to do. The idea: when you get to a waypoint there's probably already a frame sitting on the blackboard. Ignore it. Wait until a frame shows up with an `index` you haven't seen, send that one to the publisher you were given, and return `SUCCESS`. If `timeout_ticks` ticks go by without a new frame, return `FAILURE`.

3. **Write `create_perception_sweep`** (`subtrees/perception.py`). No new behavior classes here. Take the behaviors the two factory functions give you and arrange them so the drone flies to waypoint 0, takes a picture, flies to waypoint 1, takes a picture, and so on, and so the whole thing fails if any single step fails.

Keep going until both of these pass:

```bash
warg run airside test
warg run airside lint
```

## About `src/engine/ros/`

This is the code we wrote to connect your behaviors to the real drone. It only gets imported inside the Part 5 container, because it needs ROS 2 (rclpy, MAVROS), which you don't have on your laptop. That's fine and expected. Read it if you're curious, don't change it:

- `telemetry.py`: takes the drone's status from MAVROS and writes it to the `vehicle/*` keys.
- `camera_source.py`: runs your Part 2 `SimCamera` and writes each frame to `camera/latest_frame`.
- `perception_publisher.py`: the real publisher that gets handed to your behavior. It sends the image on `/perception/image` and a status message on `/perception/status`.
- `flight.py`: the behaviors that actually fly the drone: `SetModeGuided`, `Arm`, `Takeoff`, `FlyToWaypoint`, `Land`.
- `mission.py`: builds the full tree, with your code in the middle of it:

  ```text
  Mission [Sequence]
  ├── SetModeGuided
  ├── Arm
  ├── WaitForReady                  <- the finished example
  ├── Takeoff
  ├── PerceptionSweep               <- your create_perception_sweep
  │   ├── FlyToWaypoint0
  │   ├── Capture0                  <- your CaptureForPerception
  │   └── ... (waypoints 1, 2)
  └── Land
  ```

  It ticks the tree every 500 ms, stops the mission if it takes too long, and writes `mission_result.json`, which is the file the Part 5 tests read to grade your run.

In Part 5 you run this exact tree against a simulated drone without changing anything.
