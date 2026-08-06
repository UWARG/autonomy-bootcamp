# Part 2: Add a Camera

We fly a few different cameras, some real hardware and some simulated. They all have the same methods with the same names, so the rest of our code can use any of them without caring which one it got.

In this part you read a camera we already wrote, then write your own: a `SimCamera` that makes up fake frames instead of reading real hardware. Part 4 and Part 5 will take pictures with it.

## Setup

From the repository root:

```bash
warg up camera
warg run camera setup
```

## The files

| File                         | What it is                                                          |
| ---------------------------- | ------------------------------------------------------------------- |
| `src/abstract_camera.py`     | The base class every camera inherits from. **Read this first.**     |
| `src/frame.py`               | `CameraFrame`, the object you get back from a capture               |
| `src/fixed.py`               | `FixedCamera`, a finished camera you can copy from                  |
| `tests/test_fixed_camera.py` | The tests for `FixedCamera`                                         |
| `src/sim.py`                 | `SimCamera`, **your assignment** (empty methods for you to fill in) |
| `tests/test_sim_camera.py`   | The tests you have to pass. **Do not modify.**                      |

Every camera has three methods:

```text
                                    ┌──────────────────┐
                                    │  capture_frame() │
                                    ▼                  │
  off ──── initialize_camera() ───► on ────────────────┘
   ▲                                │
   └─────────── stop() ─────────────┘
```

A camera starts off, so nothing works until you initialize it.

The rules your camera has to follow:

- Calling `capture_frame()` before `initialize_camera()` or after `stop()` raises a `RuntimeError`.
- Frames come out in order. Their `index` goes 0, 1, 2, ... and each frame's timestamp is bigger than the last one's.
- For this bootcamp you can assume `capture_frame()` always succeeds, so it always gives you a frame and there's no failure case to handle. (`FixedCamera` loops back to the start of its list; your `SimCamera` just makes up the next image.)
- Each capture returns a **copy** of the image. If someone changes the array you handed them, the next frame you return must not change.

## Your task

1. Read `src/fixed.py` and `tests/test_fixed_camera.py` side by side. The tests go through the four methods in order, and `FixedCamera` shows you the three tricks you'll need: remembering whether the camera is on, making sure timestamps always go up, and copying the image before returning it. Copying this code is fine and expected.
2. Read the comment at the top of `src/sim.py`, then open `tests/test_sim_camera.py`. Those tests are the assignment: make them pass.
3. Fill in `SimCamera`, replacing every `TODO(bootcamper)` and `NotImplementedError`. The one new thing here is that you make up the image instead of reading it from somewhere. The pixels have to depend only on the frame's index: frame 3 always looks the same every time, and frame 3 does not look like frame 4. How you do that is up to you. Filling the whole image with the index works, so does a gradient, so does `numpy.random.default_rng(index)`.

## Check your work

```bash
warg run camera test
warg run camera lint
```

Both have to pass. Then add it to your PR:

```bash
git add camera
git commit -m "Implement simulated camera"
git push
```

CI runs the same two commands on your branch. Look for the **Camera** job on your pull request.

## Stuck?

- `RuntimeError` tests failing? Keep a `True`/`False` variable for whether the camera is on, like `FixedCamera._initialized` does.
- Timestamp test passing sometimes and failing others? `time.monotonic()` can give you the same number twice in a row. See how `FixedCamera._next_timestamp()` deals with it.
- The test about changing a frame failing? You returned your own array instead of a copy of it. Return `array.copy()`, or build a new array on every capture.
