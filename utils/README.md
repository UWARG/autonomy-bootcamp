# Part 3: Test Waypoint Utilities

In Part 2 you wrote code to pass tests we gave you. This part is the other way around: the code is already written and **you write the tests**.

`src/waypoint_utils.py` is real code copied out of the WARG autonomy repo. It reads waypoints out of a YAML file and puts them in clockwise order so the drone can fly a lap. Your job is to write tests good enough that we'd trust them to catch a bug before it ships.

## Setup

```bash
warg up utils
warg run utils setup
```

## What to do

1. **Read the code you're testing.** `src/waypoint_utils.py` has three functions you can call from outside: `east_north_coordinate_offset_m`, `parse_waypoints_file`, and `sort_clockwise_sweep`. It also uses `Coordinate` from `src/types.py` and the numbers in `src/constants.py`.
2. **Read the example.** `tests/test_coordinate.py` is a finished test file. Copy how it's written: small setup, one thing checked per test, `pytest.approx` when comparing decimals, `pytest.raises` when you expect an error.
3. **Write `tests/test_waypoint_utils.py`.** The file already exists with a checklist in the comment at the top: files that parse fine, files with bad data, coordinates out of range, distances that are close but not exact, empty files, clockwise ordering, where the lap starts, two waypoints at the same angle, and making sure the functions don't modify their inputs.

Tips:

- Use pytest's `tmp_path` fixture to write temporary YAML files in your tests instead of adding test files to the repo.
- Distances calculated from latitude and longitude are never exact. Compare them with `pytest.approx(..., abs=...)`, never with `==`.

## How you are graded

`grade_bootcamper_tests.py` checks three things, in this order (it reports the first two together as GATE 1 and the third as GATE 2):

1. **Your tests pass on the real code.** A test that fails on correct code is worth nothing, so this is checked first.
2. **At least 90% branch coverage** of `src/waypoint_utils.py` (`pytest --cov=src.waypoint_utils --cov-branch`). Coverage only means your tests ran the code. It doesn't mean they checked anything, which is what the next gate is for.
3. **Every mutant is killed.** `grader/mutants/` has 12 copies of the module, each with one bug put into it on purpose (the kinds of bugs are listed in `grader/mutants/MANIFEST.md`). Your tests get run against each broken copy and at least one of them has to fail. If a broken copy passes all your tests, that's a bug your tests would have let through.

## Run it

```bash
warg run utils test          # your tests against the real code
warg run utils lint
warg run utils grade-tests   # all three checks + which mutants survived
```

CI runs `lint` and `grade-tests` on your PR.

## Done when

`warg run utils grade-tests` prints `RESULT  PASS`, then:

```bash
git add utils
git commit -m "Add waypoint utility tests"
git push
```
