# Curated Mutants

Each file here is a full copy of `src/waypoint_utils.py` with exactly one bug put into it on purpose. `grade_bootcamper_tests.py` swaps each one in for the real module and reruns your `tests/test_waypoint_utils.py`. A mutant is **killed** when at least one of your tests fails against it.

We tell you what *kind* of bug each one has, but not which line was changed. Read the description, then write a test that passes on correct code and fails on code with that bug.

| Mutant | What's wrong with it |
| --- | --- |
| `mutant_01.py` | Sweep ordering: dropped the `% TWO_PI` that wraps the relative bearing around |
| `mutant_02.py` | Tie-break: waypoints at the same bearing come out farthest-first |
| `mutant_03.py` | Range check: latitude is checked against longitude's limits |
| `mutant_04.py` | Axes swapped: east and north come back in the wrong order |
| `mutant_05.py` | Start bearing: measured from home to the centroid instead of the other way around |
| `mutant_06.py` | Degrees vs radians: the north offset never gets converted to radians |
| `mutant_07.py` | Validation: lat/lon/alt values that aren't numbers get accepted |
| `mutant_08.py` | Empty input: an empty waypoints file raises instead of returning `(None, [])` |
| `mutant_09.py` | Empty input: the early return for lists of 0 or 1 waypoints is gone from `sort_clockwise_sweep` |
| `mutant_10.py` | Sweep ordering: sorted backwards (counterclockwise) |
| `mutant_11.py` | Missing keys: `alt` isn't in the list of keys that must be present |
| `mutant_12.py` | East offset: dropped the `cos(latitude)` scaling |

Don't import these files from your tests, and don't "fix" them. The grader copies your project into a temporary folder and does the swapping there.
