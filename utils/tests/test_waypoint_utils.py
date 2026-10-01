"""
TODO(bootcamper): write the tests for ``src/waypoint_utils.py`` in here.

The example below covers files that parse fine: with and without ``home``,
and files with comments and blank lines in them. The rest is yours:

- Bad data: a file whose top level isn't a mapping, waypoints missing
  ``lat``, ``lon``, or ``alt``, values that aren't numbers, YAML that
  doesn't parse, and a file that isn't there.
- Out of range: latitudes past +/-90 and longitudes past +/-180 get
  rejected.
- Nothing to work with: an empty file, an empty ``waypoints`` list, and
  ``sort_clockwise_sweep`` given a list of 0 or 1 waypoints.
- ``east_north_coordinate_offset_m``: offsets you worked out yourself,
  compared with ``pytest.approx``. Never use ``==`` on meters.
- Ordering: with no ``home``, ``sort_clockwise_sweep`` goes clockwise
  starting from north.
- With a ``home``: the order starts in home's direction instead, and goes
  back to starting at north if home is right on top of the centroid.
- Two waypoints in the same direction: the closer one comes first.
- Parsing gives you frozen ``Coordinate`` objects that can't be changed.

Graded by ``warg run utils grade-tests``: pass on the real code, 90% branch
coverage, and fail on every broken copy in ``grader/mutants/``.
"""

import pytest

from src.types import Coordinate
from src.waypoint_utils import (
    east_north_coordinate_offset_m,
    parse_waypoints_file,
    sort_clockwise_sweep,
)

# The helper and the test below are given to you.
NORTH = Coordinate(1, 0, 0)
EAST = Coordinate(0, 1, 0)
SOUTH = Coordinate(-1, 0, 0)
WEST = Coordinate(0, -1, 0)

def write_to_tmp_waypoints_file(tmp_path, text):
    """Write ``text`` to a YAML file and hand back its path.

    ``tmp_path`` is a pytest fixture: a fresh empty directory per test.
    """
    path = tmp_path / "waypoints.yaml"
    path.write_text(text)
    return path


# One test, three files. ``parametrize`` runs the test body once per
# ``(text, expected)`` pair, and ``ids`` names each run so a failure tells you
# which file broke.
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            """
            home: {lat: 1, lon: 2, alt: 3}
            waypoints:
              - {lat: 4, lon: 5, alt: 6}
            """,
            (Coordinate(1, 2, 3), [Coordinate(4, 5, 6)]),
        ),
        (
            """
            waypoints:
              - {lat: 4, lon: 5, alt: 6}
              - {lat: 7, lon: 8, alt: 9}
            """,
            (None, [Coordinate(4, 5, 6), Coordinate(7, 8, 9)]),
        ),
        (
            """
            # a lap

            home: {lat: 1, lon: 2, alt: 3}

            waypoints:
              # first leg
              - {lat: 4, lon: 5, alt: 6}
            """,
            (Coordinate(1, 2, 3), [Coordinate(4, 5, 6)]),
        ),
    ],
    ids=["home-and-waypoints", "no-home", "comments-and-blank-lines"],
)
def test_parse_waypoints_file_success(tmp_path, text, expected):
    path = write_to_tmp_waypoints_file(tmp_path, text)
    assert parse_waypoints_file(path) == expected

# kills mutant 3 (latitude checked against longitude's +/-180 limits),
# mutant 7 (non-numeric values accepted), and mutant 11 (alt not required);
# invalid-yaml and top-level-not-mapping are there for branch coverage
@pytest.mark.parametrize(
    "text",
    [
        "waypoints:\n  - {lat: 1, lon: 2}\n",
        "waypoints:\n  - {lat: abc, lon: 2, alt: 3}\n",
        "waypoints:\n  - {lat: 100, lon: 0, alt: 0}\n",
        "home: [unclosed\n",
        "- 1\n- 2\n",
    ],
    ids=[
        "missing-alt",
        "non-numeric",
        "lat-past-90",
        "invalid-yaml",
        "top-level-not-mapping",
    ],
)

# Feed the parser each bad file from the list above and check it refuses
# with a ValueError. If someone breaks a check in the parser later, it stops
# refusing, and this test fails.
def test_parse_waypoints_file_rejects_bad_data(tmp_path, text):
    # write this run's bad YAML into a temp file
    path = write_to_tmp_waypoints_file(tmp_path, text)
    # the test passes only if the line inside raises ValueError
    with pytest.raises(ValueError):
        # run the real parser on the bad file
        parse_waypoints_file(path)

# kills mutant 10 (sorted counterclockwise instead of clockwise)
def test_sort_no_home_goes_clockwise_from_north():
    result = sort_clockwise_sweep([SOUTH, WEST, NORTH, EAST])
    assert result == [NORTH, EAST, SOUTH, WEST]

# kills mutant 5 (start bearing measured from home to centroid instead of
# centroid to home) and mutant 1 (relative bearing not wrapped with % TWO_PI)
def test_sort_starts_toward_home():
    result = sort_clockwise_sweep([NORTH, EAST, SOUTH, WEST], Coordinate(0, 5, 0))
    assert result == [EAST, SOUTH, WEST, NORTH]

# kills mutant 9 (early return for 0 or 1 waypoints removed)
def test_sort_empty_list():
    result = sort_clockwise_sweep([])
    assert result == []

# kills mutant 9 (early return for 0 or 1 waypoints removed)
def test_sort_single_waypoint():
    result = sort_clockwise_sweep([NORTH])
    assert result == [NORTH]

# kills mutant 2 (waypoints at the same bearing come out farthest first)
def test_sort_closer_comes_first():
    result = sort_clockwise_sweep([Coordinate(2, 0, 0), Coordinate(1, 0, 0), Coordinate(-3, 0, 0)])
    assert result == [Coordinate(1, 0, 0), Coordinate(2, 0, 0), Coordinate(-3, 0, 0)]

#kills mutant 8 (empty fild raises instead of returning)
def test_parse_empty_file_returns_nothing(tmp_path):
    path = write_to_tmp_waypoints_file(tmp_path, "")
    assert parse_waypoints_file(path) == (None, [])

#kills mutant 4 (swap north and east) and mutant 6 (doesnt convert north to radians)
def test_offset_due_north():
    east, north = east_north_coordinate_offset_m(0, 0, 1, 0)
    assert east == pytest.approx(0, abs=1)
    assert north == pytest.approx(111195, abs=1)

#kills mutant 12 (not doing cos(latitude))
def test_offset_due_east_at_latitude_60():
    east, north = east_north_coordinate_offset_m(60, 0, 60, 1)
    assert east == pytest.approx(55598, abs=1)
    assert north == pytest.approx(0, abs=1)