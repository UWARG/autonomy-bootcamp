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

# Missing Data Tests
def test_parse_waypoints_missing_file(tmp_path):
    with pytest.raises(OSError):
        parse_waypoints_file(tmp_path / "non_existent.yaml")

def test_parse_waypoints_file_top_level_not_mapping(tmp_path):
    path = write_to_tmp_waypoints_file(tmp_path, "- 1\n- 2\n")
    with pytest.raises(ValueError):
        parse_waypoints_file(path)

def test_parse_top_level_not_mapping(tmp_path):
    path = write_to_tmp_waypoints_file(tmp_path, "- 1\n- 2\n")
    with pytest.raises(ValueError, match="expected a mapping"):
        parse_waypoints_file(path)

def test_parse_missing_lat(tmp_path):
    path = write_to_tmp_waypoints_file(tmp_path, "waypoints:\n  - {lon: 5, alt: 6}\n")
    with pytest.raises(ValueError, match=r"missing key\(s\) lat"):
        parse_waypoints_file(path)

def test_parse_missing_lon(tmp_path):
    path = write_to_tmp_waypoints_file(tmp_path, "waypoints:\n  - {lat: 4, alt: 6}\n")
    with pytest.raises(ValueError, match=r"missing key\(s\) lon"):
        parse_waypoints_file(path)

def test_parse_missing_alt(tmp_path):
    path = write_to_tmp_waypoints_file(tmp_path, "waypoints:\n  - {lat: 4, lon: 5}\n")
    with pytest.raises(ValueError, match=r"missing key\(s\) alt"):
        parse_waypoints_file(path)

def test_parse_non_numeric_value(tmp_path):
    path = write_to_tmp_waypoints_file(
        tmp_path, "waypoints:\n  - {lat: abc, lon: 5, alt: 6}\n"
    )
    with pytest.raises(ValueError, match="non-numeric"):
        parse_waypoints_file(path)

def test_parse_invalid_yaml(tmp_path):
    path = write_to_tmp_waypoints_file(tmp_path, "waypoints: [1, 2\n")
    with pytest.raises(ValueError, match="invalid YAML"):
        parse_waypoints_file(path)

# Out of Range Tests
def test_parse_lat_too_high(tmp_path):
    path = write_to_tmp_waypoints_file(
        tmp_path, "waypoints:\n  - {lat: 90.1, lon: 0, alt: 0}\n"
    )
    with pytest.raises(ValueError, match="out of range"):
        parse_waypoints_file(path)


def test_parse_lat_too_low(tmp_path):
    path = write_to_tmp_waypoints_file(
        tmp_path, "waypoints:\n  - {lat: -90.1, lon: 0, alt: 0}\n"
    )
    with pytest.raises(ValueError, match="out of range"):
        parse_waypoints_file(path)


def test_parse_lon_too_high(tmp_path):
    path = write_to_tmp_waypoints_file(
        tmp_path, "waypoints:\n  - {lat: 0, lon: 180.1, alt: 0}\n"
    )
    with pytest.raises(ValueError, match="out of range"):
        parse_waypoints_file(path)


def test_parse_lon_too_low(tmp_path):
    path = write_to_tmp_waypoints_file(
        tmp_path, "waypoints:\n  - {lat: 0, lon: -180.1, alt: 0}\n"
    )
    with pytest.raises(ValueError, match="out of range"):
        parse_waypoints_file(path)


# ---------- Nothing to work with ----------
def test_parse_empty_file(tmp_path):
    path = write_to_tmp_waypoints_file(tmp_path, "")
    assert parse_waypoints_file(path) == (None, [])


def test_parse_empty_waypoints_list(tmp_path):
    path = write_to_tmp_waypoints_file(tmp_path, "waypoints: []\n")
    assert parse_waypoints_file(path) == (None, [])


def test_sort_zero_waypoints():
    assert sort_clockwise_sweep([]) == []


def test_sort_one_waypoint():
    point = Coordinate(1, 2, 3)
    assert sort_clockwise_sweep([point]) == [point]

# Offsets
def test_offset_due_north():
    east, north = east_north_coordinate_offset_m(0, 0, 1, 0)
    assert east == pytest.approx(0.0, abs=1e-6)
    assert north == pytest.approx(111195.08, abs=0.5)

def test_offset_due_east():
    east, north = east_north_coordinate_offset_m(0, 0, 0, 1)
    assert east == pytest.approx(111195.08, abs=0.5)
    assert north == pytest.approx(0.0, abs=1e-6)

def test_offset_east_uses_cosine_of_mean_latitude():
    east, north = east_north_coordinate_offset_m(59, 0, 61, 1)
    assert east == pytest.approx(55597.54, abs=0.5)
    assert north == pytest.approx(222390.16, abs=1.0)

def test_offset_south_west_is_negative():
    east, north = east_north_coordinate_offset_m(0, 0, -1, -1)
    assert east == pytest.approx(-111190.85, abs=0.5)
    assert north == pytest.approx(-111195.08, abs=0.5)

# Ordering
NORTH = Coordinate(1, 0, 0)
EAST = Coordinate(0, 1, 0)
SOUTH = Coordinate(-1, 0, 0)
WEST = Coordinate(0, -1, 0)


def test_sort_no_home_clockwise_from_north():
    assert sort_clockwise_sweep([SOUTH, WEST, NORTH, EAST]) == [NORTH, EAST, SOUTH, WEST]


def test_sort_with_home_starts_in_home_direction():
    home = Coordinate(0, 5, 0)  # due east of the centroid
    assert sort_clockwise_sweep([SOUTH, WEST, NORTH, EAST], home) == [
        EAST,
        SOUTH,
        WEST,
        NORTH,
    ]


def test_sort_home_on_centroid_starts_north():
    home = Coordinate(0, 0, 0)
    assert sort_clockwise_sweep([SOUTH, WEST, NORTH, EAST], home) == [
        NORTH,
        EAST,
        SOUTH,
        WEST,
    ]


def test_sort_same_direction_closer_first():
    near = Coordinate(2, 0, 0)
    far = Coordinate(4, 0, 0)
    south = Coordinate(-6, 0, 0)
    assert sort_clockwise_sweep([south, far, near]) == [near, far, south]

def test_coordinate_is_frozen():
    point = Coordinate(1, 2, 3)
    with pytest.raises(AttributeError):
        point.lat = 0