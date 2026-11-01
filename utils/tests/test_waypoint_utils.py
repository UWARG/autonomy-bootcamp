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


def test_non_mapping(tmp_path):
    path = write_to_tmp_waypoints_file(tmp_path, "testing")

    with pytest.raises(ValueError):
        parse_waypoints_file(path)


@pytest.mark.parametrize(
    "text",
    [
        "waypoints:\n  - {lon: 5, alt: 6}\n",
        "waypoints:\n  - {lat: 4, alt: 6}\n",
        "waypoints:\n  - {lat: 4, lon: 5}\n",
    ],
    ids=["no-lat", "no-lon", "no-alt"],
)
def test_waypoints_missing_values(tmp_path, text):
    path = write_to_tmp_waypoints_file(tmp_path, text)

    with pytest.raises(ValueError):
        parse_waypoints_file(path)


def test_non_numeric_values(tmp_path):
    path = write_to_tmp_waypoints_file(
        tmp_path,
        "waypoints:\n  - {lat: abcdefg, lon: 5, alt: 6}\n",
    )

    with pytest.raises(ValueError):
        parse_waypoints_file(path)


def test_invalid_yaml(tmp_path):
    path = write_to_tmp_waypoints_file(tmp_path, "waypoints: [")

    with pytest.raises(ValueError):
        parse_waypoints_file(path)


def test_file_missing(tmp_path):
    path = tmp_path / "does_not_exist.yaml"

    with pytest.raises(OSError):
        parse_waypoints_file(path)


@pytest.mark.parametrize(
    "text",
    [
        "waypoints:\n  - {lat: 91, lon: 5, alt: 6}\n",
        "waypoints:\n  - {lat: -91, lon: 5, alt: 6}\n",
        "waypoints:\n  - {lat: 4, lon: 181, alt: 6}\n",
        "waypoints:\n  - {lat: 4, lon: -181, alt: 6}\n",
    ],
    ids=["lat-too-high", "lat-too-low", "lon-too-high", "lon-too-low"],
)
def test_out_of_range(tmp_path, text):
    path = write_to_tmp_waypoints_file(tmp_path, text)

    with pytest.raises(ValueError):
        parse_waypoints_file(path)


def test_empty_file(tmp_path):
    path = write_to_tmp_waypoints_file(tmp_path, "")

    assert parse_waypoints_file(path) == (None, [])


def test_empty_waypoints(tmp_path):
    path = write_to_tmp_waypoints_file(tmp_path, "waypoints: []")

    assert parse_waypoints_file(path) == (None, [])


def test_sort_no_waypoints():
    assert sort_clockwise_sweep([]) == []


def test_sort_one_waypoint():
    waypoint = Coordinate(1, 2, 3)

    assert sort_clockwise_sweep([waypoint]) == [waypoint]


def test_east_north_coordinate_offset_m():
    east, north = east_north_coordinate_offset_m(0, 0, 1, 1)

    assert east == pytest.approx(111_195, rel=0.001)
    assert north == pytest.approx(111_195, rel=0.001)


def test_east_offset_scales_with_latitude():
    east_at_equator, _ = east_north_coordinate_offset_m(0, 0, 0, 1)
    east_at_60_degrees, _ = east_north_coordinate_offset_m(60, 0, 60, 1)

    assert east_at_60_degrees == pytest.approx(east_at_equator * 0.5)


def test_sort_clockwise_without_home():
    north = Coordinate(1, 0, 0)
    east = Coordinate(0, 1, 0)
    south = Coordinate(-1, 0, 0)
    west = Coordinate(0, -1, 0)

    result = sort_clockwise_sweep([west, south, east, north])

    assert result == [north, east, south, west]


def test_sort_clockwise_with_home():
    north = Coordinate(1, 0, 0)
    east = Coordinate(0, 1, 0)
    south = Coordinate(-1, 0, 0)
    west = Coordinate(0, -1, 0)
    home = Coordinate(0, 2, 0)

    result = sort_clockwise_sweep([north, west, south, east], home)

    assert result == [east, south, west, north]


def test_sort_clockwise_with_home_at_centroid():
    north = Coordinate(1, 0, 0)
    east = Coordinate(0, 1, 0)
    south = Coordinate(-1, 0, 0)
    west = Coordinate(0, -1, 0)
    home = Coordinate(0, 0, 0)

    result = sort_clockwise_sweep([west, south, east, north], home)

    assert result == [north, east, south, west]


def test_sort_same_direction_closer_waypoint_first():
    north = Coordinate(1, 0, 0)
    east_near = Coordinate(0, 0.5, 0)
    east_far = Coordinate(0, 1, 0)
    south = Coordinate(-1, 0, 0)
    west = Coordinate(0, -1.5, 0)

    result = sort_clockwise_sweep(
        [east_far, south, west, east_near, north]
    )

    assert result == [north, east_near, east_far, south, west]


def test_parsed_coordinates_are_frozen(tmp_path):
    path = write_to_tmp_waypoints_file(
        tmp_path,
        "waypoints:\n  - {lat: 4, lon: 5, alt: 6}\n",
    )
    _, waypoints = parse_waypoints_file(path)

    with pytest.raises(AttributeError):
        waypoints[0].lat = 10
