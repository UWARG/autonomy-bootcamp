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

def test_parse_empty_file(tmp_path):
    path = write_to_tmp_waypoints_file(tmp_path, "")

    assert parse_waypoints_file(path) == (None, [])

def test_parse_empty_waypoints(tmp_path):
    path = write_to_tmp_waypoints_file(
        tmp_path,
        """
        waypoints: []
        """,
    )

    assert parse_waypoints_file(path) == (None, [])

def test_parse_bad_top_level_data(tmp_path):
    path = write_to_tmp_waypoints_file(
        tmp_path,
        """
        - 1
        - 2
        - 3
        """,
    )

    with pytest.raises(ValueError):
        parse_waypoints_file(path)


def test_parse_waypoint_missing_lat(tmp_path):
    path = write_to_tmp_waypoints_file(
        tmp_path,
        """
        waypoints:
          - lon: -80.0
            alt: 100
        """,
    )

    with pytest.raises(ValueError):
        parse_waypoints_file(path)

def test_parse_waypoint_missing_lon(tmp_path):
    path = write_to_tmp_waypoints_file(
        tmp_path,
        """
        waypoints:
          - lat: 43.0
            alt: 100
        """,
    )

    with pytest.raises(ValueError):
        parse_waypoints_file(path)

def test_parse_waypoint_missing_alt(tmp_path):
    path = write_to_tmp_waypoints_file(
        tmp_path,
        """
        waypoints:
          - lat: 43.0
            lon: -80.0
        """,
    )

    with pytest.raises(ValueError):
        parse_waypoints_file(path)

def test_parse_waypoint_non_numeric_value(tmp_path):
    path = write_to_tmp_waypoints_file(
        tmp_path,
        """
        waypoints:
          - lat: not-a-number
            lon: -80.0
            alt: 100
        """,
    )

    with pytest.raises(ValueError):
        parse_waypoints_file(path)

def test_parse_invalid_yaml(tmp_path):
    path = write_to_tmp_waypoints_file(
        tmp_path,
        """
        waypoints:
          - lat: 43.0
            lon: -80.0
            alt: [100
        """,
    )

    with pytest.raises(ValueError):
        parse_waypoints_file(path)

def test_parse_nonexistent_file(tmp_path):
    path = tmp_path / "does_not_exist.yaml"

    with pytest.raises(FileNotFoundError):
        parse_waypoints_file(path)


def test_parse_latitude_out_of_range(tmp_path):
    path = write_to_tmp_waypoints_file(
        tmp_path,
        """
        waypoints:
          - lat: 91
            lon: -80.0
            alt: 100
        """,
    )

    with pytest.raises(ValueError):
        parse_waypoints_file(path)

def test_parse_longitude_out_of_range(tmp_path):
    path = write_to_tmp_waypoints_file(
        tmp_path,
        """
        waypoints:
          - lat: 43.0
            lon: 181
            alt: 100
        """,
    )

    with pytest.raises(ValueError):
        parse_waypoints_file(path)


def test_sort_clockwise_sweep_empty():
    assert sort_clockwise_sweep([]) == []

def test_sort_clockwise_sweep_single_waypoint():
    waypoint = Coordinate(lat=43.0, lon=-80.0, alt=100)

    assert sort_clockwise_sweep([waypoint]) == [waypoint]

def test_parsed_coordinates_are_immutable(tmp_path):
    path = write_to_tmp_waypoints_file(
        tmp_path,
        """
        waypoints:
          - lat: 43.0
            lon: -80.0
            alt: 100
        """,
    )

    _, waypoints = parse_waypoints_file(path)

    with pytest.raises(AttributeError):
        waypoints[0].lat = 44.0

def test_east_north_coordinate_offset_north():
    east, north = east_north_coordinate_offset_m(
        0.0,
        0.0,
        0.001,
        0.0,
    )

    assert east == pytest.approx(0.0)
    assert north == pytest.approx(111.2, abs=0.1)

def test_east_north_coordinate_offset_east():
    east, north = east_north_coordinate_offset_m(
        0.0,
        0.0,
        0.0,
        0.001,
    )

    assert east == pytest.approx(111.2, abs=0.1)
    assert north == pytest.approx(0.0)

def test_sort_clockwise_sweep_starts_north():
    north = Coordinate(lat=1.0, lon=0.0, alt=0)
    east = Coordinate(lat=0.0, lon=1.0, alt=0)
    south = Coordinate(lat=-1.0, lon=0.0, alt=0)
    west = Coordinate(lat=0.0, lon=-1.0, alt=0)

    waypoints = [south, west, north, east]

    assert sort_clockwise_sweep(waypoints) == [
        north,
        east,
        south,
        west,
    ]

def test_sort_clockwise_sweep_starts_from_home_direction():
    north = Coordinate(lat=1.0, lon=0.0, alt=0)
    east = Coordinate(lat=0.0, lon=1.0, alt=0)
    south = Coordinate(lat=-1.0, lon=0.0, alt=0)
    west = Coordinate(lat=0.0, lon=-1.0, alt=0)

    home = Coordinate(lat=0.0, lon=2.0, alt=0)

    waypoints = [north, west, south, east]

    assert sort_clockwise_sweep(waypoints, home) == [
        east,
        south,
        west,
        north,
    ]

def test_sort_clockwise_sweep_home_at_centroid_starts_north():
    north = Coordinate(lat=1.0, lon=0.0, alt=0)
    east = Coordinate(lat=0.0, lon=1.0, alt=0)
    south = Coordinate(lat=-1.0, lon=0.0, alt=0)
    west = Coordinate(lat=0.0, lon=-1.0, alt=0)

    home = Coordinate(lat=0.0, lon=0.0, alt=0)

    waypoints = [south, west, east, north]

    assert sort_clockwise_sweep(waypoints, home) == [
        north,
        east,
        south,
        west,
    ]

def test_sort_clockwise_sweep_same_direction_closer_first():
    near_north = Coordinate(lat=1.0, lon=0.0, alt=0)
    far_north = Coordinate(lat=2.0, lon=0.0, alt=0)
    near_south = Coordinate(lat=-1.0, lon=0.0, alt=0)
    far_south = Coordinate(lat=-2.0, lon=0.0, alt=0)

    waypoints = [far_north, far_south, near_north, near_south]

    assert sort_clockwise_sweep(waypoints) == [
        near_north,
        far_north,
        near_south,
        far_south,
    ]

def test_east_north_coordinate_offset_east_at_high_latitude():
    east, north = east_north_coordinate_offset_m(
        60.0,
        0.0,
        60.0,
        0.001,
    )

    assert east == pytest.approx(55.6, abs=0.1)
    assert north == pytest.approx(0.0)