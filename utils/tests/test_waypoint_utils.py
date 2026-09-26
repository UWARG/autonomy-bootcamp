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

from dataclasses import FrozenInstanceError

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




def test_east_north_coordinate_0():
    east, north = east_north_coordinate_offset_m(43, -80, 43, -80)
    assert east == pytest.approx(0)
    assert north == pytest.approx(0)

def test_east_north_coordinate_positive():
    east, north = east_north_coordinate_offset_m(60, 60, 80, 80)
    assert east > 0
    assert north > 0

def test_east_north_coordinate_negative():
    east, north = east_north_coordinate_offset_m(80, 80, 60, 60)
    assert east < 0
    assert north < 0

def test_empty_file(tmp_path):
    path = write_to_tmp_waypoints_file(tmp_path, "")

    home, waypoints = parse_waypoints_file(path)

    assert home is None
    assert waypoints == []

def test_missing_alt(tmp_path):
    path = write_to_tmp_waypoints_file(
        tmp_path,
        """
        waypoints:
          - lat: 1
            lon: 2
        """
    )

    with pytest.raises(ValueError):
        parse_waypoints_file(path)

def test_latitude_out_of_range(tmp_path):
    path = write_to_tmp_waypoints_file(
        tmp_path,
        """
        waypoints:
          - lat: 999
            lon: 2
            alt: 3
        """
    )

    with pytest.raises(ValueError):
        parse_waypoints_file(path)


def test_instance_not_dict(tmp_path):
    path = write_to_tmp_waypoints_file(
        tmp_path,
        """
        hello world
        """
    )

    with pytest.raises(ValueError):
        parse_waypoints_file(path)

def test_not_nums(tmp_path):
    path = write_to_tmp_waypoints_file(
        tmp_path,
        """
        waypoints:
          - lat: abc
            lon: 22
            alt: 33
          """
    )
    with pytest.raises(ValueError):
        parse_waypoints_file(path)

def test_sort_empty(tmp_path):
    assert sort_clockwise_sweep([]) == []

def test_sort_one_waypoint():
    p = Coordinate(1, 2, 3)
    assert sort_clockwise_sweep([p]) == [p]

def test_file_not_found():
    with pytest.raises(FileNotFoundError):
        parse_waypoints_file("does_not_exist.yaml")

def test_bad_yaml(tmp_path):
    path = write_to_tmp_waypoints_file(
        tmp_path,
        """
        waypoints:
          - {lat: 1,
        """
    )

    with pytest.raises(ValueError):
        parse_waypoints_file(path)

def test_waypoints_not_list(tmp_path):
    path = write_to_tmp_waypoints_file(
        tmp_path,
        """
        waypoints: hello
        """
    )

    with pytest.raises(ValueError):
        parse_waypoints_file(path)

def test_missing_waypoints_key(tmp_path):
    path = write_to_tmp_waypoints_file(
        tmp_path,
        """
        home:
          lat: 1
          lon: 2
          alt: 3
        """
    )

    home, waypoints = parse_waypoints_file(path)

    assert home == Coordinate(1, 2, 3)
    assert waypoints == []

def test_longitude_out_of_range(tmp_path):
    path = write_to_tmp_waypoints_file(
        tmp_path,
        """
        waypoints:
          - lat: 1
            lon: 999
            alt: 3
        """
    )

    with pytest.raises(ValueError):
        parse_waypoints_file(path)

def test_coordinate_frozen(tmp_path):
    path = write_to_tmp_waypoints_file(
        tmp_path,
        """
        waypoints:
          - {lat: 1, lon: 2, alt: 3}
        """
    )

    _, waypoints = parse_waypoints_file(path)

    with pytest.raises(FrozenInstanceError):
        waypoints[0].lat = 99

def test_sort_clockwise_from_north():
    north = Coordinate(1, 0, 0)
    east = Coordinate(0, 1, 0)
    south = Coordinate(-1, 0, 0)
    west = Coordinate(0, -1, 0)

    result = sort_clockwise_sweep(
        [south, west, east, north]
    )

    assert result == [
        north,
        east,
        south,
        west,
    ]


def test_lat_smaller_than_90(tmp_path):
    data = ('''
    waypoints:
      - lat: -100
        lon: 0
        alt: 0''')
    f = tmp_path / "bad.yaml"
    f.write_text(data)
    with pytest.raises(ValueError):
        parse_waypoints_file(f)

def test_lat_bigger_than_90(tmp_path):
    data = ('''
    waypoints:
      - lat: 100
        lon: 0
        alt: 0''')
    f = tmp_path / "bad.yaml"
    f.write_text(data)
    with pytest.raises(ValueError):
        parse_waypoints_file(f)

def test_east_north_coordinate_offset_north_1_degree():
    east, north = east_north_coordinate_offset_m(
        0.0,
        0.0,
        1.0,
        0.0,
    )

    assert east == pytest.approx(0.0, abs=1)
    assert north == pytest.approx(111195, rel=0.01)

def test_same_bearing_sorted_by_distance():
    near = Coordinate(1.0, 0.0, 0.0)
    far = Coordinate(2.0, 0.0, 0.0)
    south = Coordinate(-3.0, 0.0, 0.0)

    ordered = sort_clockwise_sweep([far, near, south])

    north_points = [p for p in ordered if p in (near, far)]

    assert north_points == [near, far]

def test_sort_clockwise_wraparound():
    north = Coordinate(1, 0, 0)
    east = Coordinate(0, 1, 0)
    south = Coordinate(-1, 0, 0)
    west = Coordinate(0, -1, 0)

    home = Coordinate(1, 1, 0)

    result = sort_clockwise_sweep(
        [north, east, south, west],
        home,
    )

    assert result == [
        east,
        south,
        west,
        north,
    ]
def test_east_offset_scales_with_latitude():
    east, north = east_north_coordinate_offset_m(
        60.0,   # from lat
        0.0,    # from lon
        60.0,   # to lat
        1.0,    # to lon
    )

    assert north == pytest.approx(0.0, abs=1)

    # should be about half of 111 km
    assert east == pytest.approx(55597, rel=0.02)

def test_home_at_centroid_falls_back_to_north():
    north = Coordinate(1, 0, 0)
    east = Coordinate(0, 1, 0)
    south = Coordinate(-1, 0, 0)
    west = Coordinate(0, -1, 0)

    home = Coordinate(0, 0, 0)

    result = sort_clockwise_sweep(
        [south, west, east, north],
        home,
    )

    assert result == [
        north,
        east,
        south,
        west,
    ]