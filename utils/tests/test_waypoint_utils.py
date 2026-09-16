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

def test_parse_waypoints_file_empty(tmp_path):
    path = write_to_tmp_waypoints_file(tmp_path, "")
    assert parse_waypoints_file(path) == (None, [])

def test_parse_waypoints_file_top_level_not_mapping(tmp_path):
    path = write_to_tmp_waypoints_file(
        tmp_path, 
        """
        - hello
        - hi
        """,
    )

    with pytest.raises(ValueError):
        parse_waypoints_file(path)


@pytest.mark.parametrize(
        "waypoint",
        [
            "{lon: 5, alt: 6}",
            "{lat: 4, alt: 6}",
            "{lat: 4, lon: 5}",
        ],
)
def test_parse_waypoints_file_missing_coordinate(tmp_path, waypoint):
    path = write_to_tmp_waypoints_file(
        tmp_path, 
        f"""
        waypoints:
        - {waypoint}
        """
    )
    with pytest.raises(ValueError):
      parse_waypoints_file(path)

def test_parse_waypoints_file_non_numeric_value(tmp_path):
    path = write_to_tmp_waypoints_file(
        tmp_path,
        """
        waypoints:
          - {lat: hello, lon: 5, alt: 6}
        """,
    )

    with pytest.raises(ValueError):
        parse_waypoints_file(path)

def test_parse_waypoints_file_invalid_yaml(tmp_path):
    path = write_to_tmp_waypoints_file(
        tmp_path,
        """
        waypoints:
          - {lat: 4, lon: 5, alt: 6
        """,
    )

    with pytest.raises(ValueError):
        parse_waypoints_file(path)

def test_parse_waypoints_file_missing_file(tmp_path):
    path = tmp_path / "does_not_exist.yaml"

    with pytest.raises(OSError):
        parse_waypoints_file(path)



@pytest.mark.parametrize(
    "waypoint",
    [
        "{lat: 100, lon: 5, alt: 6}",
        "{lat: -93, lon: 5, alt: 6}",
        "{lat: 4, lon: 200 , alt: 6}",
        "{lat: 4, lon: -181, alt: 6}",
    ],
)
def test_parse_waypoints_file_out_of_range(tmp_path, waypoint):
    path = write_to_tmp_waypoints_file(
        tmp_path,
        f"""
        waypoints:
          - {waypoint}
        """,
    )

    with pytest.raises(ValueError):
        parse_waypoints_file(path)


def test_parse_waypoints_file_empty_waypoints(tmp_path):
    path = write_to_tmp_waypoints_file(
        tmp_path,
        """
        waypoints: []
        """,
    )

    assert parse_waypoints_file(path) == (None, [])

@pytest.mark.parametrize(
    "waypoints",
    [
        [],
        [Coordinate(3, 4, 5)],
    ],
)
def test_sort_clockwise_sweep_zero_or_one_waypoint(waypoints):
    assert sort_clockwise_sweep(waypoints) == waypoints

def test_east_north_coordinate_offset_same_point():
    east, north = east_north_coordinate_offset_m(
        -50.0, 
        23.0, 
        -50.0, 
        23.0
    )
    assert east == pytest.approx(0,0)
    assert north == pytest.approx(0,0)

def test_east_north_coordinate_offset_north():
    east, north = east_north_coordinate_offset_m(
        0.0,
        0.0,
        1.0,
        0.0,
    )

    assert east == pytest.approx(0.0)
    assert north == pytest.approx(111195, abs=1)
    
def test_east_north_coordinate_offset_at_nonzero_latitude():
    east, north = east_north_coordinate_offset_m(
        60.0,
        0.0,
        60.0,
        1.0,
    )

    assert east == pytest.approx(55597.5, abs=1)
    assert north == pytest.approx(0.0)

def test_sort_clockwise_sweep_no_home():
    north = Coordinate(1, 0, 0)
    east = Coordinate(0, 1, 0)
    south = Coordinate(-1, 0, 0)
    west = Coordinate(0, -1, 0)

    waypoints = [south, west, east, north]

    result = sort_clockwise_sweep(waypoints)

    assert result == [north, east, south, west]


def test_sort_clockwise_sweep_with_home():
    north = Coordinate(1, 0, 0)
    east = Coordinate(0, 1, 0)
    south = Coordinate(-1, 0, 0)
    west = Coordinate(0, -1, 0)

    home = Coordinate(0, -3, 0)

    waypoints = [north, west, south, east]

    result = sort_clockwise_sweep(waypoints, home)

    assert result == [west, north, east, south]

def test_sort_clockwise_sweep_home_at_centroid():
    north = Coordinate(1, 0, 0)
    east = Coordinate(0, 1, 0)
    south = Coordinate(-1, 0, 0)
    west = Coordinate(0, -1, 0)

    home = Coordinate(0, 0, 0)

    waypoints = [south, east, west, north]

    result = sort_clockwise_sweep(waypoints, home)

    assert result == [north, east, south, west]

def test_sort_clockwise_sweep_same_direction_closer_first():
    closer = Coordinate(1, 0, 0)
    farther = Coordinate(2, 0, 0)
    south = Coordinate(-4, 0, 0)

    waypoints = [farther, south, closer]

    result = sort_clockwise_sweep(waypoints)

    assert result == [closer, farther, south]

def test_parse_waypoints_file_coordinate_is_frozen(tmp_path):
    path = write_to_tmp_waypoints_file(
        tmp_path,
        """
        waypoints:
          - {lat: 4, lon: 5, alt: 6}
        """,
    )

    _, waypoints = parse_waypoints_file(path)

    with pytest.raises(AttributeError):
        waypoints[0].lat = 10