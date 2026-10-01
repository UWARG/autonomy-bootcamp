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

def test_same_coordinate():
    east, north = east_north_coordinate_offset_m(
      43.0, -80.0,
      43.0, -80.0
    )
    assert east == pytest.approx(0.0)
    assert north == pytest.approx(0.0)

def test_move_north():
    east, north = east_north_coordinate_offset_m(
      43.0, -80.0,
      43.0001, -80.0
    )
    assert east == pytest.approx(0.0)
    assert north > 0.0

def test_move_south():
    east, north = east_north_coordinate_offset_m(
      43, -80.0,
      42.99, -80.0
    )
    assert east == pytest.approx(0.0)
    assert north < 0.0

def test_move_east():
    east, north = east_north_coordinate_offset_m(
      43.0, -80.0,
      43.0, -79.99
    )
    assert east > 0.0
    assert north == pytest.approx(0.0)
def test_move_west():
    east, north = east_north_coordinate_offset_m(
      43.0, -80.0,
      43.0, -80.01
    )
    assert east < 0.0
    assert north == pytest.approx(0.0)

def test_missing_lat_raises(tmp_path):
    text = """
    waypoints:
      - {lon: -80, alt: 10}
    """

    path = write_to_tmp_waypoints_file(tmp_path, text)

    with pytest.raises(ValueError):
        parse_waypoints_file(path)

def test_coordinate_offset_actual_distance():
    east, north = east_north_coordinate_offset_m(
        43.0, -80.0,
        43.001, -79.999,
    )

    assert east == pytest.approx(81.32, abs=0.1)
    assert north == pytest.approx(111.20, abs=0.1)


@pytest.mark.parametrize(
    "text",
    [
        """
        waypoints:
          - {lon: -80, alt: 10}
        """,
        """
        waypoints:
          - {lat: 43, alt: 10}
        """,
        """
        waypoints:
          - {lat: 43, lon: -80}
        """,
    ],
)
def test_missing_coordinate_value_raises(tmp_path, text):
    path = write_to_tmp_waypoints_file(tmp_path, text)

    with pytest.raises(ValueError):
        parse_waypoints_file(path)


def test_non_numeric_coordinate_raises(tmp_path):
    text = """
    waypoints:
      - {lat: hello, lon: -80, alt: 10}
    """

    path = write_to_tmp_waypoints_file(tmp_path, text)

    with pytest.raises(ValueError):
        parse_waypoints_file(path)


def test_top_level_not_mapping_raises(tmp_path):
    text = """
    - hello
    - world
    """

    path = write_to_tmp_waypoints_file(tmp_path, text)

    with pytest.raises(ValueError):
        parse_waypoints_file(path)


def test_waypoint_not_mapping_raises(tmp_path):
    text = """
    waypoints:
      - hello
    """

    path = write_to_tmp_waypoints_file(tmp_path, text)

    with pytest.raises(ValueError):
        parse_waypoints_file(path)


def test_waypoints_not_list_raises(tmp_path):
    text = """
    waypoints: {lat: 43, lon: -80, alt: 10}
    """

    path = write_to_tmp_waypoints_file(tmp_path, text)

    with pytest.raises(ValueError):
        parse_waypoints_file(path)


def test_invalid_yaml_raises(tmp_path):
    text = """
    waypoints: [
    """

    path = write_to_tmp_waypoints_file(tmp_path, text)

    with pytest.raises(ValueError):
        parse_waypoints_file(path)


def test_missing_file_raises(tmp_path):
    path = tmp_path / "does_not_exist.yaml"

    with pytest.raises(OSError):
        parse_waypoints_file(path)


@pytest.mark.parametrize(
    "text",
    [
        """
        waypoints:
          - {lat: 91, lon: -80, alt: 10}
        """,
        """
        waypoints:
          - {lat: -91, lon: -80, alt: 10}
        """,
        """
        waypoints:
          - {lat: 43, lon: 181, alt: 10}
        """,
        """
        waypoints:
          - {lat: 43, lon: -181, alt: 10}
        """,
    ],
)
def test_out_of_range_coordinate_raises(tmp_path, text):
    path = write_to_tmp_waypoints_file(tmp_path, text)

    with pytest.raises(ValueError):
        parse_waypoints_file(path)


def test_empty_file(tmp_path):
    path = write_to_tmp_waypoints_file(tmp_path, "")

    assert parse_waypoints_file(path) == (None, [])

def test_empty_waypoints_list(tmp_path):
    text = """
    waypoints: []
    """

    path = write_to_tmp_waypoints_file(tmp_path, text)

    assert parse_waypoints_file(path) == (None, [])


def test_coordinate_is_frozen(tmp_path):
    text = """
    waypoints:
      - {lat: 43, lon: -80, alt: 10}
    """

    path = write_to_tmp_waypoints_file(tmp_path, text)
    _, waypoints = parse_waypoints_file(path)

    with pytest.raises(FrozenInstanceError):
        waypoints[0].lat = 50


def test_sort_empty_list():
    assert sort_clockwise_sweep([]) == []


def test_sort_one_waypoint():
    waypoint = Coordinate(43.0, -80.0, 10)

    assert sort_clockwise_sweep([waypoint]) == [waypoint]


def test_clockwise_order_from_north():
    north = Coordinate(0.001, 0.0, 10)
    east = Coordinate(0.0, 0.001, 10)
    south = Coordinate(-0.001, 0.0, 10)
    west = Coordinate(0.0, -0.001, 10)

    waypoints = [west, south, east, north]

    result = sort_clockwise_sweep(waypoints)

    assert result == [north, east, south, west]


def test_clockwise_order_starts_from_home_direction():
    north = Coordinate(0.001, 0.0, 10)
    east = Coordinate(0.0, 0.001, 10)
    south = Coordinate(-0.001, 0.0, 10)
    west = Coordinate(0.0, -0.001, 10)

    home = Coordinate(0.0, 0.002, 10)

    result = sort_clockwise_sweep(
        [north, east, south, west],
        home,
    )

    assert result == [east, south, west, north]


def test_home_at_centroid_starts_from_north():
    north = Coordinate(0.001, 0.0, 10)
    east = Coordinate(0.0, 0.001, 10)
    south = Coordinate(-0.001, 0.0, 10)
    west = Coordinate(0.0, -0.001, 10)

    home = Coordinate(0.0, 0.0, 10)

    result = sort_clockwise_sweep(
        [west, south, east, north],
        home,
    )

    assert result == [north, east, south, west]


def test_same_direction_closer_waypoint_first():
    close_north = Coordinate(0.001, 0.0, 10)
    far_north = Coordinate(0.002, 0.0, 10)
    south = Coordinate(-0.003, 0.0, 10)

    result = sort_clockwise_sweep(
        [far_north, south, close_north]
    )

    assert result == [close_north, far_north, south]