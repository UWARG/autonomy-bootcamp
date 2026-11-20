"""Tests for ``src/waypoint_utils.py`` in here.

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

def test_waypoint_not_a_mapping_raises(tmp_path):
    path = write_to_tmp_waypoints_file(tmp_path, 
                                       """
                                       waypoints:
                                          - 5
                                        """)
    with pytest.raises(ValueError):
        parse_waypoints_file(path)

def test_latitude_too_high_raises(tmp_path):
    path = write_to_tmp_waypoints_file(tmp_path, 
                                       """
                                        waypoints:
                                          - lat: 90.5
                                            lon: 13
                                            alt: 15
                                        """)
    with pytest.raises(ValueError):
        parse_waypoints_file(path)

def test_latitude_too_low_raises(tmp_path):
    path = write_to_tmp_waypoints_file(tmp_path, 
                                       """
                                        waypoints:
                                          - lat: -90.5
                                            lon: 13
                                            alt: 15
                                        """)
    with pytest.raises(ValueError):
        parse_waypoints_file(path)

def test_longitude_too_high_raises(tmp_path):
    path = write_to_tmp_waypoints_file(tmp_path, 
                                       """
                                        waypoints:
                                          - lat: 45
                                            lon: 180.5
                                            alt: 15
                                        """)
    with pytest.raises(ValueError):
        parse_waypoints_file(path)

def test_longitude_too_low_raises(tmp_path):
    path = write_to_tmp_waypoints_file(tmp_path, 
                                       """
                                        waypoints:
                                          - lat: 45
                                            lon: -180.5
                                            alt: 15
                                        """)
    with pytest.raises(ValueError):
        parse_waypoints_file(path)

@pytest.mark.parametrize(
    "yaml_text",
    [
        """
        waypoints:
          - lon: 50
            alt: 50
        """,
        """
        waypoints:
          - lat: 50
            alt: 50
        """,
        """
        waypoints:
          - lat: 50 
            lon: 50
        """,
    ],
    ids=["missing-lat", "missing-lon", "missing-alt"],
)
def test_missing_key_raises(tmp_path, yaml_text):
    path = write_to_tmp_waypoints_file(tmp_path, yaml_text)
    with pytest.raises(ValueError):
        parse_waypoints_file(path)

@pytest.mark.parametrize(
    "yaml_text",
    [
        """
        waypoints:
          - lat: 50
            lon: 50
            alt: "not-a-number"
        """,
        """
        waypoints:
          - lat: [1, 2]
            lon: 50
            alt: 50
        """,
    ],
    ids=["value-error", "type-error"],
)
def test_non_numeric_value_raises(tmp_path, yaml_text):
    path = write_to_tmp_waypoints_file(tmp_path, yaml_text)
    with pytest.raises(ValueError):
        parse_waypoints_file(path)

@pytest.mark.parametrize(
    "yaml_text",
    [
        """
        - 1
        - 2
        """,
        """
        waypoints: 5
        """,
        """
        waypoints: {lat: 1
        """,
    ],
    ids=["top-level-list", "waypoints-not-a-list", "invalid-yaml"],
)
def test_invalid_file_raises(tmp_path, yaml_text):
    path = write_to_tmp_waypoints_file(tmp_path, yaml_text)
    with pytest.raises(ValueError):
        parse_waypoints_file(path)

def test_empty_file_returns_none_and_empty_list(tmp_path):
    path = write_to_tmp_waypoints_file(tmp_path, "")
    home, waypoints = parse_waypoints_file(path)
    assert home is None
    assert waypoints == []

def test_sort_one_waypoint():
    waypoint = Coordinate(1, 2, 3)
    result = sort_clockwise_sweep([waypoint])
    assert result == [waypoint]

def test_sort_waypoint_from_north():
    north = Coordinate(1, 0, 0)
    east = Coordinate(0, 1, 0)
    south = Coordinate(-1, 0, 0)
    west = Coordinate(0, -1, 0)
    result = sort_clockwise_sweep([east, north, west, south])
    assert result == [north, east, south, west]

def test_sort_empty_list():
    result = sort_clockwise_sweep([])
    assert result == []

def test_sort_starts_in_home_direction():
    home = Coordinate(1, 1, 0)
    north = Coordinate(1, 0, 0)
    east = Coordinate(0, 1, 0)
    south = Coordinate(-1, 0, 0)
    west = Coordinate(0, -1, 0)
    result = sort_clockwise_sweep([north, east, south, west], home=home)
    assert result == [east, south, west, north]

def test_offset_one_degree_north():
    east, north = east_north_coordinate_offset_m(0, 0, 1, 0)
    assert north == pytest.approx(111195, abs=1)
    assert east == pytest.approx(0, abs=1)

def test_offset_one_degree_east_at_lat_60():
    east, north = east_north_coordinate_offset_m(60, 0, 60, 1)
    assert north == pytest.approx(0, abs=1) 
    assert east == pytest.approx(55597.5, abs=1)

def test_sort_same_direction_closer_first():
    north = Coordinate(1, 0, 0)
    far_north = Coordinate(2, 0, 0)
    east = Coordinate(0, 1, 0)
    south = Coordinate(-1, 0, 0)
    west = Coordinate(0, -1, 0)
    result = sort_clockwise_sweep([far_north, north, east, south, west])
    assert result == [north, far_north, east, south, west]