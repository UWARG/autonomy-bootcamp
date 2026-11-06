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
import dataclasses

from src.waypoint_utils import (
    east_north_coordinate_offset_m,
    parse_waypoints_file,
    sort_clockwise_sweep,
)
from src.types import Coordinate

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

# ****** PARSE WAYPOINTS FILE TEST ******
# Tests parse_waypoints_file() on a file with a top level that isn't a mapping
def test_file_top_level_not_mapping(tmp_path):
    text = """
Hello
waypoints:
  - {lat: 45, lon: 45, alt: 45}
"""
    path = write_to_tmp_waypoints_file(tmp_path, text)
    
    with pytest.raises(ValueError):
        parse_waypoints_file(path)


def test_file_missing_waypoint_component_lon(tmp_path):
    text = """
waypoints:
  - {lat: 2, lon: 180, alt: 2}
  - {lat: 2, alt: 2}
"""
    path = write_to_tmp_waypoints_file(tmp_path, text)

    with pytest.raises(ValueError):
        parse_waypoints_file(path)

def test_file_missing_waypoint_component_lat(tmp_path):
    text = """
waypoints:
  - {lat: 2, lon: 180, alt: 2}
  - {lon: 2, alt: 2}
"""
    path = write_to_tmp_waypoints_file(tmp_path, text)

    with pytest.raises(ValueError):
        parse_waypoints_file(path)

def test_file_missing_waypoint_component_alt(tmp_path):
    text = """
waypoints:
  - {lat: 2, lon: 180, alt: 2}
  - {lat: 2, lon: 2}
"""
    path = write_to_tmp_waypoints_file(tmp_path, text)

    with pytest.raises(ValueError):
        parse_waypoints_file(path)

def test_invalid_yaml(tmp_path):
    text = """
  waypoints:
  - {lat: 3, lon: aa, alt: 4}
  - {lat: 3, lon: 90, alt: 4}
"""
    path = write_to_tmp_waypoints_file(tmp_path, text)

    with pytest.raises(ValueError):
        parse_waypoints_file(path)

def test_non_existent_file(tmp_path):
    with pytest.raises(OSError):
        parse_waypoints_file(tmp_path)

def test_incorrect_format(tmp_path):
    text = """hello"""
    path = write_to_tmp_waypoints_file(tmp_path, text)

    with pytest.raises(ValueError):
      parse_waypoints_file(path)

def test_lat_out_of_range_top(tmp_path):
    text = """
waypoints:
  - {lat: 91, lon: 180, alt: 10}
"""
    path = write_to_tmp_waypoints_file(tmp_path, text)

    with pytest.raises(ValueError):
        parse_waypoints_file(path)

def test_lat_out_of_range_bottom(tmp_path):
    text = """
waypoints:
  - {lat: -91, lon: -180, alt: 10}
"""
    path = write_to_tmp_waypoints_file(tmp_path, text)

    with pytest.raises(ValueError):
        parse_waypoints_file(path)

def test_lon_out_of_range_top(tmp_path):
    text = """
waypoints:
  - {lat: 90, lon: 181, alt: 10}
"""
    path = write_to_tmp_waypoints_file(tmp_path, text)

    with pytest.raises(ValueError):
        parse_waypoints_file(path)

def test_lon_out_of_range_bottom(tmp_path):
    text = """
waypoints:
  - {lat: 90, lon: -181, alt: 10}
"""
    path = write_to_tmp_waypoints_file(tmp_path, text)

    with pytest.raises(ValueError):
        parse_waypoints_file(path)

def test_empty_waypoints(tmp_path):
    text = """
"""
    path = write_to_tmp_waypoints_file(tmp_path, text)

    assert parse_waypoints_file(path) == (None, [])

def test_coordinates_are_frozen(tmp_path):
    text = """
waypoints:
  - {lat: 90, lon: -180, alt: 10}
"""
    path = write_to_tmp_waypoints_file(tmp_path, text)

    _home, waypoints = parse_waypoints_file(path)

    with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
        waypoints[0].lat = 0.0

# ****** CLOCKWISE SWEEP TESTS ******

def test_sort_clockwise_sweep_no_home_valid(tmp_path):
    # centroid: {lat: 55, lon: 31.667}
    text = """
waypoints:
  - {lat: 30, lon: 90, alt: 5}
  - {lat: 25, lon: 45, alt: 6}
  - {lat: 40, lon: 30, alt: 4}
  """
    path = write_to_tmp_waypoints_file(tmp_path, text)
    parsedFile = parse_waypoints_file(path)
    sortedWaypoints = sort_clockwise_sweep(parsedFile[1], parsedFile[0])

    assert sortedWaypoints == [Coordinate(30, 90, 5), Coordinate(25, 45, 6), Coordinate(40, 30, 4)]


def test_sort_clockwise_sweep_home_valid(tmp_path):
    text = """
home: {lat: 30, lon: -95, alt: 6}
waypoints:
  - {lat: 33, lon: -85, alt: 5}
  - {lat: 24, lon: -93, alt: 6}
  - {lat: 35, lon: -94, alt: 4}
  """
    
    path = write_to_tmp_waypoints_file(tmp_path, text)
    parsedFile = parse_waypoints_file(path)
    sortedWaypoints = sort_clockwise_sweep(parsedFile[1], parsedFile[0])

    assert sortedWaypoints == [Coordinate(35, -94, 4), Coordinate(33, -85, 5), Coordinate(24, -93, 6)]

def test_sort_clockwise_sweep_home_on_centroid(tmp_path):
    text = """
home: {lat: 30, lon: -90, alt: 6}
waypoints:
  - {lat: 33, lon: -85, alt: 5}
  - {lat: 24, lon: -93, alt: 6}
  - {lat: 33, lon: -92, alt: 4}
  """
    
    path = write_to_tmp_waypoints_file(tmp_path, text)
    parsedFile = parse_waypoints_file(path)
    sortedWaypoints = sort_clockwise_sweep(parsedFile[1], parsedFile[0])

    assert sortedWaypoints == [Coordinate(33, -85, 5), Coordinate(24, -93, 6), Coordinate(33, -92, 4)]


def test_sort_clockwise_sweep_1_waypoint(tmp_path):
    text = """
    waypoints:
      - {lat: 90, lon: 30, alt: 5}
      """
    
    path = write_to_tmp_waypoints_file(tmp_path, text)
    parsedFile = parse_waypoints_file(path)
    sortedWaypoints = sort_clockwise_sweep(parsedFile[1], parsedFile[0])

    assert sortedWaypoints == [Coordinate(90, 30, 5)]

def test_sort_clockwise_sweep_0_waypoint(tmp_path):
    text = """
    waypoints:
    """

    path = write_to_tmp_waypoints_file(tmp_path, text)
    parsedFile = parse_waypoints_file(path)    
    sortedWaypoints = sort_clockwise_sweep(parsedFile[1], parsedFile[0])

    assert sortedWaypoints == []


def test_sort_clockwise_sweep_2_on_same_bearing_tie_break(tmp_path):
    # centroid: {lat: 31.75, lon: -92, alt: 6}
    text = """
waypoints: 
  - {lat: 35, lon: -90, alt: 5}  
  - {lat: 40, lon: -90, alt: 6}  
  - {lat: 25, lon: -85, alt: 6}  
  - {lat: 20, lon: -95, alt: 4}  
  """
    
    path = write_to_tmp_waypoints_file(tmp_path, text)
    parsedFile = parse_waypoints_file(path)
    sortedWaypoints = sort_clockwise_sweep(parsedFile[1], parsedFile[0])

    assert sortedWaypoints == [
        Coordinate(35.0, -90.0, 5.0), 
        Coordinate(40.0, -90.0, 6.0), 
        Coordinate(25.0, -85.0, 6.0), 
        Coordinate(20.0, -95.0, 4.0),
        ]

# ****** EAST NORTH OFFSET ******
def test_east_north_correct_offset():
    from_lat = 30
    from_lon = 30
    to_lat = 60
    to_lon = 60
    expected_offset = (2358803.858, 3335852.407)
    assert east_north_coordinate_offset_m(from_lat, from_lon, to_lat, to_lon) == pytest.approx(expected_offset, abs=1e-3)
