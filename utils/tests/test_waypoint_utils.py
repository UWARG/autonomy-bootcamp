"""
TODO(bootcamper): write the tests for ``src/waypoint_utils.py`` in here.

The example below covers files that parse fine: with and without ``home``,
and files with comments and blank lines in them. The rest is yours:

Done:
- Out of range: latitudes past +/-90 and longitudes past +/-180 get
  rejected.
- Bad data: a file whose top level isn't a mapping, waypoints missing
  ``lat``, ``lon``, or ``alt``, values that aren't numbers, YAML that
  doesn't parse, and a file that isn't there.
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

#--------------------------------
# Bad data
#--------------------------------

def test_parse_error_on_file_with_no_mapping_on_top_level(tmp_path):
    text =  """
            - {lat: 77, lon: 103, alt: 85}
            - {lat: 88, lon: 154, alt: 4}
            """
    path = write_to_tmp_waypoints_file(tmp_path, text)
    with pytest.raises(ValueError):
      parse_waypoints_file(path)

@pytest.mark.parametrize(
    ("text"),
    [
        (
            """
            {lat: 1, lon: 2, alt: 3}
            waypoints:
              - {lat: 4, lon: 5, alt: 6}
            """
        ),
        (
            """
            waypoints:
            {lat: 4, lon: 5, alt: 6}
              - {lat: 7, lon: 8, alt: 9}
            """
        ),
        (
            """
            home {lat: 1, lon: 2, alt: 3}
            waypoints:
              - {lat: 4, lon: 5, alt: 6}
            """
        ),
        (
            """
              waypoints
              - {lat: 7, lon: 8, alt: 9}
            """
        ),
        (
            """
            waypoints: {lat: 4, lon: 5, alt: 6}
            """
        ),
        (
            """
            waypoints: 
              - hello
            """
        ),
    ],
    ids=["home-coords-without-key", "waypoints-dash-missing", "colon-missing", "spacing-error", "waypoints-not-a-list", "waypoints-with-no-coords"],
)
def test_parse_coords_raises_error_on_invalid_mapping(tmp_path, text):
    path = write_to_tmp_waypoints_file(tmp_path, text)
    with pytest.raises(ValueError):
      parse_waypoints_file(path)

@pytest.mark.parametrize(
    ("text"),
    [
        (
            """
            waypoints:
              - {lat: , lon: 8, alt: 9}
            """
        ),
        (
            """
            home: {lat: 1, lon: 2, alt: 3}
            waypoints:
              - {lat: 4, lon: , alt: 6}
            """
        ),
        (
            """
              waypoints:
              - {lat: 7, lon: 8, alt: }
            """
        ),
        (
            """
              waypoints:
              - {lat: seven, lon: 8, alt: 9}
            """
        ),
        (
            """
              waypoints:
              - {lat: 7, lon: apple, alt: 9}
            """
        ),
        (
            """
              waypoints:
              - {lat: 7, lon: 8, alt: tree}
            """
        ),
    ],
    ids=["empty-lat", "empty-lon", "empty-alt", "non-numeric-lat", "non-numeric-lon", "non-numeric-alt"],
)
def test_parse_coords_raises_error_on_non_numeric_values(tmp_path, text):
    path = write_to_tmp_waypoints_file(tmp_path, text)
    with pytest.raises(ValueError):
      parse_waypoints_file(path)

@pytest.mark.parametrize(
    ("text"),
    [
        (
            """
            home: {lat: 85, lon: 30, alt: 20}
            waypoints:
              - {lon: -83, alt: 88}
            """
        ),
        (
            """
            waypoints:
              - {lat: 77, alt: 85}
              - {lat: 88, lon: 154, alt: 4}
            """
        ),
        (
            """
            waypoints:
              - {lat: 77, lon: 13}
            """
        ),
    ],
    ids=["missing-lat", "missing-lon", "missing-alt"]
)
def test_raises_error_on_missing_coord_fields(tmp_path, text):
    path = write_to_tmp_waypoints_file(tmp_path, text)
    with pytest.raises(ValueError):
      parse_waypoints_file(path)

def test_raises_error_on_non_existent_file(tmp_path):
    path = tmp_path / "waypoints.yaml"
    with pytest.raises(OSError):
      parse_waypoints_file(path)

#-------------------------------
# Out of range
#-------------------------------

@pytest.mark.parametrize(
    ("text"),
    [
        (
            """
            home: {lat: 95, lon: 30, alt: 20}
            waypoints:
              - {lat: 77, lon: -83, alt: 88}
            """
        ),
        (
            """
            waypoints:
              - {lat: 77, lon: 183, alt: 85}
              - {lat: 88, lon: 154, alt: 4}
            """
        ),
        (
            """
            waypoints:
              - {lat: -107, lon: -23, alt: -50}
            """
        ),
        (
            """
            home: {lat: 4, lon: 30, alt: 20}
            waypoints:
              - {lat: 77, lon: -83, alt: 88}
              - {lat: -17, lon: -283, alt: 8}
            """
        ),
    ],
    ids=["invalid-lat-high", "invalid-lon-high", "invalid-lat-low", "invalid-lon-low"],
)
def test_raises_error_on_coords_out_of_bounds(tmp_path, text):
    path = write_to_tmp_waypoints_file(tmp_path, text)
    with pytest.raises(ValueError):
      parse_waypoints_file(path)

#-------------------------------
# Nothing to work with
#-------------------------------

def test_parse_waypoints_on_empty_file(tmp_path):
    path = write_to_tmp_waypoints_file(tmp_path, "")
    assert parse_waypoints_file(path) == (None, [])
  
def test_parse_waypoints_on_empty_waypoints_list(tmp_path):
    text =  """
            waypoints:

            """
    path = write_to_tmp_waypoints_file(tmp_path, text)
    assert parse_waypoints_file(path) == (None, [])

def test_sort_clockwise_sweep_on_0_waypoints():
    waypoints = []
    ans = sort_clockwise_sweep(waypoints)
    assert ans == []
    assert ans is not waypoints

def test_sort_clockwise_sweep_on_1_waypoint():
    waypoints = [Coordinate(1.0, 1.0, 0.0)]
    ans = sort_clockwise_sweep(waypoints)
    assert ans == [Coordinate(1.0, 1.0, 0.0)]
    assert ans is not waypoints

#--------------------------------
# east_north_coordinate_offset_m
#--------------------------------

def test_coord_offset_on_same_coords():
    ans = east_north_coordinate_offset_m(1.0, 2.0, 1.0, 2.0)
    assert ans == pytest.approx((0, 0))
  
def test_coord_offset_on_known_coords():
    ans = east_north_coordinate_offset_m(2.0, 5.0, 3.0, 2.0)
    assert ans == pytest.approx((-333267.7, 111195.1), abs=1)
  
#--------------------------------
# sort_clockwise_sweep
#--------------------------------

def test_sort_clockwise_sweep_without_home():
    waypoints = [
      Coordinate(0.0, 1.0, 0.0),    # east
      Coordinate(0.0, -1.0, 0.0),   # west
      Coordinate(1.0, 0.0, 0.0),    # north
      Coordinate(-1.0, 0.0, 0.0)    # south
    ]
    answer = [
      Coordinate(1.0, 0.0, 0.0),    # north
      Coordinate(0.0, 1.0, 0.0),    # east
      Coordinate(-1.0, 0.0, 0.0),   # south
      Coordinate(0.0, -1.0, 0.0)    # west
    ]
    
    assert sort_clockwise_sweep(waypoints) == answer

def test_sort_clockwise_sweep_with_home():
    home = Coordinate(0.0, 5.0, 0.0)
    waypoints = [
      Coordinate(0.0, -1.0, 0.0),   # west
      Coordinate(1.0, 0.0, 0.0),    # north
      Coordinate(-1.0, 0.0, 0.0),   # south
      Coordinate(0.0, 1.0, 0.0)     # east
    ]
    answer = [
      Coordinate(0.0, 1.0, 0.0),    # east
      Coordinate(-1.0, 0.0, 0.0),   # south
      Coordinate(0.0, -1.0, 0.0),   # west
      Coordinate(1.0, 0.0, 0.0)     # north
    ]

    assert sort_clockwise_sweep(waypoints, home) == answer

def test_sort_clockwise_sweep_with_home_on_centroid():
    home = Coordinate(0.0, 0.0, 0.0)
    waypoints = [
      Coordinate(0.0, -1.0, 0.0),   # west
      Coordinate(1.0, 0.0, 0.0),    # north
      Coordinate(-1.0, 0.0, 0.0),   # south
      Coordinate(0.0, 1.0, 0.0)     # east
    ]
    answer = [
      Coordinate(1.0, 0.0, 0.0),    # north
      Coordinate(0.0, 1.0, 0.0),    # east
      Coordinate(-1.0, 0.0, 0.0),   # south
      Coordinate(0.0, -1.0, 0.0)    # west
    ]
    
    assert sort_clockwise_sweep(waypoints, home) == answer

def test_sort_clockwise_sweep_with_home_isclose_to_centroid():
    home = Coordinate(0.0, 1e-15, 0.0)
    waypoints = [
      Coordinate(0.0, -1.0, 0.0),   # west
      Coordinate(1.0, 0.0, 0.0),    # north
      Coordinate(-1.0, 0.0, 0.0),   # south
      Coordinate(0.0, 1.0, 0.0)     # east
    ]
    answer = [
      Coordinate(1.0, 0.0, 0.0),    # north
      Coordinate(0.0, 1.0, 0.0),    # east
      Coordinate(-1.0, 0.0, 0.0),   # south
      Coordinate(0.0, -1.0, 0.0)    # west
    ]
    
    assert sort_clockwise_sweep(waypoints, home) == answer

def test_sort_clockwise_sweep_with_waypoints_in_same_direction():
    waypoints = [
      Coordinate(0.0, -1.0, 0.0),   # west
      Coordinate(2.5, 0.0, 0.0),    # north, far
      Coordinate(1.0, 0.0, 0.0),    # north, near
      Coordinate(-1.0, 0.0, 0.0),   # south
      Coordinate(0.0, 1.0, 0.0)     # east
    ]
    answer = [
      Coordinate(1.0, 0.0, 0.0),    # north, near
      Coordinate(2.5, 0.0, 0.0),    # north, far
      Coordinate(0.0, 1.0, 0.0),    # east
      Coordinate(-1.0, 0.0, 0.0),   # south
      Coordinate(0.0, -1.0, 0.0)    # west
    ]

    assert sort_clockwise_sweep(waypoints) == answer

def test_editing_frozen_coords_home_raises_error(tmp_path):
    text =  """
            home: {lat: 1, lon: 3, alt: 0}
            waypoints:
            - {lat: 77, lon: 103, alt: 85}
            - {lat: 88, lon: 154, alt: 4}
            """
    path = write_to_tmp_waypoints_file(tmp_path, text)
    parsed = parse_waypoints_file(path)
    with pytest.raises(FrozenInstanceError):
      parsed[0].lat = 2

def test_editing_frozen_coords_waypoint_raises_error(tmp_path):
    text =  """
            home: {lat: 1, lon: 3, alt: 0}
            waypoints:
            - {lat: 77, lon: 103, alt: 85}
            - {lat: 88, lon: 154, alt: 4}
            """
    path = write_to_tmp_waypoints_file(tmp_path, text)
    parsed = parse_waypoints_file(path)
    with pytest.raises(FrozenInstanceError):
      parsed[1][0].lon = 2