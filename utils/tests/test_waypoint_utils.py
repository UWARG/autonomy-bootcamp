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

@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            """
            home: {lat: 1, lon: 2, alt: 3}
            waypoints:
              - {lon: 4, alt: 6}
            """,
            ValueError,
        ),
        (
            """
            home: {lat: 1, lon: 2, alt: 3}
            waypoints:
              - {lat: 4, alt: 6}
            """,
            ValueError,
        ),
        (
            """
            home: {lat: 1, lon: 2, alt: 3}
            waypoints:
              - {lat: 4, lon: 6}
            """,
            ValueError,
        ),
        (
            """
            home: {lon: 2, alt: 3}
            waypoints:
              - {lat: 4, lon: 5, alt: 6}
            """,
            ValueError,
        ),
        (
            """
            home: {lat: 2, alt: 3}
            waypoints:
              - {lat: 4, lon: 5, alt: 6}
            """,
            ValueError,
        ),
        (
            """
            home: {lat: 2, lon: 3}
            waypoints:
              - {lat: 4, lon: 5, alt: 6}
            """,
            ValueError,
        ),
    ],
    ids=["w-missing-lat", "w-missing-lon", "w-missing-alt", "h-missing-lat", "h-missing-lon", "h-missing-alt"],
)
def test_missing_data(tmp_path, text, expected):
    path = write_to_tmp_waypoints_file(tmp_path, text)
    with pytest.raises(expected):
        parse_waypoints_file(path)
        
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            """
            home: {lat: a, lon: 2, alt: 3}
            waypoints:
              - {lat: 4, lon: 5, alt: 6}
            """,
            ValueError,
        ),
        (
            """
            waypoints:
              - {lat: 4, lon: 5, alt: 6}
              - {lat: 7, lon: $3, alt: 9}
            """,
            ValueError,
        ),
        (
            """
            # a lap

            home: {lat: 1, lon: 2, alt: 3}

            waypoints:
              # first leg
              - {lat: 4, lon: , alt: wqef}
            """,
            ValueError,
        ),
    ],
    ids=["character-values", "character-number-values", "multiple-nans"],
)
def test_nan_data(tmp_path, text, expected):
    path = write_to_tmp_waypoints_file(tmp_path, text)
    with pytest.raises(expected):
        parse_waypoints_file(path)

def test_invalid_yaml(tmp_path):
    path = write_to_tmp_waypoints_file(tmp_path, "invalid_yaml: [1, 2, 3")
    with pytest.raises(ValueError):
        parse_waypoints_file(path)

def test_file_not_found(tmp_path):
    path = tmp_path / "nonexistent.yaml"
    with pytest.raises(FileNotFoundError):
        parse_waypoints_file(path)
        
@pytest.mark.parametrize(
    ("text", "expected_error"),
    [
        (
            """
            home: {lat: 91, lon: 2, alt: 3}
            waypoints:
              - {lat: 4, lon: 5, alt: 6}
            """,
            ValueError,
        ),
        (
            """
            waypoints:
              - {lat: 4, lon: 5, alt: 6}
              - {lat: 7, lon: 181, alt: 9}
            """,
            ValueError,
        ),
        (
            """
            # a lap

            home: {lat: -94, lon: 2, alt: 3}

            waypoints:
              # first leg
              - {lat: 4, lon: 0, alt: 0}
            """,
            ValueError,
        ),
        (
            """
            # a lap

            home: {lat: -5, lon: 2, alt: 3}

            waypoints:
              # first leg
              - {lat: 4, lon: -184, alt: 0}
            """,
            ValueError,
        ),
    ],
    ids=["lat-over", "long-over", "lat-under", "long-under"],
)
def test_input_out_of_bounds(tmp_path, text, expected_error):
    path = write_to_tmp_waypoints_file(tmp_path, text)
    with pytest.raises(expected_error):
        parse_waypoints_file(path)

def test_empty_file(tmp_path):
    path = write_to_tmp_waypoints_file(tmp_path, "")
    assert parse_waypoints_file(path) == (None, [])
    
def test_empty_waypoints(tmp_path):
    path = write_to_tmp_waypoints_file(tmp_path, "waypoints: []")
    assert parse_waypoints_file(path) == (None, [])

def test_sort_clockwise_sweep_empty_and_single():
    assert sort_clockwise_sweep([]) == []
    assert sort_clockwise_sweep([Coordinate(1, 2, 3)]) == [Coordinate(1, 2, 3)]
    
@pytest.mark.parametrize(
    ("from_lat", "from_lon", "to_lat", "to_lon", "expected_north", "expected_east"),
    [
        (0, 0, 0, 0, 0, 0),
        (0, 0, 1, 0, 111195, 0),
        (0, 0, 0, 1, 0, 111195),
        (0, 0, -1, 0, -111195, 0),
        (0, 0, 0, -1, 0, -111195),
        (0, 0, 1, 1, 111191, 111195),
        (60, 0, 60, 1, 0, 55597),
    ],
    ids=["same-point", "one-degree-north", "one-degree-east", "one-degree-south", "one-degree-west", "one-degree-north-east", "far"],
)
def test_offset_calculation(from_lat, from_lon, to_lat, to_lon, expected_north, expected_east):
  east, north = east_north_coordinate_offset_m(from_lat, from_lon, to_lat, to_lon)
  assert east == pytest.approx(expected_east, rel=1)
  assert north == pytest.approx(expected_north, rel=1)

@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            """
            waypoints:
              - {lat: 0, lon: 0, alt: 0}
              - {lat: 1, lon: 0, alt: 0}
              - {lat: 0, lon: 1, alt: 0}
            """,
            [Coordinate(0, 1, 0), Coordinate(0, 0, 0), Coordinate(1, 0, 0)],
        ),
        (
           """
            home: {lat: 0, lon: 2, alt: 0}
            waypoints:
              - {lat: 1, lon: 0, alt: 0}
              - {lat: 0, lon: 1, alt: 0}
              - {lat: -1, lon: 0, alt: 0}
              - {lat: 0, lon: -1, alt: 0}
            """,
            [
                Coordinate(0, 1, 0),
                Coordinate(-1, 0, 0),
                Coordinate(0, -1, 0),
                Coordinate(1, 0, 0),
            ],
        ),
        (
           """
            home: {lat: 0, lon: 2, alt: 0}
            waypoints:
              - {lat: 1, lon: 0, alt: 0}
              - {lat: 2, lon: 0, alt: 0}
              - {lat: -1, lon: 0, alt: 0}
              - {lat: -2, lon: 0, alt: 0}
            """,
            [
                Coordinate(-1, 0, 0),
                Coordinate(-2, 0, 0),
                Coordinate(1, 0, 0),
                Coordinate(2, 0, 0),
            ],
        ),
    ],
    ids=["no-home", "with-home", "samed-dir"],
)
def test_sorted_waypoints_file(tmp_path, text, expected):
    path = write_to_tmp_waypoints_file(tmp_path, text)
    home, waypoints = parse_waypoints_file(path)
    sorted_waypoints = sort_clockwise_sweep(waypoints, home)
    assert sorted_waypoints == expected
  
def test_coordinate_immutable(tmp_path):
  path = write_to_tmp_waypoints_file(
        tmp_path,
        """
        home: {lat: 1, lon: 2, alt: 3}
        waypoints:
          - {lat: 4, lon: 5, alt: 6}
        """,
    )
  home, waypoints = parse_waypoints_file(path)
  with pytest.raises(FrozenInstanceError):
    home.lat = 10
    
  with pytest.raises(FrozenInstanceError):
    waypoints[0].lat = 10

# def test_placeholder():
#     # TODO(bootcamper): delete this and write real tests. It's only here so
#     # linter doesn't complain about unused imports before you start.
#     assert callable(east_north_coordinate_offset_m)
#     assert callable(parse_waypoints_file)
#     assert callable(sort_clockwise_sweep)
