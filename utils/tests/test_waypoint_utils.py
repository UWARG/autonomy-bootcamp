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

# Bad Data Tests:  a file whose top level isn't a mapping, waypoints missing
#  ``lat``, ``lon``, or ``alt``, values that aren't numbers, YAML that
#  doesn't parse, and a file that isn't there.
def test_parse_waypoints_file_top_level_not_mapping(tmp_path):
    path = write_to_tmp_waypoints_file(tmp_path, "- 1\n- 2\n")
    with pytest.raises(ValueError):
        parse_waypoints_file(path)
 
def test_parse_waypoints_file_waypoints_missing_key(tmp_path):
    path = write_to_tmp_waypoints_file(
        tmp_path, "waypoints:\n  - {lat: 1, lon: 2}\n"
    )
    with pytest.raises(ValueError):
        parse_waypoints_file(path)
 
def test_parse_waypoints_file_waypoints_not_mapping(tmp_path):
    path = write_to_tmp_waypoints_file(tmp_path, 'waypoints:\n  - "not mapping"\n')
    with pytest.raises(ValueError):
        parse_waypoints_file(path)

def test_parse_waypoints_file_not_found(tmp_path):
    missing_path = tmp_path / "does-not-exist.yaml"
    with pytest.raises(OSError):
        parse_waypoints_file(missing_path)
 
def test_parse_waypoints_file_home_missing_key(tmp_path):
    path = write_to_tmp_waypoints_file(
        tmp_path, "home: {lat: 1, lon: 2}\nwaypoints: []\n"
    )
    with pytest.raises(ValueError):
        parse_waypoints_file(path)
 
def test_parse_waypoints_file_home_not_mapping(tmp_path):
    path = write_to_tmp_waypoints_file(
        tmp_path, 'home: "not mapping"\nwaypoints: []\n'
    )
    with pytest.raises(ValueError):
        parse_waypoints_file(path)
 
def test_parse_waypoints_file_non_numeric_value(tmp_path):
    path = write_to_tmp_waypoints_file(
        tmp_path, 'waypoints:\n  - {lat: "abc", lon: 2, alt: 3}\n'
    )
    with pytest.raises(ValueError):
        parse_waypoints_file(path)

def test_parse_waypoints_file_waypoints_not_a_list(tmp_path):
    path = write_to_tmp_waypoints_file(
        tmp_path, "home: {lat: 1, lon: 2, alt: 3}\nwaypoints: \"not a list\"\n"
    )
    with pytest.raises(ValueError):
        parse_waypoints_file(path)
 
def test_parse_waypoints_file_invalid_yaml(tmp_path):
    path = write_to_tmp_waypoints_file(tmp_path, "waypoints: [1, 2\n")
    with pytest.raises(ValueError):
        parse_waypoints_file(path)

# Out of Range tests:latitudes past +/-90 and longitudes past +/-180 get rejected.
@pytest.mark.parametrize(
    ("lat", "lon"),
    [
        (91, 0),
        (-91, 0),
        (0, 181),
        (0, -181),
    ],
)
def test_coordinate_out_of_range(tmp_path, lat, lon):
    path = write_to_tmp_waypoints_file(
        tmp_path,
        f"waypoints:\n  - {{lat: {lat}, lon: {lon}, alt: 0}}",
    )
    with pytest.raises(ValueError):
        parse_waypoints_file(path)

# Nothing to work with tests: an empty file, an empty ``waypoints`` list, and
#  ``sort_clockwise_sweep`` given a list of 0 or 1 waypoints.
def test_parse_waypoints_file_empty_file(tmp_path):
    path = write_to_tmp_waypoints_file(tmp_path, "")
    assert parse_waypoints_file(path) == (None, [])
 
@pytest.mark.parametrize(
    "content, expected_home",
    [
        ("waypoints: []\n", None),
        ("home: {lat: 1, lon: 2, alt: 3}\n", Coordinate(1, 2, 3)),
    ]
)
def test_parse_waypoints_file_empty_waypoints(tmp_path, content, expected_home):
    path = write_to_tmp_waypoints_file(tmp_path, content)
    home, waypoints = parse_waypoints_file(path)
    assert home == expected_home
    assert waypoints == []
 
def test_sort_clockwise_sweep_empty_list():
    assert sort_clockwise_sweep([]) == []
 
def test_sort_clockwise_sweep_single_waypoint():
    only = Coordinate(1.0, 2.0, 3.0)
    assert sort_clockwise_sweep([only]) == [only]

# east_north_coordinate_offset_m test: offsets you worked out yourself,
#  compared with ``pytest.approx``. Never use ``==`` on meters.
def test_east_north_coordinate_offset_m():
    east, north = east_north_coordinate_offset_m(0, 0, 1, 0)
    assert east == pytest.approx(0.0)
    assert north == pytest.approx(111195, abs=1)

def test_east_north_coordinate_offset_m_at_high_latitude():
    east, north = east_north_coordinate_offset_m(60, 0, 60, 1)
    assert east == pytest.approx(55597.5, abs=1)
    assert north == pytest.approx(0.0)

# Ordering: with no ``home``, ``sort_clockwise_sweep`` goes clockwise starting from north.
def test_clockwise_sweep_from_north_clockwise_no_home():
    north = Coordinate(1, 0, 0)
    east = Coordinate(0, 1, 0)
    south = Coordinate(-1, 0, 0)
    west = Coordinate(0, -1, 0)
 
    result = sort_clockwise_sweep([west, south, east, north])
    assert result == [north, east, south, west]
 
def test_clockwise_sweep_from_home():
    north = Coordinate(1, 0, 0)
    east = Coordinate(0, 1, 0)
    south = Coordinate(-1, 0, 0)
    west = Coordinate(0, -1, 0)
    home = Coordinate(0, 2, 0)
 
    result = sort_clockwise_sweep([north, west, south, east], home)
    assert result == [east, south, west, north]

# With a ``home``: the order starts in home's direction instead, and goes
#  back to starting at north if home is right on top of the centroid.
def test_clockwise_sweep_from_centroid():
    north = Coordinate(1, 0, 0)
    east = Coordinate(0, 1, 0)
    south = Coordinate(-1, 0, 0)
    west = Coordinate(0, -1, 0)
    home = Coordinate(0, 0, 0)

    result = sort_clockwise_sweep([south, west, north, east], home)
    assert result == [north, east, south, west]

# Two waypoints in the same direction: the closer one comes first.
def test_sort_clockwise_sweep_same_direction_closer_first():
    closer = Coordinate(1, 0, 0)
    farther = Coordinate(2, 0, 0)
    other_direction = Coordinate(-3, 0, 0)
    waypoints = [farther, other_direction, closer]
    result = sort_clockwise_sweep(waypoints)
    assert result == [closer, farther, other_direction]

# Parsing gives you frozen ``Coordinate`` objects that can't be changed.
def test_parse_waypoints_file_coordinate_cannot_be_changed(tmp_path):
    path = write_to_tmp_waypoints_file(
        tmp_path,
        "waypoints:\n  - {lat: 1, lon: 2, alt: 3}",
    )
    _, waypoints = parse_waypoints_file(path)
    with pytest.raises(AttributeError):
        waypoints[0].lat = 99
