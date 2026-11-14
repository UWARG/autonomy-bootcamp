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
    result = parse_waypoints_file(path)
    assert result == (None, [])


def test_parse_empty_waypoints_list(tmp_path):
    path = write_to_tmp_waypoints_file(tmp_path, "waypoints: []")
    result = parse_waypoints_file(path)
    assert result == (None, [])


def test_sort_clockwise_sweep_one_waypoint():
    waypoint = Coordinate(43.0, -80.0, 10.0)
    waypoints = [waypoint]
    result = sort_clockwise_sweep(waypoints)
    assert result == waypoints


def test_sort_clockwise_sweep_no_waypoints():
    waypoints = []
    result = sort_clockwise_sweep(waypoints)
    assert result == []

def test_parse_missing_file(tmp_path):
    missing_path = tmp_path / "missing.yaml"
    with pytest.raises(OSError):
        parse_waypoints_file(missing_path)

def test_parse_invalid_top_level(tmp_path):
    text = """ 
              - lat: 23.0
                lon: -30.0
                alt: 15.0
            """
    path = write_to_tmp_waypoints_file(tmp_path, text)
    with pytest.raises(ValueError, match="expected a mapping"):
      parse_waypoints_file(path)

def test_parse_invalid_yaml(tmp_path):
    text = """
            waypoints : {
           """
    path = write_to_tmp_waypoints_file(tmp_path, text)
    with pytest.raises(ValueError, match="invalid YAML"):
        parse_waypoints_file(path)

def test_parse_non_list_waypoints(tmp_path):
    text = """
            waypoints : hello
           """
    path = write_to_tmp_waypoints_file(tmp_path, text)
    with pytest.raises(ValueError, match="must be a list"):
        parse_waypoints_file(path)

def test_east_north_coordinate_offset_m_same():
    east, north = east_north_coordinate_offset_m(10, 10, 10, 10)
    assert east == pytest.approx(0.0, abs=1e-9) and north == pytest.approx(0.0, abs=1e-9)

def test_east_north_coordinate_offset_m_greater():
    east, north = east_north_coordinate_offset_m(10, 10, 20, 10)
    assert east == pytest.approx(0.0, abs=1e-9)
    assert north == pytest.approx(1111950.8)

def test_sort_clockwise_sweep():
    North = Coordinate(lat = 1, lon = 0, alt=0)
    East = Coordinate(lat = 0, lon = 1, alt=0)
    South = Coordinate(lat = -1, lon = 0, alt=0)
    West = Coordinate(lat = 0, lon = -1, alt=0)

    scrambled = [East, South, West, North]
    expected = [North, East, South, West]
    assert sort_clockwise_sweep(scrambled, North) == expected

def test_sort_clockwise_sweep_same_direction():
    north1 = Coordinate(lat = 1, lon = 0, alt=0)
    north2 = Coordinate(lat = 2, lon = 0, alt=0)
    south = Coordinate(lat=-3, lon=0, alt=0)
    scrambled = [north2, south, north1]
    expected =  [north1, north2, south]
    assert sort_clockwise_sweep(scrambled) == expected

def test_parse_waypoints_file_not_list(tmp_path):
    text = """
           waypoints : 
                - hello
           """
    path = write_to_tmp_waypoints_file(tmp_path, text)
    with pytest.raises(ValueError, match="must be a mapping"):
        parse_waypoints_file(path)

def test_parse_waypoint_missing_alt(tmp_path):
    text = """
    waypoints:
      - {lat: 1, lon: 2}
    """
    path = write_to_tmp_waypoints_file(tmp_path, text)
    with pytest.raises(ValueError, match="missing key"):
        parse_waypoints_file(path)

def test_parse_waypoint_out_of_range(tmp_path):
    text = """
    waypoints:
        - {lat: 91, lon: 0, alt: 0}
    """
    path = write_to_tmp_waypoints_file(tmp_path, text)
    with pytest.raises(ValueError, match="out of range"):
        parse_waypoints_file(path)

def test_parse_rejects_non_numeric(tmp_path):
    text = """
    waypoints:
      - {lat: idk, lon: 0, alt: 0}
    """
    path = write_to_tmp_waypoints_file(tmp_path, text)

    with pytest.raises(ValueError, match="non_numeric"):
        parse_waypoints_file(path)

def test_east_offset_scales_with_latitude():
    east, north = east_north_coordinate_offset_m(60, 0, 60, 1,)

    assert east == pytest.approx(55_597.54, abs=1.0) and north == pytest.approx(0.0, abs=1e-9)

def test_sort_clockwise_wraps_after_west_home():
    north = Coordinate(lat=1, lon=0, alt=0)
    east = Coordinate(lat=0, lon=1, alt=0)
    south = Coordinate(lat=-1, lon=0, alt=0)
    west = Coordinate(lat=0, lon=-1, alt=0)

    scrambled = [east, south, north, west]
    result = sort_clockwise_sweep(scrambled, home=west)

    assert result == [west, north, east, south]

