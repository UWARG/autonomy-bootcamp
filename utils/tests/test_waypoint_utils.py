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



def test_parse_empty_file_returns_nothing(tmp_path):
    path = write_to_tmp_waypoints_file(tmp_path, "")
    assert parse_waypoints_file(path) == (None, [])


@pytest.mark.parametrize(
    "text",
    [
        "waypoints:\n  - 5\n",
        "waypoints: 5\n",
        "waypoints: [unclosed\n",
    ],
    ids=["entry-not-a-mapping", "waypoints-not-a-list", "invalid-yaml"],
)
def test_parse_malformed_file_raises(tmp_path, text):
    path = write_to_tmp_waypoints_file(tmp_path, text)
    with pytest.raises(ValueError):
        parse_waypoints_file(path)


def test_sort_clockwise_sweep_same_direction_closer_first():
    near = Coordinate(1, 0, 0)
    far = Coordinate(2, 0, 0)
    other = Coordinate(-3, 0, 0)

    result = sort_clockwise_sweep([far, near, other])

    assert result == [near, far, other]


def test_coordinate_is_frozen():
    coordinate = Coordinate(1, 2, 3)
    with pytest.raises(FrozenInstanceError):
        coordinate.lat = 5




def test_sort_clockwise_sweep_starts_in_homes_direction():
    north = Coordinate(1, 0, 0)
    east = Coordinate(0, 1, 0)
    south = Coordinate(-1, 0, 0)
    west = Coordinate(0, -1, 0)
    home = Coordinate(0, 2, 0)

    result = sort_clockwise_sweep([south, north, west, east], home)

    assert result == [east, south, west, north]


def test_sort_clockwise_sweep_starts_middle_home_direction():
    north = Coordinate(1, 0, 0)
    east = Coordinate(0, 1, 0)
    south = Coordinate(-1, 0, 0)
    west = Coordinate(0, -1, 0)
    home = Coordinate(0, 0, 0)

    result = sort_clockwise_sweep([south, north, west, east], home)

    assert result == [north, east, south, west]






def test_sort_clockwise_sweep_starts_north_and_goes_clockwise():
    north = Coordinate(1, 0, 0)
    east = Coordinate(0, 1, 0)
    south = Coordinate(-1, 0, 0)
    west = Coordinate(0, -1, 0)

    result = sort_clockwise_sweep([south, north, west, east])

    assert result == [north, east, south, west]






@pytest.mark.parametrize(
    "waypoints",
    [
        [],
        [Coordinate(1, 2, 3)],
    ],
    ids=["empty", "single"],
)
def test_sort_clockwise_sweep_zero_or_one_returns_same(waypoints):
    assert sort_clockwise_sweep(waypoints) == waypoints






def test_parse_waypoint_on_boundary_is_accepted(tmp_path):
    path = write_to_tmp_waypoints_file(
        tmp_path, "waypoints:\n  - {lat: 90, lon: 180, alt: 0}\n"
    )
    home, waypoints = parse_waypoints_file(path)
    assert home is None
    assert waypoints == [Coordinate(90, 180, 0)]





@pytest.mark.parametrize(
    "entry",
    [
        "{lat: 90.1, lon: 0, alt: 0}",
        "{lat: -91, lon: 0, alt: 0}",  
        "{lat: 90.1, lon: 183, alt: 0}",  
        "{lat: 90.1, lon: -183, alt: 0}",  
    ],
    ids=["lat-too-high", "lat-too-low", "lon-too-high", "lon-too-low"],
)
def test_parse_waypoint_out_of_range_raises(tmp_path, entry):
    text = f"waypoints:\n  - {entry}\n"
    path = write_to_tmp_waypoints_file(tmp_path, text)
    with pytest.raises(ValueError):
        parse_waypoints_file(path)






@pytest.mark.parametrize(
    "entry",
    [
        "{lat: abc, lon: 5, alt: 6}",
        "{lat: 4, lon: aa, alt: 6}",  
        "{lat: 4, lon: 5, alt: aa}",  
    ],
    ids=["bad-lat", "bad-lon", "bad-alt"],
)
def test_parse_waypoint_non_numeric_raises(tmp_path, entry):
    text = f"waypoints:\n  - {entry}\n"
    path = write_to_tmp_waypoints_file(tmp_path, text)
    with pytest.raises(ValueError):
        parse_waypoints_file(path)








def test_parse_missing_file_raises(tmp_path):
    missing = tmp_path / "nope.yaml"
    with pytest.raises(OSError):
        parse_waypoints_file(missing)

def test_parse_top_level_list_raises(tmp_path):
    path = write_to_tmp_waypoints_file(tmp_path, "- 1\n- 2\n")
    with pytest.raises(ValueError):
        parse_waypoints_file(path)

@pytest.mark.parametrize(
    "entry",
    [
        "{lon: 5, alt: 6}",
        "{lat: 5, alt: 6}",  
        "{lon: 5, lat: 6}",  
    ],
    ids=["missing-lat", "missing-lon", "missing-alt"],
)
def test_parse_waypoint_missing_key_raises(tmp_path, entry):
    text = f"waypoints:\n  - {entry}\n"
    path = write_to_tmp_waypoints_file(tmp_path, text)
    with pytest.raises(ValueError):
        parse_waypoints_file(path)



def test_offset_same_point_is_zero():
    east, north = east_north_coordinate_offset_m(0, 0, 0, 0)
    assert east == pytest.approx(0)
    assert north == pytest.approx(0)


def test_offset_one_degree_north():
    east, north = east_north_coordinate_offset_m(0, 0, 1, 0)
    assert east == pytest.approx(0)
    assert north == pytest.approx(111195.03, abs=1)

def test_offset_one_degree_east():
    east, north = east_north_coordinate_offset_m(0, 0, 0, 1)
    assert east == pytest.approx(111195.03, abs=1)
    assert north == pytest.approx(0)

def test_offset_trip_degree_east():
    east, north = east_north_coordinate_offset_m(60, 0, 60, 1)
    assert east == pytest.approx(55597.54012, abs=1)
    assert north == pytest.approx(0)





