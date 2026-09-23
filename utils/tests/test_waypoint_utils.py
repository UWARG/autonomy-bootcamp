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

import dataclasses

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


def test_offset_one_degree_north_at_equator():
    east, north = east_north_coordinate_offset_m(0.0, 0.0, 1.0, 0.0)
    assert east == pytest.approx(0.0, abs=1.0)
    assert north == pytest.approx(111195.08, abs=1.0)


def test_offset_one_degree_east_at_equator():
    east, north = east_north_coordinate_offset_m(0.0, 0.0, 0.0, 1.0)
    assert east == pytest.approx(111195.08, abs=1.0)
    assert north == pytest.approx(0.0, abs=1.0)


def test_offset_one_degree_east_at_sixty_north():
    east, _ = east_north_coordinate_offset_m(60.0, 0.0, 60.0, 1.0)
    assert east == pytest.approx(55597.54, abs=1.0)


def test_waypoint_missing_alt_raises(tmp_path):
    path = write_to_tmp_waypoints_file(
        tmp_path,
        """
        waypoints:
          - {lat: 1, lon: 2}
        """,
    )
    with pytest.raises(ValueError):
        parse_waypoints_file(path)


def test_latitude_above_ninety_raises(tmp_path):
    path = write_to_tmp_waypoints_file(
        tmp_path,
        """
        waypoints:
          - {lat: 100, lon: 2, alt: 3}
        """,
    )
    with pytest.raises(ValueError):
        parse_waypoints_file(path)


def test_longitude_above_one_eighty_raises(tmp_path):
    path = write_to_tmp_waypoints_file(
        tmp_path,
        """
        waypoints:
          - {lat: 1, lon: 200, alt: 3}
        """,
    )
    with pytest.raises(ValueError):
        parse_waypoints_file(path)


def test_non_numeric_value_raises(tmp_path):
    path = write_to_tmp_waypoints_file(
        tmp_path,
        """
        waypoints:
          - {lat: abc, lon: 2, alt: 3}
        """,
    )
    with pytest.raises(ValueError):
        parse_waypoints_file(path)


def test_empty_file_returns_no_home_and_no_waypoints(tmp_path):
    path = write_to_tmp_waypoints_file(tmp_path, "")
    assert parse_waypoints_file(path) == (None, [])


def test_top_level_not_a_mapping_raises(tmp_path):
    path = write_to_tmp_waypoints_file(tmp_path, "- 1\n- 2\n")
    with pytest.raises(ValueError):
        parse_waypoints_file(path)


def test_waypoints_not_a_list_raises(tmp_path):
    path = write_to_tmp_waypoints_file(tmp_path, "waypoints: 5\n")
    with pytest.raises(ValueError):
        parse_waypoints_file(path)


def test_waypoint_not_a_mapping_raises(tmp_path):
    path = write_to_tmp_waypoints_file(tmp_path, "waypoints:\n  - 5\n")
    with pytest.raises(ValueError):
        parse_waypoints_file(path)


def test_invalid_yaml_raises(tmp_path):
    path = write_to_tmp_waypoints_file(tmp_path, "waypoints: [unclosed\n")
    with pytest.raises(ValueError):
        parse_waypoints_file(path)


def test_missing_file_raises(tmp_path):
    with pytest.raises(OSError):
        parse_waypoints_file(tmp_path / "nope.yaml")


def test_empty_waypoints_list_returns_empty(tmp_path):
    path = write_to_tmp_waypoints_file(tmp_path, "waypoints: []\n")
    assert parse_waypoints_file(path) == (None, [])


def test_parsed_coordinates_are_frozen(tmp_path):
    path = write_to_tmp_waypoints_file(
        tmp_path,
        """
        waypoints:
          - {lat: 1, lon: 2, alt: 3}
        """,
    )
    _, waypoints = parse_waypoints_file(path)
    with pytest.raises(dataclasses.FrozenInstanceError):
        waypoints[0].lat = 99


def test_sort_clockwise_from_north_with_no_home():
    north = Coordinate(1.0, 0.0, 0.0)
    east = Coordinate(0.0, 1.0, 0.0)
    south = Coordinate(-1.0, 0.0, 0.0)
    west = Coordinate(0.0, -1.0, 0.0)
    result = sort_clockwise_sweep([south, west, north, east])
    assert result == [north, east, south, west]


def test_sort_empty_list_returns_empty():
    assert sort_clockwise_sweep([]) == []


def test_sort_single_waypoint_returns_it_unchanged():
    only = Coordinate(1.0, 2.0, 3.0)
    assert sort_clockwise_sweep([only]) == [only]


def test_same_bearing_breaks_tie_by_nearest_first():
    near = Coordinate(1.0, 0.0, 0.0)
    far = Coordinate(2.0, 0.0, 0.0)
    anchor = Coordinate(-3.0, 0.0, 0.0)
    result = sort_clockwise_sweep([far, near, anchor])
    assert result == [near, far, anchor]


def test_home_sets_the_starting_direction():
    north = Coordinate(1.0, 0.0, 0.0)
    east = Coordinate(0.0, 1.0, 0.0)
    south = Coordinate(-1.0, 0.0, 0.0)
    west = Coordinate(0.0, -1.0, 0.0)
    home = Coordinate(-5.0, 0.0, 0.0)
    result = sort_clockwise_sweep([north, east, south, west], home=home)
    assert result == [south, west, north, east]


def test_home_at_centroid_falls_back_to_north():
    north = Coordinate(1.0, 0.0, 0.0)
    east = Coordinate(0.0, 1.0, 0.0)
    south = Coordinate(-1.0, 0.0, 0.0)
    west = Coordinate(0.0, -1.0, 0.0)
    home = Coordinate(0.0, 0.0, 0.0)
    result = sort_clockwise_sweep([north, east, south, west], home=home)
    assert result == [north, east, south, west]
