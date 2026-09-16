"""
TODO(bootcamper): write the tests for ``src/waypoint_utils.py`` in here.

The example below covers files that parse fine: with and without ``home``,
and files with comments and blank lines in them. The rest is yours:

- Bad data: 

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

def test_parse_empty_file(tmp_path):
    path = write_to_tmp_waypoints_file(tmp_path, "")
    assert parse_waypoints_file(path) == (None, [])


def test_sort_empty_waypoint_list():
    assert sort_clockwise_sweep([]) == []

def test_sort_empty_waypoint_lists():
    waypoint = Coordinate(1, 2, 3)
    assert sort_clockwise_sweep([waypoint]) == [waypoint]


@pytest.mark.parametrize(
    "text",
    [
        "- one\n- two",  
        "hello",        
        "42",           
        "true",           
        """
        home: {lon: 2, alt: 3}
        waypoints:
          - {lon: 5, alt: 6}
        """,
        """
        waypoints:
          - {lat: 4, alt: 6}
          - {lat: 7, alt: 9}
        """,
        """
        # a lap

        home: {lat: 1, lon: 2}

        waypoints:
        - {lat: 4, lon: 5}
        """,
        """
        home: {lat: "d", lon: "d", alt: "d"}
        waypoints:
        - {lat: "d", lon: "d", alt: "f"}
        """,
        """
        home: {lat: 100, lon: 100, alt: 100}
        waypoints:
          - {lat: 4, lon: 5, alt: 6}
          - {lat: 7, lon: 8, alt: 9}
        """,
        """
        home: {lat: 1, lon: 200, alt: 1}
        waypoints:
          - {lat: 4, lon: 5, alt: 6}
          - {lat: 7, lon: 8, alt: 9}
        """,
        """
        home: {lat: 1, lon: 2, alt: 1
        waypoints:
          - {lat: 4, lon: 5, alt: 6}
          - {lat: 7, lon: 8, alt: 9}
        """,

    ],
    ids=["list", "string", "number", "boolean","no lat", "no lon", "no alt", "not numbers", "lat too big", "lon too big", "no parse"],
)
def test_non_mapping_top_level_is_rejected(tmp_path, text):
    path = write_to_tmp_waypoints_file(tmp_path, text)
    with pytest.raises(ValueError):
        parse_waypoints_file(path)

def test_east_north_coordinate():
    east, north = east_north_coordinate_offset_m(0.0, 0.0, 1.0, 1.0)

    assert east == pytest.approx(111190.85)
    assert north == pytest.approx(111195.08)


def test_sort_clockwise_sweep_starts_at_north_without_home():
  north = Coordinate(1, 0, 0)
  east = Coordinate(0, 1, 0)
  south = Coordinate(-1, 0, 0)
  west = Coordinate(0, -1, 0)

  assert sort_clockwise_sweep([west, south, east, north]) == [north,east,south,west,]


def test_sort_clockwise_sweep_starts_at_home_or_north_at_centroid():
  north = Coordinate(1, 0, 0)
  east = Coordinate(0, 1, 0)
  south = Coordinate(-1, 0, 0)
  west = Coordinate(0, -1, 0)

  assert sort_clockwise_sweep([north, east, south, west], home=east) == [east, south, west, north]
  assert sort_clockwise_sweep([north, east, south, west], home=Coordinate(0, 0, 0)) == [north, east, south, west]


def test_sort_clockwise_sweep_puts_closer_same_direction_waypoint_first():
  close_north = Coordinate(1, 0, 0)
  far_north = Coordinate(2, 0, 0)
  south = Coordinate(-3, 0, 0)

  assert sort_clockwise_sweep([far_north, south, close_north]) == [close_north,far_north,south,]






# def test_placeholder():
#     # TODO(bootcamper): delete this and write real tests. It's only here so
#     # linter doesn't complain about unused imports before you start.
#     assert callable(east_north_coordinate_offset_m)
#     assert callable(parse_waypoints_file)
#     assert callable(sort_clockwise_sweep)
