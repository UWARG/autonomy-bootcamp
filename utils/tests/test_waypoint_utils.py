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

    result = parse_waypoints_file(path)

    assert result == expected

    with pytest.raises(dataclasses.FrozenInstanceError): # Checking that we can't change coordinate values
        if result[0]:
            result[0].lat = -1

        for waypoint in result[1]:
            waypoint.lat = -1


@pytest.mark.parametrize(
    "text",
    [
        """
          home: {lat: 1, lon: 2, test: 3}
          waypoints:
            - {lat: 4, lon: 5, test: 6}
        """,
        """
          home: {lat: 1, test: 2, alt: 3}
          waypoints:
            - {lat: 4, test: 5, alt: 6}
        """,
        """
          home: {test: 1, lon: 2, alt: 3}
          waypoints:
            - {test: 4, lon: 5, alt: 6}
        """,
        """
          - non-mapping top level
          home: {lat: 1, lon: 2, alt: 3}
          waypoints:
            - {lat: 4, lon: 5, alt: 6}
        """,
        """
        home: {lat: 1, lon: 2, alt: 3}
        waypoints:
          - {lat: 'a', lon: 'b', alt: 'c'}
        """,
        """
        home {lat: 1, lon: 2, alt: 3}
        waypoints
          : {lat: 4, lon: 5, alt: 6}
        """,
        """
        home: {lat: 91, lon: -1, alt: 3}
        waypoints:
          - {lat: -91, lon: 1, alt: 6}
        """,
        """
        home: {lat: 1, lon: -181, alt: 3}
        waypoints:
          - {lat: -1, lon: 181, alt: 6}
        """
    ],
    ids=["faulty-waypoint-format (alt)", "faulty-waypoint-format (lon)", "faulty-waypoint-format (lat)", 
         "top-level-mapping", "non-number values", "faulty YAML", 
         "out of range values (lat)", "out of range values (long)"
        ],
)
def test_parsing_bad_data(tmp_path, text):
    path = write_to_tmp_waypoints_file(tmp_path, text)

    with pytest.raises(ValueError):
        parse_waypoints_file(path)


def test_faulty_file_location():
    with pytest.raises(OSError):
        parse_waypoints_file("/some_faulty_path.txt")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            """""",
            (None, [])
        ), (
            """
          home: {lat: 1, lon: 2, alt: 3}
          waypoints:
          """,
            (Coordinate(1, 2, 3), [])
        )
    ],
    ids=["empty file", "empty waypoints"]
)
def test_empty_file(tmp_path, text, expected):
    path = write_to_tmp_waypoints_file(tmp_path, text)

    assert parse_waypoints_file(path) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            [[Coordinate(1, 1, 1)]],
            [Coordinate(1, 1, 1)],
        ),
        (
            [[]],
            [],
        ),
        (
            [[Coordinate(10, 10, 1), Coordinate(15, 10, 1), Coordinate(15, 20, 1)]],
            [Coordinate(15, 20, 1), Coordinate(10, 10, 1), Coordinate(15, 10, 1) ],
        ),
        (
            [[Coordinate(10, 10, 1), Coordinate(15, 10, 1), Coordinate(15, 20, 1)], Coordinate(12, 8, 1)],
            [Coordinate(15, 10, 1), Coordinate(15, 20, 1), Coordinate(10, 10, 1)],
        ),
        (
            [[Coordinate(10, 12, 1), Coordinate(17, 10, 1), Coordinate(15, 20, 1)], Coordinate(14, 14, 1)],
            [Coordinate(15, 20, 1), Coordinate(10, 12, 1), Coordinate(17, 10, 1)],
        ),
        (
            [[Coordinate(0, 0, 1), Coordinate(6, 0, 1), Coordinate(9, 0, 0)]],
            [ Coordinate(6, 0, 1), Coordinate(9, 0, 0), Coordinate(0, 0, 1)],
        )
    ],
    ids=[
        'not enough waypoints (1 waypoint)', 'not enough waypoints (0 waypoints)', 
        'check sweep dir without home', 'check sweep dir with home', 
        'check sweep dir with home on centroid', 'check overlapping waypoints'
      ]
)
def test_sort_clockwise_sweep(text, expected):
    assert sort_clockwise_sweep(*text) == expected

@pytest.mark.parametrize(
      ("text", "expected"),
      [
          (
              (23.4, 43.12, -39.34, 39.39),
              (-410751.42, -6976379.33)
          ),
          (
              (0, 0, 134.68, 7),
              (299874.59, 14975753.40)
          ),
          (
              (0, 0, 1, 1),
              (111190.84, 111195.08)
          )
      ],
      ids=[
          "east_north_coords_offset calc 1", "east_north_coords_offset calc 2", "east_north_coords_offset calc 3"
      ]
)
def test_east_north_coords_offset(text, expected):
    east, north = east_north_coordinate_offset_m(*text)
    east_expected, north_expected = expected

    assert east == pytest.approx(east_expected, abs=2)
    assert north == pytest.approx(north_expected, abs=2)