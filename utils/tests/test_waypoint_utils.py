"""
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


def test_placeholder():
    assert callable(east_north_coordinate_offset_m)
    assert callable(parse_waypoints_file)
    assert callable(sort_clockwise_sweep)
    assert sort_clockwise_sweep([Coordinate(6,8,0), Coordinate(3,4,0), Coordinate(2,5,0)], Coordinate(5,5,0)) == [Coordinate(6,8,0), Coordinate(2,5,0), Coordinate(3,4,0)]
    assert sort_clockwise_sweep([Coordinate(0,1,0), Coordinate(1,0,0), Coordinate(0,-1,0), Coordinate(-1,0,0)], Coordinate(0,10,0)) == [Coordinate(0,1,0), Coordinate(-1,0,0), Coordinate(0,-1,0), Coordinate(1,0,0)]
    assert sort_clockwise_sweep([Coordinate(0,4,0), Coordinate(0,2,0), Coordinate(0,1,0)], None) == [Coordinate(0,4,0), Coordinate(0,2,0), Coordinate(0,1,0)]
    assert east_north_coordinate_offset_m(2, 2, 100, 100) == (pytest.approx(6857778.474787729, abs=1e-3), pytest.approx(10897117.862886224, abs=1e-3))
    assert sort_clockwise_sweep([], Coordinate(5,5,0)) == []


def test_invalid_coordinate_type(tmp_path):
    path = write_to_tmp_waypoints_file(
        tmp_path,
        """
        home: {lat: Shrimp Trawler, lon: 2, alt: 3}
        waypoints:
              - {lat: 4, lon: 5, alt: 6}
        """
    )

    with pytest.raises(ValueError):
        parse_waypoints_file(path)


def test_invalid_coordinate_range(tmp_path):
    path = write_to_tmp_waypoints_file(
        tmp_path,
        """
        home: {lat: 99, lon: 99, alt: 3}
        waypoints:
              - {lat: 4, lon: 5, alt: 6}
        """
    )

    with pytest.raises(ValueError):
        parse_waypoints_file(path)


def test_missing_alt(tmp_path):
    path = write_to_tmp_waypoints_file(
        tmp_path,
        """
        home: {lat: 99, lon: 99}
        waypoints:
              - {lat: 4, lon: 5}
        """
    )

    with pytest.raises(ValueError):
        parse_waypoints_file(path)


def test_invalid_return(tmp_path):
    path = write_to_tmp_waypoints_file(
        tmp_path,
        """
        """
    )

    assert parse_waypoints_file(path) == (None, [])

def test_parse_waypoints_file_and_sweep_edge_cases(tmp_path):
    bad_yaml = write_to_tmp_waypoints_file(
        tmp_path,
        "home: [1, 2\nwaypoints: []\n",
    )
    with pytest.raises(ValueError):
        parse_waypoints_file(bad_yaml)

    bad_root = write_to_tmp_waypoints_file(
        tmp_path,
        """
        - lat: 1
          lon: 2
          alt: 3
        """,
    )
    with pytest.raises(ValueError):
        parse_waypoints_file(bad_root)

    bad_waypoints = write_to_tmp_waypoints_file(
        tmp_path,
        """
        home: {lat: 1, lon: 2, alt: 3}
        waypoints: {lat: 4, lon: 5, alt: 6}
        """,
    )
    with pytest.raises(ValueError):
        parse_waypoints_file(bad_waypoints)

    missing = tmp_path / "missing.yaml"
    with pytest.raises(FileNotFoundError):
        parse_waypoints_file(missing)