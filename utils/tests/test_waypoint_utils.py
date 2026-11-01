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


def test_empty_file(tmp_path):
    path = write_to_tmp_waypoints_file(tmp_path, "")
    result = parse_waypoints_file(path)
    assert result == (None, [])

def test_missing_lat(tmp_path):
    path = write_to_tmp_waypoints_file(
        tmp_path,
        """
        waypoints:
          - {lon: 5, alt: 6}
        """,
    )
    with pytest.raises(ValueError):
        parse_waypoints_file(path)


def test_missing_lon(tmp_path):
    path = write_to_tmp_waypoints_file(
        tmp_path,
        """
        waypoints:
          - {lat: 5, alt: 6}
        """,
    )
    with pytest.raises(ValueError):
        parse_waypoints_file(path)


def test_missing_alt(tmp_path):
    path = write_to_tmp_waypoints_file(
        tmp_path,
        """
        waypoints:
          - {lat: 5, lon: 6}
        """,
    )
    with pytest.raises(ValueError):
        parse_waypoints_file(path)

    
@pytest.mark.parametrize("lat, lon, alt", [
    ("a", "5", "6"),
    ("5", "a", "6"),
    ("5", "6", "a"),
])
def test_value_not_number(tmp_path, lat, lon, alt):
    path = write_to_tmp_waypoints_file(
        tmp_path,
        f"""
        waypoints:
          - {{lat: {lat}, lon: {lon}, alt: {alt}}}
        """,
    )

    with pytest.raises(ValueError):
        parse_waypoints_file(path)

def test_invalid_yaml(tmp_path):
    path = write_to_tmp_waypoints_file(
        tmp_path,
        """
        waypoints:
          - lat: [1, 2
        """,
    )

    with pytest.raises(ValueError):
        parse_waypoints_file(path)


def test_file_not_found(tmp_path):
    path = tmp_path / "missing.yaml"

    with pytest.raises(OSError):
        parse_waypoints_file(path)


@pytest.mark.parametrize(
    "lat, lon",
    [
        (91, 0),
        (-91, 0),
        (0, 181),
        (0, -181),
    ],
)
def test_out_of_range_coordinates(tmp_path, lat, lon):
    path = write_to_tmp_waypoints_file(
        tmp_path,
        f"""
        waypoints:
          - {{lat: {lat}, lon: {lon}, alt: 6}}
        """,
    )

    with pytest.raises(ValueError):
        parse_waypoints_file(path)

def test_empty_waypoints(tmp_path):
    path = write_to_tmp_waypoints_file(
        tmp_path,
        "waypoints: []",
    )
    assert parse_waypoints_file(path) == (None, [])

def test_invalid_home(tmp_path):
    path = write_to_tmp_waypoints_file(
        tmp_path,
        """
        home: {lat: 1, lon: 2}
        waypoints: []
        """,
    )

    with pytest.raises(ValueError):
        parse_waypoints_file(path)

def test_east_offset_scales_with_latitude():
    east, north = east_north_coordinate_offset_m(60, 0, 60, 0.001)

    assert east == pytest.approx(55.6, abs=0.2)
    assert north == pytest.approx(0, abs=0.2)


@pytest.mark.parametrize(
    "waypoints",
    [
        [],
        [Coordinate(1, 2, 3)],
    ],
)
def test_sort_clockwise_sweep_zero_or_one_waypoint(waypoints):
    assert sort_clockwise_sweep(waypoints) == waypoints



def test_waypoints_not_a_list(tmp_path):
    path = write_to_tmp_waypoints_file(
        tmp_path,
        """
        waypoints: not-a-list
        """,
    )

    with pytest.raises(ValueError):
        parse_waypoints_file(path)

def test_east_north_coordinate_offset_m():
    east, north = east_north_coordinate_offset_m(0, 0, 0.001, 0.001)

    assert east == pytest.approx(111.2, abs=0.2)
    assert north == pytest.approx(111.2, abs=0.2)

def test_sort_clockwise_sweep_starts_north_without_home():
    north = Coordinate(1, 0, 0)
    east = Coordinate(0, 1, 0)
    south = Coordinate(-1, 0, 0)
    west = Coordinate(0, -1, 0)

    result = sort_clockwise_sweep([west, south, east, north])

    assert result == [north, east, south, west]

def test_sort_clockwise_sweep_starts_in_home_direction():
    north = Coordinate(1, 0, 0)
    east = Coordinate(0, 1, 0)
    south = Coordinate(-1, 0, 0)
    west = Coordinate(0, -1, 0)
    home = Coordinate(0, 10, 0)  # East of the waypoints' centroid

    result = sort_clockwise_sweep([north, west, south, east], home)

    assert result == [east, south, west, north]



def test_sort_clockwise_sweep_same_direction_closer_first():
    near_north = Coordinate(1, 0, 0)
    far_north = Coordinate(2, 0, 0)
    east = Coordinate(0, 1, 0)
    south = Coordinate(-1, 0, 0)
    west = Coordinate(0, -1, 0)

    result = sort_clockwise_sweep(
        [far_north, west, south, east, near_north]
    )

    assert result == [near_north, far_north, east, south, west]

"""
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
"""