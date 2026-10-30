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


@pytest.mark.parametrize("text", ["", "{}", "waypoints: []", "waypoints: null"])
def test_empty_waypoints(tmp_path, text):
    path = write_to_tmp_waypoints_file(tmp_path, text)
    assert parse_waypoints_file(path) == (None, [])


@pytest.mark.parametrize(
    "text",
    [
        "[]",
        "hello",
        "waypoints: [",
        "waypoints: {lat: 1, lon: 2, alt: 3}",
    ],
)
def test_invalid_file_contents(tmp_path, text):
    path = write_to_tmp_waypoints_file(tmp_path, text)
    with pytest.raises(ValueError):
        parse_waypoints_file(path)


@pytest.mark.parametrize("section", ["home: {entry}", "waypoints: [{entry}]"])
@pytest.mark.parametrize(
    "entry",
    [
        "5",
        "{lon: 2, alt: 3}",
        "{lat: 1, alt: 3}",
        "{lat: 1, lon: 2}",
        "{lat: bad, lon: 2, alt: 3}",
        "{lat: 1, lon: bad, alt: 3}",
        "{lat: 1, lon: 2, alt: bad}",
        "{lat: null, lon: 2, alt: 3}",
        "{lat: 91, lon: 2, alt: 3}",
        "{lat: -91, lon: 2, alt: 3}",
        "{lat: 1, lon: 181, alt: 3}",
        "{lat: 1, lon: -181, alt: 3}",
    ],
)
def test_invalid_coordinate(tmp_path, section, entry):
    path = write_to_tmp_waypoints_file(tmp_path, section.format(entry=entry))
    with pytest.raises(ValueError):
        parse_waypoints_file(path)


def test_missing_file(tmp_path):
    with pytest.raises(OSError):
        parse_waypoints_file(tmp_path / "missing.yaml")


@pytest.mark.parametrize("lat, lon", [(90, 180), (-90, -180)])
def test_coordinate_boundaries(tmp_path, lat, lon):
    path = write_to_tmp_waypoints_file(
        tmp_path, f"waypoints: [{{lat: {lat}, lon: {lon}, alt: 0}}]"
    )
    assert parse_waypoints_file(path) == (None, [Coordinate(lat, lon, 0)])


def test_numeric_strings(tmp_path):
    path = write_to_tmp_waypoints_file(
        tmp_path, 'waypoints: [{lat: "1.5", lon: "-2.5", alt: "10"}]'
    )
    assert parse_waypoints_file(path) == (None, [Coordinate(1.5, -2.5, 10)])


@pytest.mark.parametrize(
    "coordinates, expected",
    [
        ((0, 0, 0, 0), (0, 0)),
        ((0, 0, 0, 1), (111195.0802, 0)),
        ((0, 0, 1, 0), (0, 111195.0802)),
        ((0, 0, 0, -1), (-111195.0802, 0)),
        ((0, 0, -1, 0), (0, -111195.0802)),
        ((59, 10, 61, 11), (55597.5401, 222390.1605)),
    ],
)
def test_coordinate_offsets(coordinates, expected):
    assert east_north_coordinate_offset_m(*coordinates) == pytest.approx(expected, abs=0.01)


@pytest.mark.parametrize("waypoints", [[], [Coordinate(1, 2, 3)]])
def test_short_sweep(waypoints):
    result = sort_clockwise_sweep(waypoints)
    assert result == waypoints
    assert result is not waypoints


@pytest.mark.parametrize(
    "home, expected",
    [
        (None, [0, 1, 2, 3]),
        (Coordinate(0, 2, 0), [1, 2, 3, 0]),
        (Coordinate(0, -2, 0), [3, 0, 1, 2]),
        (Coordinate(0, 0, 0), [0, 1, 2, 3]),
    ],
)
def test_clockwise_order(home, expected):
    north = Coordinate(1, 0, 10)
    east = Coordinate(0, 1, 10)
    south = Coordinate(-1, 0, 10)
    west = Coordinate(0, -1, 10)
    points = [north, east, south, west]
    waypoints = [south, north, west, east]
    original = list(waypoints)
    assert sort_clockwise_sweep(waypoints, home) == [points[i] for i in expected]
    assert waypoints == original


def test_same_bearing_nearest_first():
    near = Coordinate(1, 0, 10)
    far = Coordinate(2, 0, 10)
    south = Coordinate(-3, 0, 10)
    assert sort_clockwise_sweep([far, south, near]) == [near, far, south]


def test_parsed_coordinates_are_frozen(tmp_path):
    path = write_to_tmp_waypoints_file(
        tmp_path, "home: {lat: 1, lon: 2, alt: 3}\nwaypoints: [{lat: 4, lon: 5, alt: 6}]"
    )
    home, waypoints = parse_waypoints_file(path)
    for coordinate in [home, *waypoints]:
        with pytest.raises(FrozenInstanceError):
            coordinate.lat = 0
