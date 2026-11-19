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
from pathlib import Path

import pytest

from src.types import Coordinate
from src.waypoint_utils import (
    east_north_coordinate_offset_m,
    parse_waypoints_file,
    sort_clockwise_sweep,
)


def write_to_tmp_waypoints_file(tmp_path, text):
    """Write ``text`` to a YAML file and hand back its path.

    ``tmp_path`` is a pytest fixture: a fresh empty directory per test.
    """
    path = tmp_path / "waypoints.yaml"
    path.write_text(text)
    return path


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


def test_parse_waypoints_file_nonexistent_file():
    path = Path("nonexistent_file.yaml")
    with pytest.raises(OSError):
        parse_waypoints_file(path)


def test_parse_waypoints_file_invalid_yaml(tmp_path):
    path = write_to_tmp_waypoints_file(tmp_path, "key: [something")
    with pytest.raises(ValueError, match="invalid YAML"):
        parse_waypoints_file(path)


@pytest.mark.parametrize(
    "text",
    [
        "- {lat: 1, lon: 2, alt: 3}",
        "hello world",
        "39",
    ],
    ids=["top-level-list", "top-level-string", "top-level-int"],
)
def test_parse_waypoints_file_not_mapping(tmp_path, text):
    path = write_to_tmp_waypoints_file(tmp_path, text)
    with pytest.raises(ValueError, match="expected a mapping"):
        parse_waypoints_file(path)


def test_parse_waypoints_file_empty(tmp_path):
    path = write_to_tmp_waypoints_file(tmp_path, "")
    assert parse_waypoints_file(path) == (None, [])


@pytest.mark.parametrize(
    ("text", "match_msg"),
    [
        ("home: 'not-a-mapping'", "home must be a mapping"),
        ("waypoints: ['not-a-mapping']", "waypoint 1 must be a mapping"),
        ("home: {lon: 2, alt: 3}", "home is missing key\\(s\\) lat"),
        ("home: {lat: 1, lon: 2}", "home is missing key\\(s\\) alt"),
        ("waypoints: [{lat: 1, alt: 3}]", "waypoint 1 is missing key\\(s\\) lon"),
        ("waypoints: [{lat: 1, lon: 2}]", "waypoint 1 is missing key\\(s\\) alt"),
        ("home: {lat: 'abc', lon: 2, alt: 3}", "home has a non-numeric value"),
        ("waypoints: [{lat: 1, lon: 'xyz', alt: 3}]", "waypoint 1 has a non-numeric value"),
        ("home: {lat: 91.0, lon: 0.0, alt: 0.0}", "home latitude/longitude out of range"),
        ("waypoints: [{lat: 0.0, lon: 181.0, alt: 0.0}]", "waypoint 1 latitude/longitude out of range"),
    ],
    ids=[
        "home-not-mapping",
        "waypoint-not-mapping",
        "home-missing-lat",
        "home-missing-alt",
        "waypoint-missing-lon",
        "waypoint-missing-alt",
        "home-non-numeric",
        "waypoint-non-numeric",
        "home-lat-out-of-range",
        "waypoint-lon-out-of-range",
    ],
)
def test_parse_waypoints_file_bad_entry(tmp_path, text, match_msg):
    path = write_to_tmp_waypoints_file(tmp_path, text)
    with pytest.raises(ValueError, match=match_msg):
        parse_waypoints_file(path)


def test_parse_waypoints_file_waypoints_not_list(tmp_path):
    path = write_to_tmp_waypoints_file(tmp_path, "waypoints: 'hellow world'")
    with pytest.raises(ValueError, match="'waypoints' must be a list"):
        parse_waypoints_file(path)


def test_coordinate_is_frozen(tmp_path):
    path = write_to_tmp_waypoints_file(tmp_path, "waypoints: [{lat: 1, lon: 2, alt: 3}]")
    _, waypoints = parse_waypoints_file(path)
    coordinate = waypoints[0]
    with pytest.raises((FrozenInstanceError, AttributeError)):
        coordinate.lat = 5


def test_east_north_coordinate_offset_m():
    # Offset from a point to itself should be (0.0, 0.0)
    east, north = east_north_coordinate_offset_m(43.0, -80.0, 43.0, -80.0)
    assert east == pytest.approx(0.0, abs=1e-9)
    assert north == pytest.approx(0.0, abs=1e-9)

    # Offset between known coordinates
    east, north = east_north_coordinate_offset_m(43.0, -80.0, 43.001, -79.999)
    assert east == pytest.approx(81.32, abs=1.0)
    assert north == pytest.approx(111.2, abs=1.0)


@pytest.mark.parametrize(
    "waypoints",
    [
        [],
        [Coordinate(43.0, -80.0, 10.0)],
    ],
    ids=["zero-waypoints", "one-waypoint"],
)
def test_sort_clockwise_sweep_edge_cases(waypoints):
    assert sort_clockwise_sweep(waypoints) == waypoints


def test_sort_clockwise_sweep_no_home():
    # Centroid at (0, 0)
    # North (0, 1), East (1, 0), South (0, -1), West (-1, 0) to make it simple
    wps = [
        Coordinate(0.0, -1.0, 0.0),
        Coordinate(1.0, 0.0, 0.0),
        Coordinate(0.0, 1.0, 0.0),
        Coordinate(-1.0, 0.0, 0.0),
    ]
    sorted_wps = sort_clockwise_sweep(wps)
    # Clockwise should starting from North > East > South > West
    expected = [
        Coordinate(1.0, 0.0, 0.0),
        Coordinate(0.0, 1.0, 0.0),
        Coordinate(-1.0, 0.0, 0.0),
        Coordinate(0.0, -1.0, 0.0),
    ]
    assert sorted_wps == expected


def test_sort_clockwise_sweep_with_home():
    wps = [
        Coordinate(0.0, -1.0, 0.0),
        Coordinate(1.0, 0.0, 0.0),
        Coordinate(0.0, 1.0, 0.0),
        Coordinate(-1.0, 0.0, 0.0),
    ]
    # Home located East (0, 1) should make sweep start pointing East
    home = Coordinate(0.0, 1.0, 0.0)
    sorted_wps = sort_clockwise_sweep(wps, home=home)
    expected = [
        Coordinate(0.0, 1.0, 0.0),
        Coordinate(-1.0, 0.0, 0.0),
        Coordinate(0.0, -1.0, 0.0),
        Coordinate(1.0, 0.0, 0.0),
    ]
    assert sorted_wps == expected


def test_sort_clockwise_sweep_home_on_centroid():
    wps = [
        Coordinate(1.0, 0.0, 0.0),
        Coordinate(-1.0, 0.0, 0.0),
    ]
    # Centroid (0, 0) so Home on centroid (0, 0) should fall back to North start
    home = Coordinate(0.0, 0.0, 0.0)
    sorted_wps = sort_clockwise_sweep(wps, home=home)
    assert sorted_wps == [Coordinate(1.0, 0.0, 0.0), Coordinate(-1.0, 0.0, 0.0)]


def test_sort_clockwise_sweep_same_direction_distance():
    # Waypoints in the same direction from centroid, different distances
    wps = [
        Coordinate(2.0, 0.0, 0.0),  # Far North
        Coordinate(1.0, 0.0, 0.0),  # Close North
        Coordinate(-3.0, 0.0, 0.0), # South point just to balance Centroid
    ]
    sorted_wps = sort_clockwise_sweep(wps)
    # Closer one should come before the further one in the same direction
    assert sorted_wps[0] == Coordinate(1.0, 0.0, 0.0)
    assert sorted_wps[1] == Coordinate(2.0, 0.0, 0.0)