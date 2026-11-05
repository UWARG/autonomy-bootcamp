"""
Tests for waypoint parsing, coordinate offsets, and clockwise ordering.

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
import yaml

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


@pytest.mark.parametrize(
    "text", ["", "# empty file\n", "{}", "waypoints: []", "home: null"]
)
def test_empty_input_has_no_waypoints(tmp_path, text):
    path = write_to_tmp_waypoints_file(tmp_path, text)
    assert parse_waypoints_file(path) == (None, [])


@pytest.mark.parametrize("text", ["[1, 2]", "42", "hello"])
def test_top_level_must_be_mapping(tmp_path, text):
    path = write_to_tmp_waypoints_file(tmp_path, text)
    with pytest.raises(ValueError, match="expected a mapping"):
        parse_waypoints_file(path)


def test_invalid_yaml_is_rejected(tmp_path):
    path = write_to_tmp_waypoints_file(tmp_path, "waypoints: [")
    with pytest.raises(ValueError, match="invalid YAML"):
        parse_waypoints_file(path)


def test_missing_file_raises_oserror(tmp_path):
    with pytest.raises(OSError):
        parse_waypoints_file(tmp_path / "missing.yaml")


@pytest.mark.parametrize("value", ["wrong", {"lat": 1}, 7])
def test_waypoints_must_be_list(tmp_path, value):
    path = write_to_tmp_waypoints_file(tmp_path, yaml.safe_dump({"waypoints": value}))
    with pytest.raises(ValueError, match="must be a list"):
        parse_waypoints_file(path)


@pytest.mark.parametrize("location", ["home", "waypoints"])
@pytest.mark.parametrize("entry", ["wrong", [1, 2, 3], 7])
def test_coordinate_must_be_mapping(tmp_path, location, entry):
    value = entry if location == "home" else [entry]
    path = write_to_tmp_waypoints_file(tmp_path, yaml.safe_dump({location: value}))
    with pytest.raises(ValueError, match="must be a mapping"):
        parse_waypoints_file(path)


@pytest.mark.parametrize("location", ["home", "waypoints"])
@pytest.mark.parametrize("key", ["lat", "lon", "alt"])
def test_coordinate_requires_all_keys(tmp_path, location, key):
    entry = {"lat": 1, "lon": 2, "alt": 3}
    del entry[key]
    value = entry if location == "home" else [entry]
    path = write_to_tmp_waypoints_file(tmp_path, yaml.safe_dump({location: value}))
    with pytest.raises(ValueError, match="missing key"):
        parse_waypoints_file(path)


@pytest.mark.parametrize("location", ["home", "waypoints"])
@pytest.mark.parametrize("key", ["lat", "lon", "alt"])
@pytest.mark.parametrize("bad_value", ["wrong", None, []])
def test_coordinate_values_must_be_numeric(tmp_path, location, key, bad_value):
    entry = {"lat": 1, "lon": 2, "alt": 3}
    entry[key] = bad_value
    value = entry if location == "home" else [entry]
    path = write_to_tmp_waypoints_file(tmp_path, yaml.safe_dump({location: value}))
    with pytest.raises(ValueError, match="non-numeric"):
        parse_waypoints_file(path)


@pytest.mark.parametrize("location", ["home", "waypoints"])
@pytest.mark.parametrize(
    ("lat", "lon"), [(90.01, 0), (-90.01, 0), (0, 180.01), (0, -180.01)]
)
def test_out_of_range_coordinates_are_rejected(tmp_path, location, lat, lon):
    entry = {"lat": lat, "lon": lon, "alt": 3}
    value = entry if location == "home" else [entry]
    path = write_to_tmp_waypoints_file(tmp_path, yaml.safe_dump({location: value}))
    with pytest.raises(ValueError, match="out of range"):
        parse_waypoints_file(path)


@pytest.mark.parametrize(("lat", "lon"), [(90, 180), (-90, -180)])
def test_coordinate_bounds_are_inclusive(tmp_path, lat, lon):
    path = write_to_tmp_waypoints_file(
        tmp_path, yaml.safe_dump({"waypoints": [{"lat": lat, "lon": lon, "alt": -5}]})
    )
    assert parse_waypoints_file(path) == (None, [Coordinate(lat, lon, -5)])


def test_numeric_strings_are_converted(tmp_path):
    path = write_to_tmp_waypoints_file(
        tmp_path, 'waypoints: [{lat: "1.5", lon: "-2.5", alt: "3.5"}]'
    )
    assert parse_waypoints_file(path) == (None, [Coordinate(1.5, -2.5, 3.5)])


def test_parsed_coordinates_are_frozen(tmp_path):
    path = write_to_tmp_waypoints_file(
        tmp_path,
        "home: {lat: 1, lon: 2, alt: 3}\nwaypoints: [{lat: 4, lon: 5, alt: 6}]",
    )
    home, waypoints = parse_waypoints_file(path)
    for coordinate in [home, *waypoints]:
        with pytest.raises(FrozenInstanceError):
            coordinate.lat = 0


@pytest.mark.parametrize(
    ("start", "end", "expected"),
    [
        ((0, 0), (0, 0), (0, 0)),
        ((0, 0), (0, 1), (111195.08, 0)),
        ((0, 0), (1, 0), (0, 111195.08)),
        ((0, 0), (-1, -1), (-111190.846, -111195.08)),
        ((60, 10), (60, 11), (55597.54, 0)),
        ((30, 10), (50, 12), (170360.746, 2223901.605)),
    ],
)
def test_offsets_match_independently_calculated_meters(start, end, expected):
    # Reference values use a 6,371,008.8 m sphere and degrees converted to
    # radians; the east scale is cosine of the mean latitude.
    assert east_north_coordinate_offset_m(*start, *end) == pytest.approx(
        expected, abs=0.01
    )


@pytest.mark.parametrize("waypoints", [[], [Coordinate(1, 2, 3)]])
def test_short_sweeps_return_a_separate_list(waypoints):
    result = sort_clockwise_sweep(waypoints)
    assert result == waypoints
    assert result is not waypoints


@pytest.fixture
def compass_points():
    # Symmetric points have centroid (0, 0), so compass order is unambiguous.
    return [
        Coordinate(1, 0, 10),
        Coordinate(0, 1, 20),
        Coordinate(-1, 0, 30),
        Coordinate(0, -1, 40),
    ]


def test_sweep_starts_north_and_runs_clockwise_without_mutating_input(compass_points):
    north, east, south, west = compass_points
    waypoints = [south, west, north, east]
    original = list(waypoints)
    assert sort_clockwise_sweep(waypoints) == [north, east, south, west]
    assert waypoints == original


@pytest.mark.parametrize(
    ("home", "indices"),
    [
        (Coordinate(0, 2, 0), [1, 2, 3, 0]),
        (Coordinate(-2, 0, 0), [2, 3, 0, 1]),
        (Coordinate(1, 1, 0), [1, 2, 3, 0]),
    ],
)
def test_home_sets_start_direction(compass_points, home, indices):
    assert sort_clockwise_sweep(list(reversed(compass_points)), home) == [
        compass_points[index] for index in indices
    ]


def test_home_at_centroid_falls_back_to_north(compass_points):
    assert (
        sort_clockwise_sweep(list(reversed(compass_points)), Coordinate(0, 0, 100))
        == compass_points
    )


def test_same_bearing_orders_near_before_far():
    near = Coordinate(1, 0, 10)
    far = Coordinate(2, 0, 20)
    south = Coordinate(-3, 0, 30)
    # These latitudes sum to zero. near and far have exactly the same bearing.
    assert sort_clockwise_sweep([far, south, near]) == [near, far, south]
