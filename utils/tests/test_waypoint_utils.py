"""
Tests for ``src/waypoint_utils.py``.

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
import math

import pytest

from src.constants import EARTH_RADIUS_M
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




# ---------------------------------------------------------------------------
# parse_waypoints_file: more files that parse
# ---------------------------------------------------------------------------


def test_parse_empty_file_gives_no_home_and_no_waypoints(tmp_path):
    path = write_to_tmp_waypoints_file(tmp_path, "")
    assert parse_waypoints_file(path) == (None, [])


def test_parse_empty_waypoints_list(tmp_path):
    path = write_to_tmp_waypoints_file(tmp_path, "waypoints: []\n")
    assert parse_waypoints_file(path) == (None, [])


def test_parse_waypoints_key_with_no_value(tmp_path):
    path = write_to_tmp_waypoints_file(tmp_path, "waypoints:\n")
    assert parse_waypoints_file(path) == (None, [])


def test_parse_home_only(tmp_path):
    path = write_to_tmp_waypoints_file(tmp_path, "home: {lat: 1, lon: 2, alt: 3}\n")
    assert parse_waypoints_file(path) == (Coordinate(1, 2, 3), [])


def test_parse_accepts_string_path(tmp_path):
    path = write_to_tmp_waypoints_file(tmp_path, "waypoints:\n  - {lat: 4, lon: 5, alt: 6}\n")
    assert parse_waypoints_file(str(path)) == (None, [Coordinate(4, 5, 6)])


def test_parse_converts_values_to_float(tmp_path):
    path = write_to_tmp_waypoints_file(
        tmp_path,
        """
        waypoints:
          - {lat: "4.5", lon: 5, alt: "-6"}
        """,
    )
    _, waypoints = parse_waypoints_file(path)
    assert waypoints == [Coordinate(4.5, 5.0, -6.0)]
    assert all(
        isinstance(value, float)
        for value in (waypoints[0].lat, waypoints[0].lon, waypoints[0].alt)
    )


def test_parse_keeps_waypoint_order_from_file(tmp_path):
    path = write_to_tmp_waypoints_file(
        tmp_path,
        """
        waypoints:
          - {lat: 3, lon: 0, alt: 1}
          - {lat: 1, lon: 0, alt: 2}
          - {lat: 2, lon: 0, alt: 3}
        """,
    )
    _, waypoints = parse_waypoints_file(path)
    assert waypoints == [Coordinate(3, 0, 1), Coordinate(1, 0, 2), Coordinate(2, 0, 3)]


@pytest.mark.parametrize(
    ("lat", "lon"),
    [(90, 180), (-90, -180), (90, -180), (-90, 180), (0, 0)],
    ids=["max-max", "min-min", "max-lat-min-lon", "min-lat-max-lon", "zero"],
)
def test_parse_accepts_coordinates_on_the_range_limits(tmp_path, lat, lon):
    path = write_to_tmp_waypoints_file(
        tmp_path, f"waypoints:\n  - {{lat: {lat}, lon: {lon}, alt: 0}}\n"
    )
    assert parse_waypoints_file(path) == (None, [Coordinate(lat, lon, 0)])


def test_parsed_coordinates_are_frozen(tmp_path):
    path = write_to_tmp_waypoints_file(
        tmp_path,
        """
        home: {lat: 1, lon: 2, alt: 3}
        waypoints:
          - {lat: 4, lon: 5, alt: 6}
        """,
    )
    home, waypoints = parse_waypoints_file(path)
    with pytest.raises(dataclasses.FrozenInstanceError):
        home.lat = 10
    with pytest.raises(dataclasses.FrozenInstanceError):
        waypoints[0].alt = 10


# ---------------------------------------------------------------------------
# parse_waypoints_file: bad files
# ---------------------------------------------------------------------------


def test_parse_missing_file_raises_oserror(tmp_path):
    with pytest.raises(OSError):
        parse_waypoints_file(tmp_path / "does_not_exist.yaml")


def test_parse_invalid_yaml_raises_value_error(tmp_path):
    path = write_to_tmp_waypoints_file(tmp_path, "waypoints: [unclosed\n")
    with pytest.raises(ValueError, match="invalid YAML"):
        parse_waypoints_file(path)


@pytest.mark.parametrize(
    "text",
    ["- {lat: 1, lon: 2, alt: 3}\n", "just a string\n", "42\n"],
    ids=["list", "string", "number"],
)
def test_parse_top_level_not_a_mapping_raises(tmp_path, text):
    path = write_to_tmp_waypoints_file(tmp_path, text)
    with pytest.raises(ValueError, match="expected a mapping"):
        parse_waypoints_file(path)


@pytest.mark.parametrize(
    "text",
    ["waypoints: {lat: 1, lon: 2, alt: 3}\n", "waypoints: hello\n", "waypoints: 5\n"],
    ids=["mapping", "string", "number"],
)
def test_parse_waypoints_not_a_list_raises(tmp_path, text):
    path = write_to_tmp_waypoints_file(tmp_path, text)
    with pytest.raises(ValueError, match="must be a list"):
        parse_waypoints_file(path)


@pytest.mark.parametrize(
    ("entry", "missing"),
    [
        ("{lon: 5, alt: 6}", "lat"),
        ("{lat: 4, alt: 6}", "lon"),
        ("{lat: 4, lon: 5}", "alt"),
        ("{alt: 6}", "lat, lon"),
        ("{}", "lat, lon, alt"),
    ],
    ids=["no-lat", "no-lon", "no-alt", "no-lat-lon", "empty-mapping"],
)
def test_parse_waypoint_missing_keys_raises(tmp_path, entry, missing):
    path = write_to_tmp_waypoints_file(tmp_path, f"waypoints:\n  - {entry}\n")
    with pytest.raises(ValueError, match=f"missing key\\(s\\) {missing}"):
        parse_waypoints_file(path)


def test_parse_home_missing_key_raises(tmp_path):
    path = write_to_tmp_waypoints_file(tmp_path, "home: {lat: 1, lon: 2}\nwaypoints: []\n")
    with pytest.raises(ValueError, match="home is missing key"):
        parse_waypoints_file(path)


@pytest.mark.parametrize(
    "entry",
    ["5", "hello", "[1, 2, 3]"],
    ids=["number", "string", "list"],
)
def test_parse_waypoint_not_a_mapping_raises(tmp_path, entry):
    path = write_to_tmp_waypoints_file(tmp_path, f"waypoints:\n  - {entry}\n")
    with pytest.raises(ValueError, match="must be a mapping"):
        parse_waypoints_file(path)


def test_parse_home_not_a_mapping_raises(tmp_path):
    path = write_to_tmp_waypoints_file(tmp_path, "home: [1, 2, 3]\n")
    with pytest.raises(ValueError, match="home must be a mapping"):
        parse_waypoints_file(path)


@pytest.mark.parametrize(
    "entry",
    [
        "{lat: abc, lon: 5, alt: 6}",
        "{lat: 4, lon: abc, alt: 6}",
        "{lat: 4, lon: 5, alt: abc}",
        "{lat: null, lon: 5, alt: 6}",
        "{lat: [1, 2], lon: 5, alt: 6}",
        "{lat: 4, lon: {x: 1}, alt: 6}",
    ],
    ids=["lat-word", "lon-word", "alt-word", "lat-null", "lat-list", "lon-mapping"],
)
def test_parse_non_numeric_value_raises(tmp_path, entry):
    path = write_to_tmp_waypoints_file(tmp_path, f"waypoints:\n  - {entry}\n")
    with pytest.raises(ValueError, match="non-numeric"):
        parse_waypoints_file(path)


def test_parse_home_non_numeric_value_raises(tmp_path):
    path = write_to_tmp_waypoints_file(tmp_path, "home: {lat: x, lon: 2, alt: 3}\n")
    with pytest.raises(ValueError, match="home has a non-numeric"):
        parse_waypoints_file(path)


@pytest.mark.parametrize(
    ("lat", "lon"),
    [(90.0001, 0), (-90.0001, 0), (0, 180.0001), (0, -180.0001), (91, 181)],
    ids=["lat-high", "lat-low", "lon-high", "lon-low", "both"],
)
def test_parse_out_of_range_coordinates_raise(tmp_path, lat, lon):
    path = write_to_tmp_waypoints_file(
        tmp_path, f"waypoints:\n  - {{lat: {lat}, lon: {lon}, alt: 0}}\n"
    )
    with pytest.raises(ValueError, match="out of range"):
        parse_waypoints_file(path)


def test_parse_home_out_of_range_raises(tmp_path):
    path = write_to_tmp_waypoints_file(tmp_path, "home: {lat: 100, lon: 0, alt: 0}\n")
    with pytest.raises(ValueError, match="home latitude/longitude out of range"):
        parse_waypoints_file(path)


def test_parse_error_names_the_bad_waypoint(tmp_path):
    path = write_to_tmp_waypoints_file(
        tmp_path,
        """
        waypoints:
          - {lat: 1, lon: 1, alt: 1}
          - {lat: 1, lon: 1}
        """,
    )
    with pytest.raises(ValueError, match="waypoint 2 is missing"):
        parse_waypoints_file(path)


# ---------------------------------------------------------------------------
# east_north_coordinate_offset_m
# ---------------------------------------------------------------------------

# Meters in one degree along a great circle: R * pi / 180, about 111195 m.
METERS_PER_DEGREE = EARTH_RADIUS_M * math.pi / 180.0


def test_offset_same_point_is_zero():
    east, north = east_north_coordinate_offset_m(43.47, -80.54, 43.47, -80.54)
    assert east == pytest.approx(0.0, abs=1e-9)
    assert north == pytest.approx(0.0, abs=1e-9)


def test_offset_one_degree_north():
    east, north = east_north_coordinate_offset_m(0, 0, 1, 0)
    assert east == pytest.approx(0.0, abs=1e-6)
    assert north == pytest.approx(111195.08, abs=0.1)


def test_offset_one_degree_south_is_negative():
    east, north = east_north_coordinate_offset_m(0, 0, -1, 0)
    assert east == pytest.approx(0.0, abs=1e-6)
    assert north == pytest.approx(-111195.08, abs=0.1)


def test_offset_one_degree_east_at_equator():
    east, north = east_north_coordinate_offset_m(0, 0, 0, 1)
    assert east == pytest.approx(111195.08, abs=0.1)
    assert north == pytest.approx(0.0, abs=1e-6)


def test_offset_west_is_negative():
    east, north = east_north_coordinate_offset_m(0, 10, 0, 9)
    assert east == pytest.approx(-111195.08, abs=0.1)
    assert north == pytest.approx(0.0, abs=1e-6)


def test_offset_east_shrinks_with_latitude():
    # At 60 degrees, a degree of longitude is cos(60) = half as long.
    east, north = east_north_coordinate_offset_m(60, 0, 60, 1)
    assert east == pytest.approx(55597.54, abs=0.1)
    assert north == pytest.approx(0.0, abs=1e-6)


def test_offset_east_uses_mean_latitude():
    # Going from lat 0 to lat 60, the east offset is scaled by cos(30).
    east, north = east_north_coordinate_offset_m(0, 0, 60, 1)
    assert east == pytest.approx(METERS_PER_DEGREE * math.cos(math.radians(30)), abs=0.01)
    assert east == pytest.approx(96297.76, abs=0.1)
    assert north == pytest.approx(60 * METERS_PER_DEGREE, abs=0.01)


def test_offset_is_antisymmetric():
    east_ab, north_ab = east_north_coordinate_offset_m(43.47, -80.54, 43.48, -80.53)
    east_ba, north_ba = east_north_coordinate_offset_m(43.48, -80.53, 43.47, -80.54)
    assert east_ab == pytest.approx(-east_ba, abs=1e-9)
    assert north_ab == pytest.approx(-north_ba, abs=1e-9)
    assert east_ab > 0
    assert north_ab > 0


def test_offset_small_realistic_distance():
    # About 30 m north and 10 m west around the WARG field.
    east, north = east_north_coordinate_offset_m(43.471520, -80.541400, 43.471790, -80.541524)
    assert north == pytest.approx(30.02, abs=0.01)
    assert east == pytest.approx(-10.01, abs=0.01)


# ---------------------------------------------------------------------------
# sort_clockwise_sweep
# ---------------------------------------------------------------------------

# A square of waypoints around a centre that isn't (0, 0), so lat and lon
# can't be mixed up without a test noticing.
CENTER_LAT = 43.47
CENTER_LON = -80.54
D = 0.001

NORTH = Coordinate(CENTER_LAT + D, CENTER_LON, 10)
EAST = Coordinate(CENTER_LAT, CENTER_LON + D, 11)
SOUTH = Coordinate(CENTER_LAT - D, CENTER_LON, 12)
WEST = Coordinate(CENTER_LAT, CENTER_LON - D, 13)


def test_sort_empty_list():
    assert sort_clockwise_sweep([]) == []


def test_sort_single_waypoint():
    waypoints = [NORTH]
    result = sort_clockwise_sweep(waypoints)
    assert result == [NORTH]
    assert result is not waypoints


def test_sort_single_waypoint_with_home():
    assert sort_clockwise_sweep([EAST], home=WEST) == [EAST]


def test_sort_does_not_modify_input():
    waypoints = [WEST, SOUTH, EAST, NORTH]
    sort_clockwise_sweep(waypoints)
    assert waypoints == [WEST, SOUTH, EAST, NORTH]


@pytest.mark.parametrize(
    "waypoints",
    [
        [WEST, SOUTH, EAST, NORTH],
        [SOUTH, NORTH, WEST, EAST],
        [EAST, WEST, NORTH, SOUTH],
        [NORTH, EAST, SOUTH, WEST],
    ],
    ids=["reversed", "mixed-1", "mixed-2", "already-sorted"],
)
def test_sort_no_home_goes_clockwise_from_north(waypoints):
    assert sort_clockwise_sweep(waypoints) == [NORTH, EAST, SOUTH, WEST]


def test_sort_no_home_point_just_west_of_north_comes_last():
    # Bearing just under 360 degrees, so it's the very end of the sweep,
    # not the start.
    north_west = Coordinate(CENTER_LAT + D, CENTER_LON - D / 100, 0)
    north_east = Coordinate(CENTER_LAT + D, CENTER_LON + D / 100, 0)
    south = Coordinate(CENTER_LAT - 2 * D, CENTER_LON, 0)
    result = sort_clockwise_sweep([north_west, south, north_east])
    assert result == [north_east, south, north_west]


def test_sort_two_waypoints():
    assert sort_clockwise_sweep([SOUTH, NORTH]) == [NORTH, SOUTH]


@pytest.mark.parametrize(
    ("home", "expected"),
    [
        (Coordinate(CENTER_LAT + 5 * D, CENTER_LON, 0), [NORTH, EAST, SOUTH, WEST]),
        (Coordinate(CENTER_LAT, CENTER_LON + 5 * D, 0), [EAST, SOUTH, WEST, NORTH]),
        (Coordinate(CENTER_LAT - 5 * D, CENTER_LON, 0), [SOUTH, WEST, NORTH, EAST]),
        (Coordinate(CENTER_LAT, CENTER_LON - 5 * D, 0), [WEST, NORTH, EAST, SOUTH]),
    ],
    ids=["home-north", "home-east", "home-south", "home-west"],
)
def test_sort_with_home_starts_in_homes_direction(home, expected):
    assert sort_clockwise_sweep([WEST, SOUTH, EAST, NORTH], home=home) == expected


def test_sort_with_home_between_waypoints_starts_at_next_clockwise():
    # Home is to the south-east, so the sweep starts at SOUTH (the next one
    # clockwise from there), not EAST.
    home = Coordinate(CENTER_LAT - 5 * D, CENTER_LON + 5 * D, 0)
    assert sort_clockwise_sweep([NORTH, EAST, SOUTH, WEST], home=home) == [
        SOUTH,
        WEST,
        NORTH,
        EAST,
    ]


def test_sort_home_on_the_centroid_starts_at_north():
    home = Coordinate(CENTER_LAT, CENTER_LON, 0)
    assert sort_clockwise_sweep([WEST, SOUTH, EAST, NORTH], home=home) == [
        NORTH,
        EAST,
        SOUTH,
        WEST,
    ]


def test_sort_home_very_close_to_but_off_the_centroid_uses_home():
    # Home a few centimetres east of the centroid still counts as a direction.
    home = Coordinate(CENTER_LAT, CENTER_LON + 1e-6, 0)
    assert sort_clockwise_sweep([WEST, SOUTH, EAST, NORTH], home=home) == [
        EAST,
        SOUTH,
        WEST,
        NORTH,
    ]


def test_sort_same_direction_closer_comes_first():
    # Both north points sit due north of the centroid (lat 0 + offsets that
    # average to 0), so they tie on angle; the nearer one has to go first.
    near_north = Coordinate(1, 0, 0)
    far_north = Coordinate(2, 0, 0)
    south = Coordinate(-3, 0, 0)
    assert sort_clockwise_sweep([far_north, south, near_north]) == [
        near_north,
        far_north,
        south,
    ]


def test_sort_same_direction_closer_comes_first_with_home():
    near_east = Coordinate(CENTER_LAT, CENTER_LON + D, 0)
    far_east = Coordinate(CENTER_LAT, CENTER_LON + 2 * D, 0)
    west = Coordinate(CENTER_LAT, CENTER_LON - 3 * D, 0)
    home = Coordinate(CENTER_LAT, CENTER_LON + 10 * D, 0)
    assert sort_clockwise_sweep([west, far_east, near_east], home=home) == [
        near_east,
        far_east,
        west,
    ]


def test_sort_uses_centroid_not_first_waypoint():
    # The centroid of these is (CENTER_LAT, CENTER_LON + D). Measured from
    # there, NORTH-ish points split around it correctly only if the real
    # centroid is used.
    a = Coordinate(CENTER_LAT + D, CENTER_LON, 0)
    b = Coordinate(CENTER_LAT + D, CENTER_LON + 2 * D, 0)
    c = Coordinate(CENTER_LAT - 2 * D, CENTER_LON + D, 0)
    # From centroid (lat CENTER_LAT, lon CENTER_LON + D): b is north-east,
    # c is south, a is north-west.
    assert sort_clockwise_sweep([a, c, b]) == [b, c, a]


def test_sort_keeps_every_waypoint():
    waypoints = [WEST, SOUTH, EAST, NORTH, Coordinate(CENTER_LAT + D, CENTER_LON + D, 0)]
    result = sort_clockwise_sweep(waypoints)
    assert len(result) == len(waypoints)
    assert set(result) == set(waypoints)
