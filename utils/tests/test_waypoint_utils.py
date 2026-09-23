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

def test_sweep_wraps_around_from_home_direction():
    north = Coordinate(1, 0, 10)
    east = Coordinate(0, 1, 10)
    south = Coordinate(-1, 0, 10)
    west = Coordinate(0, -1, 10)
    home = Coordinate(0, 2, 10)

    result = sort_clockwise_sweep([north, east, south, west], home=home)
    assert result == [east, south, west, north]

def test_same_bearing_orders_closest_first():
    near_north = Coordinate(1, 0, 10)
    far_north = Coordinate(2, 0, 10)
    south = Coordinate(-3, 0, 10)

    result = sort_clockwise_sweep([far_north, south, near_north])

    assert result == [near_north, far_north, south]

@pytest.mark.parametrize("latitude", [90.1, -90.1])
def test_latitude_out_of_range(tmp_path, latitude):
    path = write_to_tmp_waypoints_file(
        tmp_path,
        f"""
        waypoints:
          - {{lat: {latitude}, lon: 0, alt: 10}}
        """,
    )

    with pytest.raises(ValueError, match="out of range"):
        parse_waypoints_file(path)

def test_returns_east_then_north():
    offset = east_north_coordinate_offset_m(from_lat=0, from_lon=0, to_lat=0, to_lon=1)
    assert offset == pytest.approx((111_195.08, 0.0), abs=0.1)


def test_start_bearing_points_from_centroid_to_home():
    home = Coordinate(0, 2, 10)      # east of centroid
    east_wp = Coordinate(0, 1, 10)
    west_wp = Coordinate(0, -1, 10)  # centroid = (0, 0)

    result = sort_clockwise_sweep([west_wp, east_wp], home=home)
    assert result == [east_wp, west_wp]

def test_north_offset_converts_degrees_to_radians():
    offset = east_north_coordinate_offset_m(
        from_lat=0,
        from_lon=0,
        to_lat=1,
        to_lon=0,
    )

    assert offset == pytest.approx(
        (0.0, 111_195.08),
        abs=0.1,
    )


def test_east_offset_scales_with_latitude():
    offset = east_north_coordinate_offset_m(
        from_lat=60,
        from_lon=0,
        to_lat=60,
        to_lon=1,
    )

    assert offset == pytest.approx(
        (55_597.54, 0.0),
        abs=0.1,
    )


@pytest.mark.parametrize(
    "entry",
    [
        "{lat: nope, lon: 0, alt: 10}",
        "{lat: 0, lon: nope, alt: 10}",
        "{lat: 0, lon: 0, alt: nope}",
    ],
    ids=["bad-latitude", "bad-longitude", "bad-altitude"],
)
def test_parse_rejects_non_numeric_values(tmp_path, entry):
    path = write_to_tmp_waypoints_file(
        tmp_path,
        f"""
        waypoints:
          - {entry}
        """,
    )

    with pytest.raises(ValueError, match="non-numeric"):
        parse_waypoints_file(path)


def test_empty_file_returns_no_home_and_no_waypoints(tmp_path):
    path = write_to_tmp_waypoints_file(tmp_path, "")

    assert parse_waypoints_file(path) == (None, [])


def test_empty_waypoints_list_returns_no_waypoints(tmp_path):
    path = write_to_tmp_waypoints_file(
        tmp_path,
        """
        waypoints: []
        """,
    )

    assert parse_waypoints_file(path) == (None, [])


@pytest.mark.parametrize(
    ("waypoints", "expected"),
    [
        ([], []),
        (
            [Coordinate(1, 2, 3)],
            [Coordinate(1, 2, 3)],
        ),
    ],
    ids=["empty", "single-waypoint"],
)
def test_sort_handles_zero_or_one_waypoint(waypoints, expected):
    assert sort_clockwise_sweep(waypoints) == expected


def test_sweep_without_home_starts_north_and_goes_clockwise():
    north = Coordinate(1, 0, 10)
    east = Coordinate(0, 1, 10)
    south = Coordinate(-1, 0, 10)
    west = Coordinate(0, -1, 10)

    result = sort_clockwise_sweep(
        [south, west, east, north],
    )

    assert result == [north, east, south, west]


def test_home_at_centroid_starts_sweep_from_north():
    north = Coordinate(1, 0, 10)
    east = Coordinate(0, 1, 10)
    south = Coordinate(-1, 0, 10)
    west = Coordinate(0, -1, 10)
    home = Coordinate(0, 0, 10)

    result = sort_clockwise_sweep(
        [south, west, east, north],
        home=home,
    )

    assert result == [north, east, south, west]


@pytest.mark.parametrize(
    "entry",
    [
        "{lon: 0, alt: 10}",
        "{lat: 0, alt: 10}",
        "{lat: 0, lon: 0}",
    ],
    ids=["missing-latitude", "missing-longitude", "missing-altitude"],
)
def test_parse_rejects_missing_coordinate_keys(tmp_path, entry):
    path = write_to_tmp_waypoints_file(
        tmp_path,
        f"""
        waypoints:
          - {entry}
        """,
    )

    with pytest.raises(ValueError):
        parse_waypoints_file(path)


@pytest.mark.parametrize(
    ("latitude", "longitude"),
    [
        (91, 0),
        (-91, 0),
        (0, 181),
        (0, -181),
    ],
    ids=[
        "latitude-too-high",
        "latitude-too-low",
        "longitude-too-high",
        "longitude-too-low",
    ],
)
def test_parse_rejects_coordinates_out_of_range(
    tmp_path,
    latitude,
    longitude,
):
    path = write_to_tmp_waypoints_file(
        tmp_path,
        f"""
        waypoints:
          - {{lat: {latitude}, lon: {longitude}, alt: 10}}
        """,
    )

    with pytest.raises(ValueError, match="out of range"):
        parse_waypoints_file(path)


def test_parse_rejects_non_mapping_top_level(tmp_path):
    path = write_to_tmp_waypoints_file(
        tmp_path,
        """
        - one
        - two
        """,
    )

    with pytest.raises(ValueError, match="expected a mapping"):
        parse_waypoints_file(path)


def test_parse_rejects_non_mapping_waypoint(tmp_path):
    path = write_to_tmp_waypoints_file(
        tmp_path,
        """
        waypoints:
          - 123
        """,
    )

    with pytest.raises(ValueError, match="must be a mapping"):
        parse_waypoints_file(path)


def test_parse_rejects_waypoints_that_are_not_a_list(tmp_path):
    path = write_to_tmp_waypoints_file(
        tmp_path,
        """
        waypoints:
          lat: 1
          lon: 2
          alt: 3
        """,
    )

    with pytest.raises(ValueError, match="must be a list"):
        parse_waypoints_file(path)


def test_parse_rejects_invalid_yaml(tmp_path):
    path = write_to_tmp_waypoints_file(
        tmp_path,
        """
        waypoints: [
        """,
    )

    with pytest.raises(ValueError, match="invalid YAML"):
        parse_waypoints_file(path)


def test_parse_missing_file_raises_os_error(tmp_path):
    missing_path = tmp_path / "missing.yaml"

    with pytest.raises(OSError):
        parse_waypoints_file(missing_path)


def test_parsed_coordinate_is_frozen(tmp_path):
    path = write_to_tmp_waypoints_file(
        tmp_path,
        """
        waypoints:
          - {lat: 1, lon: 2, alt: 3}
        """,
    )
    _, waypoints = parse_waypoints_file(path)

    with pytest.raises(AttributeError):
        waypoints[0].lat = 50