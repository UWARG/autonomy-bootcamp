"""Tests for waypoint_utils."""
 
import math
 
import pytest
 
from src.constants import EARTH_RADIUS_M
from src.types import Coordinate
from src.waypoint_utils import (
    east_north_coordinate_offset_m,
    parse_waypoints_file,
    sort_clockwise_sweep,
)
 
ONE_DEGREE_M = math.radians(1.0) * EARTH_RADIUS_M
 
NORTH = Coordinate(1.0, 0.0, 10.0)
EAST = Coordinate(0.0, 1.0, 10.0)
SOUTH = Coordinate(-1.0, 0.0, 10.0)
WEST = Coordinate(0.0, -1.0, 10.0)
 
 
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
 
 
@pytest.mark.parametrize(
    ("lat", "lon", "expected_east", "expected_north"),
    [
        (1.0, 0.0, 0.0, ONE_DEGREE_M),
        (-1.0, 0.0, 0.0, -ONE_DEGREE_M),
        (0.0, 1.0, ONE_DEGREE_M, 0.0),
        (0.0, -1.0, -ONE_DEGREE_M, 0.0),
    ],
    ids=["north", "south", "east", "west"],
)
def test_offset(lat, lon, expected_east, expected_north):
    east, north = east_north_coordinate_offset_m(0.0, 0.0, lat, lon)
    assert east == pytest.approx(expected_east, abs=1e-3)
    assert north == pytest.approx(expected_north, abs=1e-3)
 
 
def test_offset_east_at_latitude_60():
    east, _ = east_north_coordinate_offset_m(60.0, 0.0, 60.0, 1.0)
    assert east == pytest.approx(0.5 * ONE_DEGREE_M, abs=1e-3)
 
 
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("", (None, [])),
        ("# nothing here\n", (None, [])),
        ("waypoints: []\n", (None, [])),
        ("waypoints:\n", (None, [])),
        ("home: {lat: 1, lon: 2, alt: 3}\n", (Coordinate(1, 2, 3), [])),
    ],
    ids=["empty", "comment-only", "empty-list", "null-waypoints", "home-only"],
)
def test_parse_empty(tmp_path, text, expected):
    path = write_to_tmp_waypoints_file(tmp_path, text)
    assert parse_waypoints_file(path) == expected
 
 
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "waypoints: [{lat: 90, lon: 180, alt: 0}]\n",
            (None, [Coordinate(90, 180, 0)]),
        ),
        (
            "waypoints: [{lat: -90, lon: -180, alt: 0}]\n",
            (None, [Coordinate(-90, -180, 0)]),
        ),
    ],
    ids=["max", "min"],
)
def test_parse_limits(tmp_path, text, expected):
    path = write_to_tmp_waypoints_file(tmp_path, text)
    assert parse_waypoints_file(path) == expected
 
 
@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("waypoints: [\n", "invalid YAML"),
        ("[1, 2]\n", "expected a mapping"),
        ("waypoints: 5\n", "must be a list"),
        ("waypoints: [5]\n", "must be a mapping"),
        ("home: 5\n", "home must be a mapping"),
        ("waypoints: [{lon: 2, alt: 3}]\n", "missing key"),
        ("waypoints: [{lat: 1, alt: 3}]\n", "missing key"),
        ("waypoints: [{lat: 1, lon: 2}]\n", "missing key"),
        ("waypoints: [{lat: abc, lon: 2, alt: 3}]\n", "non-numeric"),
        ("waypoints: [{lat: [1], lon: 2, alt: 3}]\n", "non-numeric"),
        ("waypoints: [{lat: 91, lon: 0, alt: 0}]\n", "out of range"),
        ("waypoints: [{lat: -91, lon: 0, alt: 0}]\n", "out of range"),
        ("waypoints: [{lat: 0, lon: 181, alt: 0}]\n", "out of range"),
        ("waypoints: [{lat: 0, lon: -181, alt: 0}]\n", "out of range"),
    ],
    ids=[
        "bad-yaml",
        "not-a-mapping",
        "waypoints-not-list",
        "waypoint-not-mapping",
        "bad-home",
        "no-lat",
        "no-lon",
        "no-alt",
        "text-value",
        "list-value",
        "lat-high",
        "lat-low",
        "lon-high",
        "lon-low",
    ],
)
def test_parse_bad_data(tmp_path, text, message):
    path = write_to_tmp_waypoints_file(tmp_path, text)
    with pytest.raises(ValueError, match=message):
        parse_waypoints_file(path)
 
 
def test_parse_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        parse_waypoints_file(tmp_path / "missing.yaml")
 
 
def test_sort_empty():
    assert sort_clockwise_sweep([]) == []
 
 
def test_sort_one():
    assert sort_clockwise_sweep([NORTH]) == [NORTH]
 
 
def test_sort_clockwise_from_north():
    result = sort_clockwise_sweep([SOUTH, WEST, EAST, NORTH])
    assert result == [NORTH, EAST, SOUTH, WEST]
 
 
def test_sort_starts_at_home():
    home = Coordinate(0.0, 2.0, 10.0)
    result = sort_clockwise_sweep([SOUTH, WEST, EAST, NORTH], home)
    assert result == [EAST, SOUTH, WEST, NORTH]
 
 
def test_sort_home_on_centroid():
    home = Coordinate(0.0, 0.0, 10.0)
    result = sort_clockwise_sweep([SOUTH, WEST, EAST, NORTH], home)
    assert result == [NORTH, EAST, SOUTH, WEST]
 
 
def test_sort_closer_first():
    near = Coordinate(1.0, 0.0, 10.0)
    far = Coordinate(2.0, 0.0, 10.0)
    south = Coordinate(-3.0, 0.0, 10.0)
    assert sort_clockwise_sweep([far, south, near]) == [near, far, south]
 
 
def test_coordinate_is_frozen(tmp_path):
    text = "waypoints: [{lat: 1, lon: 2, alt: 3}]\n"
    path = write_to_tmp_waypoints_file(tmp_path, text)
    _, waypoints = parse_waypoints_file(path)
    with pytest.raises(AttributeError):
        waypoints[0].lat = 5.0