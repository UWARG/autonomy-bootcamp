import pytest

from src.types import Coordinate
from src.waypoint_utils import (
    east_north_coordinate_offset_m,
    parse_waypoints_file,
    sort_clockwise_sweep,
)


def write_to_tmp_waypoints_file(tmp_path, text):
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
    "bad_yaml",
    [
        "- item1\n- item2",
        "home: 'not a dict'\nwaypoints: []",
        "waypoints: 'not a list'",
        "waypoints:\n  - {lon: 5, alt: 6}",
        "waypoints:\n  - {lat: 4, alt: 6}",
        "waypoints:\n  - {lat: 4, lon: 5}",
        "waypoints:\n  - {lat: 'invalid', lon: 5, alt: 6}",
        "waypoints: [ {lat: 4, lon: 5, alt: 6",
    ],
    ids=[
        "top-not-mapping",
        "home-not-mapping",
        "waypoints-not-list",
        "missing-lat",
        "missing-lon",
        "missing-alt",
        "non-numeric",
        "yaml-syntax-error",
    ],
)
def test_parse_waypoints_file_bad_data(tmp_path, bad_yaml):
    path = write_to_tmp_waypoints_file(tmp_path, bad_yaml)
    with pytest.raises(ValueError):
        parse_waypoints_file(path)


@pytest.mark.parametrize(
    "bad_coords_yaml",
    [
        "waypoints:\n  - {lat: 90.1, lon: 0, alt: 0}",
        "waypoints:\n  - {lat: -90.1, lon: 0, alt: 0}",
        "waypoints:\n  - {lat: 0, lon: 180.1, alt: 0}",
        "waypoints:\n  - {lat: 0, lon: -180.1, alt: 0}",
    ],
    ids=["lat-high", "lat-low", "lon-high", "lon-low"],
)
def test_out_of_range_coordinates(tmp_path, bad_coords_yaml):
    path = write_to_tmp_waypoints_file(tmp_path, bad_coords_yaml)
    with pytest.raises(ValueError):
        parse_waypoints_file(path)


def test_parse_waypoints_file_not_found():
    with pytest.raises(OSError):
        parse_waypoints_file("non_existent_file.yaml")


def test_empty_file_and_empty_waypoints(tmp_path):
    path_empty = write_to_tmp_waypoints_file(tmp_path, "")
    assert parse_waypoints_file(path_empty) == (None, [])

    path_list = write_to_tmp_waypoints_file(tmp_path, "waypoints: []")
    assert parse_waypoints_file(path_list) == (None, [])


def test_coordinate_immutable():
    coord = Coordinate(1.0, 2.0, 3.0)
    with pytest.raises(AttributeError):
        coord.lat = 10.0


def test_east_north_coordinate_offset_m():
    east, north = east_north_coordinate_offset_m(0.0, 0.0, 0.001, 0.001)
    assert east > 100
    assert north > 100


def test_sort_clockwise_sweep_short_lists():
    assert sort_clockwise_sweep([]) == []
    single = [Coordinate(1, 2, 3)]
    assert sort_clockwise_sweep(single) == single


def test_sort_clockwise_sweep_ordering_no_home():
    north = Coordinate(1, 0, 0)
    east = Coordinate(0, 1, 0)
    south = Coordinate(-1, 0, 0)
    west = Coordinate(0, -1, 0)

    waypoints = [west, south, east, north]
    sorted_wp = sort_clockwise_sweep(waypoints)

    assert sorted_wp == [north, east, south, west]


def test_sort_clockwise_sweep_with_home():
    ne = Coordinate(1, 1, 0)
    se = Coordinate(-1, 1, 0)
    sw = Coordinate(-1, -1, 0)
    nw = Coordinate(1, -1, 0)
    home = Coordinate(0, 5, 0)
    sorted_wp = sort_clockwise_sweep([nw, sw, se, ne], home=home)
    assert sorted_wp == [se, sw, nw, ne]


def test_sort_clockwise_sweep_home_on_centroid():
    north = Coordinate(1, 0, 0)
    east = Coordinate(0, 1, 0)
    south = Coordinate(-1, 0, 0)
    west = Coordinate(0, -1, 0)

    home = Coordinate(0, 0, 0)
    sorted_wp = sort_clockwise_sweep([east, west, south, north], home=home)
    assert sorted_wp == [north, east, south, west]


def test_sort_clockwise_sweep_same_direction_tiebreaker():
    p1 = Coordinate(1, 1, 0)
    sorted_wp = sort_clockwise_sweep([p1, p1])
    assert sorted_wp == [p1, p1]

def test_kill_mutant_02_tiebreak_closest_first():
    p1 = Coordinate(1, 0, 0)
    p2 = Coordinate(2, 0, 0)
    anchor = Coordinate(-18, 0, 0)
    sorted_wp = sort_clockwise_sweep([p2, p1, anchor])
    assert sorted_wp.index(p1) < sorted_wp.index(p2)


def test_kill_mutant_06_north_radians_conversion():
    _, north = east_north_coordinate_offset_m(0.0, 0.0, 1.0, 0.0)
    assert north < 200000.0


def test_kill_mutant_12_cos_latitude_scaling():
    east_eq, _ = east_north_coordinate_offset_m(0.0, 0.0, 0.0, 1.0)
    east_60, _ = east_north_coordinate_offset_m(60.0, 0.0, 60.0, 1.0)
    assert east_60 == pytest.approx(east_eq * 0.5, rel=1e-3)