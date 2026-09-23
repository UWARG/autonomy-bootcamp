"""Tests for waypoint utility functions."""

from dataclasses import FrozenInstanceError

import pytest

from src.types import Coordinate
from src.waypoint_utils import (
    east_north_coordinate_offset_m,
    parse_waypoints_file,
    sort_clockwise_sweep,
)


def write_to_tmp_waypoints_file(tmp_path, text):
    """Write ``text`` to a YAML file and return its path."""
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


def test_parse_rejects_non_mapping(tmp_path):
    path = write_to_tmp_waypoints_file(
        tmp_path,
        """
        - hello
        - world
        """,
    )

    with pytest.raises(ValueError):
        parse_waypoints_file(path)


@pytest.mark.parametrize("missing", ["lat", "lon", "alt"])
def test_parse_rejects_missing_coordinate_key(tmp_path, missing):
    coordinate = {"lat": 1, "lon": 2, "alt": 3}
    coordinate.pop(missing)

    text = f"""
    waypoints:
      - {coordinate}
    """

    path = write_to_tmp_waypoints_file(tmp_path, text)

    with pytest.raises(ValueError):
        parse_waypoints_file(path)

def test_coordinate_offset_east_scales_with_latitude():
    east, north = east_north_coordinate_offset_m(
        60.0,
        0.0,
        60.0,
        0.001,
    )

    assert east == pytest.approx(55.6, abs=1.0)
    assert north == pytest.approx(0.0, abs=0.01)

def test_parse_rejects_non_numeric_coordinate(tmp_path):
    path = write_to_tmp_waypoints_file(
        tmp_path,
        """
        waypoints:
          - {lat: hello, lon: 2, alt: 3}
        """,
    )

    with pytest.raises(ValueError):
        parse_waypoints_file(path)


def test_parse_rejects_non_mapping_waypoint(tmp_path):
    path = write_to_tmp_waypoints_file(
        tmp_path,
        """
        waypoints:
          - hello
        """,
    )

    with pytest.raises(ValueError):
        parse_waypoints_file(path)


def test_parse_rejects_invalid_yaml(tmp_path):
    path = write_to_tmp_waypoints_file(
        tmp_path,
        """
        waypoints:
          - [this is: invalid
        """,
    )

    with pytest.raises(ValueError):
        parse_waypoints_file(path)


def test_parse_missing_file(tmp_path):
    path = tmp_path / "does-not-exist.yaml"

    with pytest.raises(OSError):
        parse_waypoints_file(path)


def test_parse_rejects_non_list_waypoints(tmp_path):
    path = write_to_tmp_waypoints_file(
        tmp_path,
        """
        waypoints:
          lat: 1
          lon: 2
          alt: 3
        """,
    )

    with pytest.raises(ValueError):
        parse_waypoints_file(path)


@pytest.mark.parametrize(
    ("lat", "lon"),
    [
        (90.1, 0),
        (-90.1, 0),
        (0, 180.1),
        (0, -180.1),
    ],
)
def test_parse_rejects_out_of_range_coordinates(tmp_path, lat, lon):
    path = write_to_tmp_waypoints_file(
        tmp_path,
        f"""
        waypoints:
          - {{lat: {lat}, lon: {lon}, alt: 0}}
        """,
    )

    with pytest.raises(ValueError):
        parse_waypoints_file(path)


def test_parse_empty_file(tmp_path):
    path = write_to_tmp_waypoints_file(tmp_path, "")

    assert parse_waypoints_file(path) == (None, [])


def test_parse_empty_waypoints(tmp_path):
    path = write_to_tmp_waypoints_file(
        tmp_path,
        """
        waypoints: []
        """,
    )

    assert parse_waypoints_file(path) == (None, [])


def test_coordinate_offset_same_position():
    east, north = east_north_coordinate_offset_m(
        43.0,
        -80.0,
        43.0,
        -80.0,
    )

    assert east == pytest.approx(0.0, abs=1e-9)
    assert north == pytest.approx(0.0, abs=1e-9)


def test_coordinate_offset_north():
    east, north = east_north_coordinate_offset_m(
        0.0,
        0.0,
        0.001,
        0.0,
    )

    # 0.001 degrees latitude is approximately 111.2 m.
    assert east == pytest.approx(0.0, abs=0.01)
    assert north == pytest.approx(111.2, abs=1.0)


def test_coordinate_offset_east():
    east, north = east_north_coordinate_offset_m(
        0.0,
        0.0,
        0.0,
        0.001,
    )

    assert east == pytest.approx(111.2, abs=1.0)
    assert north == pytest.approx(0.0, abs=0.01)


def test_sort_empty_list():
    assert sort_clockwise_sweep([]) == []


def test_sort_single_waypoint():
    waypoint = Coordinate(1, 2, 3)

    assert sort_clockwise_sweep([waypoint]) == [waypoint]


def test_sort_clockwise_from_north():
    north = Coordinate(1, 0, 0)
    east = Coordinate(0, 1, 0)
    south = Coordinate(-1, 0, 0)
    west = Coordinate(0, -1, 0)

    unordered = [south, west, north, east]

    result = sort_clockwise_sweep(unordered)

    assert result == [north, east, south, west]


def test_sort_starts_from_home_direction():
    north = Coordinate(1, 0, 0)
    east = Coordinate(0, 1, 0)
    south = Coordinate(-1, 0, 0)
    west = Coordinate(0, -1, 0)

    # Home lies east of the centroid.
    home = Coordinate(0, 2, 0)

    result = sort_clockwise_sweep(
        [north, east, south, west],
        home=home,
    )

    assert result == [east, south, west, north]


def test_home_at_centroid_starts_from_north():
    north = Coordinate(1, 0, 0)
    east = Coordinate(0, 1, 0)
    south = Coordinate(-1, 0, 0)
    west = Coordinate(0, -1, 0)

    home = Coordinate(0, 0, 0)

    result = sort_clockwise_sweep(
        [west, south, east, north],
        home=home,
    )

    assert result == [north, east, south, west]


def test_same_direction_closer_waypoint_first():
    near = Coordinate(1, 0, 0)
    far = Coordinate(2, 0, 0)

    # These balance the centroid so near/far remain in the same
    # direction from it.
    south_1 = Coordinate(-1, 0, 0)
    south_2 = Coordinate(-2, 0, 0)

    result = sort_clockwise_sweep(
        [far, south_1, near, south_2],
    )

    assert result.index(near) < result.index(far)


def test_parsed_coordinates_are_frozen(tmp_path):
    path = write_to_tmp_waypoints_file(
        tmp_path,
        """
        waypoints:
          - {lat: 1, lon: 2, alt: 3}
        """,
    )

    _, waypoints = parse_waypoints_file(path)
    coordinate = waypoints[0]

    with pytest.raises(FrozenInstanceError):
        coordinate.lat = 100


def test_clockwise_sort_does_not_modify_input():
    north = Coordinate(1, 0, 0)
    east = Coordinate(0, 1, 0)
    south = Coordinate(-1, 0, 0)
    west = Coordinate(0, -1, 0)

    original = [south, north, west, east]
    snapshot = list(original)

    sort_clockwise_sweep(original)

    assert original == snapshot