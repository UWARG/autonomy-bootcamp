"""Tests for waypoint parsing, coordinate offsets, and sweep ordering."""

from dataclasses import FrozenInstanceError

import pytest

from src.types import Coordinate
from src.waypoint_utils import (
    east_north_coordinate_offset_m,
    parse_waypoints_file,
    sort_clockwise_sweep,
)


def write_to_tmp_waypoints_file(tmp_path, text):
    """Write ``text`` to a YAML file and hand back its path."""
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


@pytest.mark.parametrize("text", ["", "  \n  "])
def test_empty_file_returns_no_home_or_waypoints(tmp_path, text):
    path = write_to_tmp_waypoints_file(tmp_path, text)
    assert parse_waypoints_file(path) == (None, [])


def test_empty_mapping_and_waypoint_list_return_empty_results(tmp_path):
    path = write_to_tmp_waypoints_file(tmp_path, "waypoints: []\n")
    assert parse_waypoints_file(path) == (None, [])


def test_missing_waypoints_key_defaults_to_empty_list(tmp_path):
    path = write_to_tmp_waypoints_file(tmp_path, "home: {lat: 1, lon: 2, alt: 3}\n")
    assert parse_waypoints_file(path) == (Coordinate(1, 2, 3), [])


@pytest.mark.parametrize("text", ["- one\n- two\n", "42\n", "false\n"])
def test_top_level_must_be_a_mapping(tmp_path, text):
    path = write_to_tmp_waypoints_file(tmp_path, text)
    with pytest.raises(ValueError, match="expected a mapping"):
        parse_waypoints_file(path)


def test_invalid_yaml_raises_value_error(tmp_path):
    path = write_to_tmp_waypoints_file(tmp_path, "home: [\n")
    with pytest.raises(ValueError, match="invalid YAML"):
        parse_waypoints_file(path)


def test_missing_file_keeps_the_filesystem_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        parse_waypoints_file(tmp_path / "missing.yaml")


@pytest.mark.parametrize(
    "entry",
    ["{lon: 2, alt: 3}", "{lat: 1, alt: 3}", "{lat: 1, lon: 2}"],
    ids=["missing-latitude", "missing-longitude", "missing-altitude"],
)
def test_waypoint_requires_all_three_keys(tmp_path, entry):
    path = write_to_tmp_waypoints_file(tmp_path, f"waypoints:\n  - {entry}\n")
    with pytest.raises(ValueError, match="missing key"):
        parse_waypoints_file(path)


@pytest.mark.parametrize(
    "text",
    [
        "waypoints:\n  - not-a-mapping\n",
        "waypoints: {lat: 1, lon: 2, alt: 3}\n",
        "waypoints: a-string\n",
    ],
    ids=["waypoint-is-not-a-mapping", "waypoints-is-a-mapping", "waypoints-is-a-string"],
)
def test_wrong_entry_and_waypoint_container_types_raise(tmp_path, text):
    path = write_to_tmp_waypoints_file(tmp_path, text)
    with pytest.raises(ValueError):
        parse_waypoints_file(path)


def test_home_must_be_a_coordinate_mapping(tmp_path):
    path = write_to_tmp_waypoints_file(
        tmp_path, "home: [1, 2, 3]\nwaypoints: []\n"
    )
    with pytest.raises(ValueError, match="home must be a mapping"):
        parse_waypoints_file(path)


def test_non_numeric_coordinate_value_raises_value_error(tmp_path):
    path = write_to_tmp_waypoints_file(
        tmp_path, 'waypoints:\n  - {lat: "north", lon: 2, alt: 3}\n'
    )
    with pytest.raises(ValueError, match="non-numeric"):
        parse_waypoints_file(path)


@pytest.mark.parametrize(
    ("key", "value"),
    [("lat", 91), ("lat", -91), ("lon", 181), ("lon", -181)],
    ids=[
        "latitude-too-north",
        "latitude-too-south",
        "longitude-too-east",
        "longitude-too-west",
    ],
)
def test_out_of_range_coordinates_raise(tmp_path, key, value):
    values = {"lat": 0, "lon": 0, "alt": 3}
    values[key] = value
    path = write_to_tmp_waypoints_file(
        tmp_path,
        f"waypoints:\n  - {{lat: {values['lat']}, lon: {values['lon']}, alt: {values['alt']}}}\n",
    )
    with pytest.raises(ValueError, match="out of range"):
        parse_waypoints_file(path)


def test_parsed_coordinates_are_frozen(tmp_path):
    path = write_to_tmp_waypoints_file(
        tmp_path, "home: {lat: 1, lon: 2, alt: 3}\nwaypoints: []\n"
    )
    home, _ = parse_waypoints_file(path)
    assert home == Coordinate(1, 2, 3)
    with pytest.raises(FrozenInstanceError):
        home.lat = 99


def test_east_and_north_offsets_use_compass_axes_and_radians():
    east, north = east_north_coordinate_offset_m(0, 0, 1, 2)
    assert east == pytest.approx(222_381.69, abs=1)
    assert north == pytest.approx(111_195.08, abs=1)


def test_east_offset_scales_with_latitude():
    east, north = east_north_coordinate_offset_m(60, 0, 60, 1)
    assert east == pytest.approx(55_597.54, abs=1)
    assert north == pytest.approx(0, abs=1e-9)


def test_empty_and_single_waypoint_sweeps_are_preserved():
    waypoint = Coordinate(12, 34, 5)
    assert sort_clockwise_sweep([]) == []
    assert sort_clockwise_sweep([waypoint]) == [waypoint]


def test_sweep_without_home_starts_north_and_runs_clockwise():
    north = Coordinate(1, 0, 10)
    east = Coordinate(0, 1, 20)
    south = Coordinate(-1, 0, 30)
    west = Coordinate(0, -1, 40)
    assert sort_clockwise_sweep([south, west, east, north]) == [
        north,
        east,
        south,
        west,
    ]


def test_sweep_starts_in_home_direction_and_wraps_clockwise():
    north = Coordinate(1, 0, 10)
    east = Coordinate(0, 1, 20)
    south = Coordinate(-1, 0, 30)
    west = Coordinate(0, -1, 40)
    home = Coordinate(0, -5, 0)
    assert sort_clockwise_sweep([north, east, south, west], home) == [
        west,
        north,
        east,
        south,
    ]


def test_home_at_centroid_defaults_to_north():
    north = Coordinate(1, 0, 1)
    east = Coordinate(0, 1, 2)
    south = Coordinate(-1, 0, 3)
    west = Coordinate(0, -1, 4)
    assert sort_clockwise_sweep(
        [south, west, east, north], Coordinate(0, 0, 0)
    ) == [north, east, south, west]


def test_same_bearing_waypoints_are_ordered_nearest_first():
    near_north = Coordinate(1, 0, 0)
    far_north = Coordinate(3, 0, 0)
    south = Coordinate(-4, 0, 0)
    ordered = sort_clockwise_sweep([far_north, south, near_north])
    assert ordered == [near_north, far_north, south]
