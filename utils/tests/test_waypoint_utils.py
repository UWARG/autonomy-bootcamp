
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


def test_parse_bad_top_level(tmp_path):
    path = write_to_tmp_waypoints_file(
        tmp_path,
        """
        - hello
        - world
        """,
    )

    with pytest.raises(ValueError):
        parse_waypoints_file(path)


def test_parse_waypoints_not_list(tmp_path):
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
    "text",
    [
        """
        waypoints:
          - lon: 2
            alt: 3
        """,
        """
        waypoints:
          - lat: 1
            alt: 3
        """,
        """
        waypoints:
          - lat: 1
            lon: 2
        """,
    ],
    ids=["missing-lat", "missing-lon", "missing-alt"],
)
def test_parse_missing_coordinate_field(tmp_path, text):
    path = write_to_tmp_waypoints_file(tmp_path, text)

    with pytest.raises(ValueError):
        parse_waypoints_file(path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("lat", "x"),
        ("lon", "x"),
        ("alt", "x"),
    ],
)
def test_parse_non_numeric_values(tmp_path, field, value):
    path = write_to_tmp_waypoints_file(
        tmp_path,
        f"""
        waypoints:
          - lat: {value if field == "lat" else 4}
            lon: {value if field == "lon" else 5}
            alt: {value if field == "alt" else 6}
        """,
    )

    with pytest.raises(ValueError):
        parse_waypoints_file(path)


@pytest.mark.parametrize(
    ("lat", "lon"),
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
def test_parse_out_of_range_values(tmp_path, lat, lon):
    path = write_to_tmp_waypoints_file(
        tmp_path,
        f"""
        waypoints:
          - lat: {lat}
            lon: {lon}
            alt: 5
        """,
    )

    with pytest.raises(ValueError):
        parse_waypoints_file(path)


def test_parse_invalid_yaml(tmp_path):
    path = write_to_tmp_waypoints_file(
        tmp_path,
        """
        waypoints:
          - lat: 10
            lon: 20
            alt: 5
          - lat: 30
            lon: 40
            alt: [5
        """,
    )

    with pytest.raises(ValueError):
        parse_waypoints_file(path)


def test_parse_waypoints_file_missing_file(tmp_path):
    path = tmp_path / "does_not_exist.yaml"

    with pytest.raises(FileNotFoundError):
        parse_waypoints_file(path)


def test_parse_waypoints_file_empty(tmp_path):
    path = write_to_tmp_waypoints_file(tmp_path, "")

    assert parse_waypoints_file(path) == (None, [])


def test_parse_waypoints_file_empty_waypoints(tmp_path):
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
        ([Coordinate(1, 2, 3)], [Coordinate(1, 2, 3)]),
    ],
    ids=["empty", "one-waypoint"],
)
def test_sort_clockwise_sweep_zero_or_one_waypoints(waypoints, expected):
    assert sort_clockwise_sweep(waypoints) == expected


@pytest.mark.parametrize(
    (
        "from_lat",
        "from_lon",
        "to_lat",
        "to_lon",
        "expected_east",
        "expected_north",
    ),
    [
        # East movement at 45 degrees latitude tests the cos(latitude) factor.
        (45, 0, 45, 1, 78_626.7, 0.0),

        # North movement tests conversion from degrees to radians.
        (0, 0, 1, 0, 0.0, 111_195.1),
    ],
)
def test_east_north_coordinate_offset_m(
    from_lat,
    from_lon,
    to_lat,
    to_lon,
    expected_east,
    expected_north,
):
    east, north = east_north_coordinate_offset_m(
        from_lat,
        from_lon,
        to_lat,
        to_lon,
    )

    assert east == pytest.approx(expected_east, abs=1)
    assert north == pytest.approx(expected_north, abs=1)


def test_sort_clockwise_sweep_without_home():
    north = Coordinate(1, 0, 0)
    east = Coordinate(0, 1, 0)
    south = Coordinate(-1, 0, 0)
    west = Coordinate(0, -1, 0)

    waypoints = [east, south, west, north]

    assert sort_clockwise_sweep(waypoints) == [
        north,
        east,
        south,
        west,
    ]


def test_sort_clockwise_sweep_with_home():
    north = Coordinate(1, 0, 0)
    east = Coordinate(0, 1, 0)
    south = Coordinate(-1, 0, 0)
    west = Coordinate(0, -1, 0)

    home = Coordinate(0, 1, 0)

    waypoints = [north, east, south, west]

    assert sort_clockwise_sweep(waypoints, home) == [
        east,
        south,
        west,
        north,
    ]


def test_sort_clockwise_sweep_home_at_centroid_starts_at_north():
    north = Coordinate(1, 0, 0)
    east = Coordinate(0, 1, 0)
    south = Coordinate(-1, 0, 0)
    west = Coordinate(0, -1, 0)

    home = Coordinate(0, 0, 0)

    waypoints = [east, south, west, north]

    assert sort_clockwise_sweep(waypoints, home) == [
        north,
        east,
        south,
        west,
    ]


def test_sort_clockwise_sweep_same_direction_closer_first():
    north_close = Coordinate(1, 0, 0)
    north_far = Coordinate(2, 0, 0)
    south = Coordinate(-3, 0, 0)
    east = Coordinate(0, 1, 0)
    west = Coordinate(0, -1, 0)

    waypoints = [north_far, west, south, east, north_close]

    result = sort_clockwise_sweep(waypoints)

    assert result.index(north_close) < result.index(north_far)


def test_parse_waypoints_file_coordinates_are_frozen(tmp_path):
    path = write_to_tmp_waypoints_file(
        tmp_path,
        """
        home: {lat: 1, lon: 2, alt: 3}
        waypoints:
          - {lat: 4, lon: 5, alt: 6}
        """,
    )

    home, waypoints = parse_waypoints_file(path)

    with pytest.raises((AttributeError, TypeError)):
        home.lat = 10

    with pytest.raises((AttributeError, TypeError)):
        waypoints[0].lat = 10