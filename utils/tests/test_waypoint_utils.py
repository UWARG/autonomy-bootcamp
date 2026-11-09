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

def test_east_offset_scaling():
    east, north = east_north_coordinate_offset_m(
            from_lat=60.0,
            from_lon=-0.0,
            to_lat=60.0,
            to_lon=-1.0
        )
    assert east == pytest.approx(-55597, rel=0.02)
    assert north == pytest.approx(0.0, abs=1)



def test_clockwise_sweep_point():
    assert sort_clockwise_sweep([]) == []

    waypoint = Coordinate(lat=10, lon=10, alt=10)
    assert sort_clockwise_sweep([waypoint]) == [waypoint]

def test_clockwise_sweep_north():
    north = Coordinate(lat=1,lon=0,alt=0)
    south = Coordinate(lat=-1,lon=0,alt=0)
    east = Coordinate(lat=0,lon=1,alt=0)
    west = Coordinate(lat=0,lon=-1,alt=0)

    unsorted = [east, south,north,west]

    sorted = [north,east,south,west]
    assert sort_clockwise_sweep(unsorted) == sorted

def test_broken_file(tmp_path):
    path = write_to_tmp_waypoints_file(tmp_path,"broken")
    with pytest.raises(ValueError):
        parse_waypoints_file(path)

def test_lat(tmp_path):
    yaml = """
                waypoints:
                  - {lat: 91, lon: 5, alt: 6}
                """
    path = write_to_tmp_waypoints_file(tmp_path,yaml)
    with pytest.raises(ValueError):
        parse_waypoints_file(path)

def test_lat_negative(tmp_path):
    yaml = """
                waypoints:
                  - {lat: -91, lon: 5, alt: 6}
                """
    path = write_to_tmp_waypoints_file(tmp_path,yaml)
    with pytest.raises(ValueError):
        parse_waypoints_file(path)

def test_lon_negative(tmp_path):
    yaml = """
                waypoints:
                  - {lat: 4, lon: -181, alt: 6}
                """
    path = write_to_tmp_waypoints_file(tmp_path,yaml)
    with pytest.raises(ValueError):
        parse_waypoints_file(path)

def test_lon(tmp_path):
    yaml = """
                waypoints:
                  - {lat: 4, lon: 181, alt: 6}
                """
    path = write_to_tmp_waypoints_file(tmp_path,yaml)
    with pytest.raises(ValueError):
        parse_waypoints_file(path)

def test_nonvalues(tmp_path):
    yaml = """
                waypoints:
                  - {lat: a, lon: b, alt: c}
                """
    path = write_to_tmp_waypoints_file(tmp_path,yaml)
    with pytest.raises(ValueError):
        parse_waypoints_file(path)

def test_zero_lon(tmp_path):
    yaml = """
                waypoints:
                  - {lat: 4, alt: 6}
                """
    path = write_to_tmp_waypoints_file(tmp_path,yaml)
    with pytest.raises(ValueError):
        parse_waypoints_file(path)

def test_zero_lat(tmp_path):
    yaml = """
                waypoints:
                  - {lon:4 , alt: 6}
                """
    path = write_to_tmp_waypoints_file(tmp_path,yaml)
    with pytest.raises(ValueError):
        parse_waypoints_file(path)

def test_east_north_offset():
    east, north = east_north_coordinate_offset_m(from_lat=43.0, from_lon=-80.0, 
                                                 to_lat=43.0, to_lon=-80.0)
    
    assert east == pytest.approx(0.0)
    assert north == pytest.approx(0.0)
        

def test_empty_file(tmp_path):
    path = write_to_tmp_waypoints_file(tmp_path,"")

    home, waypoints = parse_waypoints_file(path)

    assert home is None
    assert waypoints ==[]

def test_missing_file(tmp_path):
    path = tmp_path / "nonexistent.yaml"

    with pytest.raises((ValueError, FileNotFoundError)):
        parse_waypoints_file(path)

def test_empty_waypoints(tmp_path):
    path = write_to_tmp_waypoints_file(
        tmp_path,
        "waypoints: []"
    )

    home, waypoints = parse_waypoints_file(path)

    assert home is None
    assert waypoints == []

@pytest.mark.parametrize(
    "coordinates",
    [
        "{lon: 5, alt: 6}",  # Missing lat
        "{lat: 4, alt: 6}",  # Missing lon
        "{lat: 4, lon: 5}",  # Missing alt
    ]
)
def test_missing_coordinates(tmp_path, coordinates):
    yaml = f"""
    waypoints:
      - {coordinates}
    """

    path = write_to_tmp_waypoints_file(tmp_path, yaml)

    with pytest.raises(ValueError):
        parse_waypoints_file(path)


def test_same_direction_closer_first():
    close = Coordinate(lat=1, lon=0, alt=0)
    far = Coordinate(lat=2, lon=0, alt=0)
    opposite = Coordinate(lat=-3, lon=0, alt=0)

    result = sort_clockwise_sweep([far, opposite, close])

    assert result == [close, far, opposite]

def test_coordinate_is_frozen():
    coordinate = Coordinate(lat=10, lon=20, alt=30)

    with pytest.raises((AttributeError, TypeError)):
        coordinate.lat = 50


def test_north_offset():
    east, north = east_north_coordinate_offset_m(
        from_lat=43.0,
        from_lon=-80.0,
        to_lat=44.0,
        to_lon=-80.0
    )

    assert east == pytest.approx(0, abs=1)
    assert north == pytest.approx(111195, abs=200)

def test_invalid_home(tmp_path):
    yaml = """
    home: {lat: 91, lon: 5, alt: 6}
    waypoints:
      - {lat: 4, lon: 5, alt: 6}
    """

    path = write_to_tmp_waypoints_file(tmp_path, yaml)

    with pytest.raises(ValueError):
        parse_waypoints_file(path)

def test_waypoint_not_mapping(tmp_path):
    path = write_to_tmp_waypoints_file(
        tmp_path,
        """
        waypoints:
          - hello
        """
    )

    with pytest.raises(ValueError):
        parse_waypoints_file(path)

def test_waypoints_not_list(tmp_path):
    path = write_to_tmp_waypoints_file(
        tmp_path,
        """
        waypoints: hello
        """
    )

    with pytest.raises(ValueError):
        parse_waypoints_file(path)


def test_invalid_yaml_syntax(tmp_path):
    path = write_to_tmp_waypoints_file(
        tmp_path,
        """
        waypoints: [
        """
    )

    with pytest.raises(ValueError):
        parse_waypoints_file(path)


def test_clockwise_sweep_from_home():
    north = Coordinate(1, 0, 0)
    east = Coordinate(0, 1, 0)
    south = Coordinate(-1, 0, 0)
    west = Coordinate(0, -1, 0)

    home = Coordinate(0, 2, 0)

    result = sort_clockwise_sweep(
        [north, west, south, east],
        home
    )

    assert result == [east, south, west, north]


