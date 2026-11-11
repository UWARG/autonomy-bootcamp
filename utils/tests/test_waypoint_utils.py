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

from dataclasses import FrozenInstanceError

from src.waypoint_utils import (
    east_north_coordinate_offset_m,
    parse_waypoints_file,
    sort_clockwise_sweep,
)
from src.types import Coordinate

# The helper and the test below are given to you.


def write_to_tmp_waypoints_file(tmp_path, text):
    """Write ``text`` to a YAML file and hand back its path.

    ``tmp_path`` is a pytest fixture: a fresh empty directory per test.
    """
    path = tmp_path / "waypoints.yaml"
    path.write_text(text)
    return path

# --------------------------------- east_north_coordinate_offset_m ---------------------------------
@pytest.mark.parametrize(
    ("input", "expected"),
    [
        (
            [-12, 23, 2, 101], 
            [8640212.051837476, 1556731.1232694609],
        ),
        (
            [1, 1, 1, 1],
            [0, 0]
        )
    ]
)
def test_east_north_coordinate_offset_proper_output(input, expected):
    east, north = east_north_coordinate_offset_m(input[0], input[1], input[2], input[3])
    assert east == pytest.approx(expected[0])
    assert north == pytest.approx(expected[1])

# --------------------------------- parse_waypoints_file ---------------------------------
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
    
def test_parse_waypoints_file_empty_file(tmp_path):
    path = write_to_tmp_waypoints_file(tmp_path, "")
    assert parse_waypoints_file(path) == (None, [])
    
def test_parse_waypoints_file_frozen_coordinate_output(tmp_path):
    text = """
    home: {lat: 1, lon: 2, alt: 3}
    waypoints:
      - {lat: 4, lon: 5, alt: 6}
    """
    
    path = write_to_tmp_waypoints_file(tmp_path, text)
    with pytest.raises(FrozenInstanceError):
        home, waypoints = parse_waypoints_file(path)
        home.lat = 0

@pytest.mark.parametrize(
    "text",
    [
        (
            """
            home {lat: 3, lon: 5, alt: 6}
            waypoints:
              - {lat: 3, lon: 5, alt: 6}
            """
        ),
        (
            """
            home: {lat: 3, lon: 5, alt: 6}
            waypoints
              - {lat: 3, lon: 5, alt: 6}
            """
        ),
    ],
    ids=["invalid home map", "invalid waypoint map"],
)
def test_raises_error_on_invalid_mapping(tmp_path, text):
    path = write_to_tmp_waypoints_file(tmp_path, text)
    with pytest.raises(ValueError):
        parse_waypoints_file(path)


@pytest.mark.parametrize(
    "text",
    [
        (
            """
            home: {lon: 5, alt: 6}
            waypoints:
              - {lat: 3, lon: 5, alt: 6}
            """
        ),
        (
            """
            waypoints:
              - {lat: 5, alt: 6}
            """
        ),
        (
            """
            waypoints:
              - {lat: 5, lon: 6}
            """
        ),
        (
            """
            waypoints:
              - {lat: 4, lon: 5, alt: 6}
              - {lat: 7, alt: 9}
            """
        ),
    ],
    ids=["missing lat", "missing lon", "missing alt", "missing, multiple"],
)
def test_raises_error_on_missing_coordinate_fields(tmp_path, text):
    path = write_to_tmp_waypoints_file(tmp_path, text)
    with pytest.raises(ValueError):
        parse_waypoints_file(path)
      
@pytest.mark.parametrize(
    "text",
    [
        (
            """
            home: {lon: "yes", alt: 6}
            waypoints:
              - {lat: 3, lon: "no", alt: 6}
            """
        ),
        (
            """
            waypoints:
              - lat:
                  - 2
                lon:
                  - 2
                alt:
                  - 2
            """
        ),
    ],
    ids=["string field", "not dict"],
)
def test_raises_error_on_incorrect_data_type(tmp_path, text):
    path = write_to_tmp_waypoints_file(tmp_path, text)
    with pytest.raises(ValueError):
        parse_waypoints_file(path)

@pytest.mark.parametrize(
    "text",
    [
        (
            """
            home: {lat: -91, lon: 2, alt: 3}
            waypoints:
              - {lat: 4, lon: 5, alt: 6}
            """
        ),
        (
            """
            waypoints:
              - {lat: 203, lon: 20, alt: 3}
            """
        ),
        (
            """
            waypoints:
              - {lat: 4, lon: -183, alt: 6}
              - {lat: 7, lon: 8, alt: 9}
            """
        ),
        (
            """
            waypoints:
              - {lat: 7, lon: 180.01, alt: 9}
            """
        ),
    ],
    ids=["lat low", "lat high", "lon low", "lon high"],
)
def test_raises_error_on_out_of_range_lat_and_lon(tmp_path, text):
    path = write_to_tmp_waypoints_file(tmp_path, text)
    with pytest.raises(ValueError):
        parse_waypoints_file(path)

@pytest.mark.parametrize(
    "text",
    [
        (
            """
            home: {lat: 3, lon: 2, alt: 3}
             waypoints:
              - {lat: 4, lon: 5, alt: 6}
            """
        ),
        (
            """
            home: {lat: -3, lon: 2, alt: 3}
            waypoints:
              {lat: 4, lon: 5, alt: 6}
              {lat: 2, lon: 4, alt: 3}
            """
        ),
    ],
    ids=["spacing", "dashes"]
)
def test_raises_error_on_parse_failure(tmp_path, text):
    path = write_to_tmp_waypoints_file(tmp_path, text)
    with pytest.raises(ValueError):
        parse_waypoints_file(path)
        
# --------------------------------- sort_clockwise_sweep ---------------------------------
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            """
            home: {lat: 3, lon: 2, alt: 3}
            waypoints:
              - {lat: 2, lon: 4, alt: 5}
              - {lat: 4, lon: 5, alt: 6}
              - {lat: -10, lon: 10, alt: 3}
            """,
            ([Coordinate(2,4,5), Coordinate(4,5,6), Coordinate(-10,10,3)]),
        ),
        (
            """
            waypoints:
              - {lat: 4, lon: 5, alt: 6}
              - {lat: 2, lon: 4, alt: 5}
              - {lat: -10, lon: 10, alt: 3}
            """,
            ([Coordinate(-10,10,3), Coordinate(2,4,5), Coordinate(4,5,6)]),
        ),
        (
            """
            waypoints:
            """,
            [],
        ),
        (
            """
            waypoints:
              - {lat: 4, lon: 5, alt: 6}
            """,
            [Coordinate(4, 5, 6)],
        ),
    ],
    ids=["3 waypoints", "no home", "no waypoint", "1 waypoint"]
)
def test_sort_clockwise_sweep_success(tmp_path, text, expected):
    path = write_to_tmp_waypoints_file(tmp_path, text)
    home, waypoints = parse_waypoints_file(path)
    
    assert sort_clockwise_sweep(waypoints, home) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            """
            waypoints:
              - {lat: 1, lon: 0, alt: 5}
              - {lat: 2, lon: 0, alt: 6}
              - {lat: 3, lon: 0, alt: 3}
            """,
            ([Coordinate(2,0,6), Coordinate(3,0,3), Coordinate(1,0,5)]),
        ),
    ]
)
def test_sort_clockwise_equal_bearing(tmp_path, text, expected):
    path = write_to_tmp_waypoints_file(tmp_path, text)
    home, waypoints = parse_waypoints_file(path)
    
    assert sort_clockwise_sweep(waypoints, home) == expected


def test_placeholder():
    # TODO(bootcamper): delete this and write real tests. It's only here so
    # linter doesn't complain about unused imports before you start.
    assert callable(east_north_coordinate_offset_m)
    assert callable(parse_waypoints_file)
    assert callable(sort_clockwise_sweep)
