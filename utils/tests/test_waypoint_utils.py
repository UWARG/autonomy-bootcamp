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

from dataclasses import FrozenInstanceError

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

@pytest.mark.parametrize(
    "text",
    [
        """
        waypoints:
        - {lat: 80
        """,
        """
        waypoints:
        - {lat: 0, lon: abc, alt: 10}
        """,
        """
        waypoints:
        - {lon: 80, alt: 10}
        """,
        """
        waypoints:
        - {lon: 80, lat: 80}
        """,
        """
        waypoints:
        - {lat: 80, alt: 10}
        """,
    ]
)
def test_rejects_invalid_file(tmp_path, text):
    path = write_to_tmp_waypoints_file(tmp_path,text)
    with pytest.raises(ValueError):
        parse_waypoints_file(path)

def test_rejects_missing_file():
    path = "nonexistentpath/abc"
    with pytest.raises(OSError):
        parse_waypoints_file(path)

@pytest.mark.parametrize(
    "text",
    [
        """
        waypoints:
          - {lat: 91, lon: 0, alt: 10}
        """,
        """
        waypoints:
          - {lat: -91, lon: 0, alt: 10}
        """,
        """
        waypoints:
          - {lat: 0, lon: 181, alt: 10}
        """,
        """
        waypoints:
          - {lat: 0, lon: -181, alt: 10}
        """,
    ],
)
def test_rejects_coordinates_out_of_range(tmp_path, text):
    path = write_to_tmp_waypoints_file(tmp_path, text)
    with pytest.raises(ValueError):
        parse_waypoints_file(path)

def test_parse_waypoints_file_returns_on_empty_input(tmp_path):
    path = write_to_tmp_waypoints_file(tmp_path, "")
    assert parse_waypoints_file(path) == (None, [])

@pytest.mark.parametrize(
    "waypoints",
    [
        [],
        [Coordinate(0.5, 0.4, 0.2)]
    ]
)
def test_sort_clockwise_returns_on_empty_input(waypoints):
    assert sort_clockwise_sweep(waypoints) == waypoints

@pytest.mark.parametrize(
    ("values", "expected"),
    [
        (
            (5.0, 5.0, 5.0, 5.0),
            (0, 0)
        ),
        (
            (49.0, 122.0, 49.01, 122.01),
            (1111.95, 729.432)
        ),
        (
            (10.0, 12.0, 9.98, 11.97),
            (-2223.90, -3285.274)
        )
    ],
)
def test_east_north_coordinate_offest_returns_correct_values(values, expected):
    assert east_north_coordinate_offset_m(
        values[0],values[1],values[2],values[3]
    ) == (pytest.approx(expected[1]), pytest.approx(expected[0]))

@pytest.mark.parametrize(
    ("waypoints", "expected"),
    [
        (
            [
                Coordinate(0.0, 0.1, 0.0),
                Coordinate(0.1, 0.01, 0.0),
                Coordinate(-1.0, 0.0, 0.0),
                Coordinate(0.1, 0.1, 0.0),
            ],
            [
                Coordinate(0.1, 0.1, 0.0),
                Coordinate(0.0, 0.1, 0.0),
                Coordinate(-1.0, 0.0, 0.0),
                Coordinate(0.1, 0.01, 0.0),

            ],
        ),
        (
            [
                Coordinate(50.1, 49.9, 0.0),
                Coordinate(50.0, 49.9, 0.0),
                Coordinate(49.9, 49.9, 0.0),
                Coordinate(49.9, 50.0, 0.0),
            ],
            [
                Coordinate(49.9, 50.0, 0.0),
                Coordinate(49.9, 49.9, 0.0),
                Coordinate(50.0, 49.9, 0.0),
                Coordinate(50.1, 49.9, 0.0),
            ],
        ),
    ]
)
def test_sort_clockwise_sweep_no_home(waypoints, expected):
    assert sort_clockwise_sweep(waypoints) == expected

@pytest.mark.parametrize(
    ("waypoints", "home", "expected"),
    [
        (
            [
                Coordinate(0.0, 0.1, 0.0),
                Coordinate(0.1, 0.01, 0.0),
                Coordinate(-0.1, 0.0, 0.0),
                Coordinate(0.1, 0.1, 0.0),
            ],
            Coordinate(0.1, 1.0, 0.0),
            [
                Coordinate(0.0, 0.1, 0.0),
                Coordinate(-0.1, 0.0, 0.0),
                Coordinate(0.1, 0.01, 0.0),
                Coordinate(0.1, 0.1, 0.0),
            ],
        ),
        (
            [
                Coordinate(50.1, 49.9, 0.0),
                Coordinate(50.0, 49.9, 0.0),
                Coordinate(49.9, 49.9, 0.0),
                Coordinate(49.9, 50.0, 0.0),
            ],
            Coordinate(49.9, 49.8, 0.0),
            [
                Coordinate(50.0, 49.9, 0.0),
                Coordinate(50.1, 49.9, 0.0),
                Coordinate(49.9, 50.0, 0.0),
                Coordinate(49.9, 49.9, 0.0),
            ],
        ),
    ]
)
def test_sort_clockwise_sweep_with_home(waypoints, home, expected):
    assert sort_clockwise_sweep(waypoints, home) == expected
    

def test_sort_clockwise_sweep_home_is_on_centroid():
    waypoints = [
        Coordinate(-0.1, 0.1, 0.0),
        Coordinate(0.1, 0.1, 0.0),
        Coordinate(-0.1, -0.1, 0.0),
        Coordinate(0.1, -0.1, 0.0),
    ]
    home = Coordinate(0.0, 0.0, 0.0)
    expected = [
        Coordinate(0.1, 0.1, 0.0),
        Coordinate(-0.1, 0.1, 0.0),
        Coordinate(-0.1, -0.1, 0.0),
        Coordinate(0.1, -0.1, 0.0),
    ]
    assert sort_clockwise_sweep(waypoints, home) == expected

def test_sort_clockwise_sweep_waypoints_same_direction():
    waypoints = [
        Coordinate(0.0, 3.0, 0.0),
        Coordinate(0.0, -0.3, 0.0),
        Coordinate(0.0, -0.2, 0.0),
        Coordinate(0.0, -0.1, 0.0)
    ]
    expected = [
        Coordinate(0.0, 3.0, 0.0),
        Coordinate(0.0, -0.1, 0.0),
        Coordinate(0.0, -0.2, 0.0),
        Coordinate(0.0, -0.3, 0.0)
    ]
    assert sort_clockwise_sweep(waypoints) == expected

def test_parsed_coordinate_is_frozen(tmp_path):
    text = """
        waypoints:
        - {lat: 1, lon: 1, alt: 0}
    """
    path = write_to_tmp_waypoints_file(tmp_path, text)
    _, waypoints = parse_waypoints_file(path)
    with pytest.raises(FrozenInstanceError):
        waypoints[0].lat = 10
    
def test_placeholder():
    # TODO(bootcamper): delete this and write real tests. It's only here so
    # linter doesn't complain about unused imports before you start.
    assert callable(east_north_coordinate_offset_m)
    assert callable(parse_waypoints_file)
    assert callable(sort_clockwise_sweep)
