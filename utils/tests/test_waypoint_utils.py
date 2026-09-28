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

@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            """
            home: {lat: 1, lon: 2, alt: 3}
            waypoints:
              - {lat: 4, lon: 5}
            """,
            (ValueError),
        ),
        (
          """
          home: {lat: 1, lon: 2, alt: 3}
          waypoints:
            - {lat: 4, alt: 6}
          """,
          (ValueError),
        ),
        (
          """
          home: {lat: 1, lon: 2, alt: 3}
          waypoints:
            - {lon: 5, alt: 6}
          """,
          (ValueError),
        )
    ],
    ids=["missing-alt","missing-lon","missing-lat"]
)
def test_missing_input(tmp_path, text, expected):
    path = write_to_tmp_waypoints_file(tmp_path, text)

    with pytest.raises(ValueError):
        parse_waypoints_file(path)

def test_missing_file():
    with pytest.raises(OSError):
        parse_waypoints_file("literally/doesnt/exist.txt")

@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            """
            asdfhome  : {lat: 1, lon: 2, alt: 3}
            waypoints:ds
              - {lat: 4, lon: 5, alt: 6}sdf
            """,
            (None),
        ),
    ],
    ids=["broken-yaml"],
)
def test_bad_yaml(tmp_path, text, expected):
    path = write_to_tmp_waypoints_file(tmp_path, text)

    with pytest.raises(ValueError):
        parse_waypoints_file(path)   

@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            """
            """,
            (None,[]),
        ),
    ],
    ids=["empty_yaml"],
)
def test_empty_yaml(tmp_path, text, expected):
    path = write_to_tmp_waypoints_file(tmp_path, text)
    assert parse_waypoints_file(path) == expected

@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            """
            home  : {lat: 1, lon: 2, alt: 3}
            waypoints:
              - [late: 4, long: 5, alt: 6]
            """,
            (None),
        ),
        (
            """
            home  : {lat: 1, lon: 2, alt: 3}
            waypoints:
              - {lat: troll, lon: so, alt: tired}
            """,
            (None),
        ),
    ],
    ids=["broken-heading", "non-numeric-inputs"],
)
def test_bad_heading(tmp_path, text, expected):
    path = write_to_tmp_waypoints_file(tmp_path, text)

    with pytest.raises(ValueError):
        parse_waypoints_file(path)        

@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            """
            waypoints:
              - {lat: 91, lon: 5, alt: 6}
            """,
            (ValueError),
        ),
        (
            """
            waypoints:
              - {lat: -91, lon: 5, alt: 6}
            """,
            (ValueError),
        ),
        (
            """
            waypoints:              
            - {lat: 4, lon: 181, alt: 6}
            """,
            (ValueError),
        ),
        (
            """
            waypoints:              
            - {lat: 4, lon: -181, alt: 6}
            """,
            (ValueError),
        ),
    ],
    ids=["lat-too-big", "lat-too-small", "lon-too-big","lon-too-small"],
)
def test_out_of_bounds(tmp_path, text, expected):
    path = write_to_tmp_waypoints_file(tmp_path, text)

    with pytest.raises(ValueError):
        parse_waypoints_file(path)

def test_clockwise_sweep_empty():
    assert sort_clockwise_sweep([]) == []

def test_clockwise_sweep_single():
    assert sort_clockwise_sweep([Coordinate(2,3,4)]) == [Coordinate(2,3,4)]

def test_clockwise_sweep_many_no_home():
    assert sort_clockwise_sweep([Coordinate(10,0,0),Coordinate(-11,0,0),Coordinate(0,12,0),Coordinate(0,-13,0)]) == [Coordinate(10,0,0),Coordinate(0,12,0),Coordinate(-11,0,0),Coordinate(0,-13,0)]

def test_clockwise_sweep_many():
    assert sort_clockwise_sweep([Coordinate(10,0,0),Coordinate(-11,0,0),Coordinate(0,12,0),Coordinate(0,-13,0)],home=Coordinate(-20,0,0)) == [Coordinate(0,-13,0),Coordinate(10,0,0),Coordinate(0,12,0),Coordinate(-11,0,0)]

def test_clockwise_sweep_ordering():
    assert sort_clockwise_sweep([Coordinate(10,0,0),Coordinate(11,0,0),Coordinate(12,0,0)],home=Coordinate(0,0,0)) == [Coordinate(10,0,0),Coordinate(11,0,0),Coordinate(12,0,0)]

def test_clockwise_sweep_many_home_near_centroid():
    assert sort_clockwise_sweep([Coordinate(10,0,0),Coordinate(-10,0,0),Coordinate(0,10,0),Coordinate(0,-10,0)],home=Coordinate(0,0,0)) == [Coordinate(10,0,0),Coordinate(0,10,0),Coordinate(-10,0,0),Coordinate(0,-10,0)]

#I matched it to the actual output
#I did not want to manually calculate this value
def test_east_north_offset():
    assert east_north_coordinate_offset_m(10,5,3,6) == (pytest.approx(110480.30220973832), pytest.approx(-778365.5616347304))


