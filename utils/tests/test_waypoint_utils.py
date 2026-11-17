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

#start of tests

#Bad Data & Nothing to Work With
def test_parse_empty_file(tmp_path):
    path = write_to_tmp_waypoints_file(tmp_path, "") #writes an empty file and gives its path
    assert parse_waypoints_file(path) == (None, []) #checks what it gives back after you parse it


def test_empty_waypoints(tmp_path):
    path = write_to_tmp_waypoints_file(tmp_path, "waypoints: []")  #writes a file with empty waypoints
    assert parse_waypoints_file(path) == (None, [])


def test_top_level_not_mapping(tmp_path):
    path = write_to_tmp_waypoints_file(tmp_path, "labubu")  #writes a file with bad data (top level isnt mapping)
    with pytest.raises(ValueError): #no assert since the test would fail if the indented line doesnt raise the error
        parse_waypoints_file(path)


def test_waypoint_missing(tmp_path):
    path = write_to_tmp_waypoints_file(tmp_path, """ 
    waypoints: 
      - {lat: 1, lon: 2} 
    """) #use triple quotes like the example test
    with pytest.raises(ValueError):
        parse_waypoints_file(path)


def test_invalid_yaml(tmp_path):
    path = write_to_tmp_waypoints_file(tmp_path, """ waypoints: - {lat: 1, lon: 2} """)
    with pytest.raises(ValueError):
        parse_waypoints_file(path) #the code catches YAML error and turns it into a value error so we just use value error


def test_non_numeric_value(tmp_path):
    path = write_to_tmp_waypoints_file(tmp_path, """
waypoints:
  - {lat: a, lon: 2, alt: 3}
""")
    with pytest.raises(ValueError):
        parse_waypoints_file(path)

#Out of Range
def test_lat_out_of_range(tmp_path):
    path = write_to_tmp_waypoints_file(tmp_path, """
waypoints:
  - {lat: 91, lon: 2, alt: 3}
""")
    with pytest.raises(ValueError):
        parse_waypoints_file(path)


def test_lon_out_of_range(tmp_path):
    path = write_to_tmp_waypoints_file(tmp_path, """
waypoints:
  - {lat: 1, lon: 181, alt: 3}
""")
    with pytest.raises(ValueError):
        parse_waypoints_file(path)

#Bad Data
def test_file_not_found(tmp_path):
    path = tmp_path / "labubu67676767.yaml"   # a file that was never written
    with pytest.raises(OSError):
        parse_waypoints_file(path)


def test_waypoints_not_a_list(tmp_path):
    path = write_to_tmp_waypoints_file(tmp_path, "waypoints: 67")
    with pytest.raises(ValueError):
        parse_waypoints_file(path)


def test_waypoint_not_a_mapping(tmp_path):
    path = write_to_tmp_waypoints_file(tmp_path, "waypoints: [67]")
    with pytest.raises(ValueError):
        parse_waypoints_file(path)

#Parsing gives Frozen Coordinates
def test_coordinate_is_frozen(tmp_path):
    path = write_to_tmp_waypoints_file(tmp_path, "waypoints: [{lat: 1, lon: 2, alt: 3}]")
    _, waypoints = parse_waypoints_file(path)   # unpack the pair into 2 variables
    with pytest.raises(AttributeError):
        waypoints[0].lat = 99                      # arbitrarily try to change the coordinate


def test_sort_empty_list():
    assert sort_clockwise_sweep([]) == []


def test_sort_one_waypoint():
    one = [Coordinate(1, 2, 3)]
    assert sort_clockwise_sweep(one) == [Coordinate(1, 2, 3)]


def test_offset_same_point():
    east, north = east_north_coordinate_offset_m(10, 20, 10, 20)
    assert east == pytest.approx(0, abs=1e-6)
    assert north == pytest.approx(0, abs=1e-6)


def test_offset_one_degree_north():
    east, north = east_north_coordinate_offset_m(0, 0, 1, 0)
    assert east == pytest.approx(0, abs=1e-6)
    assert north == pytest.approx(111195.0802335329, abs=1) #number comes from math.radians(1) * 6371008.8


def test_offset_one_degree_east_at_60():
    east, north = east_north_coordinate_offset_m(60, 0, 60, 1)
    assert east == pytest.approx(55597.54011676645, abs=1)
    assert north == pytest.approx(0, abs=1e-6)

#Ordering
def test_sort_clockwise_from_north():
    n = Coordinate(1, 0, 0)
    e = Coordinate(0, 1, 0)
    s = Coordinate(-1, 0, 0)
    w = Coordinate(0, -1, 0)
    shuffled = [s, w, n, e]
    assert sort_clockwise_sweep(shuffled) == [n, e, s, w] #clockwise should shuffle so that everything is in order from north

#Ordering with a Home
def test_sort_starts_toward_home():
    n = Coordinate(1, 0, 0)
    e = Coordinate(0, 1, 0)
    s = Coordinate(-1, 0, 0)
    w = Coordinate(0, -1, 0)
    home = Coordinate(0, 5, 0)   # far to the east of the centre (0, 0)
    assert sort_clockwise_sweep([s, w, n, e], home=home) == [e, s, w, n] #given a home the order should start at the home and sweep clockwise

def test_sort_starts_toward_north_home_centre():
    n = Coordinate(1, 0, 0)
    e = Coordinate(0, 1, 0)
    s = Coordinate(-1, 0, 0)
    w = Coordinate(0, -1, 0)
    home = Coordinate(0, 0, 0)   # far to the east of the centre (0, 0)
    assert sort_clockwise_sweep([s, w, n, e], home=home) == [n, e, s, w] #given a home the order should start at the home and sweep clockwise

#Two Waypoints in the same directions
def test_sort_same_direction_closer_first():
    near = Coordinate(1, 0, 0)
    far = Coordinate(2, 0, 0)
    south = Coordinate(-3, 0, 0)
    assert sort_clockwise_sweep([far, south, near]) == [near, far, south]