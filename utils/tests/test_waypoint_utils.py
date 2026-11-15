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



#bad data ---------------------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            """
            home:
              - 1
              - 2
            waypoints:
              - {lat: 4, lon: 5, alt: 6}
              - {lat: 7, lon: 8, alt: 9}
            """,
            (None, [Coordinate(4, 5, 6), Coordinate(7, 8, 9)]),
        ),
        (
            """

            home: {lat: 1, lon: 2, alt: 3}

            waypoints:
              - {lat: 4, alt: 6}
              - {lon: 4, alt: 3}
            """,
            (Coordinate(1, 2, 3), [Coordinate(4, 5, 6)]),
        ),
        (
            """

            home: {lat: 1, lon: 2, alt: 3}

            waypoints:
              - {lat: 4, lon: 3, alt: 6}
              - {lat: "bob", lon: 4, alt: 3}
            """,
            (Coordinate(1, 2, 3), [Coordinate(4, 5, 6)]),
        ),
        (
            """
           
            home: {lat: 1, lon: 2, alt: 3

            waypoints:
              - {lat: 4, lon: 3, alt: 6}
              - {lat: "bob", lon: 4, alt: 3}
            """,
           (Coordinate(1, 2, 3), [Coordinate(4, 5, 6)])
        ),
        (
            """
waypoints:
- {lat: 4, lon: 5}
""",
            "gurt"
		)
       
        
    ],
    ids=["home-not-dict", "waypoints-missing-data", "lat-is-string", "invalid code", "no alt"],
)
def test_bad_data(tmp_path, text, expected):
	path = write_to_tmp_waypoints_file(tmp_path, text)
	with pytest.raises(ValueError):
		parse_waypoints_file(path)


def test_nonexistent_path():
	path = "lalala"
	with pytest.raises(OSError):
		parse_waypoints_file(path)






# bad coordinates --------------------------------------------------------------------

@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            """
            home: {lat: 1, lon: 2, alt: 3}
            waypoints:
              - {lat: 91, lon: 5, alt: 6}
            """,
            (Coordinate(1, 2, 3), [Coordinate(4, 5, 6)]),
        ),
        (
            """
            
            waypoints:
              - {lat: 4, lon: 5, alt: 6}
              - {lat: 7, lon: -181, alt: 9}
            """,
            (None, [Coordinate(4, 5, 6), Coordinate(7, 8, 9)]),
        ),
 
    ],
    ids=["lat-too-big", "lon-too-big", ],
)
def test_out_of_range_lat_lon(tmp_path, text, expected):
	path = write_to_tmp_waypoints_file(tmp_path, text)
	with pytest.raises(ValueError):
		parse_waypoints_file(path)



@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "",
            (None, []),
        ),
        (
            """
            
            waypoints: []
              
            """,
            (None, []),
        ),
 
    ],
    ids=["empty-yaml", "empty-waypoints-list", ],
)
def test_empty_stuff(tmp_path, text, expected):
	path = write_to_tmp_waypoints_file(tmp_path, text)
	assert parse_waypoints_file(path) == expected




@pytest.mark.parametrize(
    ("from_lat", "from_lon", "to_lat", "to_lon", "expected"),
    [
   
        (0, 0, 0, 1, (111195.08, 0.0)),
        (0, 0, 1, 0, (0.0, 111195.08)),
        (60, 0, 60, 1, (55597.54, 0.0)),

    ],
    ids=["east-only-offset", "north-only-offset", "east with scale"],
)
def test_coordinate_offset(from_lat, from_lon, to_lat, to_lon, expected):
	result = east_north_coordinate_offset_m(from_lat, from_lon, to_lat, to_lon)
	assert result == pytest.approx(expected, abs=0.1)


NORTH = Coordinate(1, 0, 0)
EAST = Coordinate(0, 1, 0)
SOUTH = Coordinate(-1, 0, 0)
WEST = Coordinate(0, -1, 0)


def test_sort_clockwise_sweep_starts_north_and_goes_clockwise():


    result = sort_clockwise_sweep([SOUTH, WEST, NORTH, EAST])

    assert result == [NORTH, EAST, SOUTH, WEST]

def test_sort_clockwise_sweep_trivial_lists():
    assert sort_clockwise_sweep([]) == []
    assert sort_clockwise_sweep([NORTH]) == [NORTH]

	



@pytest.mark.parametrize(
    ("home", "expected"),
    [
        (Coordinate(0, 5, 0), [EAST, SOUTH, WEST, NORTH]),
        (Coordinate(-5, 0, 0), [SOUTH, WEST, NORTH, EAST]),
        (Coordinate(0, 0, 0), [NORTH, EAST, SOUTH, WEST]),
    ],
    ids=["home-east", "home-south", "home-on-centroid"],
)
def test_sort_clockwise_sweep_with_home(home, expected):
    result = sort_clockwise_sweep([SOUTH, WEST, NORTH, EAST], home=home)
    assert result == expected


def test_sort_clockwise_sweep_closer_waypoint_first_on_same_bearing():
    near_north = Coordinate(1, 0, 0)
    far_north = Coordinate(2, 0, 0)
    far_south = Coordinate(-3, 0, 0)

    result = sort_clockwise_sweep([far_south, far_north, near_north])

    assert result == [near_north, far_north, far_south]


import dataclasses


def test_parsed_coordinates_are_frozen(tmp_path):
    path = write_to_tmp_waypoints_file(
        tmp_path,
        """
        waypoints:
          - {lat: 4, lon: 5, alt: 6}
        """,
    )
    _, waypoints = parse_waypoints_file(path)

    with pytest.raises(dataclasses.FrozenInstanceError):
        waypoints[0].lat = 99