"""
TODO(bootcamper): write the tests for ``src/waypoint_utils.py`` in here.

``tests/test_coordinate.py`` shows the style. Cover at least these:

- Files that parse fine: with and without ``home``, and files with
  comments and blank lines in them.
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

import dataclasses

import pytest

from src.types import Coordinate
from src.waypoint_utils import (
    east_north_coordinate_offset_m,
    parse_waypoints_file,
    sort_clockwise_sweep,
)


def w(tmp_path, text):
    path = tmp_path / "wp.yaml"
    path.write_text(text)
    return path


N, E, S, W = (
    Coordinate(1, 0, 0),
    Coordinate(0, 1, 0),
    Coordinate(-1, 0, 0),
    Coordinate(0, -1, 0),
)


def test_enu():
    assert east_north_coordinate_offset_m(0, 0, 1, 0) == pytest.approx(
        (0, 111195.0802335329)
    )
    assert east_north_coordinate_offset_m(60, 0, 60, 1) == pytest.approx(
        (55597.54011676647, 0)
    )


def test_parse_ok(tmp_path):
    home, waypoints = parse_waypoints_file(
        w(
            tmp_path,
            """
            # a lap

            home: {lat: 1, lon: 2, alt: 3}
            waypoints:
              - {lat: 4, lon: 5, alt: 6}
            """,
        )
    )
    assert home == Coordinate(1, 2, 3)
    assert waypoints == [Coordinate(4, 5, 6)]
    with pytest.raises(dataclasses.FrozenInstanceError):
        waypoints[0].lat = 0


def test_parse_empty(tmp_path):
    assert parse_waypoints_file(w(tmp_path, "")) == (None, [])
    assert parse_waypoints_file(w(tmp_path, "waypoints: []")) == (None, [])


@pytest.mark.parametrize(
    "text",
    [
        "- {lat: 1, lon: 2, alt: 3}",  # top level isn't a mapping
        "waypoints: 5",  # not a list
        "waypoints: [nope]",  # waypoint isn't a mapping
        "waypoints: [{lat: 1, lon: 2}]",  # no alt
        "waypoints: [{lat: x, lon: 2, alt: 3}]",  # not a number
        "waypoints: [{lat: 91, lon: 2, alt: 3}]",  # lat out of range
        "waypoints: [{lat: 1, lon: 181, alt: 3}]",  # lon out of range
        "waypoints: [oops",  # not YAML
    ],
)
def test_parse_bad(tmp_path, text):
    with pytest.raises(ValueError):
        parse_waypoints_file(w(tmp_path, text))


def test_parse_missing_file(tmp_path):
    with pytest.raises(OSError):
        parse_waypoints_file(tmp_path / "nope.yaml")


def test_sort():
    assert sort_clockwise_sweep([]) == []
    assert sort_clockwise_sweep([N]) == [N]
    # clockwise from north, from home's direction, north again when home is
    # sitting on the centroid
    assert sort_clockwise_sweep([W, S, E, N]) == [N, E, S, W]
    assert sort_clockwise_sweep([W, S, E, N], Coordinate(0, 5, 0)) == [E, S, W, N]
    assert sort_clockwise_sweep([W, S, E, N], Coordinate(0, 0, 0)) == [N, E, S, W]
    # same bearing, closer first
    near, far, opposite = Coordinate(1, 0, 0), Coordinate(2, 0, 0), Coordinate(-3, 0, 0)
    assert sort_clockwise_sweep([far, opposite, near]) == [near, far, opposite]
