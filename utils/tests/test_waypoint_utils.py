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

from src.waypoint_utils import (
    east_north_coordinate_offset_m,
    parse_waypoints_file,
    sort_clockwise_sweep,
)


def test_placeholder():
    # TODO(bootcamper): delete this and write real tests. It's only here so
    # pytest doesn't complain about an empty file before you start.
    assert callable(east_north_coordinate_offset_m)
    assert callable(parse_waypoints_file)
    assert callable(sort_clockwise_sweep)
