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
                # YAML not parseable
                home {lat: 1, lon: 2, alt: 3}
                waypoints
                  - {lat: 4}
                """,
                (ValueError)
            ),
            (
                """
                # first layer not a mapping
                - home: {lat: 1, lon: 2, alt: 3}
                - waypoints:
                  - {lat: 4}
                """,
                (ValueError)
            ),
            (
              """
              # values are not numbers
              home: {lat: a, lon: b, alt: c}
              waypoints: 
                - {lat: d}
              """,
              (ValueError)
            ),
            (
                """
                # missing 'lat', 'lon', 'alt'
                home: {lon: 2, alt: 3}
                waypoints:
                  - {lat: 4}
                """,
                (ValueError)
            ),
            (
                """
                # out of range (lat +90)
                home: {lat: 100, lon: 2, alt: 3}
                waypoints:
                  - {lat: 100}
                """,
                (ValueError)
            ),
            (
                """
                # out of range (lat -90)
                home: {lat: -100, lon: 2, alt: 3}
                waypoints:
                  - {lat: -100}
                """,
                (ValueError)
            ),
            (
                """
                # out of range (lat +180)
                home: {lat: 1, lon: 200, alt: 3}
                waypoints:
                  - {lat: 1}
                """,
                (ValueError)
            ),
            (
                """
                # out of range (lat -180)
                home: {lat: 100, lon: -200, alt: 3}
                waypoints:
                  - {lat: 1}
                """,
                (ValueError)
            )
        ],
        ids=["malformed-yaml", "first-layer-not-mapping", "values-not-numbers", "missing-field", "out-of-range-lat-plus-ninety", "out-of-range-lat-negative-ninety", "out-of-range-lon-plus-one-eighty", "out-of-range-lon-negative-one-eighty"]
)
def test_parse_waypoints_file_failure_malformed_data(tmp_path, text, expected):
    path = write_to_tmp_waypoints_file(tmp_path, text)
    with pytest.raises(expected):
      parse_waypoints_file(path)

def test_parse_waypoints_file_failure_missing_file(tmp_path):
    with pytest.raises(OSError):
        parse_waypoints_file(tmp_path / "no-path")


def test_placeholder():
    # TODO(bootcamper): delete this and write real tests. It's only here so
    # linter doesn't complain about unused imports before you start.
    assert callable(east_north_coordinate_offset_m)
    assert callable(parse_waypoints_file)
    assert callable(sort_clockwise_sweep)
