"""
Example tests for :class:`src.types.Coordinate`.

Finished, and the model for what you write in ``test_waypoint_utils.py``:
one small fixture, one test per behavior named after it, ``pytest.approx``
for decimals, ``pytest.raises`` for failures.
"""

import dataclasses

import pytest

from src.types import Coordinate


@pytest.fixture
def coordinate() -> Coordinate:
    """A coordinate near the University of Waterloo campus."""
    return Coordinate(43.4717, -80.5414, 15.0)


def test_construction_and_field_access(coordinate):
    assert coordinate.lat == pytest.approx(43.4717)
    assert coordinate.lon == pytest.approx(-80.5414)
    assert coordinate.alt == pytest.approx(15.0)


def test_coordinates_with_equal_fields_are_equal(coordinate):
    assert coordinate == Coordinate(43.4717, -80.5414, 15.0)
    assert coordinate != Coordinate(43.4717, -80.5414, 20.0)


def test_coordinate_is_immutable(coordinate):
    # Coordinate is a frozen dataclass, so anyone holding one knows it will
    # never change on them. Assigning to any field has to raise.
    with pytest.raises(dataclasses.FrozenInstanceError):
        coordinate.lat = 0.0

    with pytest.raises(dataclasses.FrozenInstanceError):
        coordinate.alt = 100.0


def test_str_is_human_readable(coordinate):
    assert str(coordinate) == "(43.4717, -80.5414, 15.0)"
