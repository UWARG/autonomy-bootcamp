"""Dataclasses used by more than one bootcamp project."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Coordinate:
    """A latitude/longitude coordinate with relative altitude."""

    lat: float
    lon: float
    alt: float  # Relative altitude in meters

    def __str__(self) -> str:
        return f"({self.lat}, {self.lon}, {self.alt})"
