"""A quick check that the simulator boots and the drone can fly at all.

Arm, take off, land, disarm. No waypoints, no pictures. It runs first so a
broken setup (Docker, images, MAVLink) fails here instead of looking like a
bug in your behavior tree::

    warg run integration smoke
"""

from tests.harness import run_mission

#: How long arm, take off, land, and disarm get before we call it failed.
SMOKE_BUDGET_S = 180.0

#: The steps that all have to finish for this check to pass.
REQUIRED_PHASES = ("arm", "takeoff", "land")


def test_smoke_mission_succeeds() -> None:
    result = run_mission("smoke", timeout_s=SMOKE_BUDGET_S)

    assert result.get("result") == "success", result
    assert result.get("mode") == "smoke", result

    phases = result.get("phases", {})
    for phase in REQUIRED_PHASES:
        assert phases.get(phase) is True, (f"phase {phase!r} did not pass", result)

    duration_s = result.get("duration_s", -1.0)
    assert 0 < duration_s <= SMOKE_BUDGET_S, result
