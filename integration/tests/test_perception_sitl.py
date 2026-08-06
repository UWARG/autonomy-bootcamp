"""The real Part 5 check: the whole mission flown against the simulator.

Arms, takes off, visits the three waypoints taking a picture at each, lands,
disarms. This is the gate CI runs::

    warg run integration test
"""

from tests.harness import run_mission

#: How long the full three-waypoint mission gets before we call it failed.
PERCEPTION_BUDGET_S = 300.0

#: The waypoints file has exactly three waypoints, one picture at each.
EXPECTED_WAYPOINTS = 3
EXPECTED_CAPTURES = 3

#: These steps have to show up in this order. The engine writes ``phases``
#: as it goes, so the JSON keys are in the order the steps happened.
ORDERED_PHASES = ("arm", "takeoff", "land")


def test_perception_mission_succeeds() -> None:
    result = run_mission("perception", timeout_s=PERCEPTION_BUDGET_S)

    assert result.get("result") == "success", result
    assert result.get("mode") == "perception", result

    phases = result.get("phases", {})
    assert phases, ("no phases reported", result)
    failed = [name for name, passed in phases.items() if passed is not True]
    assert not failed, (f"phases failed: {failed}", result)

    # The main steps have to be there, in the order they should happen.
    keys = list(phases)
    missing = [p for p in ORDERED_PHASES if p not in keys]
    assert not missing, (f"phases missing: {missing}", result)
    positions = [keys.index(p) for p in ORDERED_PHASES]
    assert positions == sorted(positions), (f"phases out of order: {keys}", result)
    if "disarm" in keys:
        assert keys.index("disarm") > keys.index("land"), (f"disarm before land: {keys}", result)

    assert result.get("captures") == EXPECTED_CAPTURES, result
    assert result.get("waypoints_visited") == EXPECTED_WAYPOINTS, result

    duration_s = result.get("duration_s", -1.0)
    assert 0 < duration_s <= PERCEPTION_BUDGET_S, result
