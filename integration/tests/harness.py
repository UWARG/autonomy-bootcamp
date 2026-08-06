"""Runs the Part 5 mission from your machine. Docker only, no ROS.

Starts the containers, waits, and hands back the result dict the engine
wrote to ``/results/mission_result.json``::

    {
      "result": "success" | "failure",
      "mode": "smoke" | "perception",
      "phases": {phase_name: bool, ...},   # in the order they happen
      "captures": int,
      "waypoints_visited": int,
      "duration_s": float,
      "detail": str
    }

``MISSION_MODE`` and ``MISSION_TIMEOUT_S`` go to the engine, which is
expected to give up and exit on its own. The timeout here is only a last
resort for a stuck container.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

INTEGRATION_DIR = Path(__file__).resolve().parents[1]
COMPOSE_FILE = INTEGRATION_DIR / "compose.yaml"

#: Where the airside engine writes its result inside the shared volume.
RESULT_PATH = "/results/mission_result.json"

#: Extra time on top of the mission's own limit, to cover downloading and
#: building images and starting containers. The first run has to build the
#: airside image, which can take several minutes.
SETUP_GRACE_S = 600.0

#: How long to let ``docker compose down -v`` run before giving up on it.
TEARDOWN_TIMEOUT_S = 120.0


def run_mission(mode: str, timeout_s: float = 240.0) -> dict[str, Any]:
    """Run one mission and give back what was in ``mission_result.json``.

    Always returns a result dict, never raises: if the run times out or
    writes nothing usable, we make a failure dict with the same fields and
    put the details under ``"harness"``, so tests assert one way.

    Args:
        mode: ``"smoke"`` or ``"perception"``.
        timeout_s: The engine's own budget. Ours is that plus
            ``SETUP_GRACE_S``.
    """
    env = os.environ.copy()
    env["MISSION_MODE"] = mode
    env["MISSION_TIMEOUT_S"] = str(int(timeout_s))
    wall_clock_s = timeout_s + SETUP_GRACE_S

    try:
        try:
            up = _compose(
                [
                    "up",
                    "--build",
                    "--abort-on-container-exit",
                    "--exit-code-from",
                    "airside",
                ],
                env=env,
                timeout=wall_clock_s,
            )
        except subprocess.TimeoutExpired:
            _dump_logs(env)
            return _failure(
                mode,
                f"harness wall-clock timeout after {wall_clock_s:.0f}s "
                f"(mission budget {timeout_s:.0f}s); the stack was killed",
                duration_s=wall_clock_s,
                harness={"timed_out": True, "logs_tail": _logs_tail(env)},
            )

        result, problem = _read_result(env)
        if result is not None:
            if result.get("result") != "success":
                _dump_logs(env)
            return result
        _dump_logs(env)
        return _failure(
            mode,
            f"airside exited (code {up.returncode}) but no usable "
            f"{RESULT_PATH} was found: {problem}",
            harness={
                "returncode": up.returncode,
                "stdout_tail": up.stdout[-2000:],
                "stderr_tail": up.stderr[-2000:],
            },
        )
    finally:
        try:
            _compose(["down", "-v", "--remove-orphans"], env=env, timeout=TEARDOWN_TIMEOUT_S)
        except (OSError, subprocess.TimeoutExpired):
            pass  # Shutting down is a nice-to-have. Never hide the result.


def _compose(
    args: list[str],
    *,
    env: dict[str, str],
    timeout: float,
) -> subprocess.CompletedProcess[str]:
    """Run ``docker compose`` using this project's compose file."""
    return subprocess.run(
        ["docker", "compose", "-f", str(COMPOSE_FILE), *args],
        cwd=INTEGRATION_DIR,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _read_result(env: dict[str, str]) -> tuple[dict[str, Any] | None, str]:
    """Copy the result file out of the airside container and read it.

    Has to run between ``up`` and ``down``: ``cp`` works on a stopped
    container but not a deleted one.

    Returns:
        ``(result, "")`` if it worked, otherwise ``(None, what went wrong)``.
    """
    with tempfile.TemporaryDirectory() as tmp:
        dest = Path(tmp) / "mission_result.json"
        try:
            cp = _compose(["cp", f"airside:{RESULT_PATH}", str(dest)], env=env, timeout=60)
        except subprocess.TimeoutExpired:
            return None, "docker compose cp timed out"
        if cp.returncode != 0:
            return None, f"docker compose cp failed: {cp.stderr.strip()[-500:]}"
        if not dest.is_file():
            return None, "result file missing after copy"
        try:
            result = json.loads(dest.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            return None, f"result file unreadable: {exc}"
        if not isinstance(result, dict):
            return None, f"result JSON is not an object: {type(result).__name__}"
        return result, ""


def _failure(
    mode: str,
    detail: str,
    *,
    duration_s: float = 0.0,
    harness: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a failure result with the same fields the engine uses.

    Same shape means the tests never have to special-case a run that died
    before the engine could report anything.
    """
    return {
        "result": "failure",
        "mode": mode,
        "phases": {},
        "captures": 0,
        "waypoints_visited": 0,
        "duration_s": duration_s,
        "detail": detail,
        "harness": harness or {},
    }


def _logs_tail(env: dict[str, str], lines: int = 120) -> str:
    """Grab the end of the logs, so a timeout says something useful."""
    try:
        proc = _compose(["logs", "--no-color", f"--tail={lines}"], env=env, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return "<unavailable>"
    return proc.stdout[-8000:]


def _dump_logs(env: dict[str, str]) -> None:
    """Write the full logs to ``compose-logs.txt`` so CI can keep them.

    Has to run before shutdown, since ``down -v`` takes the logs with it.
    Failures here are swallowed: diagnostics must never hide the result.
    """
    try:
        proc = _compose(["logs", "--no-color"], env=env, timeout=60)
        (INTEGRATION_DIR / "compose-logs.txt").write_text(proc.stdout + proc.stderr)
    except (OSError, subprocess.TimeoutExpired):
        pass
