# Part 5: Run on ROS 2 and SITL

This is where everything you built gets used at once. Your `SimCamera` (Part 2), the waypoint utilities (Part 3), and your behavior tree (Part 4) fly a whole mission against a simulated ArduPilot drone. There's no code to write here, you just run it and get it passing.

## Prerequisites

- Docker running, with about 5 GB free for docker images.
- Parts 2–4 passing on your laptop (`warg run camera test`, `warg run utils test`, `warg run airside test`).

## Run it

```bash
warg up integration            # check out this project (and sitl/, airside/, ...)
warg run integration setup     # test tooling (uv) + build both docker images

warg run integration smoke     # check 1: the simulator boots, MAVROS connects,
                               #          arm -> takeoff -> land -> disarm
warg run integration test      # check 2: the full mission with your camera code
```

The first run is slow. It downloads the simulator image and builds the airside image (ROS 2 Humble + MAVROS) with your code in it. Later runs reuse what's already built and take a few minutes.

## What actually happens

`tests/harness.py` runs `compose.yaml` with `docker compose`:

1. The `sitl` container starts the simulated drone, built from `sitl/`. It listens on TCP port 5760, and nothing else starts until Docker confirms it's up.
2. The `airside` container gets built from `airside/docker/Dockerfile`, using the **repo root** as the build context so `camera/` and `utils/` get copied in too. It then starts the engine, which connects MAVROS to `tcp://sitl:5760`.
3. The mission runs on its own. Every step checks the drone's actual status and moves on when it's ready.
   - **smoke** mode: arm, take off, land, disarm.
   - **perception** mode (the normal one): arm, take off, fly to the three waypoints in `airside/src/engine/config/waypoints.yaml`, take a picture at each one with your `SimCamera` and `CaptureForPerception`, send them out on `/perception/image` and `/perception/status`, land, disarm.
4. When it's done the engine writes `/results/mission_result.json` to a shared volume. The test harness copies that file out, shuts the containers down (`down -v`), and checks the JSON: did it succeed, did the steps happen in the right order, did it visit three waypoints, did it take three pictures, did it finish in time.

The harness also has its own overall time limit on top of the engine's `MISSION_TIMEOUT_S`, so a stuck container can't hang your terminal or CI forever.

## Troubleshooting

**The simulator build failed, or you want to watch it.** `warg run integration setup` hides nothing, but the ArduPilot compile is long and quiet. To build just that image with full output:

```bash
cd integration
docker compose build --progress=plain sitl
```

A build that dies part way through is usually the network (it clones ArduPilot and all its submodules) or disk space. Both are safe to retry: Docker keeps the layers it already finished. If you want to start over from nothing, `docker compose build --no-cache sitl`, and expect the full 15–30 minutes again.

**A test failed. Read the mission result first.** The failure message prints the whole result dict, including `detail` and which steps came back false. If you see a `harness` key in there, the run never produced a result file at all (usually a timeout), and the message includes the end of the logs.

**Looking at the logs.** While a run is going, or right after you start one by hand and before it shuts down:

```bash
cd integration
MISSION_MODE=smoke docker compose up --build --abort-on-container-exit --exit-code-from airside
docker compose logs airside   # the engine and MAVROS
docker compose logs sitl      # the simulator
docker compose down -v        # clean up when you're done
```

If you see `MAVROS: connection timeout` or no heartbeat, the simulator container is usually the problem. Check `docker compose ps` and the `sitl` logs.

**Timeouts.** The mission stops itself after `MISSION_TIMEOUT_S` (180 s by default in compose), and the harness allows ten minutes on top of that for builds and startup. That is nowhere near enough for a cold simulator build, which is exactly why `setup` builds the images first. If the _harness_ times out, run `warg run integration setup` and try again. If the _mission_ itself times out, look at which step came back false. Stuck before `arm` is usually a connection problem (MAVROS isn't connected, or the drone isn't in GUIDED). Stuck partway through the waypoints is usually a Part 4 bug, and you can reproduce it much faster with `warg run airside test`.

**Leftover containers or volumes.** The harness always cleans up after itself, but if you kill a manual run partway through it can leave things behind. Run `docker compose down -v --remove-orphans` from `integration/` to reset everything, including the `results` volume.
