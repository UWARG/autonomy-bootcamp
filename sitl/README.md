# sitl

The simulated drone used in Part 5. It's [ArduCopter SITL](https://ardupilot.org/dev/docs/sitl-simulator-software-in-the-loop.html) (software-in-the-loop) in a container: a fake quadcopter that takes off from a fixed spot in Waterloo, always with the same settings, at a speed you can turn up or down.

**You don't need to build or run this during the bootcamp.** `integration/compose.yaml` downloads a prebuilt image. This folder exists so anyone can rebuild that image and see how it's set up.

## What's in the image

- ArduPilot at the **`Copter-4.5.7`** tag, built for the `sitl` board in a build stage on `ubuntu:22.04`.
- A small runtime stage with just the `arducopter` binary and two settings files:
  - `copter.parm`: ArduPilot's own default settings, taken from the same tag.
  - `config/bootcamp.parm`: our overrides, with comments explaining each one (no RC failsafes, no flaky pre-arm checks, disarms quickly after landing, no logging).
- It runs the `quad` model straight from the `arducopter` binary. There's no `sim_vehicle.py` and no MAVProxy in the image.

MAVLink comes out on **TCP port 5760** (SITL `serial0`, device string `tcp:0:wait`). The `:wait` part means the simulation doesn't start until something connects to it, so no simulated time gets wasted while the airside container is still booting. One quirk of SITL device strings: for small `N`, `tcp:N` means port `5760 + N` (anything above 1000 is treated as an actual port number), and `tcp:0:wait` is ArduPilot's default for `serial0`.

### Environment variables

| Variable      | Default                     | Meaning                                    |
| ------------- | --------------------------- | ------------------------------------------ |
| `SIM_SPEEDUP` | `2`                               | How much faster than real time to run      |
| `SIM_HOME`    | `43.4338267,-80.5773236,336,90`   | `lat,lon,alt_m,heading_deg` starting point |

`SIM_HOME` has to be within a few hundred meters of the waypoints in `airside/src/engine/config/waypoints.yaml`. If the drone starts too far away, `FlyToWaypoint` gives up after 60 s. The default is exactly the `home` from that file, and `integration/compose.yaml` sets it again explicitly so the two stay in sync even if the published image changes.

## Build, publish, pin

- **Build it yourself** (only needed if you're changing the image):

  ```bash
  warg run sitl build          # docker build -t bootcamp-sitl .
  ```

  The first build takes a long time. It clones ArduPilot with all its submodules and compiles the simulator.

- **Publishing:** pushing a tag runs `.github/workflows/publish-sitl-image.yml`, which builds this Dockerfile and pushes `ghcr.io/uwarg/bootcamp-sitl:<version>`.

- **Pinning:** `integration/compose.yaml` points at `ghcr.io/uwarg/bootcamp-sitl:0.1.0`. Once the image is published, swap that tag for the image **digest** (`ghcr.io/uwarg/bootcamp-sitl@sha256:...`) so everyone runs the exact same bytes.

## Running it on its own

Useful if you're debugging the simulator itself:

```bash
docker run --rm -p 5760:5760 ghcr.io/uwarg/bootcamp-sitl:0.1.0
# or, if you built it locally:
docker run --rm -p 5760:5760 -e SIM_SPEEDUP=1 bootcamp-sitl
```

The container sits there until a MAVLink client connects. From your machine you can point any ground station at `tcp://127.0.0.1:5760`, for example:

```bash
mavproxy.py --master tcp:127.0.0.1:5760
```

The `-w` flag wipes the simulated drone's saved settings on every start, so every run begins from the same state. Nothing carries over between runs.
