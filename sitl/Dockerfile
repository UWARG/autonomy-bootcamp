# syntax=docker/dockerfile:1
#
# ArduCopter SITL, pinned to a known ArduPilot release.
#
# This image provides flight dynamics only: a quadcopter simulator with a
# fixed home location and a deterministic parameter set. The bootcamper's code
# never runs here -- the airside container connects to this simulator over
# MAVLink (TCP port 5760).
#
# Built locally through integration/compose.yaml (`build:
# ../sitl`), which `warg run integration setup` triggers. Nothing is
# published: the pinned ARDUPILOT_TAG below is what makes every bootcamper's
# simulator identical.

ARG ARDUPILOT_TAG=Copter-4.5.7

# ---------------------------------------------------------------------------
# Build stage: compile the SITL binary from the pinned ArduPilot tag.
# ---------------------------------------------------------------------------
FROM ubuntu:22.04 AS build

ARG ARDUPILOT_TAG
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
  ca-certificates \
  g++ \
  gawk \
  gcc \
  git \
  make \
  pkg-config \
  python3 \
  python3-dev \
  python3-pip \
  python3-setuptools \
  python3-wheel \
  && rm -rf /var/lib/apt/lists/*

# empy is pinned: 4.x is incompatible with ArduPilot's code generators.
RUN python3 -m pip install --no-cache-dir empy==3.3.4 future pexpect

# Build as a non-root user (matches upstream ArduPilot container practice).
RUN useradd --create-home builder
USER builder

RUN git clone --depth 1 --branch "${ARDUPILOT_TAG}" \
  https://github.com/ArduPilot/ardupilot.git /home/builder/ardupilot

WORKDIR /home/builder/ardupilot

# Shallow submodule fetch when the server allows it; fall back to a full
# fetch for any submodule pinned to a commit a shallow fetch cannot reach.
RUN git submodule update --init --recursive --depth 1 \
  || git submodule update --init --recursive

RUN ./waf configure --board sitl \
  && ./waf copter

# ---------------------------------------------------------------------------
# Runtime stage: just the SITL binary and its parameter files.
# ---------------------------------------------------------------------------
FROM ubuntu:22.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive

# The base image already ships libstdc++/libc; install explicitly anyway so
# the runtime dependency is visible and survives base-image slimming.
RUN apt-get update && apt-get install -y --no-install-recommends \
  libstdc++6 \
  && rm -rf /var/lib/apt/lists/*

COPY --from=build /home/builder/ardupilot/build/sitl/bin/arducopter /usr/local/bin/arducopter
# Stock SITL copter defaults from the same pinned tag, layered under our
# bootcamp overrides (see --defaults in CMD below).
COPY --from=build /home/builder/ardupilot/Tools/autotest/default_params/copter.parm /sitl/params/copter.parm
COPY config/bootcamp.parm /sitl/params/bootcamp.parm

# Non-root runtime user. arducopter writes eeprom.bin (and terrain data, if
# enabled) to its working directory, so run from a home it owns.
RUN useradd --create-home sitl
USER sitl
WORKDIR /home/sitl

# Tunables (overridden by integration/compose.yaml):
#   SIM_SPEEDUP  simulation speed multiplier
#   SIM_HOME     lat,lon,alt_m,heading_deg -- must be within a few hundred
#                meters of the waypoints in
#                airside/src/engine/config/waypoints.yaml, or FlyToWaypoint
#                times out before ever reaching the sweep area. The default
#                IS that file's home.
ENV SIM_SPEEDUP=2 \
  SIM_HOME=43.4338267,-80.5773236,336,90

# MAVLink on serial0, exposed as a TCP server.
#
# NOTE on the --serial0 device string: for SITL "tcp:N" with a small N
# means "listen on BASE_PORT + N" (base port 5760) -- that is how the stock
# defaults serial1=tcp:2 / serial2=tcp:3 become the well-known ports
# 5762/5763 (values above 1000 are taken as literal ports). "tcp:0:wait" is
# the stock serial0 default: listen on 5760, and ":wait" blocks the
# simulation until the first client connects, so no sim time elapses before
# MAVROS -- or the compose healthcheck -- attaches.
EXPOSE 5760

# -w wipes the virtual eeprom on every start so the --defaults chain is
# applied fresh: deterministic parameters on every container run.
CMD ["/bin/bash", "-c", "exec /usr/local/bin/arducopter \
  -w \
  --model quad \
  --speedup \"${SIM_SPEEDUP}\" \
  --home \"${SIM_HOME}\" \
  --defaults /sitl/params/copter.parm,/sitl/params/bootcamp.parm \
  --serial0 tcp:0:wait"]
