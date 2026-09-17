# Airside engine image: ROS 2 Humble + MAVROS + py_trees.
#
# Deliberately simpler than the monorepo image: no colcon workspace, no
# rosbridge/rosapi. The repo is copied to /bootcamp and put on PYTHONPATH,
# and the launch file is run by absolute path.
#
# Build context is the repo root (see integration/compose.yaml):
#   docker build -f airside/docker/Dockerfile .

FROM ros:humble-ros-base

ENV DEBIAN_FRONTEND=noninteractive

# Pin the ROS apt repo to a dated snapshot instead of the rolling index.
# The live Humble index dropped ros-humble-mavros and ros-humble-mavros-extras
# in the September 2026 sync, and a rolling index means the image can break
# between one bootcamper's build and the next. The base image ships the live
# repo as a deb822 stanza symlinked into /etc/apt/sources.list.d; this
# replaces it. Snapshots are signed by a different key than the live repo
# (ROS Snapshot builder, fingerprint 4B63CF8FDE49746E98FA01DDAD19BAB3CBF125EA),
# vendored next to this file.
ARG ROS_SNAPSHOT=2026-08-07
COPY airside/docker/ros-snapshots.asc /usr/share/keyrings/ros-snapshots.asc
RUN rm /etc/apt/sources.list.d/ros2.sources \
  && printf 'Types: deb\nURIs: http://snapshots.ros.org/humble/%s/ubuntu\nSuites: jammy\nComponents: main\nSigned-By: /usr/share/keyrings/ros-snapshots.asc\n' \
     "${ROS_SNAPSHOT}" > /etc/apt/sources.list.d/ros2-snapshot.sources

RUN apt-get update && apt-get install -y --no-install-recommends \
    geographiclib-tools \
    python3-pip \
    ros-humble-mavros \
    ros-humble-mavros-extras \
    ros-humble-py-trees \
    ros-humble-py-trees-ros \
  && rm -rf /var/lib/apt/lists/*

# MAVROS needs the GeographicLib datasets for global-frame conversions.
RUN /opt/ros/humble/lib/mavros/install_geographiclib_datasets.sh

RUN pip install --no-cache-dir numpy pyyaml

COPY . /bootcamp

# Repo root: cross-project imports (camera.src.sim, utils.src.waypoint_utils).
# airside/src: the engine package itself (engine.ros.mission, ...).
ENV PYTHONPATH=/bootcamp:/bootcamp/airside/src

COPY airside/docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["ros2", "launch", "/bootcamp/airside/src/engine/launch/perception.launch.py"]
