#!/usr/bin/env bash
# Sources the ROS environment, then runs the container command.

set -eo pipefail

source /opt/ros/humble/setup.bash

exec "$@"
