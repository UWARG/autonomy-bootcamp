"""
Starts MAVROS and the mission. Container only.

Launched by full path, with the mission running as a plain Python module,
so there's no colcon workspace to build. When the mission exits everything
else goes with it, which is how the container stops on its own.
"""

import os

from launch import LaunchDescription
from launch.actions import EmitEvent, ExecuteProcess, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch_ros.actions import Node

_FCU_URL = os.environ.get("FCU_URL", "tcp://sitl:5760")


def generate_launch_description() -> LaunchDescription:
    mission = ExecuteProcess(
        cmd=["python3", "-m", "engine.ros.mission"],
        name="mission",
        output="screen",
    )

    return LaunchDescription(
        [
            Node(
                package="mavros",
                executable="mavros_node",
                namespace="mavros",
                output="both",
                respawn=True,
                respawn_delay=2.0,
                parameters=[
                    {
                        "fcu_url": _FCU_URL,
                        "fcu_protocol": "v2.0",
                        "tgt_system": 1,
                        "tgt_component": 1,
                    }
                ],
            ),
            mission,
            RegisterEventHandler(
                OnProcessExit(
                    target_action=mission,
                    on_exit=[EmitEvent(event=Shutdown(reason="mission finished"))],
                )
            ),
        ]
    )
