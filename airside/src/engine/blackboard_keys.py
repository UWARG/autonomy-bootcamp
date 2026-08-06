"""
Every blackboard key name, in one place.

Import from here instead of typing the string. A typo then fails at import
instead of quietly reading a key that doesn't exist.

Written by ``engine/ros/`` in the container, by fixtures in the tests.
"""

# True once MAVROS has heard from the drone's flight controller.
VEHICLE_CONNECTED = "vehicle/connected"

# True while the drone is armed, meaning the motors can spin.
VEHICLE_ARMED = "vehicle/armed"

# The flight mode the drone is in, like "GUIDED" or "STABILIZE".
VEHICLE_MODE = "vehicle/mode"

# The newest CameraFrame from the camera, or None if nothing has been
# captured yet.
LATEST_FRAME = "camera/latest_frame"

# The waypoints to visit, in order (Coordinate objects from utils).
WAYPOINTS = "mission/waypoints"

# How many waypoints we've reached, which is also the index of the next one.
WAYPOINT_INDEX = "mission/waypoint_index"
