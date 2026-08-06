"""AbstractCamera: the base class every camera inherits from.

Fake, simulated, or real hardware, they all work the same way, so the code
taking pictures never has to know which one it got. Turn it on, capture as
long as you like, turn it off.

Each method below says what it does and why.
"""

import abc

from .frame import CameraFrame


class AbstractCamera(abc.ABC):
    """Base class all cameras inherit from."""

    @abc.abstractmethod
    def initialize_camera(self) -> bool:
        """Turn the camera on.

        Nothing gets opened in ``__init__``, so building a camera is always
        cheap and can't fail. This is where the real work happens, and you
        can call it again after ``stop()`` to bring the camera back.

        Returns:
            True if it worked, False if the camera could not be turned on.
        """

    @abc.abstractmethod
    def capture_frame(self) -> CameraFrame:
        """Take a picture.

        For this bootcamp you can assume capturing always succeeds, so this
        always gives you a frame and callers have nothing to check.

        Hands out a copy, so whoever gets a frame can do what they like with
        it without breaking the camera. Timestamps always go up, which is how
        callers tell which frame came first.

        Raises:
            RuntimeError: If the camera isn't on. That's a bug in the calling
                code, so it fails loudly rather than quietly doing nothing.
        """

    @abc.abstractmethod
    def stop(self) -> None:
        """Turn the camera off and let go of whatever it was holding.

        Safe to call any time, as often as you like, so shutdown code never
        has to check first.
        """
