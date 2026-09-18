from __future__ import annotations

import py_trees

from engine import blackboard_keys


class CaptureForPerception(py_trees.behaviour.Behaviour):

    def __init__(self, name: str, publisher, timeout_ticks: int = 20) -> None:
        super().__init__(name=name)
        self._publisher = publisher
        self._timeout_ticks = timeout_ticks
        self._initial_frame_index: int | None = None
        self._ticks_elapsed: int = 0

        self.blackboard = self.attach_blackboard_client(name=self.name)
        self.blackboard.register_key(
            key=blackboard_keys.LATEST_FRAME, access=py_trees.common.Access.READ
        )

    def _latest_frame(self):
        try:
            return self.blackboard.get(blackboard_keys.LATEST_FRAME)
        except KeyError:
            return None

    def initialise(self) -> None:
        frame = self._latest_frame()
        self._initial_frame_index = frame.index if frame is not None else None
        self._ticks_elapsed = 0

    def update(self) -> py_trees.common.Status:
        frame = self._latest_frame()

        if frame is not None and frame.index != self._initial_frame_index:
            self._publisher.publish_image(frame)
            self._publisher.publish_status(
                {"phase": "capture", "frame_index": frame.index}
            )
            return py_trees.common.Status.SUCCESS

        self._ticks_elapsed += 1
        if self._ticks_elapsed >= self._timeout_ticks:
            return py_trees.common.Status.FAILURE

        return py_trees.common.Status.RUNNING