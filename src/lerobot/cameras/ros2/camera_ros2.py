# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Provides the ROS2Camera class for cameras that stream their frames over ROS 2.

The camera itself is expected to be attached to another host on the local network
(e.g. an Orbbec Gemini 336 driven by the `orbbec_camera` wrapper), while leRobot
subscribes to its image topic (default: ``/camera/color/image_raw``) from the
robot host. The stream must be a `sensor_msgs/msg/Image` topic with an 8-bit
encoding (`rgb8`, `bgr8`, `rgba8`, `bgra8` or `mono8`), which covers the Orbbec
color streams.

Receiving a stream from another host requires a sourced ROS 2 environment with
the same ``ROS_DOMAIN_ID`` (and RMW implementation) as the camera host, plus DDS
discovery reaching it over the LAN. Check with
``ros2 topic hz /camera/color/image_raw`` before connecting.

Example:
    ```python
    from lerobot.cameras.ros2 import ROS2Camera, ROS2CameraConfig

    camera = ROS2Camera(ROS2CameraConfig(topic="/camera/color/image_raw", width=640, height=480))
    camera.connect()
    color_image = camera.read()
    print(color_image.shape)  # (480, 640, 3)
    camera.disconnect()
    ```
"""

import itertools
import logging
import time
from threading import Condition, Event, Thread
from typing import Any

import cv2
import numpy as np

from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..camera import Camera
from ..configs import ColorMode
from ..utils import get_cv2_rotation
from .configuration_ros2 import ROS2CameraConfig

logger = logging.getLogger(__name__)

# Seconds `connect(warmup=True)` waits for the first frame before giving up.
CONNECT_TIMEOUT_S = 10.0

# Process-wide counter used to generate unique ROS 2 node names.
_NODE_COUNTER = itertools.count(1)

# Supported `sensor_msgs/msg/Image` encodings, mapped to (channels, OpenCV
# conversion code). A code of None means the frame is already BGR.
_IMAGE_ENCODINGS: dict[str, tuple[int, int | None]] = {
    "rgb8": (3, cv2.COLOR_RGB2BGR),
    "bgr8": (3, None),
    "rgba8": (4, cv2.COLOR_RGBA2BGR),
    "bgra8": (4, cv2.COLOR_BGRA2BGR),
    "mono8": (1, cv2.COLOR_GRAY2BGR),
}


def image_msg_to_bgr(msg: Any) -> np.ndarray:
    """Converts a `sensor_msgs/msg/Image` message into a BGR uint8 array.

    Args:
        msg: The image message to convert.

    Returns:
        np.ndarray: A (height, width, 3) BGR uint8 array.

    Raises:
        ValueError: If the encoding is unsupported or the message data is invalid.
    """
    if msg.encoding not in _IMAGE_ENCODINGS:
        raise ValueError(
            f"Unsupported image encoding '{msg.encoding}'. Supported encodings: {sorted(_IMAGE_ENCODINGS)}."
        )
    channels, to_bgr = _IMAGE_ENCODINGS[msg.encoding]

    height, width, step = int(msg.height), int(msg.width), int(msg.step)
    if height <= 0 or width <= 0 or step < width * channels:
        raise ValueError(f"Invalid image geometry (height={height}, width={width}, step={step}).")

    data = np.frombuffer(msg.data, dtype=np.uint8)
    if data.size < height * step:
        raise ValueError(
            f"Image data is truncated: got {data.size} bytes, expected at least {height * step}."
        )

    # `step` is the row stride in bytes and may include padding; reshaping the
    # trimmed buffer copies the rows when they are padded.
    pixels = data[: height * step].reshape(height, step)[:, : width * channels]
    image = np.ascontiguousarray(
        pixels.reshape(height, width) if channels == 1 else pixels.reshape(height, width, channels)
    )

    return image if to_bgr is None else cv2.cvtColor(image, to_bgr)


class ROS2Camera(Camera):
    """
    Manages a camera that is streamed over a ROS 2 image topic.

    Frames are decoded by a subscription callback that is spun in a background
    thread, so `read` and `async_read` only wait for the stream to deliver a
    frame. If the process already has an active `rclpy` context (e.g. a robot
    driver that uses ROS 2 in the same process), the camera reuses it; otherwise
    a context is initialized on `connect` and shut down on `disconnect`.
    """

    def __init__(self, config: ROS2CameraConfig):
        """
        Initializes the ROS2Camera instance.

        Args:
            config: The configuration settings for the camera.
        """
        super().__init__(config)

        self.config = config
        self.topic = config.topic
        self.color_mode = config.color_mode
        self.rotation: int | None = get_cv2_rotation(config.rotation)

        # Lazy references set on connect().
        self._rclpy: Any | None = None
        self._owns_rclpy_context = False
        self._node: Any | None = None
        self._executor: Any | None = None
        self._spin_thread: Thread | None = None

        self._frame_condition = Condition()
        self._latest_frame: np.ndarray | None = None
        self._frame_count = 0
        self._new_frame_event = Event()
        self._warned_encodings: set[str] = set()

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.topic})"

    @property
    def is_connected(self) -> bool:
        """Checks if the camera node is created and its ROS 2 context is alive."""
        return self._node is not None and self._rclpy.ok()

    @staticmethod
    def _import_rclpy() -> Any:
        """Lazily imports and returns the `rclpy` module.

        Raises:
            ImportError: If `rclpy` is not installed or fails to import (e.g.
                because the ROS 2 environment has not been sourced).
        """
        try:
            import rclpy

            return rclpy
        except Exception as e:
            raise ImportError(
                "The `rclpy` package could not be imported. Source a ROS 2 installation in the same "
                "Python environment as `lerobot` (e.g. `source /opt/ros/humble/setup.bash`) to use the "
                f"ROS2 camera backend. Import error: {e}"
            ) from e

    @staticmethod
    def _ensure_rclpy_context(rclpy: Any) -> bool:
        """Initializes a ROS 2 context if there is none, reusing an existing one.

        Returns:
            bool: True if this camera initialized the context (and is therefore
            responsible for shutting it down), False if an existing one is reused.
        """
        if rclpy.ok():
            return False

        from rclpy.signals import SignalHandlerOptions

        # Keep leRobot in charge of signal handling, so that Ctrl-C still raises
        # `KeyboardInterrupt` in the main thread (e.g. to stop a recording loop).
        rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
        return True

    @staticmethod
    def find_cameras(timeout_s: float = 3.0) -> list[dict[str, Any]]:
        """
        Lists the `sensor_msgs/msg/Image` topics advertised on the ROS 2 graph.

        This includes topics published by other hosts on the same network, so it
        can be used to discover a remote camera such as an Orbbec Gemini 336.

        Args:
            timeout_s: How long to keep collecting graph updates, in seconds.
                DDS discovery is incremental, so a short window is used to give
                remote publishers time to show up.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, where each dictionary
            contains 'type' (always 'ROS2'), 'id' (the topic name), a
            human-readable name and the number of publishers.

        Raises:
            ImportError: If `rclpy` is not installed.
        """
        rclpy = ROS2Camera._import_rclpy()
        owns_context = ROS2Camera._ensure_rclpy_context(rclpy)
        node = rclpy.create_node(f"lerobot_ros2_camera_discovery_{next(_NODE_COUNTER)}")

        try:
            image_topics: list[str] = []
            deadline = time.monotonic() + timeout_s
            while time.monotonic() < deadline:
                for topic_name, topic_types in node.get_topic_names_and_types():
                    if "sensor_msgs/msg/Image" in topic_types and topic_name not in image_topics:
                        image_topics.append(topic_name)
                time.sleep(0.1)

            return [
                {
                    "name": f"ROS2 image topic {topic_name}",
                    "type": "ROS2",
                    "id": topic_name,
                    "publisher_count": node.count_publishers(topic_name),
                }
                for topic_name in sorted(image_topics)
            ]
        finally:
            node.destroy_node()
            if owns_context:
                rclpy.try_shutdown()

    def connect(self, warmup: bool = True):
        """
        Connects to the ROS 2 image topic specified in the configuration.

        Creates a node, subscribes to the topic and spins the subscription in a
        background thread. If `warmup` is True, the call blocks until the first
        frame is received (up to `CONNECT_TIMEOUT_S`) so that `connect` fails fast
        when the remote stream is not reachable.

        Args:
            warmup: If True (default), wait for the first frame before returning.
                If False, return as soon as the subscription is active.

        Raises:
            DeviceAlreadyConnectedError: If the camera is already connected.
            ImportError: If `rclpy` is not installed.
            ConnectionError: If no frame is received before the timeout.
        """
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} is already connected.")

        rclpy = self._import_rclpy()
        self._rclpy = rclpy
        self._owns_rclpy_context = self._ensure_rclpy_context(rclpy)

        try:
            from rclpy.qos import QoSProfile, ReliabilityPolicy
            from sensor_msgs.msg import Image

            # NOTE: `best_effort` is compatible with both best-effort and reliable
            # publishers, and a queue depth of 1 keeps only the freshest frame.
            qos_profile = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)

            node_name = f"lerobot_ros2_camera_{next(_NODE_COUNTER)}_{self.topic.strip('/').replace('/', '_')}"
            self._node = rclpy.create_node(node_name)
            self._node.create_subscription(Image, self.topic, self._image_callback, qos_profile)
            self._start_spin_thread()

            if warmup:
                self._wait_for_first_frame()
        except Exception:
            self._release_resources(shutdown_context=self._owns_rclpy_context)
            raise

        logger.info(f"{self} connected.")

    def _image_callback(self, msg: Any) -> None:
        """Decodes an incoming image message and publishes it to the readers."""
        try:
            frame = image_msg_to_bgr(msg)
        except Exception as e:
            if msg.encoding not in self._warned_encodings:
                self._warned_encodings.add(msg.encoding)
                logger.warning(f"{self}: dropping frame that could not be decoded: {e}")
            return

        with self._frame_condition:
            self._latest_frame = frame
            self._frame_count += 1
            self._frame_condition.notify_all()
        self._new_frame_event.set()

    def _start_spin_thread(self) -> None:
        """Adds the node to a private executor and spins it in a daemon thread."""
        from rclpy.executors import SingleThreadedExecutor

        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._spin_thread = Thread(target=self._spin, args=(), name=f"{self}_spin_loop")
        self._spin_thread.daemon = True
        self._spin_thread.start()

    def _spin(self) -> None:
        """Internal loop run by the background thread, i.e. the executor spin."""
        try:
            self._executor.spin()
        except Exception as e:
            # A normal shutdown returns from `spin()` without raising, so anything
            # caught here means the subscription is no longer being processed.
            if self._node is not None:
                logger.error(f"{self} spin thread stopped unexpectedly: {e}")

    def _stop_spin_thread(self) -> None:
        """Shuts the executor down and waits for the background thread to join."""
        if self._executor is not None:
            self._executor.shutdown(timeout_sec=1.0)

        if self._spin_thread is not None:
            self._spin_thread.join(timeout=2.0)
            self._spin_thread = None

        self._executor = None

    def _wait_for_first_frame(self) -> None:
        """Waits until the first frame is received, or raises after the timeout."""
        if not self._new_frame_event.wait(timeout=CONNECT_TIMEOUT_S):
            raise ConnectionError(self._no_stream_message(CONNECT_TIMEOUT_S))

    def _no_stream_message(self, timeout_s: float) -> str:
        """Builds a diagnostic message for missing frames on the topic."""
        publisher_count = 0 if self._node is None else self._node.count_publishers(self.topic)
        return (
            f"No image was received on topic '{self.topic}' within {timeout_s:.1f}s "
            f"(publishers discovered on this host: {publisher_count}). Check that the camera host is "
            "publishing the topic (e.g. `ros2 topic hz /camera/color/image_raw`), that both hosts use "
            "the same ROS_DOMAIN_ID/RMW implementation, and that DDS discovery can reach the camera "
            "host over the LAN."
        )

    def read(self, color_mode: ColorMode | None = None, timeout_ms: int = 1000) -> np.ndarray:
        """
        Reads the next frame from the ROS 2 topic, blocking until it arrives.

        Args:
            color_mode (Optional[ColorMode]): The target color mode (RGB or BGR).
                If None, uses the instance's default `self.color_mode`.
            timeout_ms (int): Maximum time in milliseconds to wait for the next
                frame. Defaults to 1000ms.

        Returns:
            np.ndarray: The captured frame as a NumPy array (height, width, channels),
                processed according to `color_mode` and rotation.

        Raises:
            DeviceNotConnectedError: If the camera is not connected.
            TimeoutError: If no new frame arrives within `timeout_ms`.
            ValueError: If an invalid `color_mode` is requested.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        with self._frame_condition:
            target_frame_count = self._frame_count + 1
            deadline = time.monotonic() + timeout_ms / 1000.0
            while self._frame_count < target_frame_count:
                remaining_s = deadline - time.monotonic()
                if remaining_s <= 0:
                    raise TimeoutError(self._no_stream_message(timeout_ms / 1000.0))
                self._frame_condition.wait(remaining_s)
            frame = self._latest_frame

        return self._postprocess_image(frame, color_mode)

    def _postprocess_image(self, image: np.ndarray, color_mode: ColorMode | None = None) -> np.ndarray:
        """
        Applies resizing, color conversion, and rotation to a decoded BGR frame.

        The output is resized so that its shape matches the configured
        `height`/`width` (if set), even when a 90°/270° rotation is applied.

        Args:
            image (np.ndarray): The decoded BGR frame (H, W, 3).
            color_mode (Optional[ColorMode]): The target color mode (RGB or BGR).
                If None, uses the instance's default `self.color_mode`.

        Returns:
            np.ndarray: The processed image frame.

        Raises:
            ValueError: If the requested `color_mode` is invalid.
            RuntimeError: If the decoded frame is not a 3-channel image.
        """
        requested_color_mode = self.color_mode if color_mode is None else color_mode

        if requested_color_mode not in (ColorMode.RGB, ColorMode.BGR):
            raise ValueError(
                f"Invalid color mode '{requested_color_mode}'. Expected {ColorMode.RGB} or {ColorMode.BGR}."
            )

        if image.ndim != 3 or image.shape[2] != 3:
            raise RuntimeError(f"{self} frame shape={image.shape} does not match expected 3 channels (BGR).")

        if self.width is not None and self.height is not None:
            if self.rotation in (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE):
                # The frame is transposed by the rotation, so the pre-rotation size
                # is swapped to keep the final shape at (height, width).
                target_size = (self.height, self.width)
            else:
                target_size = (self.width, self.height)
            if (image.shape[1], image.shape[0]) != target_size:
                image = cv2.resize(image, target_size)

        if requested_color_mode == ColorMode.RGB:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.rotation in (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180):
            image = cv2.rotate(image, self.rotation)

        return image

    def async_read(self, timeout_ms: float = 200) -> np.ndarray:
        """
        Reads the latest available frame asynchronously.

        This method returns the most recent frame delivered by the subscription,
        waiting up to `timeout_ms` for a frame to become available. It does not
        block on the network itself.

        Args:
            timeout_ms (float): Maximum time in milliseconds to wait for a frame
                to become available. Defaults to 200ms (0.2 seconds).

        Returns:
            np.ndarray: The latest captured frame, processed according to configuration.

        Raises:
            DeviceNotConnectedError: If the camera is not connected.
            TimeoutError: If no frame data becomes available within the specified timeout.
            RuntimeError: If no frame is available although the event is set.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if not self._new_frame_event.wait(timeout=timeout_ms / 1000.0):
            raise TimeoutError(self._no_stream_message(timeout_ms / 1000.0))

        with self._frame_condition:
            frame = self._latest_frame
            self._new_frame_event.clear()

        if frame is None:
            raise RuntimeError(f"Internal error: Event set but no frame available for {self}.")

        return self._postprocess_image(frame)

    def disconnect(self):
        """
        Disconnects from the ROS 2 topic and releases resources.

        Stops the background spin thread, destroys the camera node and, if the
        camera was the one that initialized the ROS 2 context, shuts it down.

        Raises:
            DeviceNotConnectedError: If the camera is already disconnected.
        """
        if self._node is None and self._spin_thread is None:
            raise DeviceNotConnectedError(
                f"Attempted to disconnect {self}, but it appears already disconnected."
            )

        self._release_resources(shutdown_context=self._owns_rclpy_context)

        logger.info(f"{self} disconnected.")

    def _release_resources(self, shutdown_context: bool = False) -> None:
        """Stops background threads, destroys the node and resets the frame state."""
        self._stop_spin_thread()

        if self._node is not None:
            try:
                self._node.destroy_node()
            finally:
                self._node = None

        with self._frame_condition:
            self._latest_frame = None
            self._frame_count = 0
            self._frame_condition.notify_all()
        self._new_frame_event.clear()

        if shutdown_context and self._rclpy is not None:
            # NOTE: Only shut down the ROS 2 context if this camera initialized it,
            # so that other ROS 2 users in the same process (e.g. a robot driver)
            # keep working.
            self._rclpy.try_shutdown()
            self._owns_rclpy_context = False
