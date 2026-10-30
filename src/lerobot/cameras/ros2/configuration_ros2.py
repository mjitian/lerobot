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

from dataclasses import dataclass

from ..configs import CameraConfig, ColorMode, Cv2Rotation


@CameraConfig.register_subclass("ros2")
@dataclass
class ROS2CameraConfig(CameraConfig):
    """Configuration class for cameras streamed over a ROS 2 topic.

    This backend subscribes to a `sensor_msgs/msg/Image` topic, which makes it
    suitable for cameras that are plugged into another host on the same network
    and published over ROS 2, e.g. an Orbbec Gemini 336 driven by the
    `orbbec_camera` wrapper on the camera host:

    ```python
    # On the camera host: ros2 launch orbbec_camera gemini_336.launch.py

    # On the leRobot host, as part of a robot's camera configuration:
    cameras = {"front": ROS2CameraConfig(topic="/camera/color/image_raw", width=640, height=480)}
    ```

    Attributes:
        topic: Name of the ROS 2 image topic to subscribe to. Defaults to
            `/camera/color/image_raw`, the Orbbec color topic.
        color_mode: Color mode for image output (RGB or BGR). Defaults to RGB.
        rotation: Image rotation setting (0°, 90°, 180°, or 270°). Defaults to no rotation.
        fps: Informational only; the frame rate is dictated by the camera host.
        width: Requested output frame width in pixels. Received frames are resized
            to this size. If None, the native stream size is used.
        height: Requested output frame height in pixels. If None, the native
            stream size is used.

    Note:
        - Requires a ROS 2 environment (`rclpy`, `sensor_msgs`) in the same Python
          environment as `lerobot`, e.g. `source /opt/ros/humble/setup.bash`.
        - Both hosts must use the same `ROS_DOMAIN_ID` (and RMW implementation) and
          DDS discovery must be able to reach the camera host over the LAN. Check
          with `ros2 topic hz /camera/color/image_raw` before connecting.
        - `width` and `height` must be set together (or both left as None).
    """

    topic: str = "/camera/color/image_raw"
    color_mode: ColorMode = ColorMode.RGB
    rotation: Cv2Rotation = Cv2Rotation.NO_ROTATION

    def __post_init__(self):
        if self.color_mode not in (ColorMode.RGB, ColorMode.BGR):
            raise ValueError(
                f"`color_mode` is expected to be {ColorMode.RGB.value} or {ColorMode.BGR.value}, but {self.color_mode} is provided."
            )

        if self.rotation not in (
            Cv2Rotation.NO_ROTATION,
            Cv2Rotation.ROTATE_90,
            Cv2Rotation.ROTATE_180,
            Cv2Rotation.ROTATE_270,
        ):
            raise ValueError(
                f"`rotation` is expected to be in {(Cv2Rotation.NO_ROTATION, Cv2Rotation.ROTATE_90, Cv2Rotation.ROTATE_180, Cv2Rotation.ROTATE_270)}, but {self.rotation} is provided."
            )

        if not self.topic or not self.topic.startswith("/"):
            raise ValueError(
                f"`topic` is expected to be a fully qualified ROS 2 topic name (e.g. '/camera/color/image_raw'), but '{self.topic}' is provided."
            )

        if (self.width is None) != (self.height is None):
            raise ValueError(
                f"`width` and `height` need to be set together, but width={self.width} and height={self.height} are provided."
            )
