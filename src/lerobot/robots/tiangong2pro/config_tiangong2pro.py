#!/usr/bin/env python

from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.cameras.ros2 import ROS2CameraConfig

from ..config import RobotConfig


def tiangong2pro_cameras_config() -> dict[str, CameraConfig]:
    """Default camera configuration for the Astrabot robot.

    Mirrors the Dora data-collection pipeline
    (`astrabot_data_collection_xr1_thor_test.yaml`): an Orbbec Gemini 336 head
    camera streamed over ROS 2 from another host, plus two USB wrist cameras.
    """
    return {
        # Head camera: Orbbec Gemini 336, published by the camera host over ROS 2.
        "head": ROS2CameraConfig(
            topic="/camera/color/image_raw",
            fps=30,
            width=640,
            height=480,
        ),
        # Wrist cameras (USB webcams). NOTE: device aliases are swapped in the
        # Dora pipeline (`/dev/r_arm_cam` feeds the left-wrist observation and
        # vice versa).
        "left_wrist": OpenCVCameraConfig(
            index_or_path="/dev/left_camera", fps=30, width=640, height=480, fourcc="MJPG"
        ),
        "right_wrist": OpenCVCameraConfig(
            index_or_path="/dev/right_camera", fps=30, width=640, height=480, fourcc="MJPG"
        ),
    }


@RobotConfig.register_subclass("tiangong2pro")
@dataclass
class Tiangong2ProConfig(RobotConfig):
    cameras: dict[str, CameraConfig] = field(default_factory=tiangong2pro_cameras_config)
