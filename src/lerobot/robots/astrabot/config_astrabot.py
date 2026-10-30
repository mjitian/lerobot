#!/usr/bin/env python

from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.cameras.zed import ZEDCameraConfig
from lerobot.cameras.zed.configuration_zed import ZedDepthMode, ZedResolution, ZedView

from ..config import RobotConfig


def astrabot_cameras_config() -> dict[str, CameraConfig]:
    """Default camera configuration for the Astrabot robot.

    Mirrors the Dora data-collection pipeline
    (`astrabot_data_collection_xr1_thor_test.yaml`): a ZED 2i head camera
    (right eye) plus two USB wrist cameras.
    """
    return {
        # Head camera: ZED 2i right eye (`head_camera/right_image`), 640x480 @ 15Hz.
        "head": ZEDCameraConfig(
            serial_number="36753969",
            fps=30,
            width=640,
            height=480,
            view=ZedView.RIGHT,
            resolution=ZedResolution.HD720,
            depth_mode=ZedDepthMode.QUALITY,  # match single_zed_node.py default
        ),
        # Wrist cameras (USB webcams). NOTE: device aliases are swapped in the
        # Dora pipeline (`/dev/r_arm_cam` feeds the left-wrist observation and
        # vice versa).
        "left_wrist": OpenCVCameraConfig(
            index_or_path="/dev/r_arm_cam", fps=30, width=640, height=480
        ),
        "right_wrist": OpenCVCameraConfig(
            index_or_path="/dev/l_arm_cam", fps=30, width=640, height=480
        ),
    }


@RobotConfig.register_subclass("astrabot")
@dataclass
class AstrabotConfig(RobotConfig):
    cameras: dict[str, CameraConfig] = field(default_factory=astrabot_cameras_config)