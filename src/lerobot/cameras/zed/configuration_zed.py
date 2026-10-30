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
from enum import Enum

from ..configs import CameraConfig, ColorMode, Cv2Rotation


class ZedView(str, Enum):
    """Which eye of the ZED stereo camera to use as the color stream."""

    LEFT = "left"
    RIGHT = "right"


class ZedResolution(str, Enum):
    """ZED camera resolutions (presets offered by the ZED SDK)."""

    HD2K = "HD2K"
    HD1080 = "HD1080"
    HD720 = "HD720"
    VGA = "VGA"


class ZedDepthMode(str, Enum):
    """ZED depth computation modes."""

    NONE = "NONE"
    PERFORMANCE = "PERFORMANCE"
    QUALITY = "QUALITY"
    ULTRA = "ULTRA"


@CameraConfig.register_subclass("zed")
@dataclass
class ZEDCameraConfig(CameraConfig):
    """Configuration class for Stereolabs ZED stereo cameras.

    This class provides specialized configuration options for ZED cameras using
    the `pyzed` SDK (``pyzed.sl``), including stereo view selection (left/right
    eye), resolution presets, depth mode, exposure control, and device
    identification via serial number.

    Example configurations for a ZED 2i used as a head camera (right eye):
    ```python
    # Basic configuration
    ZEDCameraConfig(fps=30, width=640, height=480, view=ZedView.RIGHT)

    # Full configuration
    ZEDCameraConfig(
        serial_number="SN00000000",  # Optional, needed to disambiguate multiple ZEDs
        fps=30,
        width=640,
        height=480,
        view=ZedView.RIGHT,
        resolution=ZedResolution.HD720,
        color_mode=ColorMode.RGB,
        rotation=Cv2Rotation.NO_ROTATION,
        depth_mode=ZedDepthMode.PERFORMANCE,
        enable_auto_exposure=True,
        color_exposure=100,
    )
    ```

    Attributes:
        serial_number: Unique serial number identifying the camera. If None, the
            first available ZED is opened.
        view: Which stereo eye to output ("left" or "right"). Defaults to left.
        resolution: ZED capture resolution preset (HD2K, HD1080, HD720, VGA).
            Defaults to HD720.
        fps: Requested frames per second for the color stream.
        width: Requested output frame width in pixels. The captured frame is
            resized to this size regardless of the capture resolution.
        height: Requested output frame height in pixels.
        color_mode: Color mode for image output (RGB or BGR). Defaults to RGB.
        rotation: Image rotation setting (0°, 90°, 180°, or 270°). Defaults to no rotation.
        warmup_s: Time reading frames before returning from connect (in seconds).
        depth_mode: ZED depth computation mode. Note that ZED computes depth on
            every grab even if it is never read; set to NONE to save compute.
        enable_auto_exposure: Whether to use automatic gain/exposure (AEC_AGC).
        color_exposure: Manual exposure value in [0..100], used only when
            `enable_auto_exposure` is False.

    Note:
        - Requires the `pyzed` package (ZED SDK) installed in the same Python
          environment as `lerobot`.
        - Only 3-channel color output (RGB/BGR) is currently supported.
    """

    serial_number: str | None = None
    view: ZedView = ZedView.LEFT
    resolution: ZedResolution = ZedResolution.HD720
    color_mode: ColorMode = ColorMode.RGB
    rotation: Cv2Rotation = Cv2Rotation.NO_ROTATION
    warmup_s: int = 1
    depth_mode: ZedDepthMode = ZedDepthMode.PERFORMANCE
    enable_auto_exposure: bool = True
    color_exposure: int = 100

    def __post_init__(self):
        if self.view not in (ZedView.LEFT, ZedView.RIGHT):
            raise ValueError(
                f"`view` is expected to be {ZedView.LEFT.value} or {ZedView.RIGHT.value}, but {self.view} is provided."
            )

        if self.resolution not in (
            ZedResolution.HD2K,
            ZedResolution.HD1080,
            ZedResolution.HD720,
            ZedResolution.VGA,
        ):
            raise ValueError(
                f"`resolution` is expected to be in {tuple(r.value for r in ZedResolution)}, but {self.resolution} is provided."
            )

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

        if self.depth_mode not in (
            ZedDepthMode.NONE,
            ZedDepthMode.PERFORMANCE,
            ZedDepthMode.QUALITY,
            ZedDepthMode.ULTRA,
        ):
            raise ValueError(
                f"`depth_mode` is expected to be in {tuple(m.value for m in ZedDepthMode)}, but {self.depth_mode} is provided."
            )

        if not 0 <= self.color_exposure <= 100:
            raise ValueError(
                f"`color_exposure` is expected to be in [0, 100], but {self.color_exposure} is provided."
            )
