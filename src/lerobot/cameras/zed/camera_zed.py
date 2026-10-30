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
Provides the ZEDCamera class for capturing frames from Stereolabs ZED stereo cameras.

This module wraps the `pyzed` SDK (``pyzed.sl``) into the leRobot `Camera`
interface, mirroring the structure of `RealSenseCamera`. It supports selecting
which stereo eye (left/right) to output, resolution presets, depth mode
configuration (used only at init time; depth is not exposed), manual/auto
exposure, and device identification via serial number.

The ZED SDK returns BGRA frames on CPU; these are converted to RGB (or BGR)
and resized to the configured width/height before being returned.
"""

import logging
import time
from threading import Event, Lock, Thread
from typing import Any

import cv2
import numpy as np

from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..camera import Camera
from ..configs import ColorMode
from ..utils import get_cv2_rotation
from .configuration_zed import (
    ZEDCameraConfig,
    ZedDepthMode,
    ZedResolution,
    ZedView,
)

logger = logging.getLogger(__name__)


class ZEDCamera(Camera):
    """
    Manages interactions with Stereolabs ZED stereo cameras for frame recording.

    This class provides an interface similar to `RealSenseCamera` but tailored
    for ZED devices, leveraging the `pyzed` SDK (`pyzed.sl`). The camera is
    identified by its serial number when several ZEDs are connected.

    Since a ZED is a stereo camera, each instance outputs a single eye
    (configurable via `view`, defaulting to the left eye). Use two instances
    (or two robot cameras) to capture both eyes if needed.

    Example:
        ```python
        from lerobot.cameras.zed import ZEDCamera, ZEDCameraConfig
        from lerobot.cameras.zed.configuration_zed import ZedView, ZedResolution
        from lerobot.cameras import ColorMode

        # Basic usage (left eye, default resolution HD720)
        config = ZEDCameraConfig(fps=30, width=640, height=480)
        camera = ZEDCamera(config)
        camera.connect()

        # Read 1 frame synchronously
        color_image = camera.read()
        print(color_image.shape)  # (480, 640, 3)

        # Read 1 frame asynchronously
        async_image = camera.async_read()

        # When done, properly disconnect the camera using
        camera.disconnect()

        # Example using the right eye (e.g. as a head camera)
        right_config = ZEDCameraConfig(
            serial_number="SN00000000",  # Optional, to disambiguate multiple ZEDs
            fps=30,
            width=640,
            height=480,
            view=ZedView.RIGHT,
            resolution=ZedResolution.HD720,
            color_mode=ColorMode.RGB,
        )
        right_camera = ZEDCamera(right_config)
        # ... connect, read, disconnect ...
        ```
    """

    def __init__(self, config: ZEDCameraConfig):
        """
        Initializes the ZEDCamera instance.

        Args:
            config: The configuration settings for the camera.
        """
        super().__init__(config)

        self.config = config

        self.serial_number = config.serial_number
        self.view = config.view
        self.resolution = config.resolution
        self.color_mode = config.color_mode
        self.warmup_s = config.warmup_s
        self.depth_mode = config.depth_mode
        self.enable_auto_exposure = config.enable_auto_exposure
        self.color_exposure = config.color_exposure

        # Lazy reference to the `pyzed.sl` module, set on connect().
        self._sl_module = None

        self.zed: Any | None = None
        self.runtime_params: Any | None = None

        self.thread: Thread | None = None
        self.stop_event: Event | None = None
        self.frame_lock: Lock = Lock()
        self.latest_frame: np.ndarray | None = None
        self.new_frame_event: Event = Event()

        self.rotation: int | None = get_cv2_rotation(config.rotation)

        if self.height and self.width:
            self.capture_width, self.capture_height = self.width, self.height
            if self.rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE]:
                self.capture_width, self.capture_height = self.height, self.width
        else:
            # Filled from the native stream in `_configure_capture_settings`.
            self.capture_width, self.capture_height = None, None

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.serial_number or self.view.value})"

    @staticmethod
    def _import_pyzed() -> Any:
        """Lazily imports and returns the `pyzed.sl` module.

        Raises:
            ImportError: If `pyzed` is not installed or fails to import (e.g.
                because of an incompatible numpy version).
        """
        try:
            import pyzed.sl as sl

            return sl
        except Exception as e:
            raise ImportError(
                "The `pyzed` package could not be imported. Install the ZED SDK and "
                "`pyzed` in the same Python environment as `lerobot` (see "
                "https://www.stereolabs.com/developers/release). Note that the ZED "
                "SDK requires numpy <= 1.26.4. "
                f"Import error: {e}"
            ) from e

    @property
    def is_connected(self) -> bool:
        """Checks if the ZED camera is connected and opened."""
        return self.zed is not None and self.zed.is_opened()

    @staticmethod
    def find_cameras() -> list[dict[str, Any]]:
        """
        Detects available ZED cameras connected to the system.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, where each dictionary
            contains 'type', 'id' (serial number), 'name', camera model, max FPS
            and camera state.

        Raises:
            ImportError: If `pyzed` is not installed.
        """
        sl = ZEDCamera._import_pyzed()

        found_cameras_info = []
        device_list = sl.Camera.get_device_list()

        for device in device_list:
            camera_info = {
                "name": f"ZED {device.camera_model.name}",
                "type": "ZED",
                "id": str(device.serial_number),
                "serial_number": str(device.serial_number),
                "camera_model": device.camera_model.name,
                "max_fps": device.max_fps,
                "camera_state": device.camera_state.name,
            }
            found_cameras_info.append(camera_info)

        return found_cameras_info

    def connect(self, warmup: bool = True):
        """
        Connects to the ZED camera specified in the configuration.

        Initializes the `pyzed.sl` camera, configures the requested settings
        (resolution, FPS, depth mode, exposure), opens the device, and optionally
        reads a few warmup frames.

        Raises:
            DeviceAlreadyConnectedError: If the camera is already connected.
            ConnectionError: If no ZED device is detected or the camera fails to open.
        """
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} is already connected.")

        sl = self._import_pyzed()
        self._sl_module = sl

        self.zed = sl.Camera()
        self.runtime_params = sl.RuntimeParameters()
        init_params = sl.InitParameters(
            camera_resolution=self._sl_resolution(),
            depth_mode=self._sl_depth_mode(),
            coordinate_units=sl.UNIT.METER,
            depth_minimum_distance=0.3,
            depth_maximum_distance=20.0,
        )
        if self.fps is not None:
            init_params.camera_fps = self.fps

        if self.serial_number is not None:
            try:
                init_params.set_from_serial_number(int(self.serial_number))
            except ValueError:
                logger.warning(
                    f"Serial number '{self.serial_number}' is not an integer, ignoring..."
                )

        err = self.zed.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            self.zed = None
            raise ConnectionError(
                f"Failed to open {self} (error: {err})."
                "Run `python -m lerobot.find_cameras zed` to find available cameras."
            )

        self._configure_exposure()
        self._configure_capture_settings()

        if warmup:
            time.sleep(
                1
            )  # NOTE: ZED cameras need a bit of time to warm up before the first read.
            start_time = time.time()
            while time.time() - start_time < self.warmup_s:
                self.read()
                time.sleep(0.1)

        logger.info(f"{self} connected.")

    def _sl_resolution(self) -> Any:
        """Maps the configured ZED resolution to the pyzed enum."""
        sl = self._sl_module
        return {
            ZedResolution.HD2K: sl.RESOLUTION.HD2K,
            ZedResolution.HD1080: sl.RESOLUTION.HD1080,
            ZedResolution.HD720: sl.RESOLUTION.HD720,
            ZedResolution.VGA: sl.RESOLUTION.VGA,
        }[self.resolution]

    def _sl_depth_mode(self) -> Any:
        """Maps the configured ZED depth mode to the pyzed enum."""
        sl = self._sl_module
        return {
            ZedDepthMode.NONE: sl.DEPTH_MODE.NONE,
            ZedDepthMode.PERFORMANCE: sl.DEPTH_MODE.PERFORMANCE,
            ZedDepthMode.QUALITY: sl.DEPTH_MODE.QUALITY,
            ZedDepthMode.ULTRA: sl.DEPTH_MODE.ULTRA,
        }[self.depth_mode]

    def _sl_view(self) -> Any:
        """Returns the pyzed VIEW corresponding to the configured stereo eye."""
        sl = self._sl_module
        return sl.VIEW.RIGHT if self.view == ZedView.RIGHT else sl.VIEW.LEFT

    def _configure_exposure(self) -> None:
        """Configures auto or manual exposure according to the configuration."""
        sl = self._sl_module
        if self.enable_auto_exposure:
            self.zed.set_camera_settings(sl.VIDEO_SETTINGS.AEC_AGC, 1)
        else:
            self.zed.set_camera_settings(sl.VIDEO_SETTINGS.AEC_AGC, 0)
            self.zed.set_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE, self.color_exposure)

    def _configure_capture_settings(self) -> None:
        """Sets fps, width, and height from the device stream if not already configured.

        Uses the camera information reported by the ZED SDK to fill unset
        attributes. Original capture dimensions are always stored in
        `capture_width`/`capture_height`, while `width`/`height` reflect the
        output dimensions (swapped when a 90° rotation is applied).
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"Cannot validate settings for {self} as it is not connected.")

        camera_info = self.zed.get_camera_information()
        stream = camera_info.camera_configuration

        if self.fps is None:
            self.fps = stream.fps

        if self.width is None or self.height is None:
            actual_width = int(round(stream.resolution.width))
            actual_height = int(round(stream.resolution.height))
            if self.rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE]:
                self.width, self.height = actual_height, actual_width
            else:
                self.width, self.height = actual_width, actual_height
            self.capture_width, self.capture_height = actual_width, actual_height

    def read(self, color_mode: ColorMode | None = None, timeout_ms: int = 200) -> np.ndarray:
        """
        Reads a single frame (color) synchronously from the camera.

        This is a blocking call. It waits for the next frame from the ZED camera,
        retrieves the configured stereo eye, and post-processes it (resize, color
        conversion, rotation).

        Args:
            color_mode (Optional[ColorMode]): The target color mode (RGB or BGR).
                If None, uses the instance's default `self.color_mode`.
            timeout_ms (int): Maximum time in milliseconds to wait for a frame.
                Defaults to 200ms.

        Returns:
            np.ndarray: The captured color frame as a NumPy array
                (height, width, channels), processed according to `color_mode`
                and rotation.

        Raises:
            DeviceNotConnectedError: If the camera is not connected.
            RuntimeError: If grabbing or retrieving a frame from the camera fails.
            ValueError: If an invalid `color_mode` is requested.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        sl = self._sl_module
        start_time = time.perf_counter()

        # NOTE: pyzed >= 5.x `Camera.grab` only accepts a single positional argument
        # (the runtime parameters); the timeout is no longer passed as a second arg.
        if self.zed.grab(self.runtime_params) != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"{self} read failed.")

        mat = sl.Mat()
        self.zed.retrieve_image(mat, self._sl_view(), sl.MEM.CPU)
        bgra = mat.get_data()
        if bgra is None:
            raise RuntimeError(f"{self} read failed: no image data retrieved.")

        color_image_processed = self._postprocess_image(bgra, color_mode)

        read_duration_ms = (time.perf_counter() - start_time) * 1e3
        logger.debug(f"{self} read took: {read_duration_ms:.1f}ms")

        return color_image_processed

    def _postprocess_image(self, image: np.ndarray, color_mode: ColorMode | None = None) -> np.ndarray:
        """
        Applies resizing, color conversion, and rotation to a raw ZED BGRA frame.

        The ZED SDK returns BGRA frames on CPU. This method always converts to
        RGB first, optionally converts to BGR, and applies the configured rotation.

        Args:
            image (np.ndarray): The raw BGRA frame (H, W, 4).
            color_mode (Optional[ColorMode]): The target color mode (RGB or BGR).
                If None, uses the instance's default `self.color_mode`.

        Returns:
            np.ndarray: The processed image frame.

        Raises:
            ValueError: If the requested `color_mode` is invalid.
            RuntimeError: If the raw frame dimensions or channel count are invalid.
        """
        requested_color_mode = self.color_mode if color_mode is None else color_mode

        if requested_color_mode not in (ColorMode.RGB, ColorMode.BGR):
            raise ValueError(
                f"Invalid color mode '{requested_color_mode}'. Expected {ColorMode.RGB} or {ColorMode.BGR}."
            )

        h, w, c = image.shape

        if self.capture_width and self.capture_height and (w != self.capture_width or h != self.capture_height):
            image = cv2.resize(image, (self.capture_width, self.capture_height))
            h, w, c = image.shape

        if c != 4:
            raise RuntimeError(
                f"{self} frame channels={c} do not match expected 4 channels (BGRA) from the ZED SDK."
            )

        # ZED delivers BGRA on CPU; drop alpha and convert to RGB.
        processed_image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)

        if requested_color_mode == ColorMode.BGR:
            processed_image = cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR)

        if self.rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180]:
            processed_image = cv2.rotate(processed_image, self.rotation)

        return processed_image

    def _read_loop(self):
        """
        Internal loop run by the background thread for asynchronous reading.

        On each iteration:
        1. Reads a color frame with 500ms timeout
        2. Stores result in latest_frame (thread-safe)
        3. Sets new_frame_event to notify listeners

        Stops on DeviceNotConnectedError, logs other errors and continues.
        """
        while not self.stop_event.is_set():
            try:
                color_image = self.read(timeout_ms=500)

                with self.frame_lock:
                    self.latest_frame = color_image
                self.new_frame_event.set()

            except DeviceNotConnectedError:
                break
            except Exception as e:
                logger.warning(f"Error reading frame in background thread for {self}: {e}")

    def _start_read_thread(self) -> None:
        """Starts or restarts the background read thread if it's not running."""
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=0.1)
        if self.stop_event is not None:
            self.stop_event.set()

        self.stop_event = Event()
        self.thread = Thread(target=self._read_loop, args=(), name=f"{self}_read_loop")
        self.thread.daemon = True
        self.thread.start()

    def _stop_read_thread(self) -> None:
        """Signals the background read thread to stop and waits for it to join."""
        if self.stop_event is not None:
            self.stop_event.set()

        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=2.0)

        self.thread = None
        self.stop_event = None

    def async_read(self, timeout_ms: float = 200) -> np.ndarray:
        """
        Reads the latest available frame data (color) asynchronously.

        This method retrieves the most recent color frame captured by the background
        read thread. It does not block waiting for the camera hardware directly,
        but may wait up to timeout_ms for the background thread to provide a frame.

        Args:
            timeout_ms (float): Maximum time in milliseconds to wait for a frame
                to become available. Defaults to 200ms (0.2 seconds).

        Returns:
            np.ndarray: The latest captured frame (color image), processed
                according to configuration.

        Raises:
            DeviceNotConnectedError: If the camera is not connected.
            TimeoutError: If no frame data becomes available within the specified timeout.
            RuntimeError: If the background thread died unexpectedly or another error occurs.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if self.thread is None or not self.thread.is_alive():
            self._start_read_thread()

        if not self.new_frame_event.wait(timeout=timeout_ms / 1000.0):
            thread_alive = self.thread is not None and self.thread.is_alive()
            raise TimeoutError(
                f"Timed out waiting for frame from camera {self} after {timeout_ms} ms. "
                f"Read thread alive: {thread_alive}."
            )

        with self.frame_lock:
            frame = self.latest_frame
            self.new_frame_event.clear()

        if frame is None:
            raise RuntimeError(f"Internal error: Event set but no frame available for {self}.")

        return frame

    def disconnect(self):
        """
        Disconnects from the camera and releases resources.

        Stops the background read thread (if running) and closes the ZED camera.

        Raises:
            DeviceNotConnectedError: If the camera is already disconnected.
        """
        if not self.is_connected and self.thread is None:
            raise DeviceNotConnectedError(
                f"Attempted to disconnect {self}, but it appears already disconnected."
            )

        if self.thread is not None:
            self._stop_read_thread()

        if self.zed is not None:
            self.zed.close()
            self.zed = None
            self.runtime_params = None
            self._sl_module = None

        logger.info(f"{self} disconnected.")
