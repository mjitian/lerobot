#!/usr/bin/env python

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

import logging
from functools import cached_property
from typing import Any

import numpy as np

from lerobot.cameras.utils import make_cameras_from_configs
from .config_tiangong2pro import Tiangong2ProConfig
from scipy.spatial.transform import Rotation as R

from ..robot import Robot
from .robot_control_tiangong2pro import Tiangong2ProRobotController

logger = logging.getLogger(__name__)


class Tiangong2Pro(Robot):

    config_class = Tiangong2ProConfig
    name = "tiangong2pro"

    def __init__(self, config: Tiangong2ProConfig):
        super().__init__(config)
        self.config = config
        self.cameras = make_cameras_from_configs(config.cameras)
        self.robot_controller = Tiangong2ProRobotController(robot_type = "tiangong2pro", enable_filter = False)
        self.robot_controller.setup()
        print("Tiangong2Pro was initialized ...")
    @property
    def _motors_ft(self) -> dict[str, type]:
        # Scalar features only: leRobot's `hw_to_dataset_features` collects `float`
        # features into `observation.state` (vector features like `tuple[float, N]`
        # are silently dropped). Each arm state is expanded into per-component
        # scalars, mirroring the 34-dim state vector documented in
        # `astrabot_obs_action_api.md` (joint_left(7) + joint_right(7) +
        # ee_pose_left(9) + ee_pose_right(9) + gripper_left(1) + gripper_right(1)).
        return {
            **{f"observation_states_joint_angle_left.j{i}": float for i in range(7)},
            **{f"observation_states_joint_angle_right.j{i}": float for i in range(7)},
            **{f"observation_states_ee_pose_left.e{i}": float for i in range(9)},
            **{f"observation_states_ee_pose_right.e{i}": float for i in range(9)},
            "observation_states_gripper_left": float,
            "observation_states_gripper_right": float,
        }

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return {
            **{f"action_left.e{i}": float for i in range(10)},
            **{f"action_right.e{i}": float for i in range(10)},
        }

    @property
    def is_connected(self) -> bool:
        return (
            all(cam.is_connected for cam in self.cameras.values())
        )

    def connect(self, calibrate: bool = True) -> None:

        for cam in self.cameras.values():
            cam.connect()

    def reset(self) -> None:
        """Return both arms to the initial joint pose (same as the teleop long-press-B reset)."""
        self.robot_controller.reset()

    @property
    def is_calibrated(self) -> bool:
        return True
    
    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def setup_motors(self) -> None:
        pass

    def get_observation(self) -> dict[str, Any]:
        obs = {}

        # State from ROS2 subscriptions (/joint_states, TF, gripper), mapped to
        # the scalar features declared in `_motors_ft`.
        state = self.robot_controller.get_state()
        for i in range(7):
            obs[f"observation_states_joint_angle_left.j{i}"] = float(state["joint_angle_l"][i])
            obs[f"observation_states_joint_angle_right.j{i}"] = float(state["joint_angle_r"][i])
        for i in range(9):
            obs[f"observation_states_ee_pose_left.e{i}"] = float(state["ee_pose_l"][i])
            obs[f"observation_states_ee_pose_right.e{i}"] = float(state["ee_pose_r"][i])
        obs["observation_states_gripper_left"] = float(state["gripper_l"])
        obs["observation_states_gripper_right"] = float(state["gripper_r"])

        # Camera images
        for cam_name, cam in self.cameras.items():
            obs[cam_name] = cam.async_read()
        return obs


    @staticmethod
    def _rot6d_to_quat(rot6d: np.ndarray) -> np.ndarray:
        """将 rotation_6d (旋转矩阵前两行) 转换为四元数 [x, y, z, w]。

        rotation_6d = [r1..r6] 是 3×3 旋转矩阵的前两行。
        通过 Gram-Schmidt 正交化补全第三行，再转为四元数。
        """
        a1 = rot6d[:3].astype(np.float64)
        a2 = rot6d[3:6].astype(np.float64)
        b1 = a1 / np.linalg.norm(a1)
        b2 = a2 - np.dot(b1, a2) * b1
        b2 = b2 / np.linalg.norm(b2)
        b3 = np.cross(b1, b2)
        rot_mat = np.vstack([b1, b2, b3])
        return R.from_matrix(rot_mat).as_quat()  # [x, y, z, w]

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        # Actions come as `action_left.e0..e9` / `action_right.e0..e9`, each arm
        # being a 10-dim vector: [x, y, z, rot6d(6), gripper].
        left_action = np.array([action[f"action_left.e{i}"] for i in range(10)], dtype=float)
        right_action = np.array([action[f"action_right.e{i}"] for i in range(10)], dtype=float)

        # /reference/pose expects xyzw quaternions, so convert rot6d.
        left_pose = np.concatenate([left_action[:3], self._rot6d_to_quat(left_action[3:9])])
        right_pose = np.concatenate([right_action[:3], self._rot6d_to_quat(right_action[3:9])])
        self.robot_controller.step_policy(left_pose, right_pose)

        # `step_gripper` reads the gripper command at index 7, so build an
        # 8-dim command with the policy gripper value placed there.
        left_gripper_cmd = np.zeros(8)
        left_gripper_cmd[7] = left_action[9]
        right_gripper_cmd = np.zeros(8)
        right_gripper_cmd[7] = right_action[9]
        self.robot_controller.step_gripper(left_gripper_cmd, "left")
        self.robot_controller.step_gripper(right_gripper_cmd, "right")

        return action

    def disconnect(self):
        
        for cam in self.cameras.values():
            cam.disconnect()
