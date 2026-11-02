"""
Tiangong 2 Pro (天工2.0 Pro) teleop robot controller.

Ported from dora_nodes/astrabot/robot_control_odom.py with the robot-specific
surface replaced:

- arm joint names: shoulder_pitch/roll/yaw_{l,r}_joint, elbow_pitch/yaw_{l,r}_joint,
  wrist_pitch/roll_{l,r}_joint,
- head: 3 joints (head_yaw / head_pitch / head_roll), observation only,
- base frame: "pelvis" (the URDF has no base_link),
- no chassis: no /odom or /ecu/cmd_vel subscription, no chassis state/action,
- gripper: observation comes from the VR trigger (same as the Astrabot node);
  commanding the gripper is opt-in via GRIPPER_CMD_TOPIC_LEFT / GRIPPER_CMD_TOPIC_RIGHT
  so the same code runs on the simulation stack (which has no gripper) and on the
  real robot (set the topics when known).

The /reference/pose JSON contract is unchanged (xyzw quaternion), which is what the
tiangong2pro MPC stack (astrabot_arbitration -> /reference/cmd -> MPC/MRT) consumes.
"""
import os
import time
import json
import numpy as np
import threading
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64, Float64MultiArray, String
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose, Point, Quaternion, TransformStamped
from tf2_ros import Buffer, TransformListener, TransformBroadcaster
from typing import Literal

from .transform_util import Transform, Rotation
from .utils import track_frequency


class Tiangong2ProRobotController(Node):
    # Joint names
    LEFT_JOINTS_NAME = ["shoulder_pitch_l_joint", "shoulder_roll_l_joint", "shoulder_yaw_l_joint",
                        "elbow_pitch_l_joint", "elbow_yaw_l_joint", "wrist_pitch_l_joint",
                        "wrist_roll_l_joint"]
    RIGHT_JOINTS_NAME = ["shoulder_pitch_r_joint", "shoulder_roll_r_joint", "shoulder_yaw_r_joint",
                         "elbow_pitch_r_joint", "elbow_yaw_r_joint", "wrist_pitch_r_joint",
                         "wrist_roll_r_joint"]
    HEAD_JOINTS_NAME = ["head_yaw_joint", "head_pitch_joint", "head_roll_joint"]

    # Initial dual-arm pose, taken from the repo-root `init_pos` record: the
    # per-joint readback of motors 11-17 / 21-27 plus the two TCP poses that
    # `tf2_echo pelvis {left,right}_tcp_link` printed at that moment.
    # Joint order matches LEFT/RIGHT_JOINTS_NAME, i.e. the order the forward
    # position controller expects in its 14-dim command.
    INITIAL_JOINTS = {  # rad
        'left': [1.2743, 0.5710, 0.6135, -2.0426, 0.5849, -0.4138, -0.6306],
        'right': [0.8031, -0.7696, 0.1040, -2.0409, -0.3665, -0.1834, 0.0980],
    }
    INITIAL_TCP_POSE = {  # (position, orientation xyzw), pelvis frame
        'left': ([0.055, 0.424, 0.155], [-0.085, -0.581, -0.294, 0.754]),
        'right': ([0.135, -0.314, 0.068], [0.112, -0.564, 0.276, 0.770]),
    }
    # Joint command topic of the forward position controller (the name is fixed
    # by task.info's follow_topic_name).
    ARM_CMD_TOPIC = "/tiangong2pro_arm_forward_position_controller/commands"

    # Frame names
    BASE_LINK = "pelvis"
    LEFT_TCP_LINK = "left_tcp_link"
    RIGHT_TCP_LINK = "right_tcp_link"

    # TF child frames for the VR targets, parented to BASE_LINK. They let RViz
    # show the commanded target next to the measured left/right_tcp_link, which
    # is what the frame-mismatch probe in docs/vr_teleop_architecture.md needs.
    VR_TARGET_FRAME = {'left': "left_vr_target", 'right': "right_vr_target"}
    # step_teleop runs on the 5 ms pub tick; no point streaming TF at 200 Hz.
    TF_PUBLISH_INTERVAL = 0.02

    # Gripper constants
    GRIPPER_OPEN_THRESHOLD = 0.9
    GRIPPER_CLOSE_THRESHOLD = 0.1
    GRIPPER_SMOOTHING_ALPHA = 0.3  # Lower is smoother but responds more slowly.
    GRIPPER_BINARY_OPEN_THRESHOLD = 0.6
    GRIPPER_BINARY_CLOSE_THRESHOLD = 0.3
    GRIPPER_ACTION_INDEX = 7

    # Reset constants
    RESET_POSE_TOLERANCE = 0.01  # m
    RESET_ROTATION_TOLERANCE = 0.05  # rad
    RESET_POSE_TIMEOUT = 1.0  # s
    RESET_CARTESIAN_DURATION = 3.0  # s, minimum-jerk Cartesian trajectory

    RESET_DURATION = 2.0  # s, joint-space interpolation after reaching the target pose
    RESET_PUB_PERIOD = 0.01  # s, 100 Hz Cartesian/joint streaming and pose polling

    DEBUG = False

    def __init__(self, robot_type: str = "tiangong2pro", enable_filter: bool = False):
        # Ensure rclpy is initialized before creating the ROS 2 node, so the controller
        # works both standalone and embedded in LeRobot (same as the Astrabot controller).
        if not rclpy.ok():
            rclpy.init()
        super().__init__("tiangong2pro_robot_node")
        self.robot_type = robot_type
        # Kept for API compatibility: like the Astrabot controller, there is no VR smoothing.
        self.enable_vr_filter = enable_filter
        self._init_ros_components()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

    def _init_ros_components(self):
        # publishers
        self.pub_dict = {
            'action': self.create_publisher(String, "/reference/pose", 10),
            'arm_cmd': self.create_publisher(Float64MultiArray, self.ARM_CMD_TOPIC, 10),
            'vr_left_ee': self.create_publisher(Pose, "/tiangong2pro/left_ee_vr/pose", 10),
            'vr_right_ee': self.create_publisher(Pose, "/tiangong2pro/right_ee_vr/pose", 10),
        }

        # Gripper command publishers are opt-in: the simulation stack has no gripper,
        # the real robot topics are configured via env.
        for side, var in (('left', "GRIPPER_CMD_TOPIC_LEFT"), ('right', "GRIPPER_CMD_TOPIC_RIGHT")):
            topic = os.getenv(var, f"/{side}_gripper_controller/commands").strip()
            if topic:
                self.pub_dict[f'{side}_gripper'] = self.create_publisher(Float64, topic, 10)
                print(f"[tiangong2pro_robot_controller]: {side} gripper command topic: {topic}")
            else:
                self.pub_dict[f'{side}_gripper'] = None

        self.tf_broadcaster = TransformBroadcaster(self)
        self._last_vr_tf_time = 0.0
        self._joint_states_received = False

        self.create_subscription(JointState, "/joint_states", self._joint_states_callback, 10)

    def setup(self):
        print("[tiangong2pro_robot_controller]: Setting up ....")
        self.tf_timeout = 5.0

        # Initialize Joint Positions
        self.joint_positions = {
            'left': [0.0] * len(self.LEFT_JOINTS_NAME),
            'right': [0.0] * len(self.RIGHT_JOINTS_NAME),
            'head': [0.0] * len(self.HEAD_JOINTS_NAME),
        }
        # Initialize Gripper State
        self.gripper_state = {
            'left': {'cmd': 0.0, 'state': 0.0, 'filtered': 0.0},
            'right': {'cmd': 0.0, 'state': 0.0, 'filtered': 0.0}
        }

        # Initialize Transform
        self.tf_init = Transform(
            rotation=Rotation.from_euler(seq="ZYX", angles=[0., 0., 0.], degrees=True),
            translation=np.zeros(3)
        )
        # Initialize ee States
        self.ee_states = {
            'left': {
                'origin': self.tf_init,
                'vr_ee_pose': self.tf_init,
                'current_ee_pose': self.tf_init
            },
            'right': {
                'origin': self.tf_init,
                'vr_ee_pose': self.tf_init,
                'current_ee_pose': self.tf_init
            }
        }

        #hold gripper
        self.gripper_control_state = {
            'left': {'hold': False, 'trigger': False},
            'right': {'hold': False, 'trigger': False}
        }

        self.is_radian = True
        self.start_spin_thread()

        if not self._wait_for_tf():
            raise Exception("[tiangong2pro_robot_controller]: failed to connect TF")

        print("[tiangong2pro_robot_controller]: Setup completed ....")

    def start_spin_thread(self):
        self.spin_thread = threading.Thread(target=rclpy.spin, args=(self,))
        self.spin_thread.daemon = True
        self.spin_thread.start()

    def stop_spin_thread(self, timeout: float = 2.0) -> None:
        """Join the spin threads after rclpy.shutdown().

        A daemon thread that is still joinable when the interpreter exits makes
        the C++ layer abort with "terminate called without an active exception"
        (the process dies on SIGABRT, which makes `dora run` report failure even
        though the dataflow itself finished cleanly). Call this *after*
        rclpy.shutdown(), which is what makes these threads return.
        """
        for name in ("spin_thread", "tf_listener"):
            holder = getattr(self, name, None)
            thread = getattr(holder, "thread", None) if name == "tf_listener" else holder
            if thread is None:
                continue
            try:
                thread.join(timeout=timeout)
            except Exception as e:
                print(f"[tiangong2pro_robot_controller]: could not join {name}: {e}")

    def _wait_for_tf(self, timeout: float = 10.0) -> bool:
        print("[tiangong2pro_robot_controller]: Waiting for TF connection ...")
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                left_ok = self.tf_buffer.can_transform(self.BASE_LINK, self.LEFT_TCP_LINK, rclpy.time.Time())
                right_ok = self.tf_buffer.can_transform(self.BASE_LINK, self.RIGHT_TCP_LINK, rclpy.time.Time())

                if left_ok and right_ok:
                    print("[tiangong2pro_robot_controller]: TF connected successfully ...")
                    return True

                time.sleep(1.0)
            except Exception as e:
                print(f"[tiangong2pro_robot_controller]: TF check error: {e} ...")
                time.sleep(1.0)

        print("[tiangong2pro_robot_controller]: TF connection timeout ...")
        return False

    def abs_quat_to_rot6d(
        self,
        relative_teleop_data: np.ndarray,
        side: Literal['left', 'right'],
        return_raw: bool = True,
        return_filtered: bool = True,
    ) -> dict:
        """
        Relative value to absolute value
        Convert quaternion to rotation6d

        Args:
            relative_teleop_data: Input teleop data [x, y, z, qx, qy, qz, qw, ...]
            side: 'left' or 'right' arm
            return_raw: Whether to return raw VR data
            return_filtered: Whether to also return the target in the same encoding

        Returns:
            dict: Contains requested data types:
                'raw_vr_data': np.ndarray (if return_raw=True)
                'filter_vr_data': np.ndarray (if return_filtered=True)

        Raises:
            ValueError: If both return_raw and return_filtered are False
        """
        if not return_raw and not return_filtered:
            raise ValueError("At least one of return_raw or return_filtered must be True")

        tf_teleop = Transform(
            rotation=Rotation.from_quat(relative_teleop_data[3:7]),
            translation=relative_teleop_data[:3]
        )
        self.fixed_teleop_to_ee = Transform(
            rotation=Rotation.from_euler(seq="ZYX", angles=[0., -90., 0.], degrees=True),
            translation=np.zeros(3)
        )

        abs_vr_pose = self.ee_states[side]['origin'] * self.fixed_teleop_to_ee.inverse() * tf_teleop * self.fixed_teleop_to_ee

        # Keep the raw target around: step_teleop reads it back (it is what
        # /reference/pose carries) and it is the pose broadcast as
        # BASE_LINK -> *_vr_target in TF.
        self.ee_states[side]['vr_ee_pose'] = abs_vr_pose

        if self.DEBUG:
            track_frequency(f"robot_control_abs_quat_to_rot6d", print_interval=5.0)

        result = {}

        # Get raw VR data if requested
        if return_raw:
            abs_vr_pose_6d = abs_vr_pose.to_rep(to_rep="rotation_6d")
            raw_vr_data = np.concatenate([abs_vr_pose_6d, [self.gripper_state[side]['cmd']]])
            result['raw_vr_data'] = raw_vr_data

        # Get the (unfiltered) target in the same encoding, if requested
        if return_filtered:
            if 'raw_vr_data' in result:
                filtered_vr_data = result['raw_vr_data'].copy()
            else:
                abs_vr_pose_6d = abs_vr_pose.to_rep(to_rep="rotation_6d")
                filtered_vr_data = np.concatenate([abs_vr_pose_6d, [self.gripper_state[side]['cmd']]])
            result['filter_vr_data'] = filtered_vr_data

        return result

    def quat_to_rot6d(self, tf: np.ndarray, side: Literal['left', 'right']) -> np.ndarray:
        tf_teleop = Transform(
            rotation=Rotation.from_quat(tf[3:7]),
            translation=tf[:3]
        )
        target_ee = tf_teleop.to_rep(to_rep="rotation_6d")

        return np.concatenate([target_ee, [self.gripper_state[side]['cmd']]])

    def step_gripper(self, action: np.ndarray, side: Literal['left', 'right']) -> None:
        gripper_position = float(action[self.GRIPPER_ACTION_INDEX])


        target = float(np.clip(gripper_position, 0.0, 1.0))
        previous = self.gripper_state[side]['filtered']
        filtered = previous + self.GRIPPER_SMOOTHING_ALPHA * (target - previous)
        gripper_position = self.gripper_state[side]['cmd']
        if filtered >= self.GRIPPER_BINARY_OPEN_THRESHOLD:
            gripper_position = 1.0
        elif filtered <= self.GRIPPER_BINARY_CLOSE_THRESHOLD:
            gripper_position = 0.0
        self.gripper_state[side]['filtered'] = filtered
        self.gripper_state[side]['cmd'] = gripper_position
        self.gripper_state[side]['state'] = gripper_position
        # if side == "right":
        #     print(f'girpper_position = {gripper_position}', flush=True)

        publisher = self.pub_dict.get(f'{side}_gripper')
        if publisher is not None:
            msg = Float64()
            # msg.data = self.gripper_state[side]['cmd']
            msg.data = gripper_position
            publisher.publish(msg)

    def step_teleop(self):
        if self.DEBUG:
            track_frequency(f"robot_control_step_teleop", print_interval=5.0)

        left_pos = self.ee_states['left']['vr_ee_pose'].translation.data
        left_quat = self.ee_states['left']['vr_ee_pose'].rotation.as_quat()
        right_pos = self.ee_states['right']['vr_ee_pose'].translation.data
        right_quat = self.ee_states['right']['vr_ee_pose'].rotation.as_quat()
        self.publish_vr_ee_pose(np.concatenate([left_pos, left_quat]), "left")
        self.publish_vr_ee_pose(np.concatenate([right_pos, right_quat]), "right")

        self.publish_vr_tf_both()

        msg_data = {
            "left_arm": {
                "position": [float(left_pos[0]), float(left_pos[1]), float(left_pos[2])],
                "orientation": [float(left_quat[0]), float(left_quat[1]), float(left_quat[2]), float(left_quat[3])]
            },
            "right_arm": {
                "position": [float(right_pos[0]), float(right_pos[1]), float(right_pos[2])],
                "orientation": [float(right_quat[0]), float(right_quat[1]), float(right_quat[2]), float(right_quat[3])]
            }
        }
        message = String()
        message.data = json.dumps(msg_data)
        self.pub_dict['action'].publish(message)

    def step_policy(self, left_action, right_action):
        msg_data = {
            "left_arm": {
                "position": [float(left_action[0]),
                             float(left_action[1]),
                             float(left_action[2])],
                "orientation": [float(left_action[3]),
                                float(left_action[4]),
                                float(left_action[5]),
                                float(left_action[6])]
            },
            "right_arm": {
                "position": [float(right_action[0]),
                             float(right_action[1]),
                             float(right_action[2])],
                "orientation": [float(right_action[3]),
                                float(right_action[4]),
                                float(right_action[5]),
                                float(right_action[6])]
            }
        }
        message = String()
        message.data = json.dumps(msg_data)
        self.pub_dict['action'].publish(message)

    def get_state(self):
        try:
            l_ee_pose, r_ee_pose = self.get_arm_ee_pose()

            for side, ee_pose in zip(['left', 'right'], [l_ee_pose, r_ee_pose]):
                self.ee_states[side]['current_ee_pose'] = Transform(
                    rotation=Rotation.from_quat(ee_pose[3:]),
                    translation=ee_pose[:3]
                )
            ee_pose_l = self.ee_states['left']['current_ee_pose'].to_rep(to_rep="rotation_6d")
            ee_pose_r = self.ee_states['right']['current_ee_pose'].to_rep(to_rep="rotation_6d")

            state = {
                "ee_pose_l": ee_pose_l,
                "joint_angle_l": self.joint_positions['left'],
                "gripper_l": self.gripper_state['left']['state'],
                "ee_pose_r": ee_pose_r,
                "joint_angle_r": self.joint_positions['right'],
                "gripper_r": self.gripper_state['right']['state'],
                "joint_angle_head": self.joint_positions['head'],
                }

            return state

        except Exception as e:
            print(f"[tiangong2pro_robot_controller]: Error in getState: {e} ...")
            raise

    def get_arm_ee_pose(self):
        try:
            transforms = {}
            for side, link in [('left', self.LEFT_TCP_LINK), ('right', self.RIGHT_TCP_LINK)]:
                tcp = self.tf_buffer.lookup_transform(self.BASE_LINK, link, rclpy.time.Time())
                pos = [tcp.transform.translation.x, tcp.transform.translation.y, tcp.transform.translation.z]
                ori = [tcp.transform.rotation.x, tcp.transform.rotation.y,
                      tcp.transform.rotation.z, tcp.transform.rotation.w]
                transforms[side] = np.array(pos + ori)

            return transforms['left'], transforms['right']
        except Exception:
            print(f"[tiangong2pro_robot_controller]: Failed to get the end-effector pose ...")
            raise

    def publish_vr_ee_pose(self, ee_pose: np.ndarray, side: Literal['left', 'right']):
        try:
            if ee_pose is None or len(ee_pose) < 7:
                print(f"[tiangong2pro_robot_controller]: Invalid {side} ee_pose: {ee_pose} ...")
                return

            pose_msg = Pose(
                position=Point(x=float(ee_pose[0]), y=float(ee_pose[1]), z=float(ee_pose[2])),
                orientation=Quaternion(x=float(ee_pose[3]), y=float(ee_pose[4]),
                                     z=float(ee_pose[5]), w=float(ee_pose[6]))
            )
            self.pub_dict[f'vr_{side}_ee'].publish(pose_msg)
        except Exception as e:
            print(f"[tiangong2pro_robot_controller]: Error publishing {side} ee_pose: {e} ...")

    def publish_vr_tf_both(self) -> None:
        """Broadcast both VR target frames, rate-limited to TF_PUBLISH_INTERVAL."""
        now = time.perf_counter()
        if now - self._last_vr_tf_time < self.TF_PUBLISH_INTERVAL:
            return
        self._last_vr_tf_time = now

        for side in ('left', 'right'):
            self.publish_vr_tf(side)

    def publish_vr_tf(self, side: Literal['left', 'right']) -> None:
        """Broadcast a VR target pose into TF as BASE_LINK -> *_vr_target.

        The target is the pose computed in abs_quat_to_rot6d and handed to
        /reference/pose. It is expressed in the pelvis frame, so RViz can overlay
        it on the measured left/right_tcp_link to spot a teleop frame mismatch.
        """
        try:
            tf = self.ee_states[side]['vr_ee_pose']
            quat = tf.rotation.as_quat()  # xyzw
            pos = tf.translation

            transform = TransformStamped()
            transform.header.stamp = self.get_clock().now().to_msg()
            transform.header.frame_id = self.BASE_LINK
            transform.child_frame_id = self.VR_TARGET_FRAME[side]
            transform.transform.translation.x = float(pos[0])
            transform.transform.translation.y = float(pos[1])
            transform.transform.translation.z = float(pos[2])
            transform.transform.rotation.x = float(quat[0])
            transform.transform.rotation.y = float(quat[1])
            transform.transform.rotation.z = float(quat[2])
            transform.transform.rotation.w = float(quat[3])
            self.tf_broadcaster.sendTransform(transform)
        except Exception as e:
            print(f"[tiangong2pro_robot_controller]: Error publishing {side} VR TF: {e} ...")

    def set_new_origin(self, side: Literal['left', 'right']) -> None:
        # Keep the idle side's target at its measured pose too: step_teleop()
        # publishes both arms as soon as either hand trigger is held.
        self.ee_states[side]['origin'] = self.ee_states[side]['current_ee_pose']
        self.ee_states[side]['vr_ee_pose'] = self.ee_states[side]['current_ee_pose']

    def _joint_states_callback(self, msg):
        self._joint_states_received = True
        joint_configs = [
            ('left', self.LEFT_JOINTS_NAME),
            ('right', self.RIGHT_JOINTS_NAME),
            ('head', self.HEAD_JOINTS_NAME),
        ]

        for side, joint_names in joint_configs:
            for idx, name in enumerate(joint_names):
                if name in msg.name:
                    self.joint_positions[side][idx] = msg.position[msg.name.index(name)]

    def reset(self) -> None:
        """Return both arms to the initial pose (long-press B).

        A minimum-jerk Cartesian trajectory from the current TCP poses to
        INITIAL_TCP_POSE is streamed to the MPC. Only after both arms reach the
        target are the joints interpolated to INITIAL_JOINTS.
        """
        # Interpolating from the all-zero defaults would command a large jump.
        if not self._joint_states_received:
            raise RuntimeError(
                "[tiangong2pro_robot_controller]: no /joint_states received yet, refusing to reset."
            )

        # target_poses = {
        #     side: np.array(position + orientation, dtype=float)
        #     for side, (position, orientation) in self.INITIAL_TCP_POSE.items()
        # }
        # try:
        #     start_poses = dict(zip(('left', 'right'), self.get_arm_ee_pose(), strict=True))
        # except Exception as e:
        #     raise RuntimeError(
        #         "[tiangong2pro_robot_controller]: failed to read the current TCP poses, "
        #         "refusing to reset."
        #     ) from e

        # start_rotations = {
        #     side: Rotation.from_quat(pose[3:]) for side, pose in start_poses.items()
        # }
        # rotation_deltas = {
        #     side: (
        #         Rotation.from_quat(target_poses[side][3:]) * start_rotations[side].inv()
        #     ).as_rotvec()
        #     for side in start_poses
        # }
        # trajectory_start = time.perf_counter()
        # while True:
        #     ratio = min(
        #         (time.perf_counter() - trajectory_start) / self.RESET_CARTESIAN_DURATION,
        #         1.0,
        #     )
        #     # Quintic time scaling gives zero velocity and acceleration at both ends.
        #     blend = ratio**3 * (10.0 - 15.0 * ratio + 6.0 * ratio**2)
        #     waypoints = {}
        #     for side, start_pose in start_poses.items():
        #         rotation = Rotation.from_rotvec(blend * rotation_deltas[side]) * start_rotations[side]
        #         waypoints[side] = np.concatenate(
        #             ((1.0 - blend) * start_pose[:3] + blend * target_poses[side][:3], rotation.as_quat())
        #         )
        #     self.step_policy(waypoints['left'], waypoints['right'])
        #     if ratio >= 1.0:
        #         break
        #     time.sleep(self.RESET_PUB_PERIOD)
        # print("[tiangong2pro_robot_controller]: reset Cartesian trajectory sent ...")

        # deadline = time.perf_counter() + self.RESET_POSE_TIMEOUT
        # while True:
        #     try:
        #         current_poses = dict(zip(('left', 'right'), self.get_arm_ee_pose(), strict=True))
        #         position_errors = []
        #         rotation_errors = []
        #         for side, current_pose in current_poses.items():
        #             target_pose = target_poses[side]
        #             position_errors.append(np.linalg.norm(current_pose[:3] - target_pose[:3]))
        #             rotation_error = (
        #                 Rotation.from_quat(target_pose[3:])
        #                 * Rotation.from_quat(current_pose[3:]).inv()
        #             )
        #             rotation_errors.append(np.linalg.norm(rotation_error.as_rotvec()))
        #         if (
        #             max(position_errors) <= self.RESET_POSE_TOLERANCE
        #             and max(rotation_errors) <= self.RESET_ROTATION_TOLERANCE
        #         ):
        #             break
        #     except Exception:
        #         pass

        #     if time.perf_counter() >= deadline:
        #         raise RuntimeError(
        #             "[tiangong2pro_robot_controller]: timed out waiting for the reset pose; "
        #             "joint reset was not started."
        #         )
        #     time.sleep(self.RESET_PUB_PERIOD)

        # print("[tiangong2pro_robot_controller]: reset pose reached, adjusting joints ...")
        start = np.array([self.joint_positions[side] for side in ('left', 'right')], dtype=float)
        target = np.array([self.INITIAL_JOINTS[side] for side in ('left', 'right')], dtype=float)

        cmd = Float64MultiArray()
        t0 = time.perf_counter()
        while True:
            ratio = min((time.perf_counter() - t0) / self.RESET_DURATION, 1.0)
            cmd.data = ((1.0 - ratio) * start + ratio * target).ravel().tolist()
            self.pub_dict['arm_cmd'].publish(cmd)
            if ratio >= 1.0:
                break
            time.sleep(self.RESET_PUB_PERIOD)

        # cmd.data = target.ravel().tolist()
        # self.pub_dict['arm_cmd'].publish(cmd)
        print(f"[tiangong2pro_robot_controller]: initial joint command sent to "
              f"{self.ARM_CMD_TOPIC} ({len(cmd.data)} joints) ...")

        # The reset bypasses the teleop path, so update the saved robot state
        # here: joint angles plus the EE anchors (origin / current / VR targets).
        for side in ('left', 'right'):
            self.joint_positions[side] = [float(v) for v in self.INITIAL_JOINTS[side]]
            position, orientation = self.INITIAL_TCP_POSE[side]
            init_pose = Transform(
                rotation=Rotation.from_quat(orientation),
                translation=np.array(position, dtype=float)
            )
            self.ee_states[side]['origin'] = init_pose
            self.ee_states[side]['current_ee_pose'] = init_pose
            self.ee_states[side]['vr_ee_pose'] = init_pose

    def renew_ee_pose(self):
        # The commanded target goes back to the origin.
        for side in ('left', 'right'):
            self.ee_states[side]['vr_ee_pose'] = self.ee_states[side]['origin']

    def release(self):
        print("[tiangong2pro_robot_controller]: release ....")
        # Note: spin_thread is daemon thread, will be cleaned up automatically when main exits
        self.destroy_node()


def main():
    rclpy.init()
    try:
        robot_type = os.getenv("ROBOT_TYPE", "tiangong2pro")
        node = Tiangong2ProRobotController(robot_type=robot_type)
        node.setup()  # This already starts a spin thread

        time.sleep(1.0)
        node.get_state()
        time.sleep(1.0)

    except Exception as e:
        print(f"Error in main: {e}")
    finally:
        if 'node' in locals():
            node.release()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
