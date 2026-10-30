"""
Author: yanjun.wang@quantgroup.com
Created: 2025-04-18
Description: astrabot robot
"""
import os
import time
import json
import numpy as np
import threading
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64, String
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose, Point, Quaternion
from tf2_ros import Buffer, TransformListener
from typing import Literal
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import UInt32MultiArray
from collections import deque
from scipy.spatial.transform import Rotation as R

import sys
sys.path.append("..")
from .transform_util import Transform, Rotation
from .utils import track_frequency




class AstrabotRobotController(Node):
    # Joint names
    LEFT_JOINTS_NAME = ["left_arm_1_joint", "left_arm_2_joint", "left_arm_3_joint", "left_arm_4_joint",
                        "left_arm_5_joint", "left_arm_6_joint", "left_arm_7_joint"]
    RIGHT_JOINTS_NAME = ["right_arm_1_joint", "right_arm_2_joint", "right_arm_3_joint", "right_arm_4_joint",
                        "right_arm_5_joint", "right_arm_6_joint", "right_arm_7_joint"]
    HEAD_JOINTS_NAME = ["head_pitch_joint", "head_yaw_joint"]
    
    # Frame names
    BASE_LINK = "base_link"
    # BASE_LINK = "odom"
    LEFT_TCP_LINK = "left_tcp_link"
    RIGHT_TCP_LINK = "right_tcp_link"

    # Gripper constants
    GRIPPER_OPEN_THRESHOLD = 0.9
    GRIPPER_CLOSE_THRESHOLD = 0.1
    GRIPPER_ACTION_INDEX = 7
    
    # Filter constants
    DEFAULT_WINDOW_SIZE = 20
    DEFAULT_MAX_DT_THRESHOLD = 0.5
    
    # Reset position constants


    # 抓取孔板初始位姿第一版
    # DEFAULT_LEFT_RESET_POSITION = [0.22677, 0.43133, 0.91769]
    # DEFAULT_LEFT_RESET_ORIENTATION = [-0.12058, 0.74318, -0.64789, 0.1157]
    # DEFAULT_RIGHT_RESET_POSITION = [0.172, -0.377, 0.875]
    # DEFAULT_RIGHT_RESET_ORIENTATION = [0.513, -0.379, 0.613, -0.467]

    # 抓取孔板初始位姿第二版(xyzw)
    DEFAULT_LEFT_RESET_POSITION = [0.271, 0.454, 0.923]
    DEFAULT_LEFT_RESET_ORIENTATION = [-0.308, 0.698, -0.577, 0.291]
    DEFAULT_RIGHT_RESET_POSITION = [0.302, -0.280, 0.900]
    DEFAULT_RIGHT_RESET_ORIENTATION = [-0.668, 0.251, -0.355, 0.604]

    # DEFAULT_LEFT_RESET_POSITION = [0.43019, 0.40167, 0.95295]# <-xr1 fangwan
    # DEFAULT_LEFT_RESET_ORIENTATION = [0.69743, 0.02863, 0.71607, 0.0043522]# <-xr1 fangwan
    # DEFAULT_RIGHT_RESET_POSITION = [0.29662, -0.41482, 0.96381]# <-xr1 fangwan
    # DEFAULT_RIGHT_RESET_ORIENTATION = [-0.64599, 0.2426, -0.2524, 0.67834]# <-xr1 fangwan

    # DEFAULT_LEFT_RESET_POSITION = [0.31955, 0.45057, 0.90892]# <-xr1 chucan
    # DEFAULT_LEFT_RESET_ORIENTATION = [0.34809, -0.63131, 0.64286, -0.25889]# <-xr1 chucan
    # DEFAULT_RIGHT_RESET_POSITION = [0.30538, -0.4339, 0.90428]# <-xr1 chucan 
    # DEFAULT_RIGHT_RESET_ORIENTATION = [-0.66588, 0.36316, -0.28034, 0.58832]# <-xr1 chucan

    # DEFAULT_LEFT_RESET_POSITION = [0.31789, 0.35648, 1.048]# <-xr1 shangliao2(slh/flh)
    # DEFAULT_LEFT_RESET_ORIENTATION = [0.13236, -0.62202, 0.72251, -0.27121]# <-xr1 shangliao2
    # DEFAULT_RIGHT_RESET_POSITION = [0.44444, -0.3512, 1.0897]# <-xr1 shangliao2
    # DEFAULT_RIGHT_RESET_ORIENTATION = [0.013894, 0.68368, -0.024735, 0.72923]# <-xr1 shangliao2

    # DEFAULT_LEFT_RESET_POSITION = [0.31792, 0.35594, 1.0484]# <-xr1 shangliao3(xlh)
    # DEFAULT_LEFT_RESET_ORIENTATION = [0.13343, -0.62205, 0.72212, -0.27163]# <-xr1 shangliao3
    # DEFAULT_RIGHT_RESET_POSITION = [0.35655, -0.1535, 1.3192]# <-xr1 shangliao3
    # DEFAULT_RIGHT_RESET_ORIENTATION = [-0.49767, 0.51945, -0.4841, 0.49814]# <-xr1 shangliao3

    # DEFAULT_LEFT_RESET_POSITION = [0.31955, 0.45057, 0.90892]# <-xr1 chucan (daomi)
    # DEFAULT_LEFT_RESET_ORIENTATION = [0.34809, -0.63131, 0.64286, -0.25889]# <-xr1 chucan (daomi)
    # DEFAULT_RIGHT_RESET_POSITION = [0.32195, -0.30493, 0.93528]# <-xr1 chucan (daomi)
    # DEFAULT_RIGHT_RESET_ORIENTATION = [-0.60908, 0.28341, -0.29157, 0.68095]# <-xr1 chucan (daomi)

    # DEFAULT_LEFT_RESET_POSITION = [0.1892, 0.349,1.114] #[0.26255, 0.30625, 1.0111](chongdian) #[0.1892, 0.349,1.114] (dafan) #[0.042814, 0.31257, 0.86241](shangliao)
    # DEFAULT_LEFT_RESET_ORIENTATION = [-0.381, 0.708, -0.548, 0.229] #[0.30405, -0.61654, 0.6682, -0.28451](chongdian) #[-0.381, 0.708, -0.548, 0.229] (dafan) #[-0.21327, 0.6674, -0.60693, 0.37514]
    # DEFAULT_RIGHT_RESET_POSITION = [0.1892, -0.349, 1.114] #[0.24616, -0.25157, 1.0208](chongdian) #[0.1892, -0.349, 1.114] (dafan) #[0.054747, -0.33047, 0.85755]
    # DEFAULT_RIGHT_RESET_ORIENTATION = [-0.727, 0.225, -0.267, 0.590] #[-0.5803, 0.26716, -0.31357, 0.70254](chongdian) #[-0.727, 0.225, -0.267, 0.590] (dafan) #[-0.67723, 0.25641, -0.30999, 0.61605]
    


    # DEFAULT_LEFT_RESET_POSITION = [0.042814, 0.31257, 0.86241] #<-zhuawanju#[0.22935, 0.33542, 1.1964]#<-shangliao 2 #[0.12003, 0.28594, 1.2652]#<-shangliao 1
    # DEFAULT_LEFT_RESET_ORIENTATION = [-0.21327, 0.6674, -0.60693, 0.37514] #<-zhuawanju#[0.22547, -0.54876, 0.7682, -0.2406]#<-shangliao 2 #[0.26479, -0.55028, 0.71139, -0.34784]#<-shangliao 1
    # DEFAULT_RIGHT_RESET_POSITION = [0.054747, -0.33047, 0.85755] #<-zhuawanju#[-0.10993, -0.32275, 0.96656]#<-shangliao 2 #[-0.10993, -0.32275, 0.96656]#<-shangliao 1
    # DEFAULT_RIGHT_RESET_ORIENTATION = [-0.67723, 0.25641, -0.30999, 0.61605] #<-zhuawanju #[-0.49696, 0.41388, -0.51961, 0.55834]#<-shangliao 2 #[-0.49696, 0.41388, -0.51961, 0.55834]#<-shangliao 1
    # Gripper conversion constants
    EVT11_GRIPPER_MAX_VALUE = 1000

    DEBUG = False

    def __init__(self, robot_type: Literal['evt1b', 'evt11'] = "evt1b", enable_filter = False):
        # Ensure rclpy is initialized before creating the ROS 2 node. This makes the
        # controller usable both as a standalone node (where the caller already called
        # rclpy.init()) and when embedded in LeRobot's Astrabot integration.
        if not rclpy.ok():
            rclpy.init()
        super().__init__("astrabot_robot_node")
        self.robot_type = robot_type
        self.enable_vr_filter = enable_filter
        self._init_ros_components()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)


    def _init_ros_components(self):
        # publishers
        self.pub_dict = {
            'action': self.create_publisher(String, "/reference/pose", 10),
            'left_gripper': self.create_publisher(Float64, "/rm_left/rm_driver/teleop_gripper_float", 10),
            'right_gripper': self.create_publisher(Float64, "/rm_right/rm_driver/teleop_gripper_float", 10),
        }

        self.create_subscription(JointState, "/joint_states", self._joint_states_callback, 10)

    def setup(self):
        print("[astrabot_robot]: Setting up ....")
        self.tf_timeout = 5.0

        # Initialize Joint Positions
        self.joint_positions = {
            'left': [0.0] * len(self.LEFT_JOINTS_NAME),
            'right': [0.0] * len(self.RIGHT_JOINTS_NAME),
        }
        # Initialize Gripper State
        self.gripper_state = {
            'left': {'cmd': 0.0, 'state': 0.0},
            'right': {'cmd': 0.0, 'state': 0.0}
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
                'vr_ee_pose_filter': self.tf_init,
                'current_ee_pose': self.tf_init
            },
            'right': {
                'origin': self.tf_init,
                'vr_ee_pose': self.tf_init,
                'vr_ee_pose_filter': self.tf_init,
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
            raise Exception("[astrabot_robot_controller]: failed to connect TF")

        print("[astrabot_robot_controller]: Setup completed ....")


    def start_spin_thread(self):
        self.spin_thread = threading.Thread(target=rclpy.spin, args=(self,))
        self.spin_thread.daemon = True
        self.spin_thread.start()


    def _wait_for_tf(self, timeout: float = 10.0) -> bool:
        print("[astrabot_robot_controller]: Waiting for TF connection ...")
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                left_ok = self.tf_buffer.can_transform(self.BASE_LINK, self.LEFT_TCP_LINK, rclpy.time.Time())
                right_ok = self.tf_buffer.can_transform(self.BASE_LINK, self.RIGHT_TCP_LINK, rclpy.time.Time())

                if left_ok and right_ok:
                    print("[astrabot_robot_controller]: TF connected successfully ...")
                    return True

                time.sleep(1.0)
            except Exception as e:
                print(f"[astrabot_robot_controller]: TF check error: {e} ...")
                time.sleep(1.0)

        print("[astrabot_robot_controller]: TF connection timeout ...")
        return False


    def step_gripper(self, action: np.ndarray, side: Literal['left', 'right']) -> None:
        gripper_position = float(action[self.GRIPPER_ACTION_INDEX])

        control_state = self.gripper_control_state[side]
        if not control_state['hold']:
            if gripper_position > self.GRIPPER_OPEN_THRESHOLD:
                control_state['trigger'] = True
            elif control_state['trigger'] and gripper_position < self.GRIPPER_CLOSE_THRESHOLD:
                control_state['hold'] = True
                control_state['trigger'] = False
        else:
            if gripper_position > self.GRIPPER_OPEN_THRESHOLD:
                control_state['trigger'] = True
            elif control_state['trigger'] and gripper_position < self.GRIPPER_CLOSE_THRESHOLD:
                control_state['hold'] = False
                control_state['trigger'] = False

        if control_state['hold']:
            gripper_position = 1.0

        self.gripper_state[side]['cmd'] = 1.0 if gripper_position > 0.4 else 0.0
        self.gripper_state[side]['state'] = 1.0 if gripper_position > 0.4 else 0.0
        msg = Float64()
        msg.data = gripper_position
        self.pub_dict[f'{side}_gripper'].publish(msg)


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
                }

            return state

        except Exception as e:
            print(f"[astrabot_robot_controller]: Error in getState: {e} ...")
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
            print(f"[astrabot_robot_controller]: Failed to get the end-effector pose ...")
            raise


    def _joint_states_callback(self, msg):
        joint_configs = [
            ('left', self.LEFT_JOINTS_NAME),
            ('right', self.RIGHT_JOINTS_NAME),
        ]

        for side, joint_names in joint_configs:
            for idx, name in enumerate(joint_names):
                if name in msg.name:
                    self.joint_positions[side][idx] = msg.position[msg.name.index(name)]


    def reset(self) -> None:
        msg_data = {
            "left_arm": {
                "position": self.DEFAULT_LEFT_RESET_POSITION,
                "orientation": self.DEFAULT_LEFT_RESET_ORIENTATION
            },
            "right_arm": {
                "position": self.DEFAULT_RIGHT_RESET_POSITION,
                "orientation": self.DEFAULT_RIGHT_RESET_ORIENTATION
            }
        }

        msg = String()
        msg.data = json.dumps(msg_data)
        self.pub_dict['action'].publish(msg)
        

    def release(self):
        print("[astrabot_robot_controller]: release ....")
        # Note: spin_thread is daemon thread, will be cleaned up automatically when main exits
        # But we could add explicit cleanup if needed in the future
        self.destroy_node()





def main():
    rclpy.init()
    try:
        # Get robot type from environment variable
        robot_type = os.getenv("ROBOT_TYPE", "evt1b")
        node = AstrabotRobotController(robot_type=robot_type)
        node.setup()  # This already starts a spin thread

        time.sleep(1.0)
        for i in range(1):
            # action = np.random.rand(1, 16)  # Comment out since process method doesn't exist
            # node.process(action)            # Comment out since process method doesn't exist
            node.get_state()

        # Wait a bit more to let things settle
        time.sleep(1.0)

    except Exception as e:
        print(f"Error in main: {e}")
    finally:
        if 'node' in locals():
            node.release()
        rclpy.shutdown()




if __name__ == "__main__":
    main()
