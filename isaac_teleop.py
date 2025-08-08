from avp_stream.isaac_env import IsaacVisualizerEnv
from avp_stream.isaac_tasks import IsaacTasks
from avp_stream import VisionProStreamer
import time 
from typing import * 
import numpy as np 
import torch

from pathlib import Path
from dex_retargeting.constants import RobotName, RetargetingType, HandType, get_config_path
from dex_retargeting.retargeting_config import RetargetingConfig
from dex_retargeting.seq_retarget import SeqRetargeting
import copy
from avp_stream.utils.se3_utils import * 

from sensor_msgs.msg import PointCloud
from geometry_msgs.msg import Point32
import std_msgs
import rospy

from avp_stream.tasks.shadow_hand_stack_blocks import *

ROT_X = np.array([[[1, 0, 0, 0], 
                    [0, 0, -1, 0], 
                    [0, 1, 0, 0],
                    [0, 0, 0, 1]]], dtype = np.float64)

ROT_Y = np.array([[[0, 0, 1, 0], 
                    [0, 1,  0, 0], 
                    [-1, 0, 0, 0],
                    [0, 0, 0, 1]]], dtype = np.float64)

ROT_Y_ = np.array([[[0, 0, -1, 0], 
                    [0, 1,  0, 0], 
                    [1, 0, 0, 0],
                    [0, 0, 0, 1]]], dtype = np.float64)

OPERATOR2MANO_RIGHT = np.array(
    [
        [0, 0, -1],
        [-1, 0, 0],
        [0, 1, 0],
    ]
)

OPERATOR2MANO_LEFT = np.array(
    [
        [0, 0, -1],
        [1, 0, 0],
        [0, -1, 0],
    ]
)
APPLE2MEDIAPIPE = [
    1,2,3,4,
    6,7,8,9, 
    11,12,13,14, 
    16,17,18,19,
    21,22,23,24
]
class IsaacVisualizer:

    def __init__(self, args): 
        # self.s = VisionProStreamer(args.ip, args.record)
        # self.env = IsaacVisualizerEnv(args)
        args.task = "ShadowHandStackBlocks"
        self.env = eval(args.task)(args)

        # self.retargeting_type = RetargetingType.vector
        self.retargeting_type = RetargetingType.custom
        self.robot_name = RobotName.shadow

        self.right_config_path = get_config_path(self.robot_name, self.retargeting_type, HandType.right)
        self.left_config_path = get_config_path(self.robot_name, self.retargeting_type, HandType.left)

        print("self.right_config_path: ", self.right_config_path)
        print("self.left_config_path: ", self.left_config_path)

        self.robot_dir = Path(__file__).parent.parent / "assets" / "robots"

        RetargetingConfig.set_default_urdf_dir(str(self.robot_dir))

        self.right_retargeting = RetargetingConfig.load_from_file(self.right_config_path).build()
        self.left_retargeting = RetargetingConfig.load_from_file(self.left_config_path).build()



    def right_hand_retarget(self, transformations):

        right_wrist = transformations['right_wrist'] #@ VISIONOS_RIGHT_HAND_TO_LEAP 
        right_fingers = torch.cat([right_wrist @ finger for finger in transformations['right_fingers']], dim = 0)

        right_wrist_rot = (right_wrist.numpy() @ ROT_X @ ROT_Y_)[0][0:3,0:3]
        # print("right_wrist_rot: ", right_wrist_rot.shape)
        keypoint_3d = []
        keypoint_3d.append( right_wrist[0][0:3,3] )
        for idx in APPLE2MEDIAPIPE:
            keypoint_3d.append( right_fingers[idx][0:3,3] )  
        keypoint_3d = torch.stack( keypoint_3d )
        keypoint_3d = keypoint_3d.numpy()
        keypoint_3d = keypoint_3d.reshape(21,-1)
        keypoint_3d_world = copy.deepcopy(keypoint_3d)
        # print("keypoint_3d: ", keypoint_3d.shape)
        keypoint_3d = keypoint_3d - keypoint_3d[0:1, :]

        joint_pos = keypoint_3d @ right_wrist_rot
        retargeting_type = self.right_retargeting.optimizer.retargeting_type

        # print("retargeting_type: ", retargeting_type)

        indices = self.right_retargeting.optimizer.target_link_human_indices
        
        origin_indices = indices[0, :]
        task_indices = indices[1, :]
        # print("origin_indices: ", origin_indices)
        # print("task_indices: ", task_indices)        
        ref_value = joint_pos[task_indices, :] - joint_pos[origin_indices, :]

        qpos = self.right_retargeting.retarget(ref_value)
        return qpos


    def left_hand_retarget(self, transformations):

        left_wrist = transformations['left_wrist'] #@ VISIONOS_LEFT_HAND_TO_LEAP 
        left_fingers = torch.cat([left_wrist @ finger for finger in transformations['left_fingers']], dim = 0)

        left_wrist_rot = (left_wrist.numpy() @ ROT_X @ ROT_Y)[0][0:3,0:3]

        keypoint_3d = []
        keypoint_3d.append( left_wrist[0][0:3,3] )
        for idx in APPLE2MEDIAPIPE:
            keypoint_3d.append( left_fingers[idx][0:3,3] )  
        keypoint_3d = torch.stack( keypoint_3d )
        keypoint_3d = keypoint_3d.numpy()
        keypoint_3d = keypoint_3d.reshape(21,-1)
        keypoint_3d_world = copy.deepcopy(keypoint_3d)
        keypoint_3d = keypoint_3d - keypoint_3d[0:1, :]
        joint_pos = keypoint_3d @ left_wrist_rot
        retargeting_type = self.left_retargeting.optimizer.retargeting_type
        indices = self.left_retargeting.optimizer.target_link_human_indices
        
        origin_indices = indices[0, :]
        task_indices = indices[1, :]
        ref_value = joint_pos[task_indices, :] - joint_pos[origin_indices, :]

        qpos = self.left_retargeting.retarget(ref_value)
        return qpos

    def run(self):
        data = np.load("test.npy", allow_pickle = True)
        for j in range(2):
            for i in range(1000):
                # time.sleep(0.05)

                latest = data[i]

                transformations = copy.deepcopy(latest)
                visionos_head = transformations['head']

                right_qpos = self.right_hand_retarget(transformations)
                left_qpos = self.left_hand_retarget(transformations)
                print("right_qpos: ", right_qpos)
                print("left_qpos: ", left_qpos)                
                action = np.concatenate( [right_qpos, left_qpos] )
                print("action: ", action.shape)
                # self.env.step(np2tensor(latest, self.env.device))
            # np.save("test.npy", data, allow_pickle=True)

# python3 example/retarget_debug_node.py \
#   --robot-name shadow \
#   --retargeting-type custom \
#   --hand-type right \
#   --output-path example/data/shadow_hand.pkl  

def np2tensor(data: Dict[str, np.ndarray], device) -> Dict[str, torch.Tensor]:  
    for key in data.keys():
        data[key] = torch.tensor(data[key].clone().detach(), dtype = torch.float32, device = device)
    return data


if __name__ == "__main__":
    import argparse 
    import os 

    parser = argparse.ArgumentParser()
    parser.add_argument('--ip', type = str, default = "192.168.1.99")
    parser.add_argument('--record', action = 'store_true')
    parser.add_argument('--follow', action = 'store_true', help = "The viewpoint follows the users head")
    args = parser.parse_args()

    vis = IsaacVisualizer(args)
    # vis.run()
