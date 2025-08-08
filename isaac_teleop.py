from avp_stream.isaac_env import IsaacVisualizerEnv
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
        self.env = IsaacVisualizerEnv(args)

        self.retargeting_type = 'custom'
        self.robot_name = RobotName.shadow

        self.right_config_path = get_config_path(self.robot_name, self.retargeting_type, HandType.right)

        self.robot_dir = Path(__file__).parent.parent / "assets" / "robots"

        RetargetingConfig.set_default_urdf_dir(str(self.robot_dir))
        self.right_retargeting = RetargetingConfig.load_from_file(self.right_config_path).build()
   
        print("retargeting_type: ", self.retargeting_type)



    def run(self):
        data = np.load("test.npy", allow_pickle = True)
        for i in range(1000):
            latest = data[i]

            transformations = copy.deepcopy(latest)
            visionos_head = transformations['head']

            right_wrist = transformations['right_wrist'] #@ VISIONOS_RIGHT_HAND_TO_LEAP 
            left_wrist = transformations['left_wrist'] # @ VISIONOS_LEFT_HAND_TO_LEAP

            right_fingers = torch.cat([right_wrist @ finger for finger in transformations['right_fingers']], dim = 0)
            left_fingers = torch.cat([left_wrist @ finger for finger in transformations['left_fingers']], dim = 0)
            print("right_wrist: ", right_wrist.shape)

            right_wrist_rot = (right_wrist.numpy() @ ROT_X @ ROT_Y_)[0][0:3,0:3]
            print("right_wrist_rot: ", right_wrist_rot.shape)
            # pos + xyzw
            # print("transformations: ", transformations['right_fingers'].shape)
            # # left_pos_7d = mat2posquat(left_wrist @ ROT_X @ ROT_Y )
            keypoint_3d = []

            keypoint_3d.append( right_wrist[0][0:3,3] )
            for idx in APPLE2MEDIAPIPE:
                keypoint_3d.append( transformations['right_fingers'][idx - 1][0:3,3] )
            for ele in keypoint_3d:
                print("ele: ",ele.shape)            
            keypoint_3d = torch.stack( keypoint_3d )
            keypoint_3d = keypoint_3d.numpy()
            keypoint_3d = keypoint_3d.reshape(21,-1)
            print("keypoint_3d: ", keypoint_3d.shape)
            # root_3d = keypoint_3d[0:1, :]

            keypoint_3d = keypoint_3d - keypoint_3d[0:1, :]

            joint_pos = keypoint_3d @ right_wrist_rot @ OPERATOR2MANO_RIGHT    
            # print("joint_pos:\n", joint_pos)
            # # # print("mediapipe_wrist_rot:\n", mediapipe_wrist_rot)
            
            retargeting_type = self.right_retargeting.optimizer.retargeting_type
            indices = self.right_retargeting.optimizer.target_link_human_indices
            
            # print("retargeting_type: ", retargeting_type)

            # ref_value = None

            if retargeting_type == "POSITION":
                indices = indices
                ref_value = joint_pos[indices, :]
            else:
                origin_indices = indices[0, :]
                task_indices = indices[1, :]
                ref_value = joint_pos[task_indices, :] - joint_pos[origin_indices, :]

            qpos = self.right_retargeting.retarget(ref_value)
            print("qpos: ", qpos)

            # target_vec = ref_value * self.retargeting.optimizer.scaling
            # robot_pos = self.retargeting.optimizer.get_position(qpos, ref_value)

            # target_joint_pointcloud = PointCloud()
            # target_joint_pointcloud.header = header
            # for i in range( target_vec.shape[0] ):
            #     target_joint_pointcloud.points.append( Point32( target_vec[i][0], target_vec[i][1], target_vec[i][2])) 
            # self.target_joint_pc_publisher.publish(target_joint_pointcloud)

            # robot_joint_pointcloud = PointCloud()
            # robot_joint_pointcloud.header = header
            # for i in range( len(robot_pos) ):
            #     robot_joint_pointcloud.points.append( Point32( robot_pos[i][0], robot_pos[i][1], robot_pos[i][2])) 
            # self.robot_joint_pc_publisher.publish(robot_joint_pointcloud)

            self.env.step(np2tensor(latest, self.env.device))
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
    
    # rospy.init_node("mocap_node")

    vis = IsaacVisualizer(args)
    vis.run()
