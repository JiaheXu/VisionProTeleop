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

from sensor_msgs.msg import PointCloud
from geometry_msgs.msg import Point32
import std_msgs
import rospy

from avp_stream.tasks.shadow_hand_stack_blocks import *
from avp_stream.tasks.shadow_hand_base import *


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
        self.s = VisionProStreamer(args.ip, args.record)


    def run(self):
        
        data = []
        for i in range(10000):
            latest = self.s.latest
            print(latest)
            data.append(latest)
            time.sleep(0.05)

        np.save("test.npy", data, allow_pickle=True)

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
    vis.run()
