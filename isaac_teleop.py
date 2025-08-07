from avp_stream.isaac_env import IsaacVisualizerEnv
from avp_stream import VisionProStreamer
import time 
from typing import * 
import numpy as np 
import torch 

from dex_retargeting.constants import RobotName, RetargetingType, HandType, get_config_path
from dex_retargeting.retargeting_config import RetargetingConfig
from dex_retargeting.seq_retarget import SeqRetargeting

class IsaacVisualizer:

    def __init__(self, args): 
        self.s = VisionProStreamer(args.ip, args.record)
        self.env = IsaacVisualizerEnv(args)

    def run(self):

        while True: 
            latest = self.s.latest

            # mediapipe_wrist_rot = self.detector.estimate_frame_from_hand_points(keypoint_3d)
            # # print("mediapipe_wrist_rot: ", mediapipe_wrist_rot)
            # eul = self.rot2eul(mediapipe_wrist_rot)

            # joint_pos = keypoint_3d @ mediapipe_wrist_rot @ self.detector.operator2mano        
            # print("joint_pos:\n", joint_pos)
            # # print("mediapipe_wrist_rot:\n", mediapipe_wrist_rot)
            
            # retargeting_type = self.retargeting.optimizer.retargeting_type
            # indices = self.retargeting.optimizer.target_link_human_indices
            
            # # print("retargeting_type: ", retargeting_type)
            
            # if retargeting_type == "POSITION":
            #     indices = indices
            #     ref_value = joint_pos[indices, :]
            # else:
            #     origin_indices = indices[0, :]
            #     task_indices = indices[1, :]
            #     ref_value = joint_pos[task_indices, :] - joint_pos[origin_indices, :]

            # qpos = self.retargeting.retarget(ref_value)

            self.env.step(np2tensor(latest, self.env.device)) 


def np2tensor(data: Dict[str, np.ndarray], device) -> Dict[str, torch.Tensor]:  
    for key in data.keys():
        data[key] = torch.tensor(data[key], dtype = torch.float32, device = device)
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
