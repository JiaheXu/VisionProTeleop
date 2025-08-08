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

        # self.retargeting_type = 'custom'
        # self.robot_name = 'shadow'

        # self.right_config_path = get_config_path(self.robot_name, self.retargeting_type, 'right_hand')

        # self.robot_dir = Path(__file__).parent.parent / "assets" / "robots"

        # RetargetingConfig.set_default_urdf_dir(str(robot_dir))
        # self.right_retargeting = RetargetingConfig.load_from_file(self.right_config_path).build()
   
        # print("retargeting_type: ", retargeting_type)
        # rospy.init_node("triangulation_node")
        # constructor_node = constructor(retargeting, output_path, str(config_path))

    def run(self):
        data = []
        # while True: 
        for i in range(1000):
            latest = self.s.latest
            data.append(latest)
            # mediapipe_wrist_rot = self.detector.estimate_frame_from_hand_points(keypoint_3d)
            # # print("mediapipe_wrist_rot: ", mediapipe_wrist_rot)
            # eul = self.rot2eul(mediapipe_wrist_rot)

            # joint_pos = keypoint_3d @ mediapipe_wrist_rot @ self.detector.operator2mano        
            # print("joint_pos:\n", joint_pos)
            # # print("mediapipe_wrist_rot:\n", mediapipe_wrist_rot)
            
            # retargeting_type = self.retargeting.optimizer.retargeting_type
            # indices = self.retargeting.optimizer.target_link_human_indices
            
            # print("retargeting_type: ", retargeting_type)

            # ref_value = None

            # if retargeting_type == "POSITION":
            #     indices = indices
            #     ref_value = joint_pos[indices, :]
            # else:
            #     origin_indices = indices[0, :]
            #     task_indices = indices[1, :]
            #     ref_value = joint_pos[task_indices, :] - joint_pos[origin_indices, :]

            # qpos = self.retargeting.retarget(ref_value)
            # print("qpos: ", qpos)
            self.env.step(np2tensor(latest, self.env.device))
        np.save("test.npy", data, allow_pickle=True)

# python3 example/retarget_debug_node.py \
#   --robot-name shadow \
#   --retargeting-type custom \
#   --hand-type right \
#   --output-path example/data/shadow_hand.pkl  

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
