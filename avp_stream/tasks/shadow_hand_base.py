import isaacgym
import torch 
from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch

import numpy as np
import torch
import time
from pathlib import Path

from avp_stream import VisionProStreamer
from avp_stream.utils.isaac_utils import * 
from avp_stream.utils.se3_utils import * 
from avp_stream.utils.trn_constants import * 
from copy import deepcopy
from typing import * 

CUR_PATH = (Path(__file__).parent.resolve())
ASSET_PATH = CUR_PATH.parent
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

class ShadowHandBase: 

    def __init__(self, args):

        self.args = args 

        # acquire gym interface
        self.gym = gymapi.acquire_gym()
 
        # set torch device
        self.device = 'cpu'  # i'll just fix this to CUDA 

        # configure sim
        self.sim_params = default_sim_params(use_gpu = True if self.device == 'cuda:0' else False) 

        # create sim
        self.sim = self.gym.create_sim(0, 0, gymapi.SIM_PHYSX, self.sim_params)
        if self.sim is None:
            raise Exception("Failed to create sim")
        
        # load assets
        self.num_envs = 1

        # create viewer
        self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
        if self.viewer is None:
            raise Exception("Failed to create viewer")

        # create env 
        self._load_asset()
        self.create_env() 

        # setup viewer camera
        middle_env = self.num_envs // 2
        setup_viewer_camera(self.gym, self.envs[middle_env], self.viewer)

        # ==== prepare tensors =====
        # from now on, we will use the tensor API that can run on CPU or GPU
        self.gym.prepare_sim(self.sim)
        self.initialize_tensors()



    def _load_asset(self):

        # self.axis = load_axis(self.gym, self.sim, self.device, 'normal', f'{CUR_PATH}/assets')
        # self.small_axis = load_axis(self.gym, self.sim, self.device, 'small', f'{CUR_PATH}/assets')
        # self.huge_axis = load_axis(self.gym, self.sim, self.device, 'huge', f'{CUR_PATH}/assets')
        self.axis = load_axis(self.gym, self.sim, self.device, 'normal', f'{ASSET_PATH}/assets')
        self.small_axis = load_axis(self.gym, self.sim, self.device, 'small', f'{ASSET_PATH}/assets')
        self.huge_axis = load_axis(self.gym, self.sim, self.device, 'huge', f'{ASSET_PATH}/assets')
        asset_options = gymapi.AssetOptions()
        asset_options.disable_gravity = True
        asset_options.fix_base_link = True
        self.sphere = self.gym.create_sphere(self.sim, 0.008, asset_options)

        # load hand asset
        # asset_root = f'{ASSET_PATH}/assets'
        # asset_file = 'urdf/shadow_hand/shadow_hand_right.urdf'
        asset_root = f'{ASSET_PATH.parent}/assets'
        asset_file = 'robots/shadow_hand/shadow_hand_right.urdf'


        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.use_mesh_materials = True
        asset_options.disable_gravity = True
        asset_options.flip_visual_attachments = False
        self.right_hand_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

        # asset_root = f'{ASSET_PATH}/assets'
        # asset_file = 'urdf/shadow_hand/shadow_hand_left.urdf'
        asset_root = f'{ASSET_PATH.parent}/assets'
        asset_file = 'robots/shadow_hand/shadow_hand_left.urdf'
        
        print("asset_root: ", asset_root)
        print("asset_file: ", asset_file)

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.use_mesh_materials = True
        asset_options.disable_gravity = True
        asset_options.flip_visual_attachments = False
        self.left_hand_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

    def create_env(self):
        spacing = 1.0
        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)

        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1)
        self.gym.add_ground(self.sim, plane_params)

        # create env
        self.envs = []
        self.robot_actor_idxs_over_sim = [] 
        self.env_side_actor_idxs_over_sim = []


        env = self.gym.create_env(self.sim, env_lower, env_upper, 1)
        self.envs.append(env)
        
        self.env_axis = self.gym.create_actor(env, self.huge_axis, gymapi.Transform(), 'env_axis', 0 )
        self.head_axis = self.gym.create_actor(env, self.axis, gymapi.Transform(), 'head', 1)

        self.right_wrist_axis = self.gym.create_actor(env, self.axis, gymapi.Transform(), 'right_wrist', 2)
        self.left_wrist_axis = self.gym.create_actor(env, self.axis, gymapi.Transform(), 'left_wrist', 3)

        self.right_hand = self.gym.create_actor(env, self.right_hand_asset, gymapi.Transform(), "right_hand", 4)
        self.left_hand = self.gym.create_actor(env, self.left_hand_asset, gymapi.Transform(), "left_hand", 5)

        # get array of DOF names
        dof_names = self.gym.get_asset_dof_names(self.right_hand_asset)
        print("dof_names: ", dof_names)
        # get array of DOF properties
        dof_props = self.gym.get_asset_dof_properties(self.right_hand_asset)

        # create an array of DOF states that will be used to update the actors
        num_dofs = self.gym.get_asset_dof_count(self.right_hand_asset)
        print("num_dofs: ", num_dofs)
        
        dof_states = np.zeros(num_dofs, dtype=gymapi.DofState.dtype)

        # get list of DOF types
        dof_types = [self.gym.get_asset_dof_type(self.right_hand_asset, i) for i in range(num_dofs)]

        # get the position slice of the DOF state array
        dof_positions = dof_states['pos']

        # get the limit-related slices of the DOF properties array
        stiffnesses = dof_props['stiffness']
        dampings = dof_props['damping']
        armatures = dof_props['armature']
        has_limits = dof_props['hasLimits']
        lower_limits = dof_props['lower']
        upper_limits = dof_props['upper']

        props = self.gym.get_actor_dof_properties(env, self.right_hand)
        props["driveMode"] = (
            gymapi.DOF_MODE_POS, gymapi.DOF_MODE_POS, gymapi.DOF_MODE_POS, gymapi.DOF_MODE_POS, gymapi.DOF_MODE_POS, gymapi.DOF_MODE_POS,
            gymapi.DOF_MODE_POS, gymapi.DOF_MODE_POS, gymapi.DOF_MODE_POS, gymapi.DOF_MODE_POS, gymapi.DOF_MODE_POS, gymapi.DOF_MODE_POS,
            gymapi.DOF_MODE_POS, gymapi.DOF_MODE_POS, gymapi.DOF_MODE_POS, gymapi.DOF_MODE_POS, gymapi.DOF_MODE_POS, gymapi.DOF_MODE_POS,
            gymapi.DOF_MODE_POS, gymapi.DOF_MODE_POS, gymapi.DOF_MODE_POS, gymapi.DOF_MODE_POS, #gymapi.DOF_MODE_POS, gymapi.DOF_MODE_POS,
            #gymapi.DOF_MODE_POS, gymapi.DOF_MODE_POS, gymapi.DOF_MODE_POS, gymapi.DOF_MODE_POS, gymapi.DOF_MODE_POS, gymapi.DOF_MODE_POS
        )
        props["stiffness"] = (
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, #1.0, 1.0,
            #1.0, 1.0, 1.0, 1.0, 1.0, 1.0
        )
        Tval = 0.1
        Rval = 0.5

        props["damping"] = (
            0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
            0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
            0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
            0.1, 0.1, 0.1, 0.1,
        )            

        self.gym.set_actor_dof_properties(env, self.right_hand, props)
        self.gym.set_actor_dof_states(env, self.right_hand, dof_states, gymapi.STATE_ALL)

        self.gym.set_actor_dof_properties(env, self.left_hand, props)
        self.gym.set_actor_dof_states(env, self.left_hand, dof_states, gymapi.STATE_ALL)

    def initialize_tensors(self): 
        
        refresh_tensors(self.gym, self.sim)
        # get jacobian tensor
        # get rigid body state tensor
        _rb_states = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rb_states = gymtorch.wrap_tensor(_rb_states).view(self.num_envs, -1, 13)

        # get actor root state tensor
        _root_states = self.gym.acquire_actor_root_state_tensor(self.sim)
        root_states = gymtorch.wrap_tensor(_root_states).view(self.num_envs, -1, 13)
        self.root_state = root_states

        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        self.gym.step_graphics(self.sim)
        self.gym.draw_viewer(self.viewer, self.sim, False)
        self.gym.sync_frame_time(self.sim)

    # will be overloaded
    def step(self, transformation: Dict[str, torch.Tensor], sync_frame_time = False): 

        self.simulate()

        new_root_state = self.modify_root_state(transformation)
        env_side_actor_idxs = torch.arange(0, 6, dtype = torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(new_root_state), gymtorch.unwrap_tensor(env_side_actor_idxs), len(env_side_actor_idxs))

        # update viewer
        self.render(sync_frame_time)

    def move_camera(self):

        head_xyz = self.visionos_head[:, :3, 3]
        head_ydir = self.visionos_head[:, :3, 1]

        cam_pos = head_xyz - head_ydir * 0.5
        cam_target = head_xyz + head_ydir * 0.5
        cam_target[..., -1] -= 0.2

        cam_pos = gymapi.Vec3(*cam_pos[0])
        cam_target = gymapi.Vec3(*cam_target[0])

        self.gym.viewer_camera_look_at(self.viewer, self.envs[0], cam_pos, cam_target)

    def simulate(self): 
        # step the physics
        self.gym.simulate(self.sim)

        # refresh tensors
        refresh_tensors(self.gym, self.sim)


    def render(self, sync_frame_time = True): 

        # update viewer
        if self.args.follow:
            self.move_camera()
        self.gym.step_graphics(self.sim)
        self.gym.draw_viewer(self.viewer, self.sim, False)
        if sync_frame_time:
            self.gym.sync_frame_time(self.sim)

    def modify_root_state(self, transformations): 

        new_root_state = self.root_state

        self.visionos_head = transformations['head'] 
        self.sim_right_wrist = transformations['right_wrist'] #@ VISIONOS_RIGHT_HAND_TO_LEAP 
        self.sim_left_wrist = transformations['left_wrist'] # @ VISIONOS_LEFT_HAND_TO_LEAP

        sim_right_fingers = torch.cat([self.sim_right_wrist @ finger for finger in transformations['right_fingers']], dim = 0)
        sim_left_fingers = torch.cat([self.sim_left_wrist @ finger for finger in transformations['left_fingers']], dim = 0)

        self.sim_right_fingers = sim_right_fingers 
        self.sim_left_fingers = sim_left_fingers 

        new_root_state = deepcopy(self.root_state)
        new_root_state[:, 1, :7] = mat2posquat(self.visionos_head )
        new_root_state[:, 2, :7] = mat2posquat(self.sim_right_wrist @ ROT_X @ ROT_Y_) # right wrist axis
        new_root_state[:, 3, :7] = mat2posquat(self.sim_left_wrist @ ROT_X @ ROT_Y ) # left wrist axis

        new_root_state[:, 4, :7] = mat2posquat(self.sim_right_wrist @ ROT_X @ ROT_Y_) #right hand root
        new_root_state[:, 5, :7] = mat2posquat(self.sim_left_wrist @ ROT_X @ ROT_Y) #left hand root

        print("new_root_state: ", new_root_state.shape)
        new_root_state = new_root_state.view(-1, 13)
        order = torch.tensor([
            0, 1, 2, 3,
            12, 13, 14, 15, 16,
            4, 5, 6, 7,
            8, 9, 10, 11,
            17, 18, 19, 20, 21
        ])
        # isaaqcgym dof_names:  [
        #     'FFJ4', 'FFJ3', 'FFJ2', 'FFJ1',
        #     'LFJ5', 'LFJ4', 'LFJ3', 'LFJ2', 'LFJ1', 
        #     'MFJ4', 'MFJ3', 'MFJ2', 'MFJ1', 
        #     'RFJ4', 'RFJ3', 'RFJ2', 'RFJ1', 
        #     'THJ5', 'THJ4', 'THJ3', 'THJ2', 'THJ1'
        # ]

        # dexretarget: target_joint_names:  [
        #    'FFJ4', 'FFJ3', 'FFJ2', 'FFJ1', 
        #    'MFJ4', 'MFJ3', 'MFJ2', 'MFJ1', 
        #    'RFJ4', 'RFJ3', 'RFJ2', 'RFJ1', 
        #    'LFJ5', 'LFJ4', 'LFJ3', 'LFJ2', 'LFJ1', 
        #    'THJ5', 'THJ4', 'THJ3', 'THJ2', 'THJ1'
        # ]

        self.gym.set_actor_dof_position_targets(self.envs[0], self.right_hand, transformations['right_action'][order])
        self.gym.set_actor_dof_position_targets(self.envs[0], self.left_hand, transformations['left_action'][order])
        return new_root_state


def np2tensor(data: Dict[str, np.ndarray], device) -> Dict[str, torch.Tensor]:  
    for key in data.keys():
        data[key] = torch.tensor(data[key], dtype = torch.float32, device = device)
    return data


if __name__=="__main__": 

    import argparse 
    import os 

    parser = argparse.ArgumentParser()
    parser.add_argument('--ip', type = str, required = True)
    parser.add_argument('--record', action = 'store_true')
    parser.add_argument('--follow', action = 'store_true', help = "The viewpoint follows the users head")
    args = parser.parse_args()

    s = VisionProStreamer(args.ip, args.record)

    env = IsaacVisualizerEnv(args)
    while True: 
        t0 = time.time()
        latest = s.latest
        env.step(np2tensor(latest, env.device)) 
        print(time.time() - t0)

