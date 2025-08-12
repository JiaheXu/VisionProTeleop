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

ROT_Z = np.array([[[ 0, -1, 0, 0], 
                    [1,  0, 0, 0], 
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]]], dtype = np.float64)

class ShadowHandStackBlocks: 

    def __init__(self, args):

        self.args = args 

        # acquire gym interface
        self.gym = gymapi.acquire_gym()
 
        # set torch device
        self.device = 'cpu'  # i'll just fix this to CUDA 

        # configure sim
        self.sim_params = default_sim_params(use_gpu = True if self.device == 'cuda:0' else False) 

        self.sim_params.physx.solver_type = 1  # TGS solver (more stable for grasp)
        # avp version
        # self.sim_params.physx.num_position_iterations = 12
        # self.sim_params.physx.num_velocity_iterations = 4
        
        # bidex version
        self.sim_params.physx.num_position_iterations = 8
        self.sim_params.physx.num_velocity_iterations = 0
        self.sim_params.physx.contact_offset = 0.002
        self.sim_params.physx.rest_offset = 0.0

        # gpt version
        # self.sim_params.physx.num_position_iterations = 24
        # self.sim_params.physx.num_velocity_iterations = 12
        # self.sim_params.physx.contact_offset = 0.002  # smaller global contact offset
        # self.sim_params.physx.rest_offset = 0.001

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
        self.actors = []
    # def _env_init(self):


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
        # asset_file = 'robots/shadow_hand/shadow_hand_right.urdf'
        asset_file = 'mjcf/open_ai_assets/hand_new/shadow_hand_right.xml'
        hand_asset_options = gymapi.AssetOptions()
        hand_asset_options.fix_base_link = True
        hand_asset_options.disable_gravity = True
        hand_asset_options.flip_visual_attachments = False
        hand_asset_options.collapse_fixed_joints = True
        hand_asset_options.thickness = 0.001
        hand_asset_options.angular_damping = 100
        hand_asset_options.linear_damping = 100
        hand_asset_options.use_physx_armature = True
        self.right_hand_asset = self.gym.load_asset(self.sim, asset_root, asset_file, hand_asset_options)

        # asset_root = f'{ASSET_PATH}/assets'
        # asset_file = 'urdf/shadow_hand/shadow_hand_left.urdf'
        asset_root = f'{ASSET_PATH.parent}/assets'
        # asset_file = 'robots/shadow_hand/shadow_hand_left.urdf'
        asset_file = 'mjcf/open_ai_assets/hand_new/shadow_hand_left.xml'
        # print("asset_root: ", asset_root)
        # print("asset_file: ", asset_file)
        self.left_hand_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        
        asset_file = 'urdf/objects/cube_multicolor.urdf'
        block_asset_options = gymapi.AssetOptions()
        block_asset_options.density = 100
        block_asset_options.fix_base_link = False
        self.block1_asset = self.gym.load_asset(self.sim, asset_root, asset_file, block_asset_options)
        self.block2_asset = self.gym.load_asset(self.sim, asset_root, asset_file, block_asset_options)


        # create table asset
        table_dims = gymapi.Vec3(1.0, 1.0, 0.05)
        table_asset_options = gymapi.AssetOptions()
        table_asset_options.fix_base_link = True
        table_asset_options.flip_visual_attachments = True
        table_asset_options.collapse_fixed_joints = True
        table_asset_options.disable_gravity = True
        table_asset_options.thickness = 0.001
        self.table_asset = self.gym.create_box(self.sim, table_dims.x, table_dims.y, table_dims.z, table_asset_options)

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
        
        self.env_axis = self.gym.create_actor(env, self.huge_axis, gymapi.Transform() , 'env_axis', 1, 0, 0)
        self.head_axis = self.gym.create_actor(env, self.axis, gymapi.Transform(), 'head', 1, 0, 0)

        self.right_wrist_axis = self.gym.create_actor(env, self.axis, gymapi.Transform(), 'right_wrist', 1, 0, 0)
        self.left_wrist_axis = self.gym.create_actor(env, self.axis, gymapi.Transform(), 'left_wrist', 1, 0, 0)

        # add right hand
        right_hand_start_pose = gymapi.Transform()
        right_hand_start_pose.p = gymapi.Vec3(-0.2, -0.2, 0.55)
        right_hand_start_pose.r = gymapi.Quat().from_euler_zyx(0, 0, 0)
        self.right_hand = self.gym.create_actor(env, self.right_hand_asset, right_hand_start_pose, "right_hand", 1, 0, 0)

        # add left hand
        left_hand_start_pose = gymapi.Transform()
        left_hand_start_pose.p = gymapi.Vec3(-0.2, 0.2, 0.55)
        left_hand_start_pose.r = gymapi.Quat().from_euler_zyx(0, 0, 0)
        self.left_hand = self.gym.create_actor(env, self.left_hand_asset, left_hand_start_pose, "left_hand", 1, 0, 0)


        
        # add table
        table_pose = gymapi.Transform()
        table_pose.p = gymapi.Vec3(0.0, 0.0, 0.55)
        table_pose.r = gymapi.Quat().from_euler_zyx(0, 0, 0)
        self.table = self.gym.create_actor(env, self.table_asset, table_pose, "table", 1, -1, 0)

        # add block1
        block1_start_pose = gymapi.Transform()
        block1_start_pose.p = gymapi.Vec3(0.2, 0.1, 0.6)
        block1_start_pose.r = gymapi.Quat().from_euler_zyx(0, 0, 0)
        self.block1 = self.gym.create_actor(env, self.block1_asset, block1_start_pose, "block1", 1, 0, 0)

        # add block2
        block2_start_pose = gymapi.Transform()
        block2_start_pose.p = gymapi.Vec3(0.2, -0.1, 0.6)
        block2_start_pose.r = gymapi.Quat().from_euler_zyx(0, 0, 0)
        self.block2 = self.gym.create_actor(env, self.block2_asset, block2_start_pose, "block2", 1, 0, 0)

        # get array of DOF names
        dof_names = self.gym.get_asset_dof_names(self.right_hand_asset)
        print("dof_names: ", dof_names)
        dof_props = self.gym.get_asset_dof_properties(self.right_hand_asset)
        num_dofs = self.gym.get_asset_dof_count(self.right_hand_asset)
        print("num_dofs: ", num_dofs)
        
        dof_states = np.zeros(num_dofs, dtype=gymapi.DofState.dtype)

        props = self.gym.get_actor_dof_properties(env, self.right_hand)

        props["driveMode"] = tuple([gymapi.DOF_MODE_POS] * num_dofs)
        props["stiffness"] = tuple([100.0] * num_dofs)   # high stiffness for following targets
        props["damping"]   = tuple([1.0] * num_dofs)     # increased damping to avoid oscillation
        props["effort"]    = tuple([100.0] * num_dofs)   # strong actuator torque/force for grasping

        self.shadow_hand_default_dof_pos = torch.zeros(num_dofs, dtype=torch.float)

        self.gym.set_actor_dof_properties(env, self.right_hand, props)
        self.gym.set_actor_dof_states(env, self.right_hand, dof_states, gymapi.STATE_ALL)
        self.gym.set_actor_dof_properties(env, self.left_hand, props)
        self.gym.set_actor_dof_states(env, self.left_hand, dof_states, gymapi.STATE_ALL)

        # apply DOF properties to both hands
        self.gym.set_actor_dof_properties(env, self.right_hand, props)
        self.gym.set_actor_dof_states(env, self.right_hand, dof_states, gymapi.STATE_ALL)
        self.gym.set_actor_dof_properties(env, self.left_hand, props)
        self.gym.set_actor_dof_states(env, self.left_hand, dof_states, gymapi.STATE_ALL)


        # -------------------- Shape properties for blocks and hands --------------------
        friction_val = 10.0
        rolling_friction_val = 1.0
        torsional_friction_val = 1.0
        contact_offset_small = 0.0015
        rest_offset_small = 0.001
        restitution_val = 0.0

        # update block1 shapes
        block1_shape_props = self.gym.get_actor_rigid_shape_properties(env, self.block1)
        for prop in block1_shape_props:
            prop.friction = friction_val
            # small rolling/torsional friction helps fingers keep the block from spinning out
            if hasattr(prop, "rollingFriction"):
                prop.rollingFriction = rolling_friction_val
            if hasattr(prop, "torsionalFriction"):
                prop.torsionalFriction = torsional_friction_val
            prop.restitution = restitution_val
            prop.contact_offset = contact_offset_small
            prop.rest_offset = rest_offset_small
        self.gym.set_actor_rigid_shape_properties(env, self.block1, block1_shape_props)

        # update block2 shapes (if present)
        block2_shape_props = self.gym.get_actor_rigid_shape_properties(env, self.block2)
        for prop in block2_shape_props:
            prop.friction = friction_val
            if hasattr(prop, "rollingFriction"):
                prop.rollingFriction = rolling_friction_val
            if hasattr(prop, "torsionalFriction"):
                prop.torsionalFriction = torsional_friction_val
            prop.restitution = restitution_val
            prop.contact_offset = contact_offset_small
            prop.rest_offset = rest_offset_small
        self.gym.set_actor_rigid_shape_properties(env, self.block2, block2_shape_props)

        # update right hand shape props
        right_hand_shape_props = self.gym.get_actor_rigid_shape_properties(env, self.right_hand)
        for prop in right_hand_shape_props:
            prop.friction = friction_val
            if hasattr(prop, "rollingFriction"):
                prop.rollingFriction = rolling_friction_val
            if hasattr(prop, "torsionalFriction"):
                prop.torsionalFriction = torsional_friction_val
            prop.restitution = restitution_val
            prop.contact_offset = contact_offset_small
            prop.rest_offset = rest_offset_small
        self.gym.set_actor_rigid_shape_properties(env, self.right_hand, right_hand_shape_props)

        # update left hand shape props
        left_hand_shape_props = self.gym.get_actor_rigid_shape_properties(env, self.left_hand)
        for prop in left_hand_shape_props:
            prop.friction = friction_val
            if hasattr(prop, "rollingFriction"):
                prop.rollingFriction = rolling_friction_val
            if hasattr(prop, "torsionalFriction"):
                prop.torsionalFriction = torsional_friction_val
            prop.restitution = restitution_val
            prop.contact_offset = contact_offset_small
            prop.rest_offset = rest_offset_small
        self.gym.set_actor_rigid_shape_properties(env, self.left_hand, left_hand_shape_props)



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

        ###############################################################################
        # Original version
        ###############################################################################
        # self.simulate()
        # new_root_state = self.modify_root_state(transformation)
        # env_side_actor_idxs = torch.arange(0, 6, dtype = torch.int32)
        # self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(new_root_state), gymtorch.unwrap_tensor(env_side_actor_idxs), len(env_side_actor_idxs))
        # # update viewer
        # self.render(sync_frame_time)
        ###############################################################################

        #print("actions.dtype: ", actions.dtype)
        new_root_state = self.pre_physics_step(transformation)
        self.simulate()
        # self.load_observations()
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

    def pre_physics_step(self, transformations): 

        new_root_state = self.root_state

        self.visionos_head = transformations['head'] 
        self.sim_right_wrist = transformations['right_wrist'] #@ VISIONOS_RIGHT_HAND_TO_LEAP 
        self.sim_left_wrist = transformations['left_wrist'] # @ VISIONOS_LEFT_HAND_TO_LEAP

        # sim_right_fingers = torch.cat([self.sim_right_wrist @ finger for finger in transformations['right_fingers']], dim = 0)
        # sim_left_fingers = torch.cat([self.sim_left_wrist @ finger for finger in transformations['left_fingers']], dim = 0)

        # self.sim_right_fingers = sim_right_fingers 
        # self.sim_left_fingers = sim_left_fingers 

        new_root_state = deepcopy(self.root_state)
        new_root_state[:, 1, :7] = mat2posquat(self.visionos_head )
        new_root_state[:, 2, :7] = mat2posquat(transformations['right_wrist_sim']) # right wrist axis
        new_root_state[:, 3, :7] = mat2posquat(transformations['left_wrist_sim']) # left wrist axis

        new_root_state[:, 4, :7] = mat2posquat(transformations['right_wrist_sim'])#right hand root
        new_root_state[:, 5, :7] = mat2posquat(transformations['left_wrist_sim']) #left hand root

        # print("new_root_state: ", new_root_state.shape)
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
        # transformations['right_action'] = transformations['right_action'][order]
        # transformations['left_action'] = transformations['left_action'][order]
        # transformations['right_action'] *= 0.0
        # transformations['left_action'] *= 0.0

        self.gym.set_actor_dof_position_targets(self.envs[0], self.right_hand, transformations['right_action'])
        self.gym.set_actor_dof_position_targets(self.envs[0], self.left_hand, transformations['left_action'])

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

