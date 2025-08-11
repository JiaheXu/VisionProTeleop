    def compute_observations(self):
        """
        Compute the observations of all environment. The core function is self.compute_full_state(True), 
        which we will introduce in detail there

        """
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)

        if self.obs_type in ["point_cloud"]:
            self.gym.render_all_camera_sensors(self.sim)
            self.gym.start_access_image_tensors(self.sim)

        self.object_pose = self.root_state_tensor[self.object_indices, 0:7]
        self.object_pos = self.root_state_tensor[self.object_indices, 0:3]
        self.object_rot = self.root_state_tensor[self.object_indices, 3:7]
        self.object_linvel = self.root_state_tensor[self.object_indices, 7:10]
        self.object_angvel = self.root_state_tensor[self.object_indices, 10:13]

        self.block_pose = self.root_state_tensor[self.block_indices, 0:7]
        self.block_pos = self.root_state_tensor[self.block_indices, 0:3]
        self.block_rot = self.root_state_tensor[self.block_indices, 3:7]
        self.block_linvel = self.root_state_tensor[self.block_indices, 7:10]
        self.block_angvel = self.root_state_tensor[self.block_indices, 10:13]

        self.block_right_handle_pos = self.rigid_body_states[:, self.num_shadow_hand_bodies * 2, 0:3]
        self.block_right_handle_rot = self.rigid_body_states[:, self.num_shadow_hand_bodies * 2, 3:7]
        self.block_right_handle_pos = self.block_right_handle_pos + quat_apply(self.block_right_handle_rot, to_torch([0, 1, 0], device=self.device).repeat(self.num_envs, 1) * 0.)
        self.block_right_handle_pos = self.block_right_handle_pos + quat_apply(self.block_right_handle_rot, to_torch([1, 0, 0], device=self.device).repeat(self.num_envs, 1) * 0.0)
        self.block_right_handle_pos = self.block_right_handle_pos + quat_apply(self.block_right_handle_rot, to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.0)

        self.block_left_handle_pos = self.rigid_body_states[:, self.num_shadow_hand_bodies * 2 + 1, 0:3]
        self.block_left_handle_rot = self.rigid_body_states[:, self.num_shadow_hand_bodies * 2 + 1, 3:7]
        self.block_left_handle_pos = self.block_left_handle_pos + quat_apply(self.block_left_handle_rot, to_torch([0, 1, 0], device=self.device).repeat(self.num_envs, 1) * 0.0)
        self.block_left_handle_pos = self.block_left_handle_pos + quat_apply(self.block_left_handle_rot, to_torch([1, 0, 0], device=self.device).repeat(self.num_envs, 1) * 0.0)
        self.block_left_handle_pos = self.block_left_handle_pos + quat_apply(self.block_left_handle_rot, to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.0)

        self.left_hand_pos = self.rigid_body_states[:, self.hand_center_idx +  self.num_shadow_hand_bodies, 0:3]
        self.left_hand_rot = self.rigid_body_states[:, self.hand_center_idx +  self.num_shadow_hand_bodies, 3:7]
        self.left_hand_pos = self.left_hand_pos + quat_apply(self.left_hand_rot, to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.08)
        self.left_hand_pos = self.left_hand_pos + quat_apply(self.left_hand_rot, to_torch([0, 1, 0], device=self.device).repeat(self.num_envs, 1) * -0.02)

        self.right_hand_pos = self.rigid_body_states[:, self.hand_center_idx, 0:3]
        self.right_hand_rot = self.rigid_body_states[:, self.hand_center_idx, 3:7]
        self.right_hand_pos = self.right_hand_pos + quat_apply(self.right_hand_rot, to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.08)
        self.right_hand_pos = self.right_hand_pos + quat_apply(self.right_hand_rot, to_torch([0, 1, 0], device=self.device).repeat(self.num_envs, 1) * -0.02)

        # right hand finger
        self.right_hand_ff_pos = self.rigid_body_states[:, self.ff_idx, 0:3]
        self.right_hand_ff_rot = self.rigid_body_states[:, self.ff_idx, 3:7]
        self.right_hand_ff_pos = self.right_hand_ff_pos + quat_apply(self.right_hand_ff_rot, to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.02)
        self.right_hand_mf_pos = self.rigid_body_states[:, self.mf_idx, 0:3]
        self.right_hand_mf_rot = self.rigid_body_states[:, self.mf_idx, 3:7]
        self.right_hand_mf_pos = self.right_hand_mf_pos + quat_apply(self.right_hand_mf_rot, to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.02)
        self.right_hand_rf_pos = self.rigid_body_states[:, self.rf_idx, 0:3]
        self.right_hand_rf_rot = self.rigid_body_states[:, self.rf_idx, 3:7]
        self.right_hand_rf_pos = self.right_hand_rf_pos + quat_apply(self.right_hand_rf_rot, to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.02)
        self.right_hand_lf_pos = self.rigid_body_states[:, self.lf_idx, 0:3]
        self.right_hand_lf_rot = self.rigid_body_states[:, self.lf_idx, 3:7]
        self.right_hand_lf_pos = self.right_hand_lf_pos + quat_apply(self.right_hand_lf_rot, to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.02)
        self.right_hand_th_pos = self.rigid_body_states[:, self.th_idx, 0:3]
        self.right_hand_th_rot = self.rigid_body_states[:, self.th_idx, 3:7]
        self.right_hand_th_pos = self.right_hand_th_pos + quat_apply(self.right_hand_th_rot, to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.02)

        self.left_hand_ff_pos = self.rigid_body_states[:, self.ff_idx +  self.num_shadow_hand_bodies, 0:3]
        self.left_hand_ff_rot = self.rigid_body_states[:, self.ff_idx +  self.num_shadow_hand_bodies, 3:7]
        self.left_hand_ff_pos = self.left_hand_ff_pos + quat_apply(self.left_hand_ff_rot, to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.02)
        self.left_hand_mf_pos = self.rigid_body_states[:, self.mf_idx + self.num_shadow_hand_bodies, 0:3]
        self.left_hand_mf_rot = self.rigid_body_states[:, self.mf_idx + self.num_shadow_hand_bodies, 3:7]
        self.left_hand_mf_pos = self.left_hand_mf_pos + quat_apply(self.left_hand_mf_rot, to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.02)
        self.left_hand_rf_pos = self.rigid_body_states[:, self.rf_idx + self.num_shadow_hand_bodies, 0:3]
        self.left_hand_rf_rot = self.rigid_body_states[:, self.rf_idx + self.num_shadow_hand_bodies, 3:7]
        self.left_hand_rf_pos = self.left_hand_rf_pos + quat_apply(self.left_hand_rf_rot, to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.02)
        self.left_hand_lf_pos = self.rigid_body_states[:, self.lf_idx  + self.num_shadow_hand_bodies, 0:3]
        self.left_hand_lf_rot = self.rigid_body_states[:, self.lf_idx  + self.num_shadow_hand_bodies, 3:7]
        self.left_hand_lf_pos = self.left_hand_lf_pos + quat_apply(self.left_hand_lf_rot, to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.02)
        self.left_hand_th_pos = self.rigid_body_states[:, self.th_idx + self.num_shadow_hand_bodies, 0:3]
        self.left_hand_th_rot = self.rigid_body_states[:, self.th_idx + self.num_shadow_hand_bodies, 3:7]
        self.left_hand_th_pos = self.left_hand_th_pos + quat_apply(self.left_hand_th_rot, to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.02)

        self.goal_pose = self.goal_states[:, 0:7]
        self.goal_pos = self.goal_states[:, 0:3]
        self.goal_rot = self.goal_states[:, 3:7]

        self.fingertip_state = self.rigid_body_states[:, self.fingertip_handles][:, :, 0:13]
        self.fingertip_pos = self.rigid_body_states[:, self.fingertip_handles][:, :, 0:3]
        self.fingertip_another_state = self.rigid_body_states[:, self.fingertip_another_handles][:, :, 0:13]
        self.fingertip_another_pos = self.rigid_body_states[:, self.fingertip_another_handles][:, :, 0:3]

        if self.obs_type == "full_state":
            self.compute_full_state()
        elif self.obs_type == "point_cloud":
            self.compute_point_cloud_observation()

        if self.asymmetric_obs:
            self.compute_full_state(True)
    def compute_full_state(self, asymm_obs=False):
        """
        Compute the observations of all environment. The observation is composed of three parts: 
        the state values of the left and right hands, and the information of objects and target. 
        The state values of the left and right hands were the same for each task, including hand 
        joint and finger positions, velocity, and force information. The detail 428-dimensional 
        observational space as shown in below:

        Index       Description
        0 - 23	    right shadow hand dof position
        24 - 47	    right shadow hand dof velocity
        48 - 71	    right shadow hand dof force
        72 - 136	right shadow hand fingertip pose, linear velocity, angle velocity (5 x 13)
        137 - 166	right shadow hand fingertip force, torque (5 x 6)
        167 - 169	right shadow hand base position
        170 - 172	right shadow hand base rotation
        173 - 198	right shadow hand actions
        199 - 222	left shadow hand dof position
        223 - 246	left shadow hand dof velocity
        247 - 270	left shadow hand dof force
        271 - 335	left shadow hand fingertip pose, linear velocity, angle velocity (5 x 13)
        336 - 365	left shadow hand fingertip force, torque (5 x 6)
        366 - 368	left shadow hand base position
        369 - 371	left shadow hand base rotation
        372 - 397	left shadow hand actions
        398 - 404	object pose
        405 - 407	object linear velocity
        408 - 410	object angle velocity
        411 - 417	goal pose
        418 - 421	goal rot - object rot
        422 - 424	block1 position
        425 - 427	block2 position
        """


        num_ft_states = 13 * int(self.num_fingertips / 2)  # 65
        num_ft_force_torques = 6 * int(self.num_fingertips / 2)  # 30

        self.obs_buf[:, 0:self.num_shadow_hand_dofs] = unscale(self.shadow_hand_dof_pos,
                                                            self.shadow_hand_dof_lower_limits, self.shadow_hand_dof_upper_limits)
        self.obs_buf[:, self.num_shadow_hand_dofs:2*self.num_shadow_hand_dofs] = self.vel_obs_scale * self.shadow_hand_dof_vel
        self.obs_buf[:, 2*self.num_shadow_hand_dofs:3*self.num_shadow_hand_dofs] = self.force_torque_obs_scale * self.dof_force_tensor[:, :self.num_shadow_hand_dofs]

        fingertip_obs_start = self.num_shadow_hand_dofs * 3
        self.obs_buf[:, fingertip_obs_start:fingertip_obs_start + num_ft_states] = self.fingertip_state.reshape(self.num_envs, num_ft_states)
        self.obs_buf[:, fingertip_obs_start + num_ft_states:fingertip_obs_start + num_ft_states +
                    num_ft_force_torques] = self.force_torque_obs_scale * self.vec_sensor_tensor[:, :num_ft_force_torques]
        
        hand_pose_start = fingertip_obs_start + num_ft_force_torques + num_ft_states
        self.obs_buf[:, hand_pose_start:hand_pose_start + 3] = self.right_hand_pos
        self.obs_buf[:, hand_pose_start+3:hand_pose_start+4] = get_euler_xyz(self.hand_orientations[self.hand_indices, :])[0].unsqueeze(-1)
        self.obs_buf[:, hand_pose_start+4:hand_pose_start+5] = get_euler_xyz(self.hand_orientations[self.hand_indices, :])[1].unsqueeze(-1)
        self.obs_buf[:, hand_pose_start+5:hand_pose_start+6] = get_euler_xyz(self.hand_orientations[self.hand_indices, :])[2].unsqueeze(-1)

        action_obs_start = hand_pose_start + 6
        self.obs_buf[:, action_obs_start:action_obs_start + self.action_dim] = self.actions[:, :self.action_dim]

        # another_hand
        another_hand_start = action_obs_start + self.action_dim
        self.obs_buf[:, another_hand_start:self.num_shadow_hand_dofs + another_hand_start] = unscale(self.shadow_hand_another_dof_pos, self.shadow_hand_dof_lower_limits, self.shadow_hand_dof_upper_limits)
        self.obs_buf[:, self.num_shadow_hand_dofs + another_hand_start:2*self.num_shadow_hand_dofs + another_hand_start] = self.vel_obs_scale * self.shadow_hand_another_dof_vel
        self.obs_buf[:, 2*self.num_shadow_hand_dofs + another_hand_start:3*self.num_shadow_hand_dofs + another_hand_start] = self.force_torque_obs_scale * self.dof_force_tensor[:, self.num_shadow_hand_dofs:self.num_shadow_hand_dofs*2]

        fingertip_another_obs_start = another_hand_start + self.num_shadow_hand_dofs * 3
        self.obs_buf[:, fingertip_another_obs_start:fingertip_another_obs_start + num_ft_states] = self.fingertip_another_state.reshape(self.num_envs, num_ft_states)
        self.obs_buf[:, fingertip_another_obs_start + num_ft_states:fingertip_another_obs_start + num_ft_states +
                    num_ft_force_torques] = self.force_torque_obs_scale * self.vec_sensor_tensor[:, num_ft_force_torques:]

        hand_another_pose_start = fingertip_another_obs_start + num_ft_force_torques + num_ft_states
        self.obs_buf[:, hand_another_pose_start:hand_another_pose_start + 3] = self.left_hand_pos
        self.obs_buf[:, hand_another_pose_start+3:hand_another_pose_start+4] = get_euler_xyz(self.hand_orientations[self.another_hand_indices, :])[0].unsqueeze(-1)
        self.obs_buf[:, hand_another_pose_start+4:hand_another_pose_start+5] = get_euler_xyz(self.hand_orientations[self.another_hand_indices, :])[1].unsqueeze(-1)
        self.obs_buf[:, hand_another_pose_start+5:hand_another_pose_start+6] = get_euler_xyz(self.hand_orientations[self.another_hand_indices, :])[2].unsqueeze(-1)

        action_another_obs_start = hand_another_pose_start + 6

        self.obs_buf[:, action_another_obs_start:action_another_obs_start + self.action_dim] = self.actions[:, self.action_dim:]

        obj_obs_start = action_another_obs_start + self.action_dim

        self.obs_buf[:, obj_obs_start:obj_obs_start + 7] = self.object_pose
        self.obs_buf[:, obj_obs_start + 7:obj_obs_start + 10] = self.object_linvel
        self.obs_buf[:, obj_obs_start + 10:obj_obs_start + 13] = self.vel_obs_scale * self.object_angvel
        self.obs_buf[:, obj_obs_start + 13:obj_obs_start + 13 + 7] = self.block_pose
        self.obs_buf[:, obj_obs_start + 13 + 7:obj_obs_start + 13 + 10] = self.block_linvel
        self.obs_buf[:, obj_obs_start + 13 + 10:obj_obs_start + 13 + 13] = self.vel_obs_scale * self.block_angvel
