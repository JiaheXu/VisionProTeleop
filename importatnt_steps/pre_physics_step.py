    def pre_physics_step(self, actions):
        """
        The pre-processing of the physics step. Determine whether the reset environment is needed, 
        and calculate the next movement of Shadowhand through the given action. The 52-dimensional 
        action space as shown in below:
        
        Index   Description
        0 - 19 	right shadow hand actuated joint
        20 - 22	right shadow hand base translation
        23 - 25	right shadow hand base rotation
        26 - 45	left shadow hand actuated joint
        46 - 48	left shadow hand base translation
        49 - 51	left shadow hand base rotation

        Args:
            actions (tensor): Actions of agents in the all environment 
        """
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        goal_env_ids = self.reset_goal_buf.nonzero(as_tuple=False).squeeze(-1)

        # if only goals need reset, then call set API
        if len(goal_env_ids) > 0 and len(env_ids) == 0:
            self.reset_target_pose(goal_env_ids, apply_reset=True)
        # if goals need reset in addition to other envs, call set API in reset()
        elif len(goal_env_ids) > 0:
            self.reset_target_pose(goal_env_ids)

        if len(env_ids) > 0:
            self.reset(env_ids, goal_env_ids)

        self.actions = actions.clone().to(self.device)
        if self.use_relative_control:
            targets = self.prev_targets[:, self.actuated_dof_indices] + self.shadow_hand_dof_speed_scale * self.dt * self.actions
            self.cur_targets[:, self.actuated_dof_indices] = tensor_clamp(targets,
                                                                          self.shadow_hand_dof_lower_limits[self.actuated_dof_indices], self.shadow_hand_dof_upper_limits[self.actuated_dof_indices])
        else:
                        
            self.cur_targets[:, self.actuated_dof_indices] = scale(self.actions[:, 0:self.action_dim],
                                                                   self.shadow_hand_dof_lower_limits[self.actuated_dof_indices], self.shadow_hand_dof_upper_limits[self.actuated_dof_indices])
            #print("after 1st step: ", self.cur_targets[:, self.actuated_dof_indices])

            self.cur_targets[:, self.actuated_dof_indices] = self.act_moving_average * self.cur_targets[:,
                                                                                                        self.actuated_dof_indices] + (1.0 - self.act_moving_average) * self.prev_targets[:, self.actuated_dof_indices]
            #print("after 2nd step: ", self.cur_targets[:, self.actuated_dof_indices])

            self.cur_targets[:, self.actuated_dof_indices] = tensor_clamp(self.cur_targets[:, self.actuated_dof_indices],
                                                                          self.shadow_hand_dof_lower_limits[self.actuated_dof_indices], self.shadow_hand_dof_upper_limits[self.actuated_dof_indices])
            #print("after 3rd step: ", self.cur_targets[:, self.actuated_dof_indices])
            self.cur_targets[:, self.actuated_dof_indices + self.num_shadow_hand_dofs] = scale(self.actions[:, self.action_dim: self.action_dim*2],
            #self.cur_targets[:, self.actuated_dof_indices + self.num_shadow_hand_dofs] = scale(self.actions[:, 0 : self.action_dim],
                                                                   self.shadow_hand_dof_lower_limits[self.actuated_dof_indices], self.shadow_hand_dof_upper_limits[self.actuated_dof_indices])
            #print("left after 1st step: ", self.cur_targets[:, self.actuated_dof_indices + self.num_shadow_hand_dofs])
            
            self.cur_targets[:, self.actuated_dof_indices + self.num_shadow_hand_dofs] = self.act_moving_average * self.cur_targets[:,
                                                                                                        self.actuated_dof_indices + self.num_shadow_hand_dofs] + (1.0 - self.act_moving_average) * self.prev_targets[:, self.actuated_dof_indices]
            #print("left after 2nd step: ", self.cur_targets[:, self.actuated_dof_indices + self.num_shadow_hand_dofs])

            self.cur_targets[:, self.actuated_dof_indices + self.num_shadow_hand_dofs] = tensor_clamp(self.cur_targets[:, self.actuated_dof_indices + self.num_shadow_hand_dofs],
                                                                          self.shadow_hand_dof_lower_limits[self.actuated_dof_indices], self.shadow_hand_dof_upper_limits[self.actuated_dof_indices])

        #     print("pre_physics_step.cur_targets:")
        #     print("POSE: ", self.cur_targets[:, 0:6])
        #     print("FF: ", self.cur_targets[:, 6:10])
        #     print("MF: ", self.cur_targets[:, 10:14])
        #     print("RF: ", self.cur_targets[:, 14:18])        
        #     print("LF: ", self.cur_targets[:, 18:23])
        #     print("TH: ", self.cur_targets[:, 23:28])
        # print("right - left action diff: ", Tensor.sum( Tensor.abs( self.cur_targets[:, self.actuated_dof_indices] - self.cur_targets[:, self.actuated_dof_indices + self.num_shadow_hand_dofs] ) ) )
        
        gymutil.draw_lines(self.axes_geom, self.gym, self.viewer, self.envs[0], self.goal_viz_T)

        self.prev_targets[:, self.actuated_dof_indices] = self.cur_targets[:, self.actuated_dof_indices]
        self.prev_targets[:, self.actuated_dof_indices + self.num_shadow_hand_dofs] = self.cur_targets[:, self.actuated_dof_indices + self.num_shadow_hand_dofs]

        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.cur_targets))
