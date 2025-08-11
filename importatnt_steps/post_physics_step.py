    def post_physics_step(self):
        """
        The post-processing of the physics step. Compute the observation and reward, and visualize auxiliary 
        lines for debug when needed
        
        """
        self.progress_buf += 1
        self.randomize_buf += 1
        # print("compute_observations")
        self.compute_observations()
        # print("compute_reward")
        self.compute_reward(self.actions)
        # print("get img")
        if (self.num_envs == 1):
            cam1_img = self.cam_tensors[0].cpu().numpy()
            cam2_img = self.cam_tensors[1].cpu().numpy()

            # # Convert images to a format suitable for OpenCV
            header = std_msgs.msg.Header()
            header.stamp = rospy.Time.now()
            header.frame_id = 'map'

            cam1_img = cam1_img.reshape(cam1_img.shape[0], -1, 4)[..., :3]
            cam2_img = cam2_img.reshape(cam2_img.shape[0], -1, 4)[..., :3]


            cam1_msg = bridge.cv2_to_imgmsg(cam1_img, encoding="rgb8")
            cam1_msg.header = header

            cam2_msg = bridge.cv2_to_imgmsg(cam2_img, encoding="rgb8")
            cam2_msg.header = header

            self.cam1_pub.publish(cam1_msg)
            self.cam2_pub.publish(cam2_msg)

        if self.viewer and self.debug_viz:
            # draw axes on target object
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_rigid_body_state_tensor(self.sim)

            for i in range(self.num_envs):
                self.add_debug_lines(self.envs[i], self.block_right_handle_pos[i], self.block_right_handle_rot[i])
                self.add_debug_lines(self.envs[i], self.block_left_handle_pos[i], self.block_left_handle_rot[i])

                # self.add_debug_lines(self.envs[i], self.right_hand_ff_pos[i], self.right_hand_ff_rot[i])
                # self.add_debug_lines(self.envs[i], self.right_hand_mf_pos[i], self.right_hand_mf_rot[i])
                # self.add_debug_lines(self.envs[i], self.right_hand_rf_pos[i], self.right_hand_rf_rot[i])
                # self.add_debug_lines(self.envs[i], self.right_hand_lf_pos[i], self.right_hand_lf_rot[i])
                # self.add_debug_lines(self.envs[i], self.right_hand_th_pos[i], self.right_hand_th_rot[i])

                # self.add_debug_lines(self.envs[i], self.left_hand_ff_pos[i], self.right_hand_ff_rot[i])
                # self.add_debug_lines(self.envs[i], self.left_hand_mf_pos[i], self.right_hand_mf_rot[i])
                # self.add_debug_lines(self.envs[i], self.left_hand_rf_pos[i], self.right_hand_rf_rot[i])
                # self.add_debug_lines(self.envs[i], self.left_hand_lf_pos[i], self.right_hand_lf_rot[i])
                # self.add_debug_lines(self.envs[i], self.left_hand_th_pos[i], self.right_hand_th_rot[i])
