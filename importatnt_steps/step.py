    def step(self, actions):
        #if self.dr_randomizations.get('actions', None):
        #    actions = self.dr_randomizations['actions']['noise_lambda'](actions)

        # apply actions
        #torch.cuda.synchronize()
        #print("base_task action:\n", actions)
        
        #print("actions.dtype: ", actions.dtype)
        self.pre_physics_step(actions)

        # step physics and render each frame
        for i in range(self.control_freq_inv):
            self.render()
            self.gym.simulate(self.sim)

        # to fix!
        if self.device == 'cpu':
            self.gym.fetch_results(self.sim, True)

        # compute observations, rewards, resets, ...
        self.post_physics_step()

        if self.dr_randomizations.get('observations', None):
            self.obs_buf = self.dr_randomizations['observations']['noise_lambda'](self.obs_buf)
