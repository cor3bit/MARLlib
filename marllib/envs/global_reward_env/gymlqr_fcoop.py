from marllib.envs.base_env.gymlqr import RLlibDLQR


class RLlibDLQR_FCOOP(RLlibDLQR):

    def __init__(self, env_config):
        super().__init__(env_config)

    def step(self, action_dict):
        o, r, d, info = self.env.step(action_dict)
        reward = 0
        for key in r.keys():
            reward += r[key]
        rewards = {}
        obs = {}
        for key in action_dict.keys():
            rewards[key] = reward / self.num_agents
            obs[key] = {
                "obs": o[key]
            }
        dones = {"__all__": d["__all__"]}
        return obs, rewards, dones, info
