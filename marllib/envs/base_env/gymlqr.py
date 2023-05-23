import time

from ray.rllib.env.multi_agent_env import MultiAgentEnv
from gym.spaces import Dict as GymDict, Discrete, Box
# from ray.rllib.env import PettingZooEnv, ParallelPettingZooEnv

import gym
import gymnasium
from gym_lqr import dilqr_v0

policy_mapping_dict = {
    "all_scenario": {
        "description": "D-LQR all scenarios",
        "team_prefix": ("team_",),
        "all_agents_one_policy": True,
        "one_agent_one_policy": True,
    },
}


# method from
# https://github.com/Stanford-ILIAD/PantheonRL/blob/master/pantheonrl/envs/pettingzoo.py#L27
def gymnasium_to_gym(space: gymnasium.spaces.Space) -> gym.Space:
    """
    Converter from gymnasium spaces to gym spaces for SB3 compatibility
    """
    if isinstance(space, gymnasium.spaces.box.Box):
        return gym.spaces.Box(
            space.low,
            space.high,
            dtype=space.dtype,
        )
    if isinstance(space, gymnasium.spaces.discrete.Discrete):
        return gym.spaces.Discrete(space.n)
    if isinstance(space, gymnasium.spaces.multi_discrete.MultiDiscrete):
        return gym.spaces.MultiDiscrete(space.nvec)
    if isinstance(space, gymnasium.spaces.multi_binary.MultiBinary):
        return gym.spaces.MultiBinary(space.n)

    raise NotImplementedError(f"Space {space} not implemented yet for gymnasium to gym conversion")


class RLlibDLQR(MultiAgentEnv):

    def __init__(self, env_config):
        # map = env_config["map_name"]
        # env_config.pop("map_name", None)
        # env = REGISTRY[map](**env_config)

        env = dilqr_v0.env(**env_config)

        # keep obs and action dim same across agents
        # pad_action_space_v0 will auto mask the padding actions
        # env = ss.pad_observations_v0(env)
        # env = ss.pad_action_space_v0(env)

        self.env = ParallelPettingZooEnv2(env)

        # TODO why 1 agent is used for all??
        # see wrapper
        # action_space = gymnasium_to_gym(env.action_space('agent_1'))
        # obs_space = gymnasium_to_gym(env.observation_space('agent_1'))

        # self.action_space = self.env.action_space

        self.action_space = Box(
            low=-10000000.0,
            high=10000000.0,
            shape=(self.env.action_space.shape[0],),
            dtype=self.env.action_space.dtype
        )

        self.observation_space = GymDict(
            {
                "obs": Box(
                    low=-10000000.0,
                    high=10000000.0,
                    shape=(self.env.observation_space.shape[0],),
                    dtype=self.env.observation_space.dtype
                )
            }
        )

        self.agents = self.env.agents
        self.num_agents = len(self.agents)
        # env_config["map_name"] = map
        self.env_config = env_config

    def reset(self):
        original_obs = self.env.reset()
        obs = {}
        for i in self.agents:
            obs[i] = {"obs": original_obs[i]}
        return obs

    def step(self, action_dict):
        o, r, d, info = self.env.step(action_dict)
        rewards = {}
        obs = {}
        for key in action_dict.keys():
            rewards[key] = r[key]
            obs[key] = {
                "obs": o[key]
            }
        dones = {"__all__": d["__all__"]}
        return obs, rewards, dones, info

    def close(self):
        self.env.close()

    def render(self, mode=None):
        self.env.render()
        time.sleep(0.05)
        return True

    def get_env_info(self):
        env_info = {
            "space_obs": self.observation_space,
            "space_act": self.action_space,
            "num_agents": self.num_agents,
            "episode_limit": 25,
            "policy_mapping_info": policy_mapping_dict
        }
        return env_info


# for compatibility with later versions of Gymnasium/PettingZoo
class ParallelPettingZooEnv2(MultiAgentEnv):
    def __init__(self, env):
        self.par_env = env
        # agent idx list
        self.agents = self.par_env.possible_agents

        # Get dictionaries of obs_spaces and act_spaces
        self.observation_spaces = self.par_env.observation_spaces
        self.action_spaces = self.par_env.action_spaces

        # Get first observation space, assuming all agents have equal space
        # TODO challenge assumption
        obs_space = gymnasium_to_gym(self.observation_spaces[self.agents[0]])
        self.observation_space = obs_space

        # Get first action space, assuming all agents have equal space
        action_space = gymnasium_to_gym(self.action_spaces[self.agents[0]])
        self.action_space = action_space

        # assert all(obs_space == self.observation_space
        #            for obs_space
        #            in self.par_env.observation_spaces.values()), \
        #     "Observation spaces for all agents must be identical. Perhaps " \
        #     "SuperSuit's pad_observations wrapper can help (useage: " \
        #     "`supersuit.aec_wrappers.pad_observations(env)`"
        #
        # assert all(act_space == self.action_space
        #            for act_space in self.par_env.action_spaces.values()), \
        #     "Action spaces for all agents must be identical. Perhaps " \
        #     "SuperSuit's pad_action_space wrapper can help (useage: " \
        #     "`supersuit.aec_wrappers.pad_action_space(env)`"

        self.reset()

    def reset(self):
        # TODO seed
        return self.par_env.reset()

    def step(self, action_dict):
        #  observations, rewards, terminations, truncations, infos
        obss, rews, terminations, truncations, infos = self.par_env.step(action_dict)

        truncations["__all__"] = all(truncations.values())

        return obss, rews, truncations, infos

    def close(self):
        self.par_env.close()

    def seed(self, seed=None):
        self.par_env.seed(seed)

    def render(self, mode="human"):
        return self.par_env.render(mode)
