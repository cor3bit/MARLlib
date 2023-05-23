import numpy as np

from gym_lqr import dilqr_v0

if __name__ == '__main__':

    # problem set-up
    env = dilqr_v0.env(
        env_id='HE1',  # scale
        n_agents=3,  # scale
        dt=0.1,
        deterministic=True,  # noise
        noise_multiplier=1.0,  # noise
        communication_type='ff',  # communication constraints
        communication_map=None,  # communication constraints
    )

    # initialize
    observations = env.reset(seed=1337)
    print(observations)

    # fix constant action
    const_action = np.array(env.action_space('agent_1').sample())
    actions = {agent: const_action for agent in env.agents}

    # make 5 steps with each agent
    for _ in range(5):
        observations, rewards, terminations, truncations, infos = env.step(actions)
