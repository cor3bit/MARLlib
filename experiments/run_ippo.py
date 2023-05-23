from marllib import marl

if __name__ == '__main__':
    # prepare the environment academy_pass_and_shoot_with_keeper
    # env = marl.make_env(environment_name="hanabi", map_name="Hanabi-Very-Small")
    env = marl.make_env(
        environment_name="gymlqr",
        # map_name="simple_spread",
        # force_coop=True,
        # continuous_actions=True,
    )

    # can add extra env params. remember to check env configuration before use
    # env = marl.make_env(environment_name='smac', map_name='3m', difficulty="6", reward_scale_rate=15)

    # initialize algorithm and load hyperparameters
    ippo = marl.algos.ippo(
        hyperparam_source="gymlqr",
    )

    # can add extra algorithm params. remember to check algo_config hyperparams before use
    # mappo = marl.algos.MAPPO(hyperparam_source='common', use_gae=True,  batch_episode=10, kl_coeff=0.2, num_sgd_iter=3)

    # build agent model based on env + algorithms + user preference if checked available
    model = marl.build_model(
        env,
        ippo,
        {
            "core_arch": "mlp",
            "encode_layer": "64-64",
        },
    )

    # start learning + extra experiment settings if needed. remember to check ray.yaml before use
    ippo.fit(
        env,
        model,
        stop={
            'episode_reward_mean': 1e8,
            'timesteps_total': 100_000,  # 10_000_000
        },
        local_mode=True,  # False
        num_gpus=0,
        num_workers=1,  # 5
        share_policy='individual',  # group
        checkpoint_freq=50000000, # 50
        checkpoint_end=False, # True
    )
