# config.py
DQN_CONFIG = {
    "hidden_dim": 64,
    "learning_rate": 0.001,
    "gamma": 0.99,
    "batch_size": 32,
    "replay_memory_size": 100000,
    "min_replay_memory_size": 100,
    "update_target_every": 100,
    "epsilon_start": 1.0,
    "epsilon_decay": 0.0001,
    "epsilon_min": 0.01,
    "seed": 1
}


PPO_CONFIG = {
    "actor_hidden_dim": 64,
    "critic_hidden_dim": 64,
    "actor_lr": 0.0001,
    "critic_lr": 0.0001,
    "gamma": 0.99,
    "lambda_advantage": 0.92,
    "clip_epsilon": 0.1,
    "entropy_coeff": 0.01,
    "epochs": 7,
    "batch_size": 32,
    "seed": 1
}



SAC_CONFIG = {
    "hidden_dim": 64,
    "policy_lr": 0.00001,
    "critic_lr": 0.01,
    "memory_replay_size": 100_000,
    "min_memory_replay_size": 20_000,
    "gamma": 0.99,
    "tau": 0.005,
    "alpha": 0.2,
    "batch_size": 64,
    "seed": 1
}

DQN_CONFIG_BREAKOUT = {
    "hidden_dim": 64,
    "learning_rate": 0.00025,
    "gamma": 0.99,
    "batch_size": 32,
    "replay_memory_size": 1000000,
    "min_replay_memory_size": 50000,
    "update_target_every": 10000,
    "epsilon_start": 1.0,
    "epsilon_decay": 0.000003,
    "epsilon_min": 0.05,
    "seed": 1
}

PPO_CONFIG_BREAKOUT = {
    "actor_hidden_dim": 64,
    "critic_hidden_dim": 64,
    "actor_lr": 0.00005,
    "critic_lr": 0.00005,
    "gamma": 0.98,
    "lambda_advantage": 0.95,
    "clip_epsilon": 0.15,
    "entropy_coeff": 0.01,
    "epochs": 10,
    "batch_size": 64,
    "seed": 1
}

SAC_CONFIG_BREAKOUT = {
    "hidden_dim": 64,
    "policy_lr": 0.00001,
    "critic_lr": 0.01,
    "gamma": 0.99,
    "tau": 0.005,
    "alpha": 0.2,
    "batch_size": 64,
    "seed": 1
}