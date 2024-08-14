# config_sweep.py

DQN_SWEEP_CONFIG = {
    "hidden_dim": {"values": [64, 128, 256]},
    "learning_rate": {"values": [0.0005, 0.001, 0.005]},
    "gamma": {"values": [0.95, 0.99]},
    "batch_size": {"values": [32, 64]},
    "replay_memory_size": {"values": [50000, 100000]},
    "min_replay_memory_size": {"values": [100, 500]},
    "update_target_every": {"values": [50, 100]},
    "epsilon_start": {"value": 1.0},
    "epsilon_decay": {"values": [0.0001, 0.00005]},
    "epsilon_min": {"value": 0.01},
    "seed": {"values": [1, 42]}
}

PPO_SWEEP_CONFIG = {
    "actor_hidden_dim": {"values": [64, 128]},
    "critic_hidden_dim": {"values": [64, 128]},
    "actor_lr": {"values": [0.00005, 0.0001, 0.0005]},
    "critic_lr": {"values": [0.00005, 0.0001, 0.0005]},
    "gamma": {"values": [0.95, 0.99]},
    "lambda_advantage": {"values": [0.9, 0.92, 0.95]},
    "clip_epsilon": {"values": [0.1, 0.2]},
    "entropy_coeff": {"values": [0.01, 0.02]},
    "epochs": {"values": [5, 7, 10]},
    "batch_size": {"values": [32, 64]},
    "seed": {"values": [1, 42]}
}

SAC_SWEEP_CONFIG = {
    "hidden_dim": {"values": [64, 128]},
    "policy_lr": {"values": [0.00001, 0.00005, 0.0001]},
    "critic_lr": {"values": [0.0001, 0.001, 0.01]},
    "memory_replay_size": {"values": [50000, 100000, 200000]},
    "min_memory_replay_size": {"values": [10000, 20000]},
    "gamma": {"values": [0.95, 0.99]},
    "tau": {"values": [0.005, 0.01]},
    "alpha": {"values": [0.1, 0.2, 0.3]},
    "batch_size": {"values": [32, 64]},
    "seed": {"values": [1, 42]}
}
