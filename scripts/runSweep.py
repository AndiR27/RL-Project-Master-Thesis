# class sweep like train
import argparse

import wandb
from algorithms import DQN_Agent, PPO_Agent, SAC_Agent
from environments import EnvironnementGym
from training import Config_Sweep, Trainer


def runTraining(config=None):
    with wandb.init() as run:
        agent_name = wandb.config.agent_name
        env_name = wandb.config.env_name

        # Initialize environment
        env = EnvironnementGym(env_name)
        env = env.choix_env()
        input_dim = env.observation_space.shape[0]
        output_dim = env.action_space.n
        # Define agent-specific parameter filters
        agent_params = {
            'DQN': ['hidden_dim', 'learning_rate', 'gamma', 'batch_size', 'replay_memory_size',
                    'min_replay_memory_size',
                    'update_target_every', 'epsilon_start', 'epsilon_decay', 'epsilon_min', 'seed'],
            'PPO': ['actor_hidden_dim', 'critic_hidden_dim', 'actor_lr', 'critic_lr', 'gamma', 'lambda_advantage',
                    'clip_epsilon', 'entropy_coeff', 'epochs', 'batch_size', 'seed'],
            'SAC': ['hidden_dim', 'policy_lr', 'critic_lr', 'memory_replay_size', 'min_memory_replay_size', 'gamma',
                    'tau',
                    'alpha', 'batch_size', 'seed']
        }

        # Filter relevant parameters for the specific agent
        agent_config = {key: wandb.config[key] for key in agent_params[agent_name]}
        # Creer l'agent
        if agent_name == 'DQN':
            agent_class = DQN_Agent(input_dim, output_dim, **agent_config)
        elif agent_name == 'PPO':
            agent_class = PPO_Agent(input_dim, output_dim, **agent_config)
        elif agent_name == 'SAC':
            agent_class = SAC_Agent(input_dim, output_dim, **agent_config)
        else:
            raise ValueError(f"Agent {agent_name} not supported")

        # Create trainer and start training
        trainer = Trainer(agent_class, env, num_episodes=wandb.config.num_episodes,
                          max_steps=500, num_eval_episodes=wandb.config.num_eval_episodes,
                          seed=wandb.config.seed)
        trainer.train()


def manage_sweep(agent_name, num_episodes=2000, num_eval_episodes=5, seed=42):
    sweep_configs = {
        'DQN': Config_Sweep.DQN_SWEEP_CONFIG,
        'PPO': Config_Sweep.PPO_SWEEP_CONFIG,
        'SAC': Config_Sweep.SAC_SWEEP_CONFIG
    }

    if agent_name not in sweep_configs:
        raise ValueError(f"Agent {agent_name} not supported")

    sweep_config = {
        'method': 'grid',  # or 'random' or 'bayes'
        'metric': {
            'name': 'Total Reward',
            'goal': 'maximize'
        },
        'parameters': sweep_configs[agent_name]
    }

    # Additional parameters for training, but not passed to the agent
    sweep_config['parameters'].update({
        'num_episodes': {'value': num_episodes},
        'num_eval_episodes': {'value': num_eval_episodes},
        'seed': {'value': seed},
        'agent_name': {'value': agent_name},
        'env_name': {'value': 'CartPole-v1'}  # You can make this configurable as well
    })

    sweep_id = wandb.sweep(sweep_config, project=f"{agent_name}_Sweep")
    wandb.agent(sweep_id, function=runTraining)


def runSweep():
    parser = argparse.ArgumentParser(description="Run a hyperparameter sweep for a specified agent.")
    parser.add_argument('--agent', type=str, default='SAC', help="Nom de l'agent (DQN, PPO, SAC)")
    parser.add_argument('--num_episodes', type=int, default=2000, help="Number d'episodes pour l'entrainement")
    parser.add_argument('--num_eval_episodes', type=int, default=5,
                        help="Nombre d'episodes pour l'evaluation de l'agent")
    parser.add_argument('--seed', type=int, default=42, help="Seed pour la reproductibilite de l'experience")

    args = parser.parse_args()
    manage_sweep(args.agent, args.num_episodes, args.num_eval_episodes, args.seed)


if __name__ == '__main__':
    runSweep()
