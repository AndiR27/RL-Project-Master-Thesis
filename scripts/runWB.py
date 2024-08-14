import argparse

import wandb
from algorithms import DQN_Agent, PPO_Agent, SAC_Agent
from environments import EnvironnementGym
from training import Config_hyperparameters, Trainer


# Permet de lancer l'entrainement d'un agent au choix : sert principalement de test pour les hyperparametres
def run_training(agent_name, num_episodes, num_eval_episodes, seed):
    # Initialize the sweep by passing in the sweep configuration
    run_config = {
        'DQN': Config_hyperparameters.DQN_CONFIG,
        'PPO': Config_hyperparameters.PPO_CONFIG,
        'SAC': Config_hyperparameters.SAC_CONFIG
    }
    env = EnvironnementGym('CartPole-v1')
    env = env.choix_env()
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n
    # agent_params = run_config[agent_name]
    agent_params = run_config[agent_name]

    # Creer l'agent
    if agent_name == 'DQN':
        agent_class = DQN_Agent(input_dim, output_dim, **agent_params)
    elif agent_name == 'PPO':
        agent_class = PPO_Agent(input_dim, output_dim, **agent_params)
    elif agent_name == 'SAC':
        agent_class = SAC_Agent(input_dim, output_dim, **agent_params)
    else:
        raise ValueError(f"Agent {agent_name} not supported")

    runAgent = wandb.init(project=f"{agent_name}_CartPole", config=agent_params)
    # Create trainer and start training
    trainer = Trainer(agent_class, env, num_episodes=num_episodes, max_steps=500, num_eval_episodes=num_eval_episodes,
                      seed=seed)
    trainer.train()
    # Finish the WandB run
    runAgent.finish()

def run_wb():
    parser = argparse.ArgumentParser(description="Effectuer une seule experience avec W&B")
    parser.add_argument('--agent', type=str, default='SAC', help="Nom de l'agent (DQN, PPO, SAC)")
    parser.add_argument('--num_episodes', type=int, default=2000, help="Number d'episodes pour l'entrainement")
    parser.add_argument('--num_eval_episodes', type=int, default=5,
                        help="Nombre d'episodes pour l'evaluation de l'agent")
    parser.add_argument('--seed', type=int, default=42, help="Seed pour la reproductibilite de l'experience")

    args = parser.parse_args()
    run_training(args.agent, args.num_episodes, args.num_eval_episodes, args.seed)

if __name__ == '__main__':
    run_wb()
