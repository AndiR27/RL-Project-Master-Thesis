import argparse
#import algorithms
from algorithms import *
from algorithms import PPO
from algorithms import DQN
from environments.GestionEnv import Environnement
from training import train


def parse_args():
    parser = argparse.ArgumentParser(description="RL Agent Training")

    # Arguments pour l'agent et l'entraînement
    parser.add_argument('--algo', type=str, default='ppo', choices=['ppo', 'dqn', 'sac'], help='Algorithme à utiliser')
    parser.add_argument('--env', type=str, default='CartPole-v1', choices=['CartPole-v1', 'Pacman', 'Breakout'], help='Environnement à utiliser')
    parser.add_argument('--actor_hidden_dim', type=int, default=256,
                        help='Dimension des couches cachées du réseau d\'acteur')
    parser.add_argument('--critic_hidden_dim', type=int, default=256,
                        help='Dimension des couches cachées du réseau critique')
    parser.add_argument('--actor_lr', type=float, default=0.00095, help='Taux d\'apprentissage pour l\'acteur')
    parser.add_argument('--critic_lr', type=float, default=0.0001, help='Taux d\'apprentissage pour le critique')
    parser.add_argument('--gamma', type=float, default=0.95, help='Facteur de discount')
    parser.add_argument('--lambda_advantage', type=float, default=0.91, help='Facteur de lissage pour GAE')
    parser.add_argument('--clip_epsilon', type=float, default=0.2, help='Paramètre de clipping pour PPO')
    parser.add_argument('--entropy_coeff', type=float, default=0.0028, help='Coefficient d\'entropie pour PPO')
    parser.add_argument('--epochs', type=int, default=60, help='Nombre d\'époques pour l\'entraînement')
    parser.add_argument('--batch_size', type=int, default=256, help='Taille des mini-batches')
    parser.add_argument('--num_episodes', type=int, default=2000, help='Nombre d\'épisodes pour l\'entraînement')
    parser.add_argument('--max_steps', type=int, default=500, help='Nombre maximal de pas par épisode')

    return parser.parse_args()

def choix_algo_env(args):
    # Créer l'environnement
    env = choix_env(args)
    # Définir les dimensions d'entrée et de sortie
    actor_input_dim = env.observation_space.shape[0]
    actor_output_dim = env.action_space.n
    critic_input_dim = env.observation_space.shape[0]
    agent = None
    # Sélectionner et initialiser l'agent en fonction de l'algorithme choisi
    if args.algo == 'ppo':
        agent = PPO.PPOAgent(
            actor_input_dim, actor_output_dim, critic_input_dim,
            args.actor_hidden_dim, args.critic_hidden_dim,
            args.actor_lr, args.critic_lr, args.gamma,
            args.lambda_advantage, args.clip_epsilon, args.entropy_coeff,
            args.epochs, args.batch_size
        )
    elif args.algo == 'dqn':
        agent = DQN.DQNAgent(
            actor_input_dim, actor_output_dim,
            args.critic_hidden_dim, args.critic_lr, args.gamma, args.batch_size
        )
    # elif args.algo == 'sac':
    #     agent = SACAgent(
    #         actor_input_dim, actor_output_dim, critic_input_dim,
    #         args.actor_hidden_dim, args.critic_hidden_dim,
    #         args.actor_lr, args.critic_lr, args.gamma,
    #         args.lambda_advantage, args.clip_epsilon, args.entropy_coeff,
    #         args.epochs, args.batch_size
    #     )

    return agent, env

def choix_env(args):
    env = Environnement(args.env)
    return env.choix_env()

def main():
    args = parse_args()
    agent, env = choix_algo_env(args)
    trainer = train.Trainer(agent, env, args.num_episodes, args.max_steps)
    trainer.train()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
