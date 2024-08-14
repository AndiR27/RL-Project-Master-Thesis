import argparse
# import algorithms
from algorithms import *
from algorithms import PPO
from algorithms import DQN
from algorithms.SAC import SAC_Agent
from environments.GestionEnv import EnvironnementGym
from training import train
from training.train import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description="RL Agent Training")

    # Arguments pour choisir l'algorithme et l'environnement
    parser.add_argument('--agent', type=str, default='SAC', choices=['PPO', 'DQN', 'SAC'], help='Algorithme à utiliser')
    parser.add_argument('--env', type=str, default='CartPole-v1', choices=['CartPole-v1', 'Breakout'],
                        help='Environnement à utiliser')

    # Arguments pour le Trainer
    parser.add_argument('--num_episodes', type=int, default=1000, help='Nombre d\'épisodes pour l\'entraînement')
    parser.add_argument('--num_eval_episodes', type=int, default=10, help="Nombre d'épisodes pour l'évaluation")
    parser.add_argument('--max_steps', type=int, default=500, help='Nombre maximal de pas par épisode')
    parser.add_argument('--seed', type=int, default=1, help='Graine aléatoire pour la reproductibilité')

    # Arguments communs potentiels pour les agents
    parser.add_argument('--actor_hidden_dim', type=int, default=256,
                        help='Dimension des couches cachées du réseau d\'acteur')
    parser.add_argument('--critic_hidden_dim', type=int, default=256,
                        help='Dimension des couches cachées du réseau critique')
    parser.add_argument('--gamma', type=float, default=0.99, help='Facteur de discount')
    parser.add_argument('--batch_size', type=int, default=32, help='Taille des mini-batches')

    # Arguments pour la memoire :
    parser.add_argument('--replay_memory_size', type=int, default=100000, help='Taille de la mémoire de replay')
    parser.add_argument('--min_replay_memory_size', type=int, default=100,
                        help='Taille minimale de la mémoire de replay pour commencer l\'apprentissage')

    # Learning Rates
    parser.add_argument('--actor_lr', type=float, default=0.001, help='Taux d\'apprentissage pour l\'acteur')
    parser.add_argument('--critic_lr', type=float, default=0.001, help='Taux d\'apprentissage pour le critique')
    parser.add_argument('--dqn_lr', type=float, default=0.001, help='Taux d\'apprentissage pour DQN')


    # Arguments spécifiques à PPO
    parser.add_argument('--lambda_advantage', type=float, default=0.95, help='Facteur de lissage pour GAE')
    parser.add_argument('--clip_epsilon', type=float, default=0.2, help='Paramètre de clipping pour PPO')
    parser.add_argument('--entropy_coeff', type=float, default=0.01, help='Coefficient d\'entropie pour PPO')
    parser.add_argument('--epochs', type=int, default=10, help='Nombre d\'époques pour l\'entraînement')

    # Arguments spécifiques à DQN
    parser.add_argument('--epsilon_start', type=float, default=1.0,
                        help='Valeur initiale de epsilon pour l\'exploration')
    parser.add_argument('--epsilon_decay', type=float, default=0.0001, help='Taux de décroissance de epsilon')
    parser.add_argument('--epsilon_min', type=float, default=0.01, help='Valeur minimale de epsilon')
    parser.add_argument('--update_target_every', type=int, default=100,
                        help="Fréquence de mise à jour du réseau cible dans DQN")

    # Arguments spécifiques à SAC
    parser.add_argument('--tau', type=float, default=0.005,
                        help="Paramètre tau pour les mises à jour de la cible dans SAC")
    parser.add_argument('--alpha', type=float, default=0.2, help="Coefficient d'entropie dans SAC")

    return parser.parse_args()


def choix_algo_env(args):
    # Créer l'environnement
    env = choix_env(args)
    # Définir les dimensions d'entrée et de sortie
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n

    agent = None
    if args.agent == 'PPO':
        agent = PPO.PPO_Agent(
            input_dim,
            output_dim,
            actor_hidden_dim=args.actor_hidden_dim,
            critic_hidden_dim=args.critic_hidden_dim,
            actor_lr=args.actor_lr,
            critic_lr=args.critic_lr,
            gamma=args.gamma,
            lambda_advantage=args.lambda_advantage,
            clip_epsilon=args.clip_epsilon,
            entropy_coeff=args.entropy_coeff,
            epochs=args.epochs,
            batch_size=args.batch_size,
            seed=args.seed
        )

    elif args.agent == 'DQN':
        agent = DQN.DQN_Agent(
            input_dim, output_dim,
            hidden_dim=args.actor_hidden_dim,
            learning_rate=args.dqn_lr,
            gamma=args.gamma,
            batch_size=args.batch_size,
            replay_memory_size=args.replay_memory_size,
            min_replay_memory_size=args.min_replay_memory_size,
            update_target_every=args.update_target_every,  # Cette valeur peut être rendue configurable si nécessaire
            epsilon_start=args.epsilon_start,
            epsilon_decay=args.epsilon_decay,
            epsilon_min=args.epsilon_min,
            seed=args.seed
        )
    elif args.agent == 'SAC':
        agent = SAC_Agent(
            input_dim, output_dim,
            hidden_dim=args.actor_hidden_dim,
            policy_lr=args.actor_lr,
            critic_lr=args.critic_lr,
            memory_replay_size=args.replay_memory_size,
            min_memory_replay_size=args.min_replay_memory_size,
            gamma=args.gamma,
            tau=args.tau,  # Cette valeur peut être rendue configurable si nécessaire
            alpha=args.alpha,  # Cette valeur peut être rendue configurable si nécessaire
            batch_size=args.batch_size,
            seed=args.seed
        )

    return agent, env


def choix_env(args):
    env = EnvironnementGym(args.env)
    return env.choix_env()


def run_normal():
    args = parse_args()
    agent, env = choix_algo_env(args)
    trainer = Trainer(agent, env, args.num_episodes, args.max_steps, args.num_eval_episodes, seed=42)
    trainer.train(logging_prints=True, no_eval=True)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    run_normal()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
