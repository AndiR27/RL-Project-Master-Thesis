#!/usr/bin/env python3
import wandb
from algorithms import DQN_Agent, PPO_Agent, SAC_Agent
from environments import EnvironnementGym
from training import Config_hyperparameters, Trainer

class EvalAlgos:
    def __init__(self, env_name, num_train_episodes, num_eval_episodes, max_steps, train_seeds):
        self.env_name = env_name
        self.num_train_episodes = num_train_episodes
        self.num_eval_episodes = num_eval_episodes
        self.max_steps = max_steps
        self.train_seeds = train_seeds
        self.env = EnvironnementGym(env_name).choix_env()
        self.algo_classes = {
            'DQN': DQN_Agent,
            'PPO': PPO_Agent,
            'SAC': SAC_Agent
        }
        self.configs = {
            'DQN': Config_hyperparameters.DQN_CONFIG,
            'PPO': Config_hyperparameters.PPO_CONFIG,
            'SAC': Config_hyperparameters.SAC_CONFIG
        }

    def train_and_evaluate(self):
        project_name = 'Algos_CartPole_Resultats'

        for seed in self.train_seeds:
            for algo_name, algo_class in self.algo_classes.items():
                agent_params = self.configs[algo_name]
                agent_params['seed'] = seed  # Update seed in agent params
                state_dim = self.env.observation_space.shape[0]
                action_dim = self.env.action_space.n
                agent = algo_class(state_dim, action_dim, **agent_params)

                # Training Phase
                run_train = wandb.init(project=project_name, config={
                    'algorithm': algo_name,
                    'seed': seed,
                    'env_name': self.env_name,
                    'num_episodes': self.num_train_episodes,
                    'max_steps': self.max_steps,
                    **agent_params
                }, name=f"{algo_name}_seed_{seed}", reinit=True)

                trainer = Trainer(agent, self.env, self.num_train_episodes, self.max_steps, self.num_eval_episodes, seed)
                train_rewards = trainer.train()
                run_train.finish()
                """
                for episode, reward in enumerate(train_rewards):
                    run_train.log({
                        'episode': episode,
                        'reward': reward,
                        'algorithm': algo_name,
                        'seed': seed,
                        'phase': 'training'
                    }, )
                run_train.finish()

                # Evaluation Phase
                run_eval = wandb.init(project=project_name, config={
                    'algorithm': algo_name,
                    'seed': seed,
                    'env_name': self.env_name,
                    'num_episodes': self.num_eval_episodes,
                    'max_steps': self.max_steps,
                    'phase': 'evaluation'
                }, name=f"{algo_name}_seed_{seed}_evaluation", reinit=True)

                eval_rewards = trainer.evaluate(self.num_eval_episodes)
                for episode, reward in enumerate(eval_rewards):
                    run_eval.log({
                        'episode': episode,
                        'reward': reward,
                        'algorithm': algo_name,
                        'seed': seed,
                        'phase': 'evaluation'
                    })
                run_eval.finish()
                """
def eval_algos():
    env_name = 'CartPole-v1'
    num_train_episodes = 2000
    num_eval_episodes = 5
    max_steps = 500
    train_seeds = [27, 42, 2024]

    eval_algos = EvalAlgos(env_name, num_train_episodes, num_eval_episodes, max_steps, train_seeds)
    eval_algos.train_and_evaluate()

if __name__ == '__main__':
    env_name = 'CartPole-v1'
    num_train_episodes = 2000
    num_eval_episodes = 5
    max_steps = 500
    train_seeds = [27, 42, 2024]

    eval_algos = EvalAlgos(env_name, num_train_episodes, num_eval_episodes, max_steps, train_seeds)
    eval_algos.train_and_evaluate()

