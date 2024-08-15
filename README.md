# Projet RL : Thèse de Master - DQN, PPO, SAC

Ce projet est développé dans le cadre d'une thèse de master à la Haute École de Gestion. L'objectif principal est d'explorer et de comparer plusieurs algorithmes d'apprentissage par renforcement (Reinforcement Learning - RL) appliqués à des environnements de simulation, tels que **CartPole-v1** et **Breakout**.

Les algorithmes principaux étudiés et implémentés dans ce projet sont :
- **DQN** (Deep Q-Network) : Un algorithme qui utilise les réseaux de neurones pour approximer les fonctions de valeur d'action dans les environnements avec des états et des actions discrets.
- **PPO** (Proximal Policy Optimization) : Un algorithme d'optimisation de politique qui ajuste la politique de manière stable en limitant les changements radicaux.
- **SAC** (Soft Actor-Critic) : Un algorithme qui maximise à la fois la récompense cumulative et l'entropie, favorisant l'exploration.

Le projet utilise **PyTorch** pour l'implémentation des réseaux de neurones et **Weights & Biases** (W&B) pour suivre les expériences et visualiser les résultats.

## Installation

Pour exécuter ce projet, vous devez avoir Python 3.8+ et `pip` installés.

### Cloner le dépôt

```bash
git clone https://github.com/AndiR27/RL-Project-Master-Thesis.git
cd RL-Project-Master-Thesis
```

### Configurer l'environnement

1. Créez un environnement virtuel avec Python ou conda.

   **Avec Python :**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Sur Linux/Mac
   venv\Scripts\activate     # Sur Windows
   ```

   **Avec Conda :**
   ```bash
   conda create --name rl_project python=3.8
   conda activate rl_project
   ```

2. Installez les dépendances requises :

   ```bash
   pip install -r requirements.txt
   ```

3. (Optionnel) Si vous utilisez Weights & Biases pour suivre vos expériences, connectez-vous à votre compte W&B :

   ```bash
   wandb login
   ```

## Structure du projet

Le projet est structuré en différents sous-dossiers :
- **algorithms/** : Contient les implémentations des algorithmes DQN, PPO, et SAC.
- **scripts/** : Contient les scripts principaux pour l'exécution des algorithmes, le sweep hyperparamétrique et l'évaluation des modèles.
- **utils/** : Contient des utilitaires supplémentaires pour gérer la mémoire, le logging, etc.

## Exécution des Scripts

Le fichier `main.py` permet d'exécuter différents scripts en fonction du mode choisi. Voici comment utiliser le script `main.py`.

### Commande de base

```bash
python main.py <mode>
```

### Modes disponibles

1. **Mode `sweep`** :
   Lance un sweep hyperparamétrique pour explorer différentes configurations d'algorithmes dans Weight&Biases.
   ```bash
   python main.py sweep
   ```

2. **Mode `single`** :
   Lance une exécution d'algorithme avec un ensemble spécifique de paramètres. Utilisé pour entraîner un agent unique avec des hyperparamètres précis dans Weight&Biases.
   ```bash
   python main.py single
   ```

3. **Mode `eval`** :
   Lance le script d'évaluation pour évaluer les performances des algorithmes sur un environnement donné après l'entraînement.
   ```bash
   python main.py eval
   ```

4. **Mode `normal`** :
   Exécute un entraînement simple sur un environnement spécifique avec les paramètres par défaut sans passer par Weight&Biases, les différentes informations de logging sont effectués sous forme d'affichages dans le terminal.
   ```bash
   python main.py normal
   ```

## Exemples d'exécution

### Lancer un sweep hyperparamétrique avec des hyperparamètres spécifiques
```bash
python main.py sweep --algo DQN --env CartPole-v1 --lr 0.001 --gamma 0.99 --epsilon_decay 0.01
```

### Lancer un entraînement unique pour PPO avec des hyperparamètres modifiés
```bash
python main.py single --algo PPO --env CartPole-v1 --epochs 20 --batch_size 128 --clip_epsilon 0.1 --learning_rate 0.0003
```

## Contributions

Les contributions à ce projet sont les bienvenues. Si vous avez des suggestions ou des améliorations, n'hésitez pas à ouvrir une **issue** ou à soumettre une **pull request**.

## Licence

Ce projet est sous licence MIT.

## Contact

Pour toute question, vous pouvez me contacter à l'adresse suivante : [andi.ramiqi@hesge.ch](mailto:andi.ramiqi@hesge.ch).

---

Cet exemple de README fournit des instructions claires sur l'installation, la structure du projet, l'utilisation des scripts, ainsi qu'une explication du projet et des algorithmes étudiés. Vous pouvez personnaliser davantage les sections en fonction de vos besoins spécifiques.
