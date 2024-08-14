import argparse
import sys
from scripts.Eval_Algos import eval_algos
from scripts.runSweep import runSweep
from scripts.runWB import run_wb
from scripts.runNormal import run_normal


def main():
    parser = argparse.ArgumentParser(description="Choisissez quel script exécuter.")
    parser.add_argument('mode', choices=['sweep', 'single', 'eval', 'normal'],
                        help="Choisissez le mode : sweep, single, eval ou normal")

    # Parse uniquement le mode, et conserver les arguments restants
    args, remaining_args = parser.parse_known_args()

    # Modifier sys.argv pour les scripts spécifiques
    sys.argv = [sys.argv[0]] + remaining_args

    if args.mode == 'eval':
        # Exécution du script d'évaluation sans arguments supplémentaires
        eval_algos()

    elif args.mode == 'sweep':
        # Exécution du script de sweep avec les arguments supplémentaires
        runSweep()

    elif args.mode == 'single':
        # Exécution du script runWB avec les arguments supplémentaires
        run_wb()

    elif args.mode == 'normal':
        # Exécution du script runNormal avec les arguments supplémentaires
        run_normal()

if __name__ == '__main__':
    main()
