import argparse
from typing import List, Dict

from rldec.decompose import generate_decomposition
from rldec.config import MODELS_PATH


def decompose(args: argparse.Namespace) -> Dict[str, List[str]]:
    app_name = args.APP
    model_name = args.model
    models_path = args.path
    output_path = args.output
    num_episodes = args.episodes
    select_strategy = args.strategy
    verbose = args.verbose
    return generate_decomposition(app_name, models_path, model_name, verbose, output_path, num_episodes,
                                  select_strategy)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='rldec',
        description='run rldec to decompose an application')
    parser.add_argument('APP', type=str, help='application to decompose')
    parser.add_argument("-m", "--model", help='name of the trained model', default=None)
    parser.add_argument("-p", "--path", help='path for the trained model', type=str, default=MODELS_PATH)
    parser.add_argument("-o", "--output", help='output path', type=str, default=None)
    parser.add_argument("-e", "--episodes", help='number of episodes', type=int, default=1)
    parser.add_argument("-s", "--strategy", help='selection strategy for the decomposition', type=str,
                        choices=["best", "last"], default="last")
    parser.add_argument("-v", "--verbose", help='print the final output', action="store_true")
    args = parser.parse_args()
    decompose(args)
