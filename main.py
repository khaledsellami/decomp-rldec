import argparse
import logging

from decompose_cli import decompose
from train_cli import train
from server import serve
from rldec.config import MODELS_PATH, DATA_PATH


if __name__ == "__main__":
    # Parsing input
    parser = argparse.ArgumentParser(
        prog='rldec',
        description='run rldec to decompose an application or start the RLDec server')
    subparsers = parser.add_subparsers(dest="subtask", required=True)
    # train task parser
    train_parser = subparsers.add_parser("train", description="train a rldec model on an application")
    train_parser.add_argument('APP', type=str, help='application to train decomposition on')
    train_parser.add_argument("-i", "--iterations", help='number of training iterations', type=int, default=20)
    train_parser.add_argument("-a", "--approach", help='rldec variant to use for training',
                              default="combsequential", choices=["combsequential", "sequential", "flattened"])
    train_parser.add_argument("-c", "--config", help='path for the configuration file', type=str, default=None)
    train_parser.add_argument("-f", "--feature", help='which feature to use (invalid with combsequential)',
                              default="structural", choices=["structural", "semantic"])
    train_parser.add_argument("-d", "--data", help='path for the data', type=str, default=DATA_PATH)
    train_parser.add_argument("-o", "--output", help='path for the output', type=str, default=MODELS_PATH)
    train_parser.add_argument("-m", "--model", help='name of the model to train', default=None)
    train_parser.add_argument("-r", "--repo", help='link for the github repository', type=str)
    train_parser.add_argument("-V", "--eval", help='evaluate the model during training', action="store_true")
    train_parser.add_argument("-l", "--level", help='granularity level of the analysis', type=str,
                              default="class", choices=["class", "method"])
    train_parser.add_argument("-D", "--is_distributed", action="store_true",
                              help='True is the application has a distributed architecture')
    train_parser.add_argument("-F", "--format", help='data format', type=str, default="parquet",
                              choices=["parquet", "csv"])
    # decompose task parser
    decompose_parser = subparsers.add_parser("decompose", description="decompose an application using a trained model")
    decompose_parser.add_argument('APP', type=str, help='application to decompose')
    decompose_parser.add_argument("-m", "--model", help='name of the trained model', default=None)
    decompose_parser.add_argument("-p", "--path", help='path for the trained model', type=str, default=MODELS_PATH)
    decompose_parser.add_argument("-o", "--output", help='output path', type=str, default=None)
    decompose_parser.add_argument("-e", "--episodes", help='number of episodes', type=int, default=1)
    decompose_parser.add_argument("-s", "--strategy", help='selection strategy for the decomposition', type=str,
                                  choices=["best", "last"], default="last")
    decompose_parser.add_argument("-v", "--verbose", help='print the final output', action="store_true")
    # serve task parser
    serve_parser = subparsers.add_parser("start", description="start the RLDec server")
    # configure logging
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # route the task
    args = parser.parse_args()
    if args.subtask == "train":
        train(args)
    elif args.subtask == "decompose":
        decompose(args)
    else:
        serve()
