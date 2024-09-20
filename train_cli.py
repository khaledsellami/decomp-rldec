import argparse

from rldec.train_model import train_model
from rldec.config import MODELS_PATH, DATA_PATH


def train(args: argparse.Namespace):
    app_name = args.APP
    app_repo = args.repo
    rldec_approach = args.approach
    features_name = args.feature
    config_path = args.config
    data_path = args.data
    output_path = args.output
    model_name = args.model
    eval_model = args.eval
    n_iterations = args.iterations
    data_format = args.format
    train_model(app_name, rldec_approach, features_name, n_iterations, app_repo, config_path, data_path, output_path,
                model_name, eval_model, data_format=data_format)


if __name__ == "__main__":
    train_parser = argparse.ArgumentParser(
        prog='rldec',
        description='train rldec to decompose an application')
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
    args = train_parser.parse_args()
    train(args)
