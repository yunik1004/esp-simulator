import argparse


def add_argument_classifier(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--features_dim",
        type=int,
        default=4,
        help="feature dimension of the classifier",
    )
