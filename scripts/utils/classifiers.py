import argparse


def add_argument_classifier(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--features_dim",
        type=int,
        default=64,
        help="feature dimension of the classifier",
    )
    parser.add_argument(
        "--cls_contam_ratio",
        type=float,
        default=0.0,
        help="classifier contamination ratio",
    )
