import argparse


def add_argument_algorithm(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--algo",
        type=str,
        default="ppo",
        choices=["ppo"],
        help="policy optimization algorithm",
    )
