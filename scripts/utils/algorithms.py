import argparse


def add_argument_algorithm(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--algo", type=str, default="a2c", choices=["a2c", "ppo"], help="RL algorithm"
    )
