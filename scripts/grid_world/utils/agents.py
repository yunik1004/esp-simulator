import argparse


def add_argument_train_agent(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--log_dir", type=str, default="./output", help="tensorboard log dir"
    )
    parser.add_argument(
        "--log_name", type=str, default="GridWorld", help="tensorboard log name"
    )
    parser.add_argument("--lr", type=float, default=3e-4, help="learning rate")
    parser.add_argument(
        "--n_steps",
        type=int,
        default=2048,
        help="the number of steps to run for each environment per update",
    )
    parser.add_argument(
        "--ent_coef", type=float, default=0.01, help="entropy coefficient"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=4, help="the number of epochs"
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=64,
        help="the number of concurrent environments",
    )
    parser.add_argument(
        "--num_timesteps", type=int, default=int(1e9), help="the number of timesteps"
    )
    parser.add_argument("--batch_size", type=int, default=1024, help="minibatch size")
    parser.add_argument("--gamma", type=float, default=1.0, help="discount factor")
    parser.add_argument(
        "--checkpoint_freq",
        type=int,
        default=int(1e4),
        help="checkpoint saving frequency",
    )
