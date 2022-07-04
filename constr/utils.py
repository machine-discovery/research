from typing import Optional
import argparse

CONSOLE_ARGUMENTS = None

def parse_arguments():
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument("--nb_epoch", type=int, default=1000,
                           help="number of epochs per active learning iteration")
    argparser.add_argument("--nb_gpus", type=int, default=1, help="number of GPUs, -1 to use all GPUs available")
    argparser.add_argument("--batch", type=int, default=32, help="batch size")
    argparser.add_argument("--reg_real", type=float, default=1,
                           help="regularisation weight for real part of Jacobian eigenvalue")
    argparser.add_argument("--logdir", type=str, default='./log', help="directory to store training information")
    argparser.add_argument("--logname", type=str, default='mass-spring', help="training folder name")
    argparser.add_argument("--version", type=str, default='random_version', help="training version")
    argparser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    argparser.add_argument("--hmax", type=float, default=1e-1, help="maximum numerical solver step size")
    argparser.add_argument("--solver", type=str, default='rk4', help="rk4 only")
    argparser.add_argument("--experiment", type=str, default="mass-spring", help="experiment")
    argparser.add_argument("--val_seed", type=int, default=2, help="validation data random seed")
    argparser.add_argument("--train_seed", type=int, default=1, help="training data random seed")
    argparser.add_argument("--noiseless", type=lambda x: (str(x).lower() == 'true'), default=False, help="noiseless or not")
    argparser.add_argument("--gradient_clip_val", type=float, default=0, help="gradient clipping")
    argparser.add_argument("--model", type=str, default='baseline', help="model")

    global CONSOLE_ARGUMENTS
    CONSOLE_ARGUMENTS = argparser.parse_args()
    return CONSOLE_ARGUMENTS

CONSOLE_ARGUMENTS = parse_arguments()
