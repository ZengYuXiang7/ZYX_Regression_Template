# coding : utf-8
# Author : yuxiang Zeng
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--rounds', type=int, default=5)

    parser.add_argument('--dataset', type=str, default='data')  #
    parser.add_argument('--model', type=str, default='DNN')  #

    # Experiment
    parser.add_argument('--density', type=float, default=0.60)
    parser.add_argument('--debug', type=int, default=0)
    parser.add_argument('--record', type=int, default=0)
    parser.add_argument('--program_test', type=int, default=1)
    parser.add_argument('--valid', type=int, default=1)
    parser.add_argument('--experiment', type=int, default=0)
    parser.add_argument('--verbose', type=int, default=1)
    parser.add_argument('--path', nargs='?', default='./datasets/')

    # Training tool
    parser.add_argument('--device', type=str, default='cpu')  # gpu cpu mps
    parser.add_argument('--bs', type=int, default=512)  #
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=5000)
    parser.add_argument('--decay', type=float, default=1e-4)
    parser.add_argument('--lr_step', type=int, default=10)
    parser.add_argument('--patience', type=int, default=1000)
    parser.add_argument('--saved', type=int, default=1)

    parser.add_argument('--loss_func', type=str, default='L1Loss')
    parser.add_argument('--optim', type=str, default='AdamW')

    # Hyper parameters
    parser.add_argument('--dimension', type=int, default=6)

    # Other Experiment
    parser.add_argument('--ablation', type=int, default=0)
    parser.add_argument('--classification', type=int, default=0)
    args = parser.parse_args()
    return args


def get_ml_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--rounds', type=int, default=5)

    parser.add_argument('--dataset', type=str, default='cpu')  #
    parser.add_argument('--model', type=str, default='CF')  #

    # Experiment
    parser.add_argument('--density', type=float, default=0.01)
    parser.add_argument('--debug', type=int, default=0)
    parser.add_argument('--record', type=int, default=1)
    parser.add_argument('--program_test', type=int, default=0)
    parser.add_argument('--valid', type=int, default=1)
    parser.add_argument('--experiment', type=int, default=0)
    parser.add_argument('--verbose', type=int, default=1)
    parser.add_argument('--path', nargs='?', default='./datasets/')

    # Training tool
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--dimension', type=int, default=None)
    args = parser.parse_args()
    return args