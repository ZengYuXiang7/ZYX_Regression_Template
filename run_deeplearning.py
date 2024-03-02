# coding : utf-8
# Author : yuxiang Zeng
import collections
import time

import numpy as np
import pandas as pd
import torch
import argparse
from tqdm import *

from data import experiment, DataModule
from utils.dataloader import get_dataloaders
from utils.logger import Logger
from utils.metrics import class_Metrics, ErrorMetrics
from utils.monitor import EarlyStopping
from utils.trainer import get_loss_function, get_optimizer
from utils.utils import optimizer_zero_grad, optimizer_step, lr_scheduler_step, set_settings, set_seed

global log
# torch.set_default_dtype(torch.float)


class DNNInteraction(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DNNInteraction, self).__init__()
        self.input_dim = input_dim
        self.NeuCF = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim // 2),  # FFN
            torch.nn.LayerNorm(hidden_dim // 2),  # LayerNorm
            torch.nn.ReLU(),  # ReLU
            torch.nn.Linear(hidden_dim // 2, hidden_dim // 2),  # FFN
            torch.nn.LayerNorm(hidden_dim // 2),  # LayerNorm
            torch.nn.ReLU(),  # ReLU
            torch.nn.Linear(hidden_dim // 2, output_dim)  # y
        )

    def forward(self, x):
        x = x.to(torch.float)
        outputs = self.NeuCF(x)
        return outputs


class Model(torch.nn.Module):
    def __init__(self, input_size, hidden_size, args):
        super().__init__()
        self.args = args
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.prediction_layer = DNNInteraction(input_size, args.dimension, 1)

    def initialize(self):
        pass

    def forward(self, x):
        y = self.prediction_layer(x)
        return y.flatten()

    def setup_optimizer(self, args):
        self.to(args.device)
        self.loss_function = get_loss_function(args).to(args.device)
        self.optimizer = get_optimizer(self.parameters(), lr=args.lr, decay=args.decay, args=args)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=10, threshold=0.01)

    def train_one_epoch(self, dataModule):
        loss = None
        self.train()
        torch.set_grad_enabled(True)
        t1 = time.time()
        for train_Batch in (dataModule.train_loader):
            inputs, value = train_Batch
            inputs = inputs.to(self.args.device)
            value = value.to(self.args.device)
            pred = self.forward(inputs)
            loss = self.loss_function(pred, value)
            optimizer_zero_grad(self.optimizer)
            loss.backward()
            optimizer_step(self.optimizer)
        t2 = time.time()
        self.eval()
        torch.set_grad_enabled(False)
        return loss, t2 - t1

    def valid_one_epoch(self, dataModule):
        writeIdx = 0
        val_loss = 0.
        preds = torch.zeros((len(dataModule.valid_loader.dataset),)).to(self.args.device)
        reals = torch.zeros((len(dataModule.valid_loader.dataset),)).to(self.args.device)
        for valid_Batch in (dataModule.valid_loader):
            inputs, value = valid_Batch
            inputs = inputs.to(self.args.device)
            value = value.to(self.args.device)
            pred = self.forward(inputs)
            if self.args.classification:
                pred = torch.max(pred, 1)[1]  # 获取预测的类别标签
            preds[writeIdx:writeIdx + len(pred)] = pred
            reals[writeIdx:writeIdx + len(value)] = value
            writeIdx += len(pred)
            val_loss += self.loss_function(pred, value).item()
        self.scheduler.step(val_loss)
        if self.args.classification:
            valid_error = class_Metrics(reals, preds)
        else:
            valid_error = ErrorMetrics(reals * dataModule.max_value, preds * dataModule.max_value)
        return valid_error

    def test_one_epoch(self, dataModule):
        writeIdx = 0
        preds = torch.zeros((len(dataModule.test_loader.dataset),)).to(self.args.device)
        reals = torch.zeros((len(dataModule.test_loader.dataset),)).to(self.args.device)
        for test_Batch in (dataModule.test_loader):
            inputs, value = test_Batch
            inputs = inputs.to(self.args.device)
            value = value.to(self.args.device)
            pred = self.forward(inputs)
            if self.args.classification:
                pred = torch.max(pred, 1)[1]  # 获取预测的类别标签
            preds[writeIdx:writeIdx + len(pred)] = pred
            reals[writeIdx:writeIdx + len(value)] = value
            writeIdx += len(pred)
        if self.args.classification:
            test_error = class_Metrics(reals, preds)
        else:
            test_error = ErrorMetrics(reals * dataModule.max_value, preds * dataModule.max_value)
        return test_error


def RunOnce(args, runId, Runtime, log):
    # Set seed
    set_seed(args.seed + runId)

    # Initialize
    exper = experiment(args)
    datamodule = DataModule(exper, args)
    input_dim = datamodule.train_tensor.shape[1] - 1
    model = Model(input_dim, args.dimension, args)
    monitor = EarlyStopping(args.patience)

    # Setup training tool
    model.setup_optimizer(args)
    model.max_value = datamodule.max_value
    train_time = []
    for epoch in trange(args.epochs, disable=not args.program_test):
        epoch_loss, time_cost = model.train_one_epoch(datamodule)
        valid_error = model.valid_one_epoch(datamodule)
        if args.classification:
            monitor.track(epoch, model.state_dict(), -1 * valid_error['Acc'])
        else:
            monitor.track(epoch, model.state_dict(), valid_error['MAE'])

        train_time.append(time_cost)

        if args.verbose and epoch % args.verbose == 0 and not args.program_test:
            if args.classification:
                log.only_print(f"Round={runId + 1} Epoch={epoch + 1:02d} Loss={epoch_loss:.4f} vAcc={valid_error['Acc']:.4f} vF1={valid_error['F1']:.4f} vPrecision={valid_error['P']:.4f} vRecall={valid_error['Recall']:.4f} time={sum(train_time):.1f} s")
            else:
                log.only_print(f"Round={runId + 1} Epoch={epoch + 1:02d} Loss={epoch_loss:.4f} vMAE={valid_error['MAE']:.4f} vRMSE={valid_error['RMSE']:.4f} vNMAE={valid_error['NMAE']:.4f} vNRMSE={valid_error['NRMSE']:.4f} time={sum(train_time):.1f} s")
                log.only_print(f"Acc = [1%={valid_error['Acc'][0]:.4f}, 5%={valid_error['Acc'][1]:.4f}, 10%={valid_error['Acc'][2]:.4f}]")

        if monitor.early_stop:
            break

    model.load_state_dict(monitor.best_model)

    sum_time = sum(train_time[: monitor.best_epoch])

    results = model.test_one_epoch(datamodule) if args.valid else valid_error

    if args.classification:
        log(f'Round={runId + 1} BestEpoch={monitor.best_epoch:d} Acc={results["Acc"]:.4f} F1={results["F1"]:.4f} Precision={results["P"]:.4f} Recall={results["Recall"]:.4f} Training_time={sum_time:.1f} s\n')
    else:
        log(f'Round={runId + 1} BestEpoch={monitor.best_epoch:d} MAE={results["MAE"]:.4f} RMSE={results["RMSE"]:.4f} NMAE={results["NMAE"]:.4f} NRMSE={results["NRMSE"]:.4f} Training_time={sum_time:.1f} s')
        log(f"Acc = [1%={results['Acc'][0]:.4f}, 5%={results['Acc'][1]:.4f}, 10%={results['Acc'][2]:.4f}] ")

    return results

def RunExperiments(log, args):
    log('*' * 20 + 'Experiment Start' + '*' * 20)
    metrics = collections.defaultdict(list)

    for runId in range(args.rounds):
        runHash = int(time.time())
        results = RunOnce(args, runId, runHash, log)
        for key in results:
            metrics[key].append(results[key])

    log('*' * 20 + 'Experiment Results:' + '*' * 20)

    for key in metrics:
        log(f'{key}: {np.mean(metrics[key]):.4f} ± {np.std(metrics[key]):.4f}')

    if args.record:
        log.save_result(metrics)

    log('*' * 20 + 'Experiment Success' + '*' * 20)

    return metrics


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
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--decay', type=float, default=1e-3)
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
    args = parser.parse_args([])
    return args


if __name__ == '__main__':
    args = get_args()
    set_settings(args)
    log = Logger(args)
    args.log = log
    log(str(args))
    RunExperiments(log, args)


