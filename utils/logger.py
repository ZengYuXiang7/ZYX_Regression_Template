# coding : utf-8
# Author : yuxiang Zeng
# 日志
import logging
import pickle
import sys
import time
import numpy as np
import platform

from utils.utils import makedir

class Logger:
    def save_result(self, metrics):
        args = self.args
        makedir('./results/metrics/')
        if args.dimension == None:
            address = f'./results/metrics/Machine_learning_{args.dataset}_{args.density}'
        else:
            address = f'./results/metrics/{args.model}_{args.dataset}_{args.density}_{args.dimension}'
        for key in metrics:
            pickle.dump(np.mean(metrics[key]), open(address + key + 'mean.pkl', 'wb'))
            pickle.dump(np.std(metrics[key]), open(address + key + 'std.pkl', 'wb'))

    def __init__(self, args):
        self.args = args
        makedir('./results/log/')
        if args.experiment:
            ts = time.asctime().replace(' ', '_').replace(':', '_')
            if args.dimension == None:
                address = f'./results/log/Machine_learning_{args.dataset}_{args.density}'
            else:
                address = f'./results/log/{args.dataset}_{args.density}_{args.dimension}'
            logging.basicConfig(level=logging.INFO, filename=f'{address}_{ts}.log', filemode='w')
        else:
            logging.basicConfig(level=logging.INFO, filename=f'./' + 'None.log', filemode='a')
        self.logger = logging.getLogger(self.args.model)

    # 日志记录
    def log(self, string):
        import time
        if string[0] == '\n':
            print('\n', end='')
            string = string[1:]
        final_string = time.strftime('|%Y-%m-%d %H:%M:%S| ', time.localtime(time.time())) + string
        green_string = f'\033[92m{final_string}\033[0m'
        self.logger.info(final_string[:-1])
        print(green_string)

    def __call__(self, string):
        if self.args.verbose:
            self.log(string)

    def only_print(self, string):
        import time
        if string[0] == '\n':
            print('\n', end='')
            string = string[1:]
        final_string = time.strftime('|%Y-%m-%d %H:%M:%S| ', time.localtime(time.time())) + string
        green_string = f'\033[92m{final_string}\033[0m'
        print(green_string)
