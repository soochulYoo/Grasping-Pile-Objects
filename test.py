import argparse
import torch
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from env import make_env
import matplotlib.colors as colors

def test():
    ...

def plot_success_percentage():
    ...

def plot_performance(total_data_dict, save_file_path, epoch):
    '''
        agent 1 ~ 6
        total_data_dict = {agent number: { num of objects: [] , success data list: [] }}
        x: number of objects
        y: performance (= goal achieved / number of episodes) ex) 6 success in 10 episodes, success = (count == number of objects)
    '''
    for key in total_data_dict.keys(): # each key means each agent
        data_dict = total_data_dict[key]
        for idx in range(len(data_dict['num_of_objects'])):
            plt.plot(data_dict['num_of_objects'], data_dict['success_data_list'])
    plt.title('Results')
    plt.xlabel('Number of Objects')
    plt.ylabel('Performance')
    plt.savefig(f'{save_file_path}/{epoch}_train.png')
    plt.show()


