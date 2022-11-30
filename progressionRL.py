import argparse
import train
from utils import *

'''
    python ProgressionRL.py --output ./results/dapg --config ./config/dapg.txt
'''

parser = argparse.ArgumentParser(description='Policy gradient algorithms with demonstration data')
parser.add_argument('--output', type=str, required=True, help='location to store results')
parser.add_argument('--config', type=str, required=True, help='path to config file with exp params')
args = parser.parse_args()
demo = Demo()
origin_demo_path = demo.make_origin_demo()
train(args, origin_demo_path)
# modify gym_env.py : self.N = 1 -> 2 -> 3 -> 5 -> 6
policy_demo_path = demo.make_policy_demo()
train(args, policy_demo_path)
