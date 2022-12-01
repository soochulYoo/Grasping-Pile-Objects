import argparse
from train import train
from utils import Demo
from env import make_env
import gym_grasp
from mjrl.utils.gym_env import GymEnv
import tqdm

'''
    python progressionRL.py --output ./results/dapg --config ./config/dapg.txt
'''

# parser = argparse.ArgumentParser(description='Policy gradient algorithms with demonstration data')
# parser.add_argument('--output', type=str, required=True, help='location to store results')
# parser.add_argument('--config', type=str, required=True, help='path to config file with exp params')
# args = parser.parse_args()



demo = Demo()

origin_to_single_demo_path = demo.make_origin_demo()

args = argparse.Namespace(output = './progression_results/dapg_1', config = './config/dapg.txt')
single_best_policy_path = train(args, origin_to_single_demo_path, demo.env, 1)
print("Demo Env Number: ", demo.env.env.N)

single_policy_demo_path = demo.make_policy_demo(N = 1, best_policy_path = single_best_policy_path)

single_to_double_demo_path = demo.make_progression_demo(N = 2, policy_demo_path = single_policy_demo_path)

args = argparse.Namespace(output = './progression_results/dapg_2', config = './config/dapg.txt')
env2 = make_env('grasp-v0')
env2 = GymEnv(env2.env.set_number_of_objects(2))
double_best_policy_path = train(args, single_to_double_demo_path, env2, 2)
print("Env Number: ", env2.env.N)

double_policy_demo_path = demo.make_policy_demo(N = 2, best_policy_path = double_best_policy_path)

double_to_triple_demo_path = demo.make_progression_demo(N = 3, policy_demo_path = double_policy_demo_path)

args = argparse.Namespace(output = './progression_results/dapg_3', config = './config/dapg.txt')
env3 = make_env('grasp-v0')
env3 = GymEnv(env3.env.set_number_of_objects(3))
triple_best_policy_path = train(args, double_to_triple_demo_path, env3, 3)
print("Env Number: ", env3.env.N)

triple_policy_demo_path = demo.make_policy_demo(N = 3, best_policy_path = triple_best_policy_path)

triple_to_penta_demo_path = demo.make_progression_demo(N = 5, policy_demo_path = triple_policy_demo_path)

args = argparse.Namespace(output = './progression_results/dapg_5', config = './config/dapg.txt')
env5 = make_env('grasp-v0')
env5 = GymEnv(env5.env.set_number_of_objects(5))
penta_best_policy_path = train(args, triple_to_penta_demo_path, env5, 5)
print("Env Number: ", env5.env.N)

penta_policy_demo_path = demo.make_policy_demo(N = 5, best_policy_path = penta_best_policy_path)

penta_to_hexa_demo_path = demo.make_progression_demo(N = 6, policy_demo_path = penta_policy_demo_path)

args = argparse.Namespace(output = './progression_results/dapg_6', config = './config/dapg.txt')
env6 = make_env('grasp-v0')
env6 = GymEnv(env6.env.set_number_of_objects(6))
hexa_best_policy_path = train(args, penta_to_hexa_demo_path, env6, 6)
print("Env Number: ", env6.env.N)
print("Done!")
