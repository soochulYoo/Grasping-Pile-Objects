from mjrl.utils.gym_env import GymEnv
from mjrl.policies.gaussian_mlp import MLP
from mjrl.baselines.quadratic_baseline import QuadraticBaseline
from mjrl.baselines.mlp_baseline import MLPBaseline
from mjrl.algos.npg_cg import NPG
from mjrl.algos.dapg import DAPG
from mjrl.algos.behavior_cloning import BC
from mjrl.utils.train_agent import train_agent
from mjrl.samplers.core import sample_paths
import os
import json
import mjrl.envs
import mj_envs
import time as timer
import pickle
import argparse
from env import make_env

def train(args, demo_path, env = None , N = 1):
    output_direction = args.output
    if not os.path.exists(output_direction):
        os.mkdir(output_direction)
    with open(args.config, 'r') as f:
        data = eval(f.read())

    assert 'algorithm' in data.keys()
    assert any([data['algorithm'] == a for a in ['NPG', 'BCRL', 'DAPG']])
    data['lam_0'] = 0.0 if 'lam_0' not in data.keys() else data['lam_0']
    data['lam_1'] = 0.0 if 'lam_1' not in data.keys() else data['lam_1']
    exp_file = output_direction + '/bcrl_config.json'
    with open(exp_file, 'w') as f:
        json.dump(data, f, indent=4)

    if env is None:
        env = make_env(data['env'])
    else:
        env = env
    print("Env Obs dim: ", env.spec.observation_dim)
    policy = MLP(env.spec, hidden_sizes=data['policy_size'], seed=data['seed'])
    print("Train Policy Obs dim: ", policy.n)
    baseline = MLPBaseline(env.spec, reg_coef=1e-3, batch_size=data['vf_batch_size'], epochs=data['vf_epochs'], learn_rate=data['vf_learn_rate'])

    if data['algorithm'] != 'NPG':
        print("========================================")
        print("Collecting expert demonstrations")
        print("========================================")
        # demo_paths = pickle.load(open(data['demo_file'], 'rb'))
        demo_paths = pickle.load(open(demo_path, 'rb'))
        bc_agent = BC(demo_paths, policy=policy, epochs=data['bc_epochs'], batch_size=data['bc_batch_size'],
                      lr=data['bc_learn_rate'], loss_type='MSE', set_transforms=False)
        in_shift, in_scale, out_shift, out_scale = bc_agent.compute_transformations()
        bc_agent.set_transformations(in_shift, in_scale, out_shift, out_scale)
        bc_agent.set_variance_with_data(out_scale)

        ts = timer.time()
        print("========================================")
        print("Running BC with expert demonstrations")
        print("========================================")
        bc_agent.train()
        print("========================================")
        print("BC training complete !!!")
        print("time taken = %f" % (timer.time() - ts))
        print("========================================")


    if data['algorithm'] != 'DAPG':
        demo_paths = None

    rl_agent = DAPG(env, policy, baseline, demo_paths, 
                    normalized_step_size=data['rl_step_size'],
                    lam_0 = data['lam_0'], lam_1 = data['lam_1'],
                    seed = data['seed'], save_logs=True)

    print("========================================")
    print("Starting reinforcement learning phase")
    print("========================================")

    ts = timer.time()
    train_agent(job_name=output_direction,
                agent=rl_agent,
                seed=data['seed'],
                niter=data['rl_num_iter'],
                gamma=data['rl_gamma'],
                gae_lambda=data['rl_gae'],
                num_cpu=data['num_cpu'],
                sample_mode='trajectories',
                num_traj=data['rl_num_traj'],
                save_freq=data['save_freq'],
                evaluation_rollouts=data['eval_rollouts'],
                n = N)
    print("time taken = %f" % (timer.time()-ts))
    best_policy_path = output_direction + '/iterations/best_policy.pickle'
    return best_policy_path