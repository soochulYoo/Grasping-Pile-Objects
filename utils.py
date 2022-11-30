from collections import deque, Counter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from env import make_env
import pickle
import numpy as np

def loss_func():
    ...

def optim_func():
    ...

def update_params():
    ...

class Demo:
    def __init__(self, env_name = 'grasp-v0', origin_path = './demonstrations/relocate-v0_demos.pickle'):
        '''
            origin demo -> single origin demo -> agent -> single demo(best policy)
                copy: actions, init state dict(object pos, waypoint pos, goal pos)
                paste: actions, init state dict -> set state rewind - actions
                create: init state dict(get env state), actions(policy get actions)

            single demo -> agent -> double origin demo -> double demo(best policy)
                copy: actions, init state dict(object pos, waypoint pos, goal pos)
        
        
        '''
        self.env = make_env(env_name)
        self.num_objects = self.env.env.env.N
        self.origin_demos = pickle.load(open(origin_path, 'rb'))
        self.reset()
    
    def make_origin_demo(self):
        '''
            make demo from origin 
        '''
        action_dict = {}
        for demo_idx, demo in enumerate(self.origin_demos):
            action_que = deque([])
            for action in demo['actions']:
                action_que.appendleft(action)
            action_dict[demo_idx] = action_que

        for demo_idx, demo in enumerate(self.origin_demos):
            tmp = self.env.reset()
            state_dict = self.env.get_env_state() # copy env state dict

            self.total_demos[demo_idx]['init_state_dict']['waypoint_pos'] = demo['init_state_dict']['target_pos']
            self.total_demos[demo_idx]['actions'] = demo['actions']
            state_dict['waypoint_pos'] = self.total_demos[demo_idx]['init_state_dict']['waypoint_pos']

            # rewind actions
            if demo_idx == len(self.origin_demos) - 1:
                self.total_demos[demo_idx]['init_state_dict']['goal_pos'] = self.origin_demos[0]['init_state_dict']['obj_pos']
                self.total_demos[demo_idx]['actions'] = np.concatenate((self.total_demos[demo_idx]['actions'], action_dict[0]))
            else:
                self.total_demos[demo_idx]['init_state_dict']['goal_pos']= self.origin_demos[demo_idx + 1]['init_state_dict']['obj_pos']
                self.total_demos[demo_idx]['actions'] = np.concatenate((self.total_demos[demo_idx]['actions'], action_dict[demo_idx + 1]))
            
            state_dict['goal_pos'] = self.total_demos[demo_idx]['init_state_dict']['goal_pos']
            obj_pos = demo['init_state_dict']['obj_pos']
            if self.num_objects == 1:
                self.total_demos[demo_idx]['init_state_dict']['obj_pos'] = obj_pos
                state_dict['obj_pos'] = obj_pos

            elif self.num_objects == 2:
                self.total_demos[demo_idx]['init_state_dict']['obj_pos']= obj_pos
                self.total_demos[demo_idx]['init_state_dict']['obj1_pos'] = np.array([obj_pos[0], obj_pos[1] + 0.02, obj_pos[2]])
                state_dict['obj_pos'] = obj_pos
                state_dict['obj1_pos'] = self.total_demos[demo_idx]['init_state_dict']['obj1_pos']

            elif self.num_objects == 3:
                self.total_demos[demo_idx]['init_state_dict']['obj_pos']= obj_pos
                self.total_demos[demo_idx]['init_state_dict']['obj1_pos'] = np.array([obj_pos[0], obj_pos[1] + 0.02, obj_pos[2]])
                self.total_demos[demo_idx]['init_state_dict']['obj3_pos'] = np.array([obj_pos[0], obj_pos[1] + 0.01, obj_pos[2] + 0.035])
                state_dict['obj_pos'] = obj_pos
                state_dict['obj1_pos'] = self.total_demos[demo_idx]['init_state_dict']['obj1_pos']
                state_dict['obj3_pos'] = self.total_demos[demo_idx]['init_state_dict']['obj3_pos']

            elif self.num_objects == 5:
                self.total_demos[demo_idx]['init_state_dict']['obj_pos']= obj_pos
                self.total_demos[demo_idx]['init_state_dict']['obj1_pos'] = np.array([obj_pos[0], obj_pos[1] + 0.02, obj_pos[2]])
                self.total_demos[demo_idx]['init_state_dict']['obj3_pos'] = np.array([obj_pos[0], obj_pos[1] + 0.01, obj_pos[2] + 0.035])
                self.total_demos[demo_idx]['init_state_dict']['obj2_pos'] = np.array([obj_pos[0], obj_pos[1] - 0.02, obj_pos[2]])
                self.total_demos[demo_idx]['init_state_dict']['obj4_pos'] = np.array([obj_pos[0], obj_pos[1] - 0.01, obj_pos[2] + 0.035])
                state_dict['obj_pos'] = obj_pos
                state_dict['obj1_pos'] = self.total_demos[demo_idx]['init_state_dict']['obj1_pos']
                state_dict['obj3_pos'] = self.total_demos[demo_idx]['init_state_dict']['obj3_pos']
                state_dict['obj2_pos'] = self.total_demos[demo_idx]['init_state_dict']['obj2_pos']
                state_dict['obj4_pos'] = self.total_demos[demo_idx]['init_state_dict']['obj4_pos']
            elif self.num_objects == 6:
                self.total_demos[demo_idx]['init_state_dict']['obj_pos']= obj_pos
                self.total_demos[demo_idx]['init_state_dict']['obj1_pos'] = np.array([obj_pos[0], obj_pos[1] + 0.02, obj_pos[2]])
                self.total_demos[demo_idx]['init_state_dict']['obj3_pos'] = np.array([obj_pos[0], obj_pos[1] + 0.01, obj_pos[2] + 0.035])
                self.total_demos[demo_idx]['init_state_dict']['obj2_pos'] = np.array([obj_pos[0], obj_pos[1] - 0.02, obj_pos[2]])
                self.total_demos[demo_idx]['init_state_dict']['obj4_pos'] = np.array([obj_pos[0], obj_pos[1] - 0.01, obj_pos[2] + 0.035])
                self.total_demos[demo_idx]['init_state_dict']['obj5_pos'] = np.array([obj_pos[0], obj_pos[1] , obj_pos[2] + 0.07])
                state_dict['obj_pos'] = obj_pos
                state_dict['obj1_pos'] = self.total_demos[demo_idx]['init_state_dict']['obj1_pos']
                state_dict['obj3_pos'] = self.total_demos[demo_idx]['init_state_dict']['obj3_pos']
                state_dict['obj2_pos'] = self.total_demos[demo_idx]['init_state_dict']['obj2_pos']
                state_dict['obj4_pos'] = self.total_demos[demo_idx]['init_state_dict']['obj4_pos']
                state_dict['obj5_pos'] = self.total_demos[demo_idx]['init_state_dict']['obj5_pos']

            else:
                NotImplementedError()
            
            self.env.set_env_state(state_dict)
            actions = self.total_demos[demo_idx]['actions']
            self.total_demos[demo_idx]['observations'] = self.env.get_obs()
            for action_idx, action in enumerate(actions):
                ns, r, done, info = self.env.step(action)
                if action_idx != len(actions) - 1:
                    self.total_demos[demo_idx]['observations'] = np.vstack((self.total_demos[demo_idx]['observations'], self.env.get_obs()))
                    self.total_demos[demo_idx]['rewards'] = np.concatenate([self.total_demos[demo_idx]['rewards'], np.array([r])])
                else:
                    pass

        with open(f'./demonstrations/{self.num_objects}_origin_demo.pickle', 'wb') as handle:
            pickle.dump(self.total_demos, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def reset(self):
        self.total_demos = []
        for demo_idx in range(len(self.origin_demos)):
            each_demo = {}
            each_demo['observations'] = np.array([])
            each_demo['init_state_dict'] = {}
            each_demo['init_state_dict']['waypoint_pos'] = np.array([])
            each_demo['init_state_dict']['goal_pos'] = np.array([])
            each_demo['actions'] = np.array([])
            each_demo['rewards'] = np.array([])

            if self.num_objects == 1:
                each_demo['init_state_dict']['obj_pos'] = np.array([])
            elif self.num_objects == 2:
                each_demo['init_state_dict']['obj_pos'] = np.array([])
                each_demo['init_state_dict']['obj1_pos'] = np.array([])
            elif self.num_objects == 3:
                each_demo['init_state_dict']['obj_pos'] = np.array([])
                each_demo['init_state_dict']['obj1_pos'] = np.array([])
                each_demo['init_state_dict']['obj3_pos'] = np.array([])
            elif self.num_objects == 5:
                each_demo['init_state_dict']['obj_pos'] = np.array([])
                each_demo['init_state_dict']['obj1_pos'] = np.array([])
                each_demo['init_state_dict']['obj2_pos'] = np.array([])
                each_demo['init_state_dict']['obj3_pos'] = np.array([])
                each_demo['init_state_dict']['obj4_pos'] = np.array([])
            elif self.num_objects == 6:
                each_demo['init_state_dict']['obj_pos'] = np.array([])
                each_demo['init_state_dict']['obj1_pos'] = np.array([])
                each_demo['init_state_dict']['obj2_pos'] = np.array([])
                each_demo['init_state_dict']['obj3_pos'] = np.array([])
                each_demo['init_state_dict']['obj4_pos'] = np.array([])
                each_demo['init_state_dict']['obj5_pos'] = np.array([])
            else:
                NotImplementedError
            self.total_demos.append(each_demo)
    
    def make_policy_demo(self, best_policy_path = None):
        '''
            make demo from best policy
            policy: MLP
        '''
        self.reset()
        best_policy = None
        if best_policy_path == None:
            NotImplementedError
        else:
            best_policy = pickle.load(open(best_policy_path, 'rb'))
        max_timestep = 400
        for demo_idx in range(len(self.origin_demos)):
            init_state = self.env.reset()
            initial_state_dict = self.env.get_env_state()
            initial_get_obs = self.env.get_obs()
            self.total_demos[demo_idx]['init_state_dict']['waypoint_pos'] = initial_state_dict['waypoint_pos']
            self.total_demos[demo_idx]['init_state_dict']['goal_pos'] = initial_state_dict['goal_pos']
            self.total_demos[demo_idx]['observations'] = initial_get_obs

            obj_pos = initial_state_dict['obj_pos']
            if self.num_objects == 1:
                self.total_demos[demo_idx]['init_state_dict']['obj_pos'] = obj_pos
            elif self.num_objects == 2:
                self.total_demos[demo_idx]['init_state_dict']['obj_pos']= obj_pos
                self.total_demos[demo_idx]['init_state_dict']['obj1_pos'] = np.array([obj_pos[0], obj_pos[1] + 0.02, obj_pos[2]])
            elif self.num_objects == 3:
                self.total_demos[demo_idx]['init_state_dict']['obj_pos']= obj_pos
                self.total_demos[demo_idx]['init_state_dict']['obj1_pos'] = np.array([obj_pos[0], obj_pos[1] + 0.02, obj_pos[2]])
                self.total_demos[demo_idx]['init_state_dict']['obj3_pos'] = np.array([obj_pos[0], obj_pos[1] + 0.01, obj_pos[2] + 0.035])
            elif self.num_objects == 5:
                self.total_demos[demo_idx]['init_state_dict']['obj_pos']= obj_pos
                self.total_demos[demo_idx]['init_state_dict']['obj1_pos'] = np.array([obj_pos[0], obj_pos[1] + 0.02, obj_pos[2]])
                self.total_demos[demo_idx]['init_state_dict']['obj3_pos'] = np.array([obj_pos[0], obj_pos[1] + 0.01, obj_pos[2] + 0.035])
                self.total_demos[demo_idx]['init_state_dict']['obj2_pos'] = np.array([obj_pos[0], obj_pos[1] - 0.02, obj_pos[2]])
                self.total_demos[demo_idx]['init_state_dict']['obj4_pos'] = np.array([obj_pos[0], obj_pos[1] - 0.01, obj_pos[2] + 0.035])
            elif self.num_objects == 6:
                self.total_demos[demo_idx]['init_state_dict']['obj_pos']= obj_pos
                self.total_demos[demo_idx]['init_state_dict']['obj1_pos'] = np.array([obj_pos[0], obj_pos[1] + 0.02, obj_pos[2]])
                self.total_demos[demo_idx]['init_state_dict']['obj3_pos'] = np.array([obj_pos[0], obj_pos[1] + 0.01, obj_pos[2] + 0.035])
                self.total_demos[demo_idx]['init_state_dict']['obj2_pos'] = np.array([obj_pos[0], obj_pos[1] - 0.02, obj_pos[2]])
                self.total_demos[demo_idx]['init_state_dict']['obj4_pos'] = np.array([obj_pos[0], obj_pos[1] - 0.01, obj_pos[2] + 0.035])
                self.total_demos[demo_idx]['init_state_dict']['obj5_pos'] = np.array([obj_pos[0], obj_pos[1] , obj_pos[2] + 0.07])

            done = False
            timestep = 0
            while timestep < max_timestep and done is False:
                action = best_policy.get_action(init_state)[1]['evaluation']
                self.total_demos[demo_idx]['actions'].append(action) 
                state, reward, done, _ = self.env.step(action)
                timestep += 1
                self.total_demos[demo_idx]['observations'] = np.vstack((self.total_demos[demo_idx]['observations'], self.env.get_obs()))
                self.total_demos[demo_idx]['rewards'] = np.concatenate([self.total_demos[demo_idx]['rewards'], np.array([reward])])

        name = None
        if self.num_objects == 2:
            name = 'Single_to_Double'
        elif self.num_objects == 3:
            name = 'Double_to_Triple'
        elif self.num_objects == 5:
            name = 'Triple_to_Penta'
        elif self.num_objects == 6:
            name = 'Penta_to_Hexa'
        else:
            NotImplementedError

        with open(f'./demonstrations/{name}_policy_demo.pickle', 'wb') as handle:
            pickle.dump(self.total_demos, handle, protocol=pickle.HIGHEST_PROTOCOL)

        