from collections import deque, Counter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from env import make_env
from mjrl.utils.gym_env import GymEnv
import pickle
import numpy as np

class Demo:
    def __init__(self, env = 'grasp-v0', origin_path = './demonstrations/relocate-v0_demos.pickle'):
        '''
            origin demo -> single origin demo -> agent -> single demo(best policy)
                copy: actions, init state dict(object pos, waypoint pos, goal pos)
                paste: actions, init state dict -> set state rewind - actions
                create: init state dict(get env state), actions(policy get actions)

            policy demo
                policy get action -> action

                env1 = env for policy
                from env1 get initial pos of object, waypoint, goal -> init state dict of env1
                from env1 get actions -> actions of env1

                env2 = env for demo
                set state env2 using init state dict of env1
                step actions of env2



            single demo -> agent -> double origin demo -> double demo(best policy)
                copy: actions, init state dict(object pos, waypoint pos, goal pos)
        
        
        '''
        if type(env) == str:
            self.env = make_env(env)
        else:
            self.env = env
        self.policy_env = None
        self.num_objects = self.env.env.env.N
        self.origin_demos = pickle.load(open(origin_path, 'rb'))
        self.reset()
    
    def make_origin_demo(self):
        '''
            make demo from origin 
        '''
        print("========================================")
        print("Making Original Demo...")
        print("========================================")
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
            self.total_demos[demo_idx]['observations'] = self.env.env.get_obs()
            for action_idx, action in enumerate(actions):
                ns, r, done, info = self.env.step(action)
                if action_idx != len(actions) - 1:
                    self.total_demos[demo_idx]['observations'] = np.vstack((self.total_demos[demo_idx]['observations'], self.env.get_obs()))
                    self.total_demos[demo_idx]['rewards'] = np.concatenate([self.total_demos[demo_idx]['rewards'], np.array([r])])
                else:
                    pass

        with open(f'./demonstrations/{self.num_objects}_origin_demo.pickle', 'wb') as handle:
            pickle.dump(self.total_demos, handle, protocol=pickle.HIGHEST_PROTOCOL)
        demo_path = f'./demonstrations/{self.num_objects}_origin_demo.pickle'
        print("========================================")
        print("Original Demo making complete!")
        print("========================================")
        return demo_path

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
    
    def make_policy_demo(self, N, best_policy_path = None):
        '''
            make demo from best policy
            policy: MLP
        '''
        print("========================================")
        print("Making Policy Demo...")
        print("========================================")
        self.reset()
        best_policy = None
        if best_policy_path == None:
            raise NotImplementedError
        else:
            best_policy = pickle.load(open(best_policy_path, 'rb'))
        print("Policy model obs dim: ", best_policy.n)
        self.policy_env = make_env('grasp-v0')
        self.policy_env = self.policy_env.env.set_number_of_objects(N)
        # self.policy_env = GymEnv(self.policy_env.env.set_number_of_objects(N))
        # self.policy_env = make_env('grasp-v0').env.set_number_of_objects(N)
        self.policy_num_objects = self.policy_env.N
        max_timestep = 400
        for demo_idx in range(len(self.origin_demos)):
            init_state = self.policy_env.reset()
            initial_state_dict = self.policy_env.get_env_state()
            self.total_demos[demo_idx]['init_state_dict']['waypoint_pos'] = initial_state_dict['waypoint_pos']
            self.total_demos[demo_idx]['init_state_dict']['goal_pos'] = initial_state_dict['goal_pos']

            obj_pos = initial_state_dict['obj_pos']
            # self.env.env.env.N - N -> N + 1
            # if self.policy_num_objects == 1:
            #     self.total_demos[demo_idx]['init_state_dict']['obj_pos'] = obj_pos
            if self.policy_num_objects == 1:
                self.total_demos[demo_idx]['init_state_dict']['obj_pos']= obj_pos
                self.total_demos[demo_idx]['init_state_dict']['obj1_pos'] = np.array([obj_pos[0], obj_pos[1] + 0.02, obj_pos[2]])
            elif self.policy_num_objects == 2:
                self.total_demos[demo_idx]['init_state_dict']['obj_pos']= obj_pos
                self.total_demos[demo_idx]['init_state_dict']['obj1_pos'] = np.array([obj_pos[0], obj_pos[1] + 0.02, obj_pos[2]])
                self.total_demos[demo_idx]['init_state_dict']['obj3_pos'] = np.array([obj_pos[0], obj_pos[1] + 0.01, obj_pos[2] + 0.035])
            elif self.policy_num_objects == 3:
                self.total_demos[demo_idx]['init_state_dict']['obj_pos']= obj_pos
                self.total_demos[demo_idx]['init_state_dict']['obj1_pos'] = np.array([obj_pos[0], obj_pos[1] + 0.02, obj_pos[2]])
                self.total_demos[demo_idx]['init_state_dict']['obj3_pos'] = np.array([obj_pos[0], obj_pos[1] + 0.01, obj_pos[2] + 0.035])
                self.total_demos[demo_idx]['init_state_dict']['obj2_pos'] = np.array([obj_pos[0], obj_pos[1] - 0.02, obj_pos[2]])
                self.total_demos[demo_idx]['init_state_dict']['obj4_pos'] = np.array([obj_pos[0], obj_pos[1] - 0.01, obj_pos[2] + 0.035])
            elif self.policy_num_objects == 5:
                self.total_demos[demo_idx]['init_state_dict']['obj_pos']= obj_pos
                self.total_demos[demo_idx]['init_state_dict']['obj1_pos'] = np.array([obj_pos[0], obj_pos[1] + 0.02, obj_pos[2]])
                self.total_demos[demo_idx]['init_state_dict']['obj3_pos'] = np.array([obj_pos[0], obj_pos[1] + 0.01, obj_pos[2] + 0.035])
                self.total_demos[demo_idx]['init_state_dict']['obj2_pos'] = np.array([obj_pos[0], obj_pos[1] - 0.02, obj_pos[2]])
                self.total_demos[demo_idx]['init_state_dict']['obj4_pos'] = np.array([obj_pos[0], obj_pos[1] - 0.01, obj_pos[2] + 0.035])
                self.total_demos[demo_idx]['init_state_dict']['obj5_pos'] = np.array([obj_pos[0], obj_pos[1] , obj_pos[2] + 0.07])
            else:
                raise NotImplementedError

            done = False
            timestep = 0
            action = best_policy.get_action(init_state)[1]['evaluation']
            self.total_demos[demo_idx]['actions'] = action
            while timestep < max_timestep and done is False:
                state, reward, done, _ = self.policy_env.step(action)
                timestep += 1
                # self.total_demos[demo_idx]['observations'] = np.vstack((self.total_demos[demo_idx]['observations'], self.env.get_obs()))
                # self.total_demos[demo_idx]['rewards'] = np.concatenate([self.total_demos[demo_idx]['rewards'], np.array([reward])])
                init_state = state
                action = best_policy.get_action(init_state)[1]['evaluation']
                self.total_demos[demo_idx]['actions'] = np.vstack((self.total_demos[demo_idx]['actions'], action))

        name = None
        if self.policy_num_objects == 1:
            name = 'Single'
        elif self.policy_num_objects == 2:
            name = 'Double'
        elif self.policy_num_objects == 3:
            name = 'Triple'
        elif self.policy_num_objects == 5:
            name = 'Penta'
        else:
            raise NotImplementedError

        with open(f'./demonstrations/{name}_policy_demo.pickle', 'wb') as handle:
            pickle.dump(self.total_demos, handle, protocol=pickle.HIGHEST_PROTOCOL)
        demo_path = f'./demonstrations/{name}_policy_demo.pickle'
        print("========================================")
        print("Policy Demo making complete!")
        print("========================================")
        return demo_path
    
    def make_progression_demo(self, N, policy_demo_path = None):
        print("========================================")
        print("Making Progression Demo...")
        print("========================================")
        self.reset()
        self.env = make_env('grasp-v0')
        self.env = GymEnv(self.env.env.set_number_of_objects(N))
        self.num_objects = self.env.env.N
        policy_demos = pickle.load(open(policy_demo_path, 'rb'))
        for demo_idx in range(len(policy_demos)):
            tmp = self.env.reset()
            state_dict = self.env.env.get_env_state()
            self.total_demos[demo_idx]['init_state_dict']['waypoint_pos'] = policy_demos[demo_idx]['init_state_dict']['waypoint_pos']
            state_dict['waypoint_pos'] = policy_demos[demo_idx]['init_state_dict']['waypoint_pos']
            self.total_demos[demo_idx]['init_state_dict']['goal_pos'] = policy_demos[demo_idx]['init_state_dict']['goal_pos']
            state_dict['goal_pos'] = policy_demos[demo_idx]['init_state_dict']['goal_pos']
            self.total_demos[demo_idx]['init_state_dict']['obj_pos'] = policy_demos[demo_idx]['init_state_dict']['obj_pos']
            state_dict['obj_pos'] = policy_demos[demo_idx]['init_state_dict']['obj_pos']

            if self.num_objects == 2:
                self.total_demos[demo_idx]['init_state_dict']['obj1_pos'] = policy_demos[demo_idx]['init_state_dict']['obj1_pos']
                state_dict['obj1_pos'] = policy_demos[demo_idx]['init_state_dict']['obj1_pos']
            elif self.num_objects == 3:
                self.total_demos[demo_idx]['init_state_dict']['obj1_pos'] = policy_demos[demo_idx]['init_state_dict']['obj1_pos']
                self.total_demos[demo_idx]['init_state_dict']['obj3_pos'] = policy_demos[demo_idx]['init_state_dict']['obj3_pos']
                state_dict['obj1_pos'] = policy_demos[demo_idx]['init_state_dict']['obj1_pos']
                state_dict['obj3_pos'] = policy_demos[demo_idx]['init_state_dict']['obj3_pos']
            elif self.num_objects == 5:
                self.total_demos[demo_idx]['init_state_dict']['obj1_pos'] = policy_demos[demo_idx]['init_state_dict']['obj1_pos']
                self.total_demos[demo_idx]['init_state_dict']['obj2_pos'] = policy_demos[demo_idx]['init_state_dict']['obj2_pos']
                self.total_demos[demo_idx]['init_state_dict']['obj3_pos'] = policy_demos[demo_idx]['init_state_dict']['obj3_pos']
                self.total_demos[demo_idx]['init_state_dict']['obj4_pos'] = policy_demos[demo_idx]['init_state_dict']['obj4_pos']
                state_dict['obj1_pos'] = policy_demos[demo_idx]['init_state_dict']['obj1_pos']
                state_dict['obj2_pos'] = policy_demos[demo_idx]['init_state_dict']['obj2_pos']
                state_dict['obj3_pos'] = policy_demos[demo_idx]['init_state_dict']['obj3_pos']
                state_dict['obj4_pos'] = policy_demos[demo_idx]['init_state_dict']['obj4_pos']
            elif self.num_objects == 6:
                self.total_demos[demo_idx]['init_state_dict']['obj1_pos'] = policy_demos[demo_idx]['init_state_dict']['obj1_pos']
                self.total_demos[demo_idx]['init_state_dict']['obj2_pos'] = policy_demos[demo_idx]['init_state_dict']['obj2_pos']
                self.total_demos[demo_idx]['init_state_dict']['obj3_pos'] = policy_demos[demo_idx]['init_state_dict']['obj3_pos']
                self.total_demos[demo_idx]['init_state_dict']['obj4_pos'] = policy_demos[demo_idx]['init_state_dict']['obj4_pos']
                self.total_demos[demo_idx]['init_state_dict']['obj5_pos'] = policy_demos[demo_idx]['init_state_dict']['obj5_pos']
                state_dict['obj1_pos'] = policy_demos[demo_idx]['init_state_dict']['obj1_pos']
                state_dict['obj2_pos'] = policy_demos[demo_idx]['init_state_dict']['obj2_pos']
                state_dict['obj3_pos'] = policy_demos[demo_idx]['init_state_dict']['obj3_pos']
                state_dict['obj4_pos'] = policy_demos[demo_idx]['init_state_dict']['obj4_pos']
                state_dict['obj5_pos'] = policy_demos[demo_idx]['init_state_dict']['obj5_pos']
            else:
                raise NotImplementedError

            self.env.env.set_env_state(state_dict)
            self.total_demos[demo_idx]['actions'] = policy_demos[demo_idx]['actions']
            actions = self.total_demos[demo_idx]['actions']
            self.total_demos[demo_idx]['observations'] = self.env.env.get_obs() # fix bt mjrl/gym_env/get_obs: self.env.env.get_obs() -> self.env.get_obs()
            for action_idx, action in enumerate(actions):
                ns, r, d, _ = self.env.step(action)
                if action_idx != len(actions) - 1:
                    self.total_demos[demo_idx]['observations'] = np.vstack((self.total_demos[demo_idx]['observations'], self.env.get_obs()))
                    self.total_demos[demo_idx]['rewards'] = np.concatenate([self.total_demos[demo_idx]['rewards'], np.array([r])])
                else:
                    pass
            
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
            raise NotImplementedError
        with open(f'./demonstrations/{name}_progression_demo.pickle', 'wb') as handle:
            pickle.dump(self.total_demos, handle, protocol=pickle.HIGHEST_PROTOCOL)
        demo_path = f'./demonstrations/{name}_progression_demo.pickle'
        print("========================================")
        print("Progression Demo making complete!")
        print("========================================")
        return demo_path