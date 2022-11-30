from logging import PlaceHolder
from math import radians
from operator import length_hint
import numpy as np
from gym import utils
from mjrl.envs import mujoco_env
from mujoco_py import MjViewer
import os

ADD_BONUS_REWARDS = True

class GraspEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    metadata = {
        "render.modes": [
            "human",
            "rgb_array"
        ],
        "render_fps": 20,
    }
    
    def __init__(self):
        self.N = 3 # number of objects
        self.S_grasp_sid = 0 # grasp site id
        self.hand_bid = 0
        self.obj_bid = 0 # object body id
        self.bowl_obj_bid = 0 # bowl bid
        self.goal_obj_sid = 0 # goal sid
        self.waypoint_sid = 0 # waypoint
        
        self.waypoint_check = {}
        self.waypoint_check['Object'] = 0
        self.count_check = {}
        self.count_check['Object'] = 0
        curr_dir = os.path.dirname(os.path.abspath(__file__))

        if self.N == 1:
            mujoco_env.MujocoEnv.__init__(self, curr_dir+'/assets/DAPG_box.xml', 5)
            

        elif self.N == 3:
            
            self.obj1_bid = 0
            self.obj3_bid = 0

            self.obj1_sid = 0
            self.obj3_sid = 0

            self.waypoint_check['Object1'] = 0
            self.waypoint_check['Object3'] = 0
            self.count_check['Object1'] = 0
            self.count_check['Object3'] = 0

        elif self.N == 6:
            self.obj1_bid = 0
            self.obj2_bid = 0
            self.obj3_bid = 0
            self.obj4_bid = 0
            self.obj5_bid = 0

            self.obj1_sid = 0
            self.obj2_sid = 0
            self.obj3_sid = 0
            self.obj4_sid = 0
            self.obj5_sid = 0

            self.waypoint_check['Object1'] = 0
            self.waypoint_check['Object2'] = 0
            self.waypoint_check['Object3'] = 0
            self.waypoint_check['Object4'] = 0
            self.waypoint_check['Object5'] = 0 
            self.count_check['Object1'] = 0
            self.count_check['Object2'] = 0
            self.count_check['Object3'] = 0
            self.count_check['Object4'] = 0
            self.count_check['Object5'] = 0   
        else:
            NotImplementedError
        
        if self.N == 3:
            mujoco_env.MujocoEnv.__init__(self, curr_dir+'/assets/DAPG_multi_box.xml', 5)
            self.obj1_bid = self.sim.model.body_name2id('Object1')
            self.obj3_bid = self.sim.model.body_name2id('Object3')
            self.obj1_sid = self.sim.model.site_name2id('Object1')
            self.obj3_sid = self.sim.model.site_name2id('Object3')
        elif self.N == 6:
            mujoco_env.MujocoEnv.__init__(self, curr_dir+'/assets/DAPG_hexa_box.xml', 5)

            self.obj1_bid = self.sim.model.body_name2id('Object1')
            self.obj2_bid = self.sim.model.body_name2id('Object2')
            self.obj3_bid = self.sim.model.body_name2id('Object3')
            self.obj4_bid = self.sim.model.body_name2id('Object4')
            self.obj5_bid = self.sim.model.body_name2id('Object5')

            self.obj1_sid = self.sim.model.site_name2id('Object1')
            self.obj2_sid = self.sim.model.site_name2id('Object2')
            self.obj3_sid = self.sim.model.site_name2id('Object3')
            self.obj4_sid = self.sim.model.site_name2id('Object4')
            self.obj5_sid = self.sim.model.site_name2id('Object5')
        else:
            NotImplementedError

        self.sim.model.actuator_gainprm[self.sim.model.actuator_name2id('A_WRJ1'):self.sim.model.actuator_name2id('A_WRJ0')+1,:3] = np.array([10, 0, 0])
        self.sim.model.actuator_gainprm[self.sim.model.actuator_name2id('A_FFJ3'):self.sim.model.actuator_name2id('A_THJ0')+1,:3] = np.array([1, 0, 0])
        self.sim.model.actuator_biasprm[self.sim.model.actuator_name2id('A_WRJ1'):self.sim.model.actuator_name2id('A_WRJ0')+1,:3] = np.array([0, -10, 0])
        self.sim.model.actuator_biasprm[self.sim.model.actuator_name2id('A_FFJ3'):self.sim.model.actuator_name2id('A_THJ0')+1,:3] = np.array([0, -1, 0])

        self.hand_bid = self.sim.model.body_name2id('wrist')
        self.S_grasp_sid = self.sim.model.site_name2id('S_grasp')
        self.obj_bid = self.sim.model.body_name2id('Object')
        # self.bowl_obj_bid = self.sim.model.body_name2id('bowl')
        self.goal_obj_sid = self.sim.model.site_name2id('goal')
        self.waypoint_sid = self.sim.model.site_name2id('waypoint')

        utils.EzPickle.__init__(self)
        self.act_mid = np.mean(self.model.actuator_ctrlrange, axis = 1) # -100 ~ 100 -> 0
        self.act_rng = 0.5 * (self.model.actuator_ctrlrange[:,1] - self.model.actuator_ctrlrange[:,0])
        self.action_space.high = np.ones_like(self.model.actuator_ctrlrange[:,1])
        self.action_space.low = - 1.0 * np.ones_like(self.model.actuator_ctrlrange[:,0])

    def step(self, a):
        '''
            xpos : in global Cartesian space position
            reward: velocity, appraoch, grasp, move, place
            count for performance
        '''
        a = np.clip(a, -1.0, 1.0)
        try:
            a = self.act_mid + a * self.act_rng
        except:
            a = a
        self.do_simulation(a, self.frame_skip)
        ob = self.get_obs()
        palm_pos = self.data.site_xpos[self.S_grasp_sid].ravel()
        waypoint_pos = self.data.site_xpos[self.waypoint_sid].ravel()
        goal_pos = self.data.site_xpos[self.goal_obj_sid].ravel()

        if self.N == 1:
            object_name_list = ['Object']
        elif self.N == 3:
            object_name_list = ['Object', 'Object1', 'Object3']
        elif self.N == 6:
            object_name_list = ['Object', 'Object1', 'Object2', 'Object3', 'Object4', 'Object5']
        else:
            NotImplementedError

        goal_x, goal_y, goal_z = goal_pos[0], goal_pos[1], goal_pos[2]

        l = 0.0675
        h = 0.055
        offset = 0.04
        count = 0
        reward = 0

        # reward -= 1e-2 * np.linalg.norm(self.data.qvel.ravel())
        for obj_name in object_name_list:
            object_pos = self.data.body_xpos[self.sim.model.body_name2id(f'{obj_name}')].ravel()
            # object_site_pos = self.data.site_xpos[self.sim.model.site_name2id(f'{obj_name}')].ravel()
            object_x, object_y, object_z = object_pos[0], object_pos[1], object_pos[2]
            reward += - 0.1 * np.linalg.norm(palm_pos - object_pos)

            if not self.waypoint_check[obj_name]:
                reward += - 0.1 * np.linalg.norm(object_pos - waypoint_pos)
                reward += - 0.1 * np.linalg.norm(palm_pos - waypoint_pos)
                if object_z >= offset:
                    reward += 1.0
                if np.linalg.norm(object_pos - waypoint_pos) < 0.1:
                    self.waypoint_check[obj_name] = 1

            if self.waypoint_check[obj_name] == 1:
                distance_x = np.linalg.norm(object_x - goal_x)
                distance_y = np.linalg.norm(object_y - goal_y)
                distance_z = np.linalg.norm(object_z - goal_z)
                reward += - 0.5 * np.linalg.norm(object_pos - goal_pos)
                reward += - 0.5 * np.linalg.norm(palm_pos - goal_pos)
                if distance_x <= l and distance_y <= l and distance_z <= h:
                    # count += 1
                    self.count_check[obj_name] = 1
                

        count = sum(self.count_check.values())
        reward += 100 * count
        # goal_achieved = True if np.linalg.norm(obj_pos - goal_pos) < 0.1 else False
        goal_achieved = True if count >= 1 else False
        if goal_achieved:
            done = True
        else:
            done = False

        return ob, reward, False, dict(goal_achieved = goal_achieved, count = count)

    def get_obs(self):
        qp = self.data.qpos.ravel()
        palm_pos = self.data.site_xpos[self.S_grasp_sid].ravel()
        waypoint_pos = self.data.site_xpos[self.waypoint_sid].ravel()
        goal_pos = self.data.site_xpos[self.goal_obj_sid].ravel()
        obj_pos = self.data.body_xpos[self.obj_bid].ravel()
        if self.N == 1:
            return np.concatenate([qp[:-6], palm_pos-goal_pos, palm_pos - waypoint_pos, 
                                    palm_pos-obj_pos, 
                                    waypoint_pos - obj_pos, 
                                    obj_pos-goal_pos
                                    ])
        elif self.N == 3:
            obj1_pos = self.data.body_xpos[self.obj1_bid].ravel()
            obj3_pos = self.data.body_xpos[self.obj3_bid].ravel()
            return np.concatenate([qp[:-6], palm_pos-goal_pos, palm_pos - waypoint_pos,
                palm_pos - obj_pos, palm_pos - obj1_pos, palm_pos - obj3_pos,
                waypoint_pos - obj_pos, waypoint_pos - obj1_pos, waypoint_pos - obj3_pos,
                obj_pos - goal_pos, obj1_pos - goal_pos,  obj3_pos - goal_pos
                ])
        elif self.N == 6:
            obj1_pos = self.data.body_xpos[self.obj1_bid].ravel()
            obj2_pos = self.data.body_xpos[self.obj2_bid].ravel()
            obj3_pos = self.data.body_xpos[self.obj3_bid].ravel()
            obj4_pos = self.data.body_xpos[self.obj4_bid].ravel()
            obj5_pos = self.data.body_xpos[self.obj5_bid].ravel()
            return np.concatenate([qp[:-6], palm_pos-goal_pos, palm_pos - waypoint_pos,
                palm_pos - obj_pos, palm_pos - obj1_pos, palm_pos - obj2_pos, palm_pos - obj3_pos, palm_pos - obj4_pos, palm_pos - obj5_pos,
                waypoint_pos - obj_pos, waypoint_pos - obj1_pos, waypoint_pos - obj2_pos, waypoint_pos - obj3_pos, waypoint_pos - obj4_pos, waypoint_pos - obj5_pos,
                obj_pos - goal_pos, obj1_pos - goal_pos,  obj2_pos - goal_pos, obj3_pos - goal_pos, obj4_pos - goal_pos, obj5_pos - goal_pos
                ])        
    
    def reset(self):
        for key in self.waypoint_check.keys():
            self.waypoint_check[key] = 0
            self.count_check[key] = 0
        qp = self.init_qpos.copy()
        qv = self.init_qvel.copy()
        self.set_state(qp, qv)
        self.model.body_pos[self.obj_bid, 0] = self.np_random.uniform(low = -0.25, high = 0.25)
        self.model.body_pos[self.obj_bid, 1] = self.np_random.uniform(low = 0.00, high = 0.3)

        if self.N == 3:
            self.model.body_pos[self.obj1_bid, 0] = self.model.body_pos[self.obj_bid, 0]
            self.model.body_pos[self.obj1_bid, 1] = self.model.body_pos[self.obj_bid, 1] + 0.02

            self.model.body_pos[self.obj3_bid, 0] = self.model.body_pos[self.obj_bid, 0]
            self.model.body_pos[self.obj3_bid, 1] = self.model.body_pos[self.obj_bid, 1] + 0.01

        elif self.N == 6:
            self.model.body_pos[self.obj1_bid, 0] = self.model.body_pos[self.obj_bid, 0]
            self.model.body_pos[self.obj1_bid, 1] = self.model.body_pos[self.obj_bid, 1] + 0.02
    
            self.model.body_pos[self.obj2_bid, 0] = self.model.body_pos[self.obj_bid, 0]
            self.model.body_pos[self.obj2_bid, 1] = self.model.body_pos[self.obj_bid, 1] -0.02
    
            self.model.body_pos[self.obj3_bid, 0] = self.model.body_pos[self.obj_bid, 0]
            self.model.body_pos[self.obj3_bid, 1] = self.model.body_pos[self.obj_bid, 1] + 0.01
    
            self.model.body_pos[self.obj4_bid, 0] = self.model.body_pos[self.obj_bid, 0]
            self.model.body_pos[self.obj4_bid, 1] = self.model.body_pos[self.obj_bid, 1] -0.01
    
            self.model.body_pos[self.obj5_bid, 0] = self.model.body_pos[self.obj_bid, 0]
            self.model.body_pos[self.obj5_bid, 1] = self.model.body_pos[self.obj_bid, 1]

        # x = self.np_random.uniform(low = -0.35, high = 0.35)
        # y = self.np_random.uniform(low = -0.35, high = -0.2)
        # self.model.body_pos[self.bowl_obj_bid, 0] = x
        # self.model.body_pos[self.bowl_obj_bid, 1] = y
        self.model.site_pos[self.waypoint_sid, 0] = self.np_random.uniform(low = -0.35, high = 0.35)
        self.model.site_pos[self.waypoint_sid, 1] = self.np_random.uniform(low = -0.15, high = -0.1)
        self.model.site_pos[self.goal_obj_sid, 0] = self.np_random.uniform(low = -0.35, high = 0.35)
        self.model.site_pos[self.goal_obj_sid, 1] = self.np_random.uniform(low = -0.35, high = -0.22)
        # 오히려 바꾸면 이상해진다 그대로 body 를 따라간다
        # self.model.site_pos[self.goal_obj_sid,2] = self.np_random.uniform(low=0.15, high=0.35)
        self.sim.forward()
        return self.get_obs()

    def get_env_state(self):
        qp = self.data.qpos.ravel().copy()
        qv = self.data.qvel.ravel().copy()
        hand_qpos = qp[:30]
        obj_pos  = self.data.body_xpos[self.obj_bid].ravel()
        palm_pos = self.data.site_xpos[self.S_grasp_sid].ravel()
        # bowl_pos = self.data.body_xpos[self.bowl_obj_bid].ravel()
        goal_pos = self.data.site_xpos[self.goal_obj_sid].ravel()
        waypoint_pos = self.data.site_xpos[self.waypoint_sid].ravel()
        if self.N == 1:
            return dict(hand_qpos=hand_qpos, obj_pos=obj_pos,
                        goal_pos=goal_pos, waypoint_pos = waypoint_pos, 
                        palm_pos=palm_pos,qpos=qp, qvel=qv)
        elif self.N == 3:
            obj1_pos = self.data.body_xpos[self.obj1_bid].ravel()
            obj3_pos = self.data.body_xpos[self.obj3_bid].ravel()
            return dict(hand_qpos=hand_qpos, obj_pos=obj_pos, obj1_pos = obj1_pos, obj3_pos = obj3_pos, 
                        goal_pos=goal_pos, waypoint_pos = waypoint_pos, palm_pos=palm_pos,
                        qpos=qp, qvel=qv)
        elif self.N == 6:
            obj1_pos = self.data.body_xpos[self.obj1_bid].ravel()
            obj2_pos = self.data.body_xpos[self.obj2_bid].ravel()
            obj3_pos = self.data.body_xpos[self.obj3_bid].ravel()
            obj4_pos = self.data.body_xpos[self.obj4_bid].ravel()
            obj5_pos = self.data.body_xpos[self.obj5_bid].ravel()
            return dict(hand_qpos=hand_qpos, obj_pos=obj_pos, obj1_pos = obj1_pos, obj2_pos = obj2_pos, obj3_pos = obj3_pos, obj4_pos = obj4_pos, obj5_pos = obj5_pos,
                        goal_pos=goal_pos, waypoint_pos = waypoint_pos, palm_pos=palm_pos,
                        qpos=qp, qvel=qv)
        # bowl pos add!!!!
    
    def set_env_state(self, state_dict):
        # using state dictionary input, set envorionment
        qp = state_dict['qpos']
        qv = state_dict['qvel']
        obj_pos = state_dict['obj_pos']
        goal_pos = state_dict['goal_pos']
        # bowl_pos = state_dict['bowl_pos']
        waypoint_pos = state_dict['waypoint_pos']
        self.set_state(qp, qv)
        self.model.body_pos[self.obj_bid] = obj_pos
        self.model.site_pos[self.goal_obj_sid] = goal_pos
        # self.model.body_pos[self.bowl_obj_bid] = bowl_pos
        self.model.site_pos[self.waypoint_sid] = waypoint_pos

        if self.N == 3:
            obj1_pos = state_dict['obj1_pos']
            obj3_pos = state_dict['obj3_pos']

            self.model.body_pos[self.obj1_bid] = obj1_pos
            self.model.body_pos[self.obj3_bid] = obj3_pos

        elif self.N == 6:
            obj1_pos = state_dict['obj1_pos']
            obj2_pos = state_dict['obj2_pos']
            obj3_pos = state_dict['obj3_pos']
            obj4_pos = state_dict['obj4_pos']
            obj5_pos = state_dict['obj5_pos']

            self.model.body_pos[self.obj1_bid] = obj1_pos
            self.model.body_pos[self.obj2_bid] = obj2_pos
            self.model.body_pos[self.obj3_bid] = obj3_pos
            self.model.body_pos[self.obj4_bid] = obj4_pos
            self.model.body_pos[self.obj5_bid] = obj5_pos
        
        self.sim.forward()
    
    def mj_viewer_setup(self):
        self.viewer = MjViewer(self.sim)
        self.viewer.cam.azimuth = 0 # 90
        self.sim.forward()
        self.viewer.cam.distance = 1.5
        # camera setting
        self.viewer.cam.trackbodyid = self.hand_bid
        self.viewer.cam.distance = self.model.stat.extent * 0.015
        self.viewer.cam.lookat[0] += 0.5
        self.viewer.cam.lookat[1] += 0.5
        self.viewer.cam.lookat[2] += 0.5
        self.viewer.elevation = -90


    def evaluate_count(self, paths):
        num_count = 0
        num_paths = len(paths)
        for path in paths:
            if np.sum(path['env_infos']['count']) >= 1:
                num_count += 1
        count_percentage = num_count * 100.0 / num_paths
        return count_percentage

    def evaluate_success(self, paths):
        num_success = 0
        num_paths = len(paths)
        # success if object close to target for 25 steps
        for path in paths:
            if np.sum(path['env_infos']['goal_achieved']) > 25:
                num_success += 1
        success_percentage = 100.0 * num_success / num_paths
        return success_percentage