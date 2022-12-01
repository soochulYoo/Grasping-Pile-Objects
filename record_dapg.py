from env import make_env
from mjrl.utils.gym_env import GymEnv
import skvideo.io
import pickle
import numpy as np
import tqdm

task_list = [1, 2, 3, 5, 6]
for N in tqdm(task_list):
    env = make_env('grasp-v0')
    env = GymEnv(env.env.set_number_of_objects(N))
    policy = f'./progression_results/dapg_{N}/iterations/best_policy.pickle'
    pi = pickle.load(open(policy, 'rb'))
    episodes = 1
    horizon = 300
    frame_size = (640, 480)
    camera_name = 'vil_camera'
    for ep in range(episodes):
        o = env.reset()
        d = False
        t = 0
        arrs = []
        while t < horizon and d is False:
            a = pi.get_action(o)[1]['evaluation']
            o, r, d, _ = env.step(a)
            curr_frame = env.env.sim.render(width = frame_size[0], height = frame_size[1], mode = 'offscreen', camera_name = camera_name, device_id = 0)
            arrs.append(curr_frame[::-1, :,:])
            t += 1

        skvideo.io.vwrite(f'./video/ep_{ep}th_{N}_obj.mp4', np.asarray(arrs))