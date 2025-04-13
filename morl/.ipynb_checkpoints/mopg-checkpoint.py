import os, sys
base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.append(base_dir)
sys.path.append(os.path.join(base_dir, 'externals/baselines'))
sys.path.append(os.path.join(base_dir, 'externals/pytorch-a2c-ppo-acktr-gail'))

import numpy as np
from collections import deque
from copy import deepcopy
import time
import torch

import gym
import mo_gymnasium as mo_gym
# import a2c_ppo_acktr
from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.envs import make_vec_envs, make_env
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage

from sample import Sample
from environments.building.env_building import BuildingEnv_3d, BuildingEnv_9d
from environments.building.utils_building import ParameterGenerator

import traceback

'''
Evaluate a policy sample.
'''
def evaluation(args, sample):
    #eval_env = gym.make(args.env_name)
    if args.env_name == 'building_3d':
        params = ParameterGenerator(Building='OfficeLarge', Weather='Warm_Marine', Location='ElPaso')
        eval_env = BuildingEnv_3d(params)
    elif args.env_name == 'building_9d':
        params = ParameterGenerator(Building='OfficeLarge', Weather='Warm_Marine', Location='ElPaso')
        eval_env = BuildingEnv_9d(params)
    else:
        if args.cost_objective:
            eval_env = mo_gym.make(args.env_name, cost_objective=True, max_episode_steps=500)
        else:
            eval_env = mo_gym.make(args.env_name, max_episode_steps=500)
    objs = np.zeros(args.obj_num)
    ob_rms = sample.env_params['ob_rms']
    policy = sample.actor_critic
    with torch.no_grad():
        for eval_id in range(args.eval_num):
            eval_env.seed = args.seed + eval_id
            ob = eval_env.reset(seed = args.seed + eval_id)[0]
            done = False
            gamma = 1.0
            while not done:
                if args.ob_rms:
                    ob = np.clip((ob - ob_rms.mean) / np.sqrt(ob_rms.var + 1e-8), -10.0, 10.0)
                _, action, _, _ = policy.act(torch.Tensor(ob).unsqueeze(0), None, None, deterministic=True)
                action = action.cpu().numpy()
                action = np.squeeze(action)
                ob, reward, terminated, truncated, info = eval_env.step(action)
                done = terminated or truncated
                info['obj'] = reward
                objs += gamma * info['obj']
                gamma *= args.gamma
    eval_env.close()
    objs /= args.eval_num
    return objs

'''
define a MOPG Worker.
Input:
    args: the arguments include necessary ppo parameters.
    task_id: the task_id in all parallel executed tasks.
    device: torch device
    iteration: starting iteration number
    num_updates: number of rl iterations to run.
    start_time: starting time
    results_queue: multi-processing queue to pass the optimized policies back to main process.
    done_event: multi-processing event for process synchronization.
'''
def MOPG_worker(args, task_id, task, device, iteration, num_updates, start_time, results_queue, done_event):
    scalarization = task.scalarization
    env_params, actor_critic, agent = task.sample.env_params, task.sample.actor_critic, task.sample.agent
    
    weights_str = (args.obj_num * '_{:.3f}').format(*task.scalarization.weights)

    # make envs
    envs = make_vec_envs(env_name=args.env_name, seed=args.seed, num_processes=args.num_processes, \
                        gamma=args.gamma, log_dir=None, device=device, allow_early_resets=False, \
                        cost_objective = args.cost_objective, obj_rms=args.obj_rms, ob_rms = args.ob_rms)
    if env_params['ob_rms'] is not None:
        envs.venv.ob_rms = deepcopy(env_params['ob_rms'])
    if env_params['ret_rms'] is not None:
        envs.venv.ret_rms = deepcopy(env_params['ret_rms'])
    if env_params['obj_rms'] is not None:
        envs.venv.obj_rms = deepcopy(env_params['obj_rms'])

    # build rollouts data structure
    rollouts = RolloutStorage(num_steps = args.num_steps, num_processes = args.num_processes,
                              obs_shape = envs.observation_space.shape, action_space = envs.action_space,
                              recurrent_hidden_state_size = actor_critic.recurrent_hidden_state_size, obj_num=args.obj_num)
    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=10)
    episode_lens = deque(maxlen=10)
    episode_objs = deque(maxlen=10)   # for each cost component we care
    episode_obj = np.array([None] * args.num_processes)

    total_num_updates = num_updates

    offspring_batch = []

    start_iter, final_iter = iteration, total_num_updates
    
    for j in range(start_iter, final_iter):
        torch.manual_seed(j)
        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule( \
            agent.optimizer, j * args.lr_decay_ratio, \
            total_num_updates, args.lr)
        
        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])
            
            obs, _, done, infos = envs.step(action)
            obj_tensor = torch.zeros([args.num_processes, args.obj_num])

            for idx, info in enumerate(infos):
                obj_tensor[idx] = torch.from_numpy(info['obj'])
                episode_obj[idx] = info['obj_raw'] if episode_obj[idx] is None else episode_obj[idx] + info['obj_raw']
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])
                    episode_lens.append(info['episode']['l'])
                    if episode_obj[idx] is not None:
                        episode_objs.append(episode_obj[idx])
                        episode_obj[idx] = None

            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])
            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, obj_tensor, masks, bad_masks)

        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()

        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)

        obj_rms_var = envs.obj_rms.var if envs.obj_rms is not None else None

        value_loss, action_loss, dist_entropy = agent.update(rollouts, scalarization, obj_rms_var)

        rollouts.after_update()

        env_params = {}
        env_params['ob_rms'] = deepcopy(envs.ob_rms) if envs.ob_rms is not None else None
        env_params['ret_rms'] = deepcopy(envs.ret_rms) if envs.ret_rms is not None else None
        env_params['obj_rms'] = deepcopy(envs.obj_rms) if envs.obj_rms is not None else None

        # evaluate new sample
        sample = Sample(env_params, deepcopy(actor_critic), deepcopy(agent))
        objs = evaluation(args, sample)
        sample.objs = objs
        offspring_batch.append(sample)
            
        # put results back every update_iter iterations, to avoid the multi-processing crash
        if j == final_iter - 1:
            offspring_batch = np.array(offspring_batch)
            results = {}
            results['task_id'] = task_id
            results['offspring_batch'] = offspring_batch
            if j == final_iter - 1:
                results['done'] = True
            else:
                results['done'] = False
            results_queue.put(results)
            offspring_batch = []


    envs.close()   
    
    done_event.wait()

def Extension_worker(args, sample_id, sample, device, iteration, num_updates, start_time, results_queue, done_event):
    env_params, actor_critic, agent = sample.env_params, sample.actor_critic, sample.agent
    initial_obj = sample.objs
    # make envs
    envs = make_vec_envs(env_name=args.env_name, seed=args.seed, num_processes=args.num_processes, \
                        gamma=args.gamma, log_dir=None, device=device, allow_early_resets=False, \
                        cost_objective = args.cost_objective, obj_rms=args.obj_rms, ob_rms = args.ob_rms)
    if env_params['ob_rms'] is not None:
        envs.venv.ob_rms = deepcopy(env_params['ob_rms'])
    if env_params['ret_rms'] is not None:
        envs.venv.ret_rms = deepcopy(env_params['ret_rms'])
    if env_params['obj_rms'] is not None:
        envs.venv.obj_rms = deepcopy(env_params['obj_rms'])

    # build rollouts data structure
    rollouts = RolloutStorage(num_steps = args.num_steps, num_processes = args.num_processes,
                              obs_shape = envs.observation_space.shape, action_space = envs.action_space,
                              recurrent_hidden_state_size = actor_critic.recurrent_hidden_state_size, obj_num=args.obj_num)
    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=10)
    episode_lens = deque(maxlen=10)
    episode_objs = deque(maxlen=10)   # for each cost component we care
    episode_obj = np.array([None] * args.num_processes)

    start_iter, final_iter = iteration, num_updates
    
    offspring_batch = []
    for obj in range(args.obj_num):
        reward_list = []
        sample_new = deepcopy(sample)
        env_params, actor_critic, agent = sample_new.env_params, sample_new.actor_critic, sample_new.agent
        
        if env_params['ob_rms'] is not None:
            envs.venv.ob_rms = deepcopy(env_params['ob_rms'])
        if env_params['ret_rms'] is not None:
            envs.venv.ret_rms = deepcopy(env_params['ret_rms'])
        if env_params['obj_rms'] is not None:
            envs.venv.obj_rms = deepcopy(env_params['obj_rms'])

        for j in range(final_iter):
            torch.manual_seed(j)
            
            for step in range(args.num_steps):
                # Sample actions
                with torch.no_grad():
                    value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                        rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                        rollouts.masks[step])
                
                obs, _, done, infos = envs.step(action)
                obj_tensor = torch.zeros([args.num_processes, args.obj_num])

                for idx, info in enumerate(infos):
                    obj_tensor[idx] = torch.from_numpy(info['obj'])
                    episode_obj[idx] = info['obj_raw'] if episode_obj[idx] is None else episode_obj[idx] + info['obj_raw']
                    if 'episode' in info.keys():
                        episode_rewards.append(info['episode']['r'])
                        episode_lens.append(info['episode']['l'])
                        if episode_obj[idx] is not None:
                            episode_objs.append(episode_obj[idx])
                            episode_obj[idx] = None

                # If done then clean the history of observations.
                masks = torch.FloatTensor(
                    [[0.0] if done_ else [1.0] for done_ in done])
                bad_masks = torch.FloatTensor(
                    [[0.0] if 'bad_transition' in info.keys() else [1.0]
                    for info in infos])
                rollouts.insert(obs, recurrent_hidden_states, action,
                                action_log_prob, value, obj_tensor, masks, bad_masks)

            with torch.no_grad():
                next_value = actor_critic.get_value(
                    rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                    rollouts.masks[-1]).detach()

            rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                    args.gae_lambda, args.use_proper_time_limits)

            obj_rms_var = envs.obj_rms.var if envs.obj_rms is not None else None

            value_loss, action_loss, dist_entropy = agent.ipo_update(rollouts, obj, args.obj_num, obj_rms_var)

            rollouts.after_update()

            env_params = {}
            env_params['ob_rms'] = deepcopy(envs.ob_rms) if envs.ob_rms is not None else None
            env_params['ret_rms'] = deepcopy(envs.ret_rms) if envs.ret_rms is not None else None
            env_params['obj_rms'] = deepcopy(envs.obj_rms) if envs.obj_rms is not None else None

            # evaluate new sample
            sample = Sample(env_params, deepcopy(actor_critic), deepcopy(agent))
            objs = evaluation(args, sample)
            sample.objs = objs
            offspring_batch.append(sample)

            if True:
                offspring_batch = np.array(offspring_batch)
                results = {}
                results['task_id'] = sample_id
                results['offspring_batch'] = offspring_batch
                if j == final_iter - 1 and obj == args.obj_num - 1:
                    results['done'] = True
                else:
                    results['done'] = False
                results_queue.put(results)
                offspring_batch = []

    envs.close()   
    
    done_event.wait()