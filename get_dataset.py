import numpy as np


def dataset_setting1(dataset1, dataset2, split_x, exp_num=100, num_o_o=1000, perc_o_o=0):
    """
    Returns D_e and D_o in the paper.
    """
    dataset_o = dataset_T_trajs(dataset2, num_o_o)
    dataset_o['flag'] = np.zeros_like(dataset_o['terminals'])
    dataset_e, dataset_o_extra = dataset_split_expert(dataset1, split_x, exp_num)
    dataset_e['flag'] = np.ones_like(dataset_e['terminals'])
    if dataset_o_extra is None:
        return dataset_e, dataset_o
    else:
        dataset_o_extra['flag'] = np.ones_like(dataset_o_extra['terminals'])
        for key in dataset_o.keys():
            dataset_o[key] = np.concatenate([dataset_o[key], dataset_o_extra[key]], 0)
        return dataset_e, dataset_o


def dataset_split_expert(dataset, split_x, exp_num, terminate_on_end=False):
    """
    Returns D_e and expert data in D_o in the paper.
    """
    def concat_trajectories(trajectories):
        return np.concatenate(trajectories, 0)

    N = dataset['rewards'].shape[0]
    return_traj = []
    obs_traj = [[]]
    next_obs_traj = [[]]
    action_traj = [[]]
    reward_traj = [[]]
    done_traj = [[]]

    for i in range(N-1):
        obs_traj[-1].append(dataset['observations'][i].astype(np.float32))
        next_obs_traj[-1].append(dataset['observations'][i+1].astype(np.float32))
        action_traj[-1].append(dataset['actions'][i].astype(np.float32))
        reward_traj[-1].append(dataset['rewards'][i].astype(np.float32))
        done_traj[-1].append(bool(dataset['terminals'][i]))

        final_timestep = dataset['timeouts'][i] | dataset['terminals'][i]
        if (not terminate_on_end) and final_timestep:
            # Skip this transition and don't apply terminals on the last step of an episode
            return_traj.append(np.sum(reward_traj[-1]))
            obs_traj.append([])
            next_obs_traj.append([])
            action_traj.append([])
            reward_traj.append([])
            done_traj.append([])

    inds_all = list(range(len(obs_traj)))
    inds_succ = inds_all[:exp_num]
    if split_x > 0:
        inds_o = inds_succ[-split_x:]
        inds_o = list(inds_o)
        inds_succ = list(inds_succ)
        inds_e = set(inds_succ) - set(inds_o)
        inds_e = list(inds_e)
    

        print('# select {} trajs in expert dataset as D_e'.format(len(inds_e)))
        print('# select {} trajs in expert dataset as expert data in D_o'.format(len(inds_o)))

        obs_traj_e = [obs_traj[i] for i in inds_e]
        next_obs_traj_e = [next_obs_traj[i] for i in inds_e]
        action_traj_e = [action_traj[i] for i in inds_e]
        reward_traj_e = [reward_traj[i] for i in inds_e]
        done_traj_e = [done_traj[i] for i in inds_e]

        obs_traj_o = [obs_traj[i] for i in inds_o]
        next_obs_traj_o = [next_obs_traj[i] for i in inds_o]
        action_traj_o = [action_traj[i] for i in inds_o]
        reward_traj_o = [reward_traj[i] for i in inds_o]
        done_traj_o = [done_traj[i] for i in inds_o]

        dataset_e = {
            'observations': concat_trajectories(obs_traj_e),
            'actions': concat_trajectories(action_traj_e),
            'next_observations': concat_trajectories(next_obs_traj_e),
            'rewards': concat_trajectories(reward_traj_e),
            'terminals': concat_trajectories(done_traj_e),
        }

        dataset_o = {
            'observations': concat_trajectories(obs_traj_o),
            'actions': concat_trajectories(action_traj_o),
            'next_observations': concat_trajectories(next_obs_traj_o),
            'rewards': concat_trajectories(reward_traj_o),
            'terminals': concat_trajectories(done_traj_o),
        }
        return dataset_e, dataset_o
    
    else:
        inds_o = []
        inds_succ = list(inds_succ)
        inds_e = set(inds_succ) - set(inds_o)
        inds_e = list(inds_e)
    

        print('# select {} trajs in expert dataset as D_e'.format(len(inds_e)))
        print('# select {} trajs in expert dataset as expert data in D_o'.format(len(inds_o)))

        obs_traj_e = [obs_traj[i] for i in inds_e]
        next_obs_traj_e = [next_obs_traj[i] for i in inds_e]
        action_traj_e = [action_traj[i] for i in inds_e]
        reward_traj_e = [reward_traj[i] for i in inds_e]
        done_traj_e = [done_traj[i] for i in inds_e]

        dataset_e = {
            'observations': concat_trajectories(obs_traj_e),
            'actions': concat_trajectories(action_traj_e),
            'next_observations': concat_trajectories(next_obs_traj_e),
            'rewards': concat_trajectories(reward_traj_e),
            'terminals': concat_trajectories(done_traj_e),
        }

    return dataset_e, None


def dataset_T_trajs(dataset, T, terminate_on_end=False):
    """
    Returns T trajs from dataset.
    """   
    N = dataset['rewards'].shape[0]
    return_traj = []
    obs_traj = [[]]
    next_obs_traj = [[]]
    action_traj = [[]]
    reward_traj = [[]]
    done_traj = [[]]

    for i in range(N-1):
        obs_traj[-1].append(dataset['observations'][i].astype(np.float32))
        next_obs_traj[-1].append(dataset['observations'][i+1].astype(np.float32))
        action_traj[-1].append(dataset['actions'][i].astype(np.float32))
        reward_traj[-1].append(dataset['rewards'][i].astype(np.float32))
        done_traj[-1].append(bool(dataset['terminals'][i]))

        final_timestep = dataset['timeouts'][i] | dataset['terminals'][i]
        if (not terminate_on_end) and final_timestep:
            # Skip this transition and don't apply terminals on the last step of an episode
            return_traj.append(np.sum(reward_traj[-1]))
            obs_traj.append([])
            next_obs_traj.append([])
            action_traj.append([])
            reward_traj.append([])
            done_traj.append([])

    # select T trajectories
    inds_all = list(range(len(obs_traj)))
    inds = inds_all[:T]
    inds = list(inds)

    
    obs_traj = [obs_traj[i] for i in inds]
    next_obs_traj = [next_obs_traj[i] for i in inds]
    action_traj = [action_traj[i] for i in inds]
    reward_traj = [reward_traj[i] for i in inds]
    done_traj = [done_traj[i] for i in inds]


    def concat_trajectories(trajectories):
        return np.concatenate(trajectories, 0)

    return {
        'observations': concat_trajectories(obs_traj),
        'actions': concat_trajectories(action_traj),
        'next_observations': concat_trajectories(next_obs_traj),
        'rewards': concat_trajectories(reward_traj),
        'terminals': concat_trajectories(done_traj),
    }


def dataset_concat(dataset_e, dataset_o):

    def concat_trajectories(trajectories):
        return np.concatenate(trajectories, 0)
    dataset_b = {
        'observations': concat_trajectories([dataset_e['observations'], dataset_o['observations']]),
        'actions': concat_trajectories([dataset_e['actions'], dataset_o['actions']]),
        'next_observations': concat_trajectories([dataset_e['next_observations'], dataset_o['next_observations']]),
        'rewards': concat_trajectories([dataset_e['rewards'], dataset_o['rewards']]),
        'terminals': concat_trajectories([dataset_e['terminals'], dataset_o['terminals']]),
        'flag': concat_trajectories([dataset_e['flag'], dataset_o['flag']])
    }
    return dataset_b
