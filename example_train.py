import numpy as np
import torch
from vessel import Vessel
from policy import ActorCritic
import matplotlib.pyplot as plt
import utils
import os
import glob

# Decide which device we want to run on
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple Metal (MPS)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using NVIDIA CUDA")
else:
    device = torch.device("cpu")
    print("Using CPU")

if __name__ == '__main__':

    task = 'navigating'

    max_m_episode = 100001
    max_steps = 900
    render_after = 200 # render a simulation after n episodes

    env = Vessel(task=task, max_steps=max_steps)
    ckpt_folder = os.path.join('./', task + '_ckpt')
    if not os.path.exists(ckpt_folder):
        os.mkdir(ckpt_folder)

    last_episode_id = 0
    REWARDS = []

    net = ActorCritic(input_dim=env.state_dims, output_dim=env.action_dims).to(device)
    if len(glob.glob(os.path.join(ckpt_folder, '*.pt'))) > 0:
        # load the last ckpt
        checkpoint = torch.load(glob.glob(os.path.join(ckpt_folder, '*.pt'))[-1],
                                weights_only=False
                                )
        net.load_state_dict(checkpoint['model_G_state_dict'])
        last_episode_id = checkpoint['episode_id']
        REWARDS = checkpoint['REWARDS']

    for episode_id in range(last_episode_id, max_m_episode):

        # training loop
        state = env.reset()
        rewards, log_probs, values, entropies, masks = [], [], [], [], []
        for step_id in range(max_steps):
            action, log_prob, value, entropy = net.get_action(state)
            state, reward, done, _ = env.step(action)
            rewards.append(reward)
            log_probs.append(log_prob)
            values.append(value)
            entropies.append(entropy)
            masks.append(1-done)
            if episode_id % render_after == 1:
                env.render()

            if done or step_id == max_steps-1:
                _, _, Qval, _ = net.get_action(state)
                net.update_ac(net, rewards, log_probs, values, entropies, masks, Qval, gamma=0.99)
                break

        REWARDS.append(np.sum(rewards))
        print('episode id: %d, episode reward: %.3f'
              % (episode_id, np.sum(rewards)))

        if episode_id % render_after == 1:
            plt.figure(figsize=(14, 8))
            plt.plot(REWARDS, lw=0, marker='o')
            plt.plot(utils.moving_avg(REWARDS, N=500), lw=1)
            plt.legend(['episode reward', 'moving avg'], loc=2)
            plt.xlabel('m episode')
            plt.xlim([44000, None])
            plt.ylim([-500, None])
            plt.grid(True)
            plt.ylabel('reward')
            plt.savefig(os.path.join(ckpt_folder, 'rewards_' + str(episode_id).zfill(8) + '.jpg'), dpi=300)
            plt.close()

            torch.save({'episode_id': episode_id,
                        'REWARDS': REWARDS,
                        'model_G_state_dict': net.state_dict()},
                       os.path.join(ckpt_folder, 'ckpt_' + str(episode_id).zfill(8) + '.pt'))



