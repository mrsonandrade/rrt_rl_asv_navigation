import torch
from vessel import Vessel
from policy import ActorCritic
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
    max_steps = 5000
    ckpt_dir = glob.glob(os.path.join(task+'_ckpt', '*.pt'))[-1]  # last ckpt

    env = Vessel(task=task, max_steps=max_steps)
    net = ActorCritic(input_dim=env.state_dims, output_dim=env.action_dims).to(device)
    if os.path.exists(ckpt_dir):
        checkpoint = torch.load(ckpt_dir,
                                weights_only=False
                                )
        net.load_state_dict(checkpoint['model_G_state_dict'])

    state = env.reset()
    for step_id in range(max_steps):
        action, log_prob, value, entropy = net.get_action(state)
        state, reward, done, _ = env.step(action)
        env.render(window_name='test')

        if done:
            break


