import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from ppo_atari import PPO
import os
if __name__ == "__main__":
    ENV = "PongNoFrameskip-v4"
    max_epochs = 20
    gamma = 0.99
    N = 32
    T = 2500
    batch_size = 64
    lr = 0.001
    v_loss_coef = 0.5
    max_grad_norm = 0.5
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ppo = PPO(ENV, max_epochs, N, T, batch_size, lr=lr, v_loss_coef=v_loss_coef, max_grad_norm=max_grad_norm, device=device)
    try:
        rs = ppo.run("trush")
    except KeyboardInterrupt:
        pass
    try:
        os.mkdir("pong")
    except:
        pass
    plt.plot(range(len(rs)), rs)
    plt.savefig("pong/rs-trush.png")