import matplotlib, os
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from ppo_atari import PPO
from datetime import datetime, timedelta, timezone

if __name__ == "__main__":
    ENV = "PongNoFrameskip-v4"
    max_epochs = 200
    gamma = 0.99
    N = 64
    T = 1250
    batch_size = 64
    v_loss_coef = 0.5
    max_grad_norm = 0.1
    device = torch.device("cuda:0") if torch.cuda.is_available()
    epsilon = 0.3
    lr = 0.00005
    JST = timezone(timedelta(hours=+9), "JST")
    print(datetime.now(JST))
    name = "-lr" + str(lr) + "-"
    ppo = PPO(ENV, max_epochs, N, T, batch_size, lr=lr, v_loss_coef=v_loss_coef, max_grad_norm=max_grad_norm, epsilon=epsilon, device=device)
    try:
        rs = ppo.run(name=name)
    except KeyboardInterrupt:
        pass
    print(datetime.now(JST))
    plt.plot(range(len(rs)), rs)
    plt.savefig(name[:-1]+"rs.png")
    plt.close()