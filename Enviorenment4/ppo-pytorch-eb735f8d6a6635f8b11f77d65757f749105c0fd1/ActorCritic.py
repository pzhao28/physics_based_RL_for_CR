import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

class ActorCritic(nn.Module):
    def __init__(self, ob_s, ac_s, n_mid=10):
        super().__init__()
        self.l1 = nn.Linear(ob_s, n_mid)
        self.l2 = nn.Linear(n_mid, n_mid)
        self.pi = nn.Linear(n_mid, ac_s)
        self.v = nn.Linear(n_mid, 1)

    def forward(self, x):
        h = self.h(x)
        return self.v(h), self.pi(h)

    def h(self, x):
        h = F.relu(self.l1(x))
        return F.relu(self.l2(h))

    def act(self, x):
        pi = self.pi(self.h(x))
        prob = F.softmax(pi, dim=1)
        return prob.multinomial(1)

    def value(self, x):
        return self.v(self.h(x))



class ActorCriticContinuous(nn.Module):
    def __init__(self, ob_s, a_scale, n_mid=10):
        super().__init__()
        self.l1 = nn.Linear(ob_s, n_mid)
        self.l2 = nn.Linear(n_mid, n_mid)
        self.mean = nn.Linear(n_mid, 1)
        self.std = nn.Linear(n_mid, 1)
        self.v = nn.Linear(n_mid, 1)
        self.a_scale = torch.tensor(a_scale)

    def forward(self, x):
        h = self.h(x)
        mean = self.a_scale * torch.tanh(self.mean(h))
        std = F.softplus(self.std(h)) + 1e-4
        dist = Normal(mean, std)
        return self.v(h), dist, mean, std

    def h(self, x):
        h = F.relu(self.l1(x))
        return F.relu(self.l2(h))

    def act(self, x):
        h = self.h(x)
        mean = self.a_scale * torch.tanh(self.mean(h))
        std = F.softplus(self.std(h)) + 1e-4
        dist = Normal(mean, std)
        return dist.sample()

    def value(self, x):
        return self.v(self.h(x))

class ActorCriticCNN(nn.Module):
    def __init__(self, shape, ac_s):
        super().__init__()
        self.c1 = nn.Conv2d(shape[0], 32, 8, stride=4)
        self.c2 = nn.Conv2d(32, 64, 4, stride=2)
        self.c3 = nn.Conv2d(64, 64, 3, stride=1)
        self.conv_out = self._get_conv_out(shape)
        self.l1 = nn.Linear(self.conv_out, 512)
        self.critic = nn.Linear(512, 1)
        self.actor = nn.Linear(512, ac_s)

    def _get_conv_out(self, shape):
        x = torch.zeros(1, *shape)
        h = self.cnn_layer(x)
        return int(np.prod(h.size()))

    def cnn_layer(self, x):
        h = F.relu(self.c1(x))
        h = F.relu(self.c2(h))
        return F.relu(self.c3(h))

    def phi(self, x):
        h = self.cnn_layer(x).view(-1, self.conv_out)
        return F.relu(self.l1(h))

    def forward(self, x):
        h = self.phi(x)
        value = self.critic(h)
        pi = self.actor(h)
        return value, pi

    def act(self, x):
        with torch.no_grad():
            h = self.phi(x)
            pi = self.actor(h)
            prob = F.softmax(pi, dim=1)
            return prob.multinomial(1)

    def value(self, x):
        return self.critic(self.phi(x))