import math
import torch
import torch.nn as nn

class MixController(nn.Module):

    def __init__(self, pc, descriptor, hidden=128, lam_init=-2.0, sigma_init=1.25):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(pc + descriptor + 3, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.mu = nn.Linear(hidden, 1)
        self.log_sigma = nn.Parameter(torch.tensor(math.log(sigma_init)))
        self.critic = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))
        with torch.no_grad():
            self.mu.bias.fill_(lam_init)
            self.mu.weight.mul_(0.01)

    def forward(self, global_feature, descriptors, progress, att_err, exact_cost, sample=True):
        s = self.trunk(torch.cat((global_feature, descriptors.mean(1), progress, att_err, exact_cost), -1))
        mu = self.mu(s).squeeze(-1)
        sigma = self.log_sigma.clamp(math.log(0.001), math.log(2.0)).exp()
        dist = torch.distributions.Normal(mu, sigma)
        z = dist.sample() if sample else mu
        lam = torch.sigmoid(z)
        return (lam, dist.log_prob(z) if sample else None, dist.entropy() if sample else None, self.critic(s).squeeze(-1))
