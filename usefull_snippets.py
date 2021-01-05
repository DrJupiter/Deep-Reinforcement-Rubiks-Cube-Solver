# https://github.com/openai/spinningup/blob/038665d62d569055401d91856abb287263096178/spinup/examples/pytorch/pg_math/1_simple_pg.py#L9

"""
def mlp(sizes, activation=nn.Tanh, output_activation=nn.Identity):
    # Build a feedforward neural network.
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)
"""



import numpy as np

def reward_to_go(rews):
    n = len(rews)
    rtgs = np.zeros_like(rews)
    for i in reversed(range(n)):
        rtgs[i] = rews[i] + (rtgs[i+1] if i+1 < n else 0)
    return rtgs

def reward_to_go2(rews):
    n = len(rews)
    rtgs = np.zeros_like(rews)
    for i in range(n):
        rtgs[i] = rews[i] + (rtgs[i-1] if i+1 < n else 0)
    return rtgs

GG = np.array([1,2,3,4,5,5])

print(reward_to_go(GG))
print(reward_to_go2(GG))