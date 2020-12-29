# https://github.com/openai/spinningup/blob/038665d62d569055401d91856abb287263096178/spinup/examples/pytorch/pg_math/1_simple_pg.py#L9
def mlp(sizes, activation=nn.Tanh, output_activation=nn.Identity):
    # Build a feedforward neural network.
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)
