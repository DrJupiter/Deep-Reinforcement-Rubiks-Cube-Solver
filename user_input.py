import argparse

from pathlib import Path

parser = argparse.ArgumentParser(description='Train a Rubik\'s cube solver with Double Deep Q-networking')

model = parser.add_argument_group('Model')
model.add_argument('--model', '-M', required=False, type=Path, default=None, metavar="<MODEL_PATH>", help="The path to a model which will be loaded instead of the default model." )

testing = parser.add_argument_group('Testing')

#parser.add_argument('model', required=True, metavar='M', type=Path, default=None)
args = parser.parse_args()
print(args)

# Model:
# PATH TO MODEL

# PARAMS
# PATH TO MODEL PARAMETERS

# In Agent.__init__:
# gamma, alpha, epsilon

# In Agent.learn:
# replay_time, replay_shuffle_range, replay_chance, n_steps, epoch_time, epochs
# replay_time=10_000, replay_shuffle_range=10, replay_chance=0.2, n_steps=5, epoch_time=1_000, epochs=10