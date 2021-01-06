import torch
import torch.nn as nn

# Input will be 288 due to 6*8*6 = 288, SIDES*(ALL_TILES-MIDDLE_TILES)*(LEN_COLOR_VEC)

# Output will be of length 12, since there are 12 actions.

class Model(nn.Module):

    def __init__(self, action_num):
        super().__init__()
        