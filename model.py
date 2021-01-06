import torch
import torch.nn as nn
from copy import deepcopy
from collections import deque
import numpy as np
import pycuber as pc

from main import one_hot_code, cube_shuffle, ACTIONS, SOLVED_CUBE

# Input will be 288 due to 6*8*6 = 288, SIDES*(ALL_TILES-MIDDLE_TILES)*(LEN_COLOR_VEC)

# Output will be of length 12, since there are 12 actions.


#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#Loss_fn = torch.nn.MSELoss(reduction='mean')

      
def mlp(sizes, activation=nn.PReLU(init=1), output_activation=nn.Identity()):
    # Build a feedforward neural network.
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act]
    return nn.Sequential(*layers)
class Model(nn.Module):

    def __init__(self, initial ,other, action_num):
        super(Model, self).__init__()
        self.sizes = initial + other + action_num
        self.network = mlp(sizes=initial+other+action_num)

    def forward(self, state):
        return self.network(state)


class Agent():
    def __init__(self, online, actions,target=None, epsilon=0.9, sticky = 0.0):
        
        self.online = online 
        self.target = deepcopy(online) if target is not None else target
        # We don't need to calculate the target network's parameters
        for param in self.target.parameters():
            param.requires_grad = False
        self.actions = actions
        self.epsilon = epsilon 
        self.sticky = sticky

    
    def epsilon_greedy(epsilon):
        if np.random.random() <= epsilon:
            return True
        else:
            return False

    def sticky_action(sticky):
        if np.random.random() <= sticky:
            return True
        else:
            return False

    def experience_reward(self, experience):
        return None

    def normal_reward(self, state):
        if state.__ne__(SOLVED_CUBE):
            return -0.1
        else:
            return 0.4
    
   
    # Update target Network
    def update_target_net(self):
        self.target.load_state_dict(self.online.state_dict()) 
        # We don't need to calculate the target network's parameters
        for param in self.target.parameters():
            param.requires_grad = False

    # TODO
    # which action to take based on sticky and greedy
    # n-steps accumulator,
    #   - a buffer, which we then sum
    # updating the target network
    # 
    def learn(self, replay_time, replay_chance, replay_shuffle_range, n_steps, epoch_time):
        last_action = None

        # s, a, s', r pairs

        if replay_time > 0:
            None
        else:
           # We need to generate a random state here 
           # We also need to one-hot-code the state before we pass it as input
            cube = None 
            input = one_hot_code(state)

            if np.random.random() >= 0.5:
                if self.epsilon_greedy(self.epsilon):
                    action = self.actions[np.random.randint(0, len(self.actions))]
                else:
                    action = np.argmax(self.online(input))
            else:
                if self.sticky_action(self.sticky) and last_action is not None:
                    action = last_action
                else:
                    action = np.argmax(self.online(input))



class Experience():

    __init__(s, r):
        self.states = np.array()
        self.states.push(s)
        self.rewards = deque(maxlen=4)


        

#class Generator():

class ReplayBuffer():
    def __init__(self, capacity):
        self.buffer = deque(maxlen = capacity) 
        self.capacity = capacity   
        #self.generator = generator

    def generate_moves(self, move_depth):
        actions_nums = cube_shuffle(move_depth)
        self.buffer.append([actions_nums, None]) 
                   

    # Genereates a cube based on a random move, 
    def generate_cube(self):
        buffer_location = np.random.randint(0, self.capacity)
        trajectory_actions_nums = self.buffer[buffer_location][0]
        cube_loc = self.buffer[buffer_location][1]

        if cube_loc == None:
            cube = pc.Cube()
            reverse_actions = []
            for action_num in trajectory_actions_nums:
                # Gen cube
                cube(ACTIONS[action_num])
                # Gen reverse actions (to undo find optimal moves for solver)
                reverse_actions.append(ACTIONS[action_num-6])

            self.buffer[buffer_location][1] = cube # cube_loc = cube

        return (cube_loc, reverse_actions)


    def new_buffer(self, max_move_depth=10):
        for _ in range(self.capacity):
            #self.buffer.append(self.generate_moves(np.random.randint(1, max_move_depth)))
            self.generate_moves(np.random.randint(1, max_move_depth))

    def __len__(self):
        return len(self.buffer)



    



online = Model([288], [4,4,4], [12])

#print(list(online.named_parameters()))
target = deepcopy(online)


input = torch.randn(288, requires_grad=True)
#print(online)
#print(online(input))
#print(target(input))

#print(list(online.network.named_parameters(recurse=True)))
x = torch.tensor([4.0,3.0], requires_grad=True)
#print(x.size())
#for weight in online.network.named_parameters(recurse=True):
#    weight[1] = torch.zeros_like(weight[1])

#print(list(online.named_parameters()))
#print(online.state_dict())

for param in online.named_parameters():
    print(param)

print(online.network)
exit(0)

def TPD(x):
    return x*0.1

output = online(input) 
loss = TPD(output)

online.network.zero_grad() # Reset grads

loss.backward(output) # compute the gradient of all the weights which are the params of the neural network

for param in online.parameters():
#    param = torch.zeros_like(param)
    grad = param.grad
    #param.data.sub_(-alpha*TPD*grad) -> phi_t+1 = phi_t + alpha*TPD*grad # - alpha or + alpha, nobody is sure
#    print(param.data)
    print(f"This is the grad {grad} \n This is the param: {param}")
    param.data.sub_(-10000)
    

#print(list(online.parameters()))


def update_weights(self):
    weights = self.model.get_weights()
    target_weights = self.target_model.get_weights()
    
    for i in range(len(target_weights)):
        target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
        
    self.target_model.set_weights(target_weights)




# https://github.com/higgsfield/RL-Adventure/blob/master/2.double%20dqn.ipynb

# reward =  argmax(Q(s_t+n, a, theta_online)) - optimal_move
# TPD = torch.sum( gamma**k * reward + gamma**n * Q(s_t+n, argmax(  Q(s_t+n, a, theta_online)); theta_target ))

# TPD = loss?

# loss.backward()                       (brugt ved fish ai)
# (weights * loss).mean().backward()    (brugt i REGNBUEN)
