
import torch
import torch.nn as nn
from copy import deepcopy
from collections import deque
import numpy as np
import pycuber as pc

from main import one_hot_code, cube_shuffle, ACTIONS, SOLVED_CUBE

# Input will be 288 due to 6*8*6 = 288, SIDES*(ALL_TILES-MIDDLE_TILES)*(LEN_COLOR_VEC)

# Output will be of length 12, since there are 12 actions.


# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Loss_fn = torch.nn.MSELoss(reduction='mean')


# PReLU -> ReLU   # Remove identity activation function
def mlp(sizes, activation=nn.PReLU(init=1), output_activation=nn.Identity()):
    # Build a feedforward neural network.
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act]
    return nn.Sequential(*layers)


class Model(nn.Module):
    def __init__(self, initial, other, action_num):
        super(Model, self).__init__()
        self.sizes = initial + other + action_num
        self.network = mlp(sizes=initial+other+action_num)

    def forward(self, state):
        return self.network(state)

# Make better epislon system with decay


class Agent():
    def __init__(self, online, actions, target=None, gamma=0.4, alpha=1e-08, epsilon=0.9, sticky=-1.0):

        self.online = online
        self.target = deepcopy(online) if target is not None else target
        # We don't need to calculate the target network's parameter's gradient
        for param in self.target.parameters():
            param.requires_grad = False
        self.actions = actions
        self.epsilon = epsilon
        self.sticky = sticky
        self.gamma = gamma
        self.alpha = alpha

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

    def experience_reward(self, suggested, correct):
        if suggested == correct:
            return 2
        else:
            return -2

    def normal_reward(self, state):
        if state.__ne__(SOLVED_CUBE):
            return -0.1
        else:
            return 0.4

    # Update target Network

    def update_target_net(self):
        self.target.load_state_dict(self.online.state_dict())
        # We don't need to calculate the target network's parameter's gradient
        for param in self.target.parameters():
            param.requires_grad = False

    def get_action_value(self, input, last_action):
        if np.random.random() >= 0.5:
            if self.epsilon_greedy(self.epsilon):
                act_num = np.random.randint(0, len(self.actions))
                return (self.actions[act_num], self.online(input)[act_num])
            else:
                return self.get_best_action_value(input)
        else:
            if self.sticky_action(self.sticky) and last_action is not None:
                return (last_action, self.online(input)[self.actions.index(last_action)])
            else:
                return self.get_best_action_value(input)

    def get_best_action_value(self, input):
        q_array = self.online(input)
        act_num = 0
        act_value = q_array[act_num]

        for key, value in enumerate(q_array):
            if value > act_value:
                act_num = key
                act_value = value

        return (self.actions[act_num], act_value)

    # TODO
    # which action to take based on sticky and greedy
    # n-steps accumulator,
    #   - a buffer, which we then sum
    # updating the target network
    #

    def n_tpd_iter(self, N, reward_vec, q_n, q_t):
        q_diff_gamma = self.gamma**N * q_n - q_t
        N_TPD = 0
        for x in range(N):
            N_TPD += self.gamma**x * reward_vec[x] + q_diff_gamma
        return N_TPD

    def learn(self, replay_time=10_000, replay_shuffle_range=10, replay_chance=0.2, n_steps=5, epoch_time=1_000, epochs=10):
        generator = Generator()

        for _ in range(epochs):

            last_action = None
            time = 0
            # s, a, s', r pairs

            memory = ReplayBuffer(epoch_time)
            memory.new_full_buffer(replay_shuffle_range)

            while time < epoch_time:

                if replay_time > 0 or np.random.random() >= replay_chance:

                    cube, reverse_actions = memory.generate_random_cube()
                    input = one_hot_code(cube)
                    
                    for i in range(len(reverse_actions)): 
                        action, action_val_q_t = self.get_best_action_value(input)
                        action_val_q_n = self.target(input)[ACTIONS.index(action)] 
                        
                        reward = self.experience_reward(action, reverse_actions[10-i])
                        
                        # stemmer for vi laver alt dette her til en funktion
                        n_tpd = self.n_tpd_iter(1, reward, action_val_q_n, action_val_q_t) 

                        # update online
                        self.online.network.zero_grad()

                        n_tpd.backward()

                        for param in online.parameters():
                            grad = param.grad

                            param.data.sub_(-n_tpd*grad*self.alpha)
                        
                        replay_time -= 1
                        time += 1

                

                else:
                    # We need to generate a random state here
                    # We also need to one-hot-code the state before we pass it as input
                    depth = np.random.randint(1, replay_shuffle_range+1)
                    cube = generator.generate_cube(depth)

                    # use _ if we are going to give a relative reward for early or late completion
                    for _ in range(depth):
                        input = one_hot_code(cube)

                        last_action, q_t = self.get_action_value(input, last_action)

                        reward_vector = np.zeros(shape=(n_steps))

                        for i in range(1, n_steps-1):
                            reward_vector[i] = self.normal_reward(cube(last_action))
                            time += 1
                            if cube != SOLVED_CUBE:
                                last_action, _ = self.get_action_value(one_hot_code(cube), last_action)
                            else:
                                break
                            # TODO: if cube is solved then stop

                        # last_action is the last action we would have taken in the sequence, so the action, which takes us to state n.

                        # Calculate the temporal difference for n steps
                        n_tpd = self.n_tpd_iter(n_steps, reward_vector, q_n, q_t)

                        # update online
                        self.online.network.zero_grad()

                        n_tpd.backward()

                        for param in online.parameters():
                            grad = param.grad

                            param.data.sub_(-n_tpd*grad*self.alpha)
            
            self.update_target_net()




class Generator():

    def generate_cube(self, move_depth):
        cube = pc.Cube()
        # actions_nums = cube_shuffle(move_depth)

        for action_num in cube_shuffle(move_depth):
            # Gen cube
            cube(ACTIONS[action_num])

        return cube


class ReplayBuffer():
    def __init__(self, capacity=1000):
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
        # self.generator = generator

    def generate_moves(self, move_depth):
        actions_nums = cube_shuffle(move_depth)
        self.buffer.append([None, actions_nums])

    def new_full_buffer(self, max_move_depth=10):
        for _ in range(self.capacity):
            # self.buffer.append(self.generate_moves(np.random.randint(1, max_move_depth)))
            self.generate_moves(np.random.randint(1, max_move_depth+1))

    # Genereates a cube based on a random move,
    def generate_random_cube(self):  # when unscrabeling, it should be done in the reverse order
        buffer_location = np.random.randint(0, self.capacity)
        trajectory_actions_nums = self.buffer[buffer_location][1]

        if self.buffer[buffer_location][0] is None:
            cube = pc.Cube()
            reverse_actions = []
            for action_num in trajectory_actions_nums:
                # Gen cube
                cube(ACTIONS[action_num])
                # Gen reverse actions (to find optimal moves for solver)
                reverse_actions.append(ACTIONS[action_num-6])

            self.buffer[buffer_location][0] = cube
            self.buffer[buffer_location][1] = reverse_actions

        return (self.buffer[buffer_location][0], self.buffer[buffer_location][1])

    def __len__(self):
        return len(self.buffer)


class Test():
    def __init__(self, move_depth):
        self.online = online
        self.move_depth = move_depth
        self.win_counter = 0
        self.generator = Generator()

    def get_action(self):
        return np.argmax(self.online(input))

    def solve(self):
        cube = self.generator.generate_cube()
        for num in range(self.move_depth):
            cube(self.get_action())

            if cube != SOLVED_CUBE:
                None
            else:
                self.wincounter += 1

    def solver(self, test_times=100):
        for i in range(test_times):
            solve()
        return self.win_counter


online = Model([288], [4, 4, 4], [12])

# print(list(online.named_parameters()))
with torch.no_grad():
    target = deepcopy(online)


input = torch.randn(288, requires_grad=True)
input2 = torch.randn(288, requires_grad=True)
# print(online)
# print(online(input))
# print(target(input))

# print(list(online.network.named_parameters(recurse=True)))
#x = torch.tensor([4.0, 3.0], requires_grad=True)
# print(x.size())
# for weight in online.network.named_parameters(recurse=True):
#    weight[1] = torch.zeros_like(weight[1])

# print(list(online.named_parameters()))
# print(online.state_dict())

# for param in online.named_parameters():
#    print(param)



exit(0)
output = online(input)
output2 = target(input2)
output3 = online(input2)[1]
# output.backward(output)
#
# for param in online.parameters():
#    print(param.grad)

print(online.network)


def N_TPD_ITER(gamma, N, reward_vec, q_n, q_t):
    q_diff_gamma = gamma**N * q_n - q_t
    N_TPD = 0
    for x in range(N):
        N_TPD += gamma**x * reward_vec[x] + q_diff_gamma
    return N_TPD


q_t = output[3]
q_n = output2[1]
loss = N_TPD_ITER(0.4, 5, np.array([-0.1, -0.1, -0.1, -0.1, 0.4]), q_n, q_t)

online.network.zero_grad()  # Reset grads

# compute the gradient of all the weights which are the params of the neural network
loss.backward(output[3])

alpha = 1e-4

for param in online.parameters():
    #    param = torch.zeros_like(param)
    grad = param.grad
    # param.data.sub_(-alpha*TPD*grad) -> phi_t+1 = phi_t + alpha*TPD*grad # - alpha or + alpha, nobody is sure
#    print(param.data)
#    print(f"This is the grad {grad} \n This is the param: {param}")
    param.data.sub_(-loss*grad*alpha)




def test_convergence(online, target, input, input2, alpha):
    output = online(input)
    output2 = target(input2)

    q_t = output[3]
    q_n = output2[1]

    loss = N_TPD_ITER(0.4, 5, np.array(
        [-0.1, -0.1, -0.1, -0.1, 0.4]), q_n, q_t)

    online.network.zero_grad()

    loss.backward(q_t)
    if torch.isnan(q_n) or torch.isnan(q_t):
        print(q_n, q_t)
        return True
    print(q_n, q_t)
    for param in online.parameters():
        grad = param.grad

        param.data.sub_(-loss*grad*alpha)


for _ in range(10_000):
    if test_convergence(online, target, input, input2, alpha):
        break


# print(list(online.parameters()))
print(f"q_t: {online(input)[3]} vs {q_t}")
print(f"q_t+n: {online(input2)[1]} vs {q_n} vs online original {output3}")


def update_weights(self):
    weights = self.model.get_weights()
    target_weights = self.target_model.get_weights()

    for i in range(len(target_weights)):
        target_weights[i] = weights[i] * self.tau + \
            target_weights[i] * (1 - self.tau)

    self.target_model.set_weights(target_weights)


# https://github.com/higgsfield/RL-Adventure/blob/master/2.double%20dqn.ipynb

# reward =  argmax(Q(s_t+n, a, theta_online)) - optimal_move
# TPD = torch.sum( gamma**k * reward + gamma**n * Q(s_t+n, argmax(  Q(s_t+n, a, theta_online)); theta_target ))

# TPD = loss?

# loss.backward()                       (brugt ved fish ai)
# (weights * loss).mean().backward()    (brugt i REGNBUEN)
