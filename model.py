
import torch
import torch.nn as nn
from copy import deepcopy
from collections import deque
import numpy as np
import pycuber as pc

from main import one_hot_code, cube_shuffle, ACTIONS, SOLVED_CUBE

from enum import Enum, unique

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


@unique
class Network(Enum):
    Online = 0,
    Target = 1,


class Agent():
    def __init__(self, online, actions, target=None, gamma=0.4, alpha=1e-08, epsilon=0.9, sticky=-1.0, device=None):
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu") if device is None else device
        self.online = online.to(self.device)
        self.target = deepcopy(
            online) if target is None else target.to(self.device)
        # We don't need to calculate the target network's parameter's gradient
        for param in self.target.parameters():
            param.requires_grad = False
        self.actions = actions
        self.epsilon = epsilon
        self.sticky = sticky
        self.gamma = gamma
        self.alpha = alpha

    def get_val(self, input, network, index):
        if network is Network.Online:
            return self.online(input)[index]
        else:
            return self.target(input)[index]

    def epsilon_greedy(self):
        if np.random.random() <= self.epsilon:
            return True
        else:
            return False

    def get_epsilon_act(self):
        return np.random.randint(0, len(self.actions))

    def get_epsilon_act_val(self, input, network):
        if network is Network.Online:
            num = self.get_epsilon_act()
            return (num, self.online(input)[num])
        else:
            num = self.get_epsilon_act()
            return (num, self.target(input)[num])

    def sticky_act(sticky):
        if np.random.random() <= sticky:
            return True
        else:
            return False

    # Looks at function... "Ah yes, we have reached the pinacle of coding"
#    def get_sticky_act(self, last_action):
#        return last_action
#
#    def get_sticky_act_val(self, last_action, input, network):
#        if network is Network.Online:
#            return (last_action, self.online(input)[last_action])
#        else:
#            return (last_action, self.target(input)[last_action])


#    def get_act_val(self, input, last_action, network):
#        if np.random.random() >= 0.5:
#            if self.epsilon_greedy(self.epsilon):
#                return self.get_epsilon_act_val(input, network)
#            else:
#                return self.get_best_act_val(input, network)
#        else:
#            if self.sticky_action(self.sticky) and last_action is not None:
#                return self.get_sticky_act_val(last_action, input, network)
#            else:
#                return self.get_best_act_val(input, network)

    def get_best_act(self, input, network):
        if network is Network.Online:
            return torch.argmax(self.online(input))
        else:
            return torch.argmax(self.target(input))

    def get_best_act_val(self, input, network):

        if network is Network.Online:
            q_array = self.online(input)
        else:
            q_array = self.target(input)

        act_num = 0
        act_value = q_array[act_num]

        for key, value in enumerate(q_array):
            if value > act_value:
                act_num = key
                act_value = value

        return (act_num, act_value)

    def get_act_val(self, input, network):
        if self.epsilon_greedy():
            return self.get_epsilon_act_val(input, network)
        else:
            return self.get_best_act_val(input, network)

    # TODO
    # which action to take based on sticky and greedy
    # n-steps accumulator,
    #   - a buffer, which we then sum
    # updating the target network
    #
    def experience_reward(self, suggested, correct):
        if suggested == correct:
            return 7
        else:
            return -20

    def normal_reward(self, state):
        if state.__ne__(SOLVED_CUBE):
            return -5
        else:
            return 10

    def n_tpd_iter(self, N, reward_vec, q_n, q_t):
        q_diff_gamma = self.gamma**N * q_n - q_t
        N_TPD = 0
        for x in range(N):
            N_TPD += self.gamma**x * reward_vec[x] + q_diff_gamma
        return N_TPD

    def update_online(self, loss):
        self.online.network.zero_grad()

        loss.backward()

        for param in online.parameters():
            grad = param.grad

            param.data.sub_(-loss*grad*self.alpha)  # -loss ? yes

    def update_target_net(self):
        self.target.load_state_dict(self.online.state_dict())
        # We don't need to calculate the target network's parameter's gradient
        for param in self.target.parameters():
            param.requires_grad = False

    def learn(self, replay_time=10_000, replay_shuffle_range=10, replay_chance=0.2, n_steps=5, epoch_time=1_000, epochs=10):
        generator = Generator()
        epochs_trained = 0

        for _ in range(epochs):

            #            last_action = None
            time = 0
            # s, a, s', r pairs

            memory = ReplayBuffer(epoch_time)
            memory.new_full_buffer(replay_shuffle_range)

            while time < epoch_time:
                # REPLAY
                if replay_time > 0 or np.random.random() <= replay_chance:

                    # Get random cube and the reverse of the actions that led to it
                    cube, reverse_actions = memory.generate_random_cube()
                    depth = len(reverse_actions)

                    loss = torch.zeros(depth)
                    for i in range(depth):

                        # Get the state og the c
                        input = torch.from_numpy(one_hot_code(cube)).to(
                            self.device)  # . grad = true

                        # find new best act and the corresponding act_val og online N
                        act, act_val_q_online = self.get_best_act_val(
                            input, Network.Online)
                        correct_act = reverse_actions[depth-i-1]

                        # find act_val og target N according to best_act of online
                        # act_val_q_target = self.target(
                        #    input)[ACTIONS.index(action)]
                        act_val_q_target = self.get_val(
                            input, Network.Target, act)

                        # Calculate reward based on if correct action = reverse action
                        reward = self.experience_reward(
                            ACTIONS[act], correct_act)

                        # update cube according to optimal action (reverse action)
                        cube(correct_act)

                        # Calculate loss based on TD and reward
                        # self.n_tpd_iter(1, reward, act_val_q_target, act_val_q_online)

                        loss[i] = reward + self.gamma * \
                            act_val_q_target - act_val_q_online

                        # Times are changing
                        replay_time -= 1
                        time += 1

                    # Update online weights based on loss (n_tpd)
                    self.update_online(torch.mean(loss))

                # NORMAL
                else:

                    # We need to generate a random state here
                    # We also need to one-hot-code the state before we pass it as input
                    depth = np.random.randint(1, replay_shuffle_range+1)
                    cube = generator.generate_cube(depth)

                    # use _ if we are going to give a relative reward for early or late completion
                    acc = 0
                    while acc < depth:

                        input = torch.from_numpy(
                            one_hot_code(cube)).to(self.device)

                        # TODO: REMOVE LAST ACTION
                        act, act_val_q_online = self.get_act_val(
                            input, Network.Online)

                        # Do action and add 1 to the accumulator
                        cube(ACTIONS[act])
                        acc += 1

                        # Create reward vector
                        reward_vector = torch.zeros(n_steps)

                        reward_vector[0] = self.normal_reward(cube)

                        if not (cube != SOLVED_CUBE):
                            act_val_q_target = self.get_val(
                                input, Network.Target, self.get_best_act(input, Network.Online))
                            n_tpd = self.n_tpd_iter(
                                1, reward_vector, act_val_q_target, act_val_q_online)
                            self.update_online(n_tpd)
                            break
                        else:
                            None

                        for i in range(1, n_steps-1):
                            # Select action
                            if self.epsilon_greedy():
                                act = self.get_epsilon_act()
                                # TODO: self.update_epsilon()
                            else:
                                act = self.get_best_act(torch.from_numpy(
                                    one_hot_code(cube)).to(self.device), Network.Online)

                            # Do action and add 1 to the accumulator
                            cube(ACTIONS[act])
                            acc += 1

                            # Perscribe reward
                            reward_vector[i] = self.normal_reward(cube)

                            # If Cube is solved stop prematurely
                            # and update the weights
                            if not (cube != SOLVED_CUBE):
                                input = torch.from_numpy(
                                    one_hot_code(cube)).to(self.device)
                                act_val_q_target = self.get_val(
                                    input, Network.Target, self.get_best_act(input, Network.Online))
                                n_tpd = self.n_tpd_iter(
                                    i, reward_vector, act_val_q_target, act_val_q_online)
                                self.update_online(n_tpd)
                                break

                            # TODO: if cube is solved then stop
                        else:
                            input = torch.from_numpy(
                                one_hot_code(cube)).to(self.device)
                            act_val_q_target = self.get_val(
                                input, Network.Target, self.get_best_act(input, Network.Online))

                            n_tpd = self.n_tpd_iter(
                                n_steps, reward_vector, act_val_q_target, act_val_q_online)
                            self.update_online(n_tpd)
                            continue
                        break
                    time += acc
            self.update_target_net()
            epochs_trained += 1
            #print(f"epochs trained: {epochs_trained} of {epochs}, {(epochs_trained/epochs) * 100}%", end='\r')
            print(f"epochs trained: {epochs_trained} of {epochs}, {(epochs_trained/epochs) * 100}%")
            print(self.online(torch.from_numpy(one_hot_code(generator.generate_cube(replay_shuffle_range))).to(self.device)))


class Generator():
    def generate_cube(self, move_depth):

        # get cube and define as cube
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
    # when unscrabeling, it should be done in the reverse order
    def generate_random_cube(self):
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
    def __init__(self, move_depth, network, device):
        self.device = device
        self.network = network
        self.move_depth = move_depth
        self.win_counter = 0
        self.generator = Generator()

        self.win_act_occ_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.act_occ_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    def get_action(self, input):
        return ACTIONS[torch.argmax(self.network(input))]

    def solve(self):
        cube = self.generator.generate_cube(self.move_depth)
        for num in range(self.move_depth):
            cube(self.get_action(torch.from_numpy(
                one_hot_code(cube)).to(self.device)))

            if cube != SOLVED_CUBE:
                None
            else:
                self.win_counter += 1
                break

    def solver(self, number_of_tests=100):
        for _ in range(number_of_tests):
            self.solve()
        return f"{(self.win_counter/number_of_tests) * 100}% of test-cubes solved over {number_of_tests} tests at {self.move_depth} depth, wins = {self.win_counter}"

    def solve_with_info(self):
        cube = self.generator.generate_cube(self.move_depth)
        trajectory = []
        for i in range(self.move_depth):
            action = self.get_action(torch.from_numpy(
                one_hot_code(cube)).to(self.device))
            cube(action)

            trajectory.append(action)
            self.act_occ_list[ACTIONS.index(action)] += 1

            if cube != SOLVED_CUBE:
                None
            else:
                self.win_counter += 1
                for act in trajectory:
                    # print(ACTIONS.index(act))
                    self.win_act_occ_list[ACTIONS.index(act)] += 1
                break

    def solver_with_info(self, number_of_tests=100):
        for _ in range(number_of_tests):
            self.solve_with_info()
        return f"{(self.win_counter/number_of_tests) * 100}% of test-cubes solved over {number_of_tests} tests at {self.move_depth} depth, wins = {self.win_counter}, \n win move = {self.win_act_occ_list} \n all acts = {self.act_occ_list}"


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')
print(device)


param = torch.load('./layer_3_training_3_000_000')

online = Model([288], [144, 72, 36, 18], [12]).to(device)

online.load_state_dict(param)
online.eval()


number_of_tests = 5000
#test1 = Test(2, online, device)
# print(test1.solver(number_of_tests))
#
# if test1.win_counter/number_of_tests >= 0.06:
#    torch.save(online.state_dict(), "./initially_good")
#    print("Found one")
#exit(0)

agent = Agent(online, ACTIONS, alpha=1e-08, device=device)
cube = pc.Cube()
cube("R R U")
input = torch.from_numpy(one_hot_code(cube)).to(device)
before = agent.online(input)
agent.learn(replay_time=100_000, replay_shuffle_range=3,
            replay_chance=0.3, n_steps=7, epoch_time=10_000, epochs=100)
test = Test(3, agent.online, agent.device)

after = agent.online(input)

print(f"before\n{before} vs after\n{after}")
print(test.solver_with_info(number_of_tests))


torch.save(agent.online.state_dict(), "./layer_3_training_4_000_000")

exit(0)

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

