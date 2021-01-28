import torch
import torch.nn as nn
from copy import deepcopy
from collections import deque
import numpy as np
import pycuber as pc
import math

from main import one_hot_code, cube_shuffle, ACTIONS, SOLVED_CUBE

from enum import Enum, unique

from adam_mul import AdamMul

import time
import sys

def mlp_dropout(sizes, activation=nn.PReLU(init=1), output_activation=None, p=0.0):
    # Build a feedforward neural network.
    layers = []
    for j in range(len(sizes) - 1):
        if j < len(sizes) - 2:
            act = activation
            layers += [nn.Linear(sizes[j], sizes[j + 1]), nn.Dropout(p=p), act]
        elif output_activation is None:
            layers += [nn.Linear(sizes[j], sizes[j + 1])]
        else:
            act = output_activation
            layers += [nn.Linear(sizes[j], sizes[j + 1]), act]
    return nn.Sequential(*layers)


class Model(nn.Module):
    def __init__(self, initial, other, action_num, dropout_rate=0.0):
        super(Model, self).__init__()
        self.sizes = initial + other + action_num
        self.network = mlp_dropout(
            sizes=initial + other + action_num, p=dropout_rate)

    def forward(self, state):
        return self.network(state)


@unique
class Network(Enum):
    Online = 0
    Target = 1


class Agent:
    def __init__(self, online, actions, target=None, gamma=0.4, alpha=1e-08, epsilon=0.9, sticky=-1.0, device=None, adam=False):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if device is None else device
        self.online = online.to(self.device)
        self.target = deepcopy(online) if target is None else target.to(self.device)
        # We don't need to calculate the target network's parameter's gradient
        for param in self.target.parameters():
            param.requires_grad = False
        self.actions = actions
        self.epsilon = epsilon
        self.sticky = sticky
        self.gamma = gamma
        self.alpha = alpha
        self.adam_optim = AdamMul(self.online.parameters(), lr=self.alpha) if adam else None

    def get_val(self, input, index, network):
        if network is Network.Online:
            return self.online(input)[index]
        else:
            return self.target(input)[index]

    def get_array(self, input, network):
        if network is Network.Online:
            return self.online(input)
        else:
            return self.target(input)

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
    
    def get_epsilon_act_array(self, input, network):
        if network is Network.Online:
            num = self.get_epsilon_act()
            return (num, self.online(input))
        else:
            num = self.get_epsilon_act()
            return (num, self.target(input))

    def sticky_act(sticky):
        if np.random.random() <= sticky:
            return True
        else:
            return False

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

    def get_best_act_array(self, input, network):

        if network is Network.Online:
            q_array = self.online(input)
        else:
            q_array = self.target(input)

        act_num = 0
        act_value = 0

        for key, value in enumerate(q_array):
            if value > act_value:
                act_num = key
                act_value = value

        return (act_num, q_array)

    def get_act_val(self, input, network):
        if self.epsilon_greedy():
            return self.get_epsilon_act_val(input, network)
        else:
            return self.get_best_act_val(input, network)

    def get_act_array(self, input, network):
        if self.epsilon_greedy():
            return self.get_epsilon_act_array(input, network) 
        else:
            return self.get_best_act_array(input, network)

    def experience_reward(self, suggested, correct):
        reward_vector = torch.full((len(self.actions),), -0.1).to(self.device)
        reward_vector[ACTIONS.index(correct)] = 1.
        if suggested == correct:
            return (-(0.1+0.7), reward_vector)
        else:
            return (-(-0.1+0.7), reward_vector)

    def normal_reward(self, state):
        if state.__ne__(SOLVED_CUBE):
            return 0
        else:
            return 10

    def n_tpd_iter(self, N, reward_vec, q_n, q_t):
        q_diff_gamma = self.gamma ** N * q_n - q_t
        N_TPD = 0
        for x in range(N):
            N_TPD += self.gamma ** x * reward_vec[x] + q_diff_gamma
        return N_TPD

    def update_online(self, loss, output):
        self.online.network.zero_grad()
        output.backward(output)

        for param in online.parameters():
            grad = param.grad

            param.data.sub_(-loss * grad * self.alpha)

    def update_online_adam(self, loss, factor = 1):
        self.online.network.zero_grad()
        loss.backward(loss)
        self.adam_optim.step(factor=factor)

    def update_target_net(self):
        self.target.load_state_dict(self.online.state_dict())
        # We don't need to calculate the target network's parameter's gradient
        for param in self.target.parameters():
            param.requires_grad = False

    def learn(self, replay_time=10_000, replay_shuffle_range=10, replay_chance=0.2, n_steps=5, epoch_time=1_000, epochs=10, test=False, alpha_update_frequency=(False, 5)):

        optimizer = torch.optim.Adam(self.online.parameters(), lr=self.alpha)

        if test:
            tester1 = Test(1, self.online, self.device)
            tester2 = Test(2, self.online, self.device)
            tester3 = Test(3, self.online, self.device)
            tester4 = Test(4, self.online, self.device)
            tester5 = Test(5, self.online, self.device)
            tester6 = Test(6, self.online, self.device)

            tester = Test(replay_shuffle_range, self.online, self.device)
        else:
            tester = None
        
        if alpha_update_frequency[0]:
            alpha_updater = AlphaUpdater(alpha_update_frequency[1], replay_shuffle_range)
        else:
            alpha_updater = None
        
        generator = Generator()

        memory = ReplayBuffer(epoch_time)

        for epoch in range(epochs):

            time = 0

            memory.new_full_buffer(replay_shuffle_range)
            couldnt_solve = 0                
            while time < epoch_time:
                # REPLAY
                if replay_time > 0 or np.random.random() <= replay_chance:

                    # Get random cube and the reverse of the actions that led to it
                    cube, reverse_actions = memory.generate_random_cube()
                    depth = len(reverse_actions)


                    if self.adam_optim is not None:

                        for i in range(depth):
                            input = torch.from_numpy(one_hot_code(cube)).to(self.device) 

                            act, table_online = self.get_best_act_array(input, Network.Online)
                            val_online = table_online[act]

                            val_target = self.get_val(input, act, Network.Target)

                            correct_act = reverse_actions[depth - i - 1]

                            reward, reward_vector = self.experience_reward(ACTIONS[act], correct_act)



                            TD =  -reward + self.gamma * val_target - val_online
                            
                            loss = table_online - reward_vector

                            # -factor, because the step is taken in the direction of -step_size * factor
                            # and we want a step towards the steepest ascent
                            self.update_online_adam(loss, factor=-TD)

                            cube(correct_act)

                            replay_time -= 1
                            time += 1

                    else:

                        for i in range(depth):
                            input = torch.from_numpy(one_hot_code(cube)).to(self.device) 

                            act, table_online = self.get_best_act_array(input, Network.Online)

                            table_target = self.get_array(input, Network.Target)

                            correct_act = reverse_actions[depth - i - 1]

                            reward, reward_vector = self.experience_reward(ACTIONS[act], correct_act)

                            loss_vec = table_online - reward_vector
                            TD = reward + self.gamma * table_target[act] - table_online[act]

                            self.update_online(TD, loss_vec)

                            cube(correct_act)

                            replay_time -= 1
                            time += 1
                
                else:

                    # We need to generate a random state here
                    # We also need to one-hot-code the state before we pass it as input
                    depth = np.random.randint(replay_shuffle_range + 1)
                    cube = generator.generate_cube(depth)

                    TD = 0
                    solved = False

                    for i in range(depth):
                        input = torch.from_numpy(one_hot_code(cube)).to(self.device) 

                        act, table_online = self.get_best_act_array(input, Network.Online)
                        val_online = table_online[act]

                        val_target = self.get_val(input, act, Network.Target)


                        TD += self.gamma * val_target - val_online

                        loss = table_online
                        
                        loss.backward(loss, retain_graph = True)
                        
                        cube(ACTIONS[act])

                        if not (cube != SOLVED_CUBE):
                            solved = True
                            self.update_online_adam(loss, factor= - (TD + 0.5 * (i+1)))
                            self.online.network.zero_grad()

                        replay_time -= 1
                        time += 1
                    
                    if solved is False:
                        self.adam_optim.step(factor=0.1)
                        couldnt_solve += 1

            self.update_target_net()

            self.online.eval()

            print(f"epochs trained: {epoch} of {epochs}, {(epoch/epochs) * 100}%, WILD: couldnt solve = {couldnt_solve}")


            if test and epoch % 5 == 0: 
                num_tests = 500
                print(tester1.solver_with_info(num_tests))
                print(tester2.solver_with_info(num_tests))
                print(tester3.solver_with_info(num_tests))
                print(tester4.solver_with_info(num_tests))
                print(tester5.solver_with_info(num_tests))
                print(tester6.solver_with_info(num_tests))

                print(self.online(torch.from_numpy(one_hot_code(generator.generate_cube(replay_shuffle_range))).to(self.device))) 
                torch.save(agent.online.state_dict(), "./Long_train_plus_break_modi")
                print("saved")

                #if alpha_update_frequency[0]:
                #    self.alpha = alpha_updater.update(self.alpha, tester.win_counter/num_tests, replay_shuffle_range, 0.7)
                #    print(f"The new alpha is {self.alpha}")

            self.online.train()


class AlphaUpdater:

    def __init__(self, frequency, depth):
        self.frequency = frequency
        self.buffer = []
        self.counter = 0
        self.depth = depth

    def win_rate_fun(self, win_rate):
        return (10**(-4) * win_rate + 1)/(10**3 * win_rate + 1)

    def depth_fun(self, current_depth):
        return (current_depth - self.depth+0.1)/(self.depth + 0.1)

    def andreas_fun(self, win_rate, current_depth):
        return (1/win_rate)**1.5 * 1/(current_depth * 5) * 1e-04

    def weighed_average(self, phi):
        w_avg = 0
        w_length = 0
        for i, val in zip(range(len(self.buffer)), reversed(self.buffer)):
            w_avg += val * phi**i
            w_length += i * phi**i
        return w_avg/w_length

    def stagnated(self):
        if np.std(np.array(self.buffer)[-self.frequency:]) <= 0.05:
            return True
        else:
            return False

    def update(self, alpha, win_rate, current_depth, phi=0.5):
        #self.buffer[self.counter] = win_rate
        self.buffer.append(win_rate)
        self.counter += 1
        if self.counter == self.frequency:
            if self.stagnated():
                if alpha <= 10**(-5):
                    alpha = alpha * 10**3
                else:
                    alpha = alpha * 10**(-3)
            else:
                w_mean = self.weighed_average(phi)
                alpha = self.win_rate_fun(w_mean) * self.depth_fun(current_depth) 

            self.counter = 0
            return alpha
        else:
            return alpha


class Generator:
    def generate_cube(self, move_depth):
        # get cube and define as cube
        cube = pc.Cube()

        for action_num in cube_shuffle(move_depth):
            # Gen cube
            cube(ACTIONS[action_num])

        return cube

    def generate_cube_with_info(self, move_depth):
        self.would_be_win_acts_list = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        # get cube and define as cube
        cube = pc.Cube()

        for action_num in cube_shuffle(move_depth):
            # Gen cube
            cube(ACTIONS[action_num])
            self.would_be_win_acts_list[action_num - 6] += 1

        return cube, self.would_be_win_acts_list


class ReplayBuffer:
    def __init__(self, capacity=1000):
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity

    def generate_moves(self, move_depth):
        actions_nums = cube_shuffle(move_depth)
        self.buffer.append([None, actions_nums])

    def new_full_buffer(self, max_move_depth=10):
        for _ in range(self.capacity):
            self.generate_moves(max_move_depth + 1)

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
                reverse_actions.append(ACTIONS[action_num - 6])

            self.buffer[buffer_location][0] = cube
            self.buffer[buffer_location][1] = reverse_actions

        return (self.buffer[buffer_location][0], self.buffer[buffer_location][1])

    def __len__(self):
        return len(self.buffer)


class Test:
    def __init__(self, move_depth, network, device):
        self.device = device
        self.network = network
        self.move_depth = move_depth
        self.win_counter = 0
        self.generator = Generator()

        self.win_act_occ_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.act_occ_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.gen_trajectory_act_list = np.array(
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

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
        self.win_counter = 0
        for _ in range(number_of_tests):
            pre = time.perf_counter()
            self.solve()
        return f"{(self.win_counter/number_of_tests) * 100}% of test-cubes solved over {number_of_tests} tests at {self.move_depth} depth, wins = {self.win_counter}", time_lis

    def time_solver(self, number_of_tests=100):
        self.win_counter = 0
        time_lis = np.zeros(number_of_tests)
        for _ in range(number_of_tests):
            pre = time.perf_counter()
            self.solve()
            time_lis[_]=time.perf_counter()-pre
        return self.win_counter, time_lis

    def solve_with_info(self):
        cube, get_gen_trajectory_act_list = self.generator.generate_cube_with_info(self.move_depth)
        trajectory = []

        self.gen_trajectory_act_list += get_gen_trajectory_act_list

        for i in range(self.move_depth):
            action = self.get_action(
                torch.from_numpy(one_hot_code(cube)).to(self.device)
            )
            cube(action)

            trajectory.append(action)
            self.act_occ_list[ACTIONS.index(action)] += 1

            if cube != SOLVED_CUBE:
                None
            else:
                self.win_counter += 1
                # print(trajectory)
                for act in trajectory:
                    self.win_act_occ_list[ACTIONS.index(act)] += 1
                break

    def solver_with_info(self, number_of_tests=100):
        self.win_counter = 0
        self.win_act_occ_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.act_occ_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.gen_trajectory_act_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        for _ in range(number_of_tests):
            self.solve_with_info()
        return f"{(self.win_counter/number_of_tests) * 100}% of test-cubes solved over {number_of_tests} tests at {self.move_depth} depth, wins = {self.win_counter}, \n gen acts = {self.gen_trajectory_act_list} \n win acts = {self.win_act_occ_list} \n all acts = {self.act_occ_list}"


    def max_mover_solve(self, number_of_tests=1000, modifier = 10):
        cube = self.generator.generate_cube(self.move_depth)
        for _ in range(self.move_depth*modifier):
            cube(self.get_action(torch.from_numpy(
                one_hot_code(cube)).to(self.device)))

            if cube != SOLVED_CUBE:
                None    
            else:
                self.win_counter += 1
                break

    def max_mover_solver(self, number_of_tests=1000, modifier = 10):
        self.win_counter = 0
        for _ in range(number_of_tests):
            self.max_mover_solve(number_of_tests, modifier)
        return f"{(self.win_counter/number_of_tests) * 100}% of test-cubes solved over {number_of_tests} tests at {self.move_depth} depth, wins = {self.win_counter}"


    def confidence_interval_99(self, number_of_tests):
        z = 2.576
        
        self.solver(number_of_tests)
        mean = self.win_counter/number_of_tests 
        p1 = (1-mean)**2 * self.win_counter
        p0 = (0-mean)**2 * (number_of_tests-self.win_counter)
        mse = 1/(number_of_tests-1) * (p1+p0)
        std = math.sqrt(mse)
        return (mean, mean - z * std/math.sqrt(number_of_tests), mean + z * std/math.sqrt(number_of_tests)) 

    def visual(self):
       cube = self.generator.generate_cube(self.move_depth)
       print(chr(27) + "[1J")
       print(f"Shuffled cube at depth level {self.move_depth}:\n{repr(cube)}")
       time.sleep(2)
       print("Solving cube")
       for _ in range(self.move_depth):
           print(repr(cube(self.get_action(torch.from_numpy(one_hot_code(cube)).to(self.device)))))
           time.sleep(1)
       #shuffled_cube = cube.copy()
       #n = 8
       #for _ in range(self.move_depth):
       #     print(chr(27) + "[1J")
       #     print(f"Solving cube at depth level {self.move_depth}")
       #     print("From\n{}\n{}{}\\/\nTo\n{}".format(shuffled_cube.__repr__(),f"{' '*n}||\n"*n,' '*n, repr(cube(self.get_action(torch.from_numpy(one_hot_code(cube)).to(self.device))))))
       #     time.sleep(1)

#####################################################################################################################################################################################

def generate_tests(start: int, depth: int, network, device):
    tests = []
    for i in range(start, depth+1):
        tests.append(Test(i, network, device))
    return tests



#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#print(device)
#
#online = Model([288], [288, 288, 288, 288, 288, 288, 144, 144, 144, 144, 144, 144, 72, 72], [12]).to(device)
#
#param = torch.load("./Long_train_plus_break_modi")
#online.load_state_dict(param)
#online.eval()
#
#tests = generate_tests(int(sys.argv[1]), int(sys.argv[2]), online, device)
#for depth, test in enumerate(tests):
##    print(f"Solving cube at depth level {depth+int(sys.argv[1])}")
#    time.sleep(1)
#    test.visual()
#
#exit(0)
#
#agent = Agent(online, ACTIONS, alpha=1e-05, device=device, adam=True)
#
#test = Test(10, agent.online, agent.device)
#import time
#pre = time.perf_counter()
#
#tests = generate_tests(20, 20, online, device)
#winrate, time_list = test.time_solver(10_000)
#print(np.mean(time_list), 1.96*np.std(time_list))
#
##for test in reversed(tests):
#    #print(test.move_depth, test.confidence_interval_99(10_000))
#
#
#print(time.perf_counter()-pre)
#
#
#
#
#
#exit(0)

######################################################################################################################################################################################

# choose and print optimal device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Initialize model
online = Model([288], [288, 288, 288, 288, 288, 288, 144, 144, 144, 144, 144, 144, 72, 72], [12]).to(device)

# load model
param = torch.load("./preview_model")
online.load_state_dict(param)
online.eval()  # online.train()

# define agent variables
agent = Agent(online, ACTIONS, alpha=1e-05, device=device, adam=True)

# define and mutate test cube to show example of weigts
cube = pc.Cube()
cube("B' U F")

# define cube as input
input = torch.from_numpy(one_hot_code(cube)).to(device)

# find weights before training
before = agent.online(input)

# define mass test parameters
t_depth = 6
test = Test(t_depth, agent.online, agent.device)

# print mass test results
#print(tester1.solver_with_info(1000))
#print(test.max_mover_solver(5000, 30))
#exit(0)

agent.online.train()


# start learning and define parameters to learn based on
agent.learn(
    replay_time=100_000,
    replay_shuffle_range=t_depth,
    replay_chance=0.4,
    n_steps=4,
    epoch_time=1_000,
    epochs=200_000, 
    test=True, 
    alpha_update_frequency=(False, 4),)

agent.online.eval()

# find weights after training
after = agent.online(input)

# print weights from before and after training
print(f"before\n{before} vs after\n{after}")

# prints results of mass testing after training

tester1 = Test(1, agent.online, agent.device)
tester2 = Test(2, agent.online, agent.device)
tester3 = Test(3, agent.online, agent.device)
tester4 = Test(4, agent.online, agent.device)
tester5 = Test(5, agent.online, agent.device)
tester6 = Test(6, agent.online, agent.device)

print(tester1.solver_with_info(1000))
print(tester2.solver_with_info(1000))
print(tester3.solver_with_info(5000))
print(tester4.solver_with_info(5000))
print(tester5.solver_with_info(5000))
print(tester6.solver_with_info(5000))
#print(test.solver_with_info(5000))

torch.save(agent.online.state_dict(), "./plus")

# Prints differnces in weights and bias from before training and after
#post_param = agent.online.state_dict()
#for p in param:
#    print(param[p]-post_param[p])


exit(0)
