import pycuber as pc
import numpy as np
ACTIONS = np.array(["U", "L", "F", "D", "R", "B",    "U'", "L'", "F'", "D'", "R'", "B'"])
FACES = np.array(["L", "U", "F", "D", "R", "B"])

ONE_HOT_DICT = {"red":    np.array([1,0,0,0,0,0]), 
                "green":  np.array([0,1,0,0,0,0]), 
                "blue":   np.array([0,0,1,0,0,0]), 
                "yellow": np.array([0,0,0,1,0,0]),
                "white":  np.array([0,0,0,0,1,0]),
                "orange": np.array([0,0,0,0,0,1])}

SOLVED_CUBE = pc.Cube()

def epsilon_greedy(epsilon=0.9):
    if np.random.random() <= epsilon:
        return True
    return False

def r(s):
    if s.__ne__(SOLVED_CUBE):
        return -0.1
    else:
        return 0.4



def test_all_faces():
    for face in FACES:
        print(SOLVED_CUBE.get_face(face))

def test_single_tile():
    print(SOLVED_CUBE.get_face(FACES[3])[0][0].colour)

def one_hot_code(cube):
    one_hot = []
    for face in FACES:
        side = cube.get_face(face)
        for lines in side:
            for tile in lines:
                one_hot.append(ONE_HOT_DICT[tile.colour])
                #one_hot.append(tile)

    middle = np.array([4,13,22,31,40,49])
#    one_hot = np.array(one_hot)
#    one_hot = np.delete(one_hot, middle, axis=0)
    return np.delete(np.array(one_hot),middle, axis=0).flatten()

def test_target_middle(cube):
    faces = ["L", "U", "F", "D", "R", "B"]
    for face in faces:
        side = cube.get_face(face)
        side[1][1] = "white"
        print(side)

# print(len(one_hot_code(SOLVED_CUBE)))


ACTIONS_NUM = [0,1,2,3,4,5,6,7,8,9,10,11]

def cube_shuffle(n):
    #err = 0
    act_list = ["_","_","_","_"]

    for _ in range(n):
        new_action = False
        while new_action == False:
            new_action = True    

            act = random.randint(0, len(actions)-1) 
            inverse_act = ACTIONS_NUM[act-6]

            if inverse_act == act_list[-1]:
                new_action = False

            elif act == act_list[-1] and act == act_list[-2]:
                new_action = False

            elif (inverse_act == act_list[-2]  and   (ACTIONS_NUM[act-3] == act_list[-1] or ACTIONS_NUM[act-9] == act_list[-1]))     or      (inverse_act == act_list[-3] and (ACTIONS_NUM[act-3] == act_list[-1] or ACTIONS_NUM[act-9] == act_list[-1]) or (ACTIONS_NUM[act-3] == act_list[-2] or ACTIONS_NUM[act-9] == act_list[-2]))     or     (inverse_act == act_list[-4] and (ACTIONS_NUM[act-3] == act_list[-1] or ACTIONS_NUM[act-9] == act_list[-1]) or (ACTIONS_NUM[act-3] == act_list[-2] or ACTIONS_NUM[act-9] == act_list[-2]) or (ACTIONS_NUM[act-3] == act_list[-3] or ACTIONS_NUM[act-9] == act_list[-3])):
                new_action = False
                #err += 1
            
            if new_action == True:
                act_list.append(act)

    return np.array(act_list[4:]) #, err

# def find_state(move_list, val_of_wanted_state):
    