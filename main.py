import pycuber as pc
import numpy as np
actions = ["U", "L", "F", "D", "R", "B",    "U'", "L'", "F'", "D'", "R'", "B'"]

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
    faces = ["L", "U", "F", "D", "R", "B"]
    for face in faces:
        print(SOLVED_CUBE.get_face(face))

def test_single_tile():
    faces = ["L", "U", "F", "D", "R", "B"]

    print(SOLVED_CUBE.get_face(faces[3])[0][0].colour)

def one_hot_code(cube):
    faces = ["L", "U", "F", "D", "R", "B"]
    one_hot_dict = {"red":    np.array([1,0,0,0,0,0]), 
                    "green":  np.array([0,1,0,0,0,0]), 
                    "blue":   np.array([0,0,1,0,0,0]), 
                    "yellow": np.array([0,0,0,1,0,0]),
                    "white":  np.array([0,0,0,0,1,0]),
                    "orange": np.array([0,0,0,0,0,1])}
    one_hot = []
    for face in faces:
        side = cube.get_face(face)
        print(side)
        for lines in side:
            for tile in lines:
                one_hot.append(one_hot_dict[tile.colour])
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

print(len(one_hot_code(SOLVED_CUBE)))
#test_target_middle(SOLVED_CUBE)

#SOLVED_CUBE('U')




#actions = ["U", "L", "F", "D", "R", "B",    "U'", "L'", "F'", "D'", "R'", "B'"]
actions_num = [0,1,2,3,4,5,6,7,8,9,10,11]
#act_short = ["U", "L", "F", "D", "R", "B"]
act_short_num = [0,1,2,3,4,5]

def cube_shuffle(n):

    act_list = ["_","_","_","_"]

    for _ in range(n):
        new_action = False
        while new_action == False:
            
            new_action = True    

            act = random.randint(0, len(actions)-1) 
            inverse_act = actions_num[act-6]

            if inverse_act == act_list[-1]:
                new_action = False

            elif act == act_list[-1] and act == act_list[-2]:
                new_action = False

            elif act > 5:
                act2 = act-6
                if (inverse_act == act_list[-1] or inverse_act == act_list[-2] or inverse_act == act_list[-3] or inverse_act == act_list[-4]) and (act_short_num[act2-3] != act_list[-1] or act_short_num[act2-3] != act_list[-2] or act_short_num[act2-3] != act_list[-3] or act_short_num[act2-3] != act_list[-4]):
                    new_action = False

            elif act <= 5:
                if (inverse_act == act_list[-1] or inverse_act == act_list[-2] or inverse_act == act_list[-3] or inverse_act == act_list[-4]) and (act_short_num[act-3] != act_list[-1] or act_short_num[act-3] != act_list[-2] or act_short_num[act-3] != act_list[-3] or act_short_num[act-3] != act_list[-4]):
                    new_action = False


            if new_action == True:
                act_list.append(act)

    #print(act_list[4:])

    return act_list[4:]