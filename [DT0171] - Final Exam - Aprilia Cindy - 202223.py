import numpy as np
from collections import deque

# Define the grid size
grid_size = (5, 10) #Because 0,0 still exist in array, but will be considered as unaccessible

# Define the starting position of the agent
starting_state = (3, 4)

# Define the positions of the cooking tools (egg beater), frying pan and oven
egg_beater_pos = [(3, 1), (3, 8)]
frying_pan_pos = (4, 1)
oven_pos = (4, 8)

# Define the positions of the special gates
gate_pos = [(1, 4), (1, 6)]

# Define the actions the agent can take (up, down, left, right)
possible_actions = ['up', 'down', 'left', 'right']

# Define the learning rate and discount factor
alpha = 0.1
gamma = 0.9

#Randomize of the way taken heuristic moves
epsilon = 1
isRandom = True

#How many testing
num_episodes = 1000

# Define the reward 
def get_reward(current_state, next_state, num_steps, prev_state, fulfiled_prev):
    reward = 0
    if check_walls(current_state, next_state):
        reward = -10
    if ((current_state == prev_state) and (prev_state != (0,0))): #Not new, but backtracking
        reward = -1
    if ((next_state == prev_state) and (prev_state != (0,0))): #Not new, but backtracking
        reward = -1
    if current_state == next_state: #Not new, but backtracking
        reward = -1
    if ((fulfiled_prev == False) and (next_state in egg_beater_pos)):
        reward = 100 - num_steps
    if ((fulfiled_prev) and (next_state == frying_pan_pos or next_state == oven_pos)):
        reward = 100 - num_steps
    return reward

#Check Walls :
def check_walls(current_state, next_state):
    #From/To (1,1) to/from (2,1)
    if ((current_state == (1,1)) and (next_state == (2,1))):
        return True
    elif ((current_state == (2,1)) and (next_state == (1,1))):
        return True

    #From/To (1,2) to/from (2,2)
    elif ((current_state == (1,2)) and (next_state == (2,2))):
        return True
    elif ((current_state == (2,2)) and (next_state == (1,2))):
        return True

    #From/To (2,2) to/from (3,2)
    elif ((current_state == (2,2)) and (next_state == (3,2))):
        return True
    elif ((current_state == (3,2)) and (next_state == (2,2))):
        return True

    #From/To (2,3) to/from (3,3)
    elif ((current_state == (2,3)) and (next_state == (3,3))):
        return True
    elif ((current_state == (3,3)) and (next_state == (2,3))):
        return True

    #From/To (3,1) to/from (4,1)
    elif ((current_state == (3,1)) and (next_state == (4,1))):
        return True
    elif ((current_state == (4,1)) and (next_state == (3,1))):
        return True

    #From/To (3,1) to/from (3,2)
    elif ((current_state == (3,1)) and (next_state == (3,2))):
        return True
    elif ((current_state == (3,2)) and (next_state == (3,1))):
        return True

    #From/To (1,8) to/from (2,8)
    elif ((current_state == (1,8)) and (next_state == (2,8))):
        return True
    elif ((current_state == (2,8)) and (next_state == (1,8))):
        return True

    #From/To (3,8) to/from (4,8)
    elif ((current_state == (3,8)) and (next_state == (4,8))):
        return True
    elif ((current_state == (4,8)) and (next_state == (3,8))):
        return True

    #From/To (3,9) to/from (4,9)
    elif ((current_state == (3,9)) and (next_state == (4,9))):
        return True
    elif ((current_state == (4,9)) and (next_state == (3,9))):
        return True

    #From/To (2,7) to/from (2,8)
    elif ((current_state == (2,7)) and (next_state == (2,8))):
        return True
    elif ((current_state == (2,8)) and (next_state == (2,7))):
        return True

    #From/To (3,7) to/from (3,8)
    elif ((current_state == (3,7)) and (next_state == (3,8))):
        return True
    elif ((current_state == (3,8)) and (next_state == (3,7))):
        return True

    #From/To (1,4) to/from (1,5)
    elif ((current_state == (1,4)) and (next_state == (1,5))):
        return True
    elif ((current_state == (1,5)) and (next_state == (1,4))):
        return True

    #From/To (1,5) to/from (1,6)
    elif ((current_state == (1,5)) and (next_state == (1,6))):
        return True
    elif ((current_state == (1,6)) and (next_state == (1,5))):
        return True

    #From/To (2,4) to/from (2,5)
    elif ((current_state == (2,4)) and (next_state == (2,5))):
        return True
    elif ((current_state == (2,5)) and (next_state == (2,4))):
        return True

    #From/To (2,5) to/from (2,6)
    elif ((current_state == (2,5)) and (next_state == (2,6))):
        return True
    elif ((current_state == (2,6)) and (next_state == (2,5))):
        return True

    #From/To (3,4) to/from (3,5)
    elif ((current_state == (3,4)) and (next_state == (3,5))):
        return True
    elif ((current_state == (3,5)) and (next_state == (3,4))):
        return True

    #From/To (3,5) to/from (3,6)
    elif ((current_state == (3,5)) and (next_state == (3,6))):
        return True
    elif ((current_state == (3,6)) and (next_state == (3,5))):
        return 

    #From/To (4,4) to/from (4,5)
    elif ((current_state == (4,4)) and (next_state == (4,5))):
        return True
    elif ((current_state == (4,5)) and (next_state == (4,4))):
        return True

    #From/To (4,5) to/from (4,6)
    elif ((current_state == (4,5)) and (next_state == (4,6))):
        return True
    elif ((current_state == (4,6)) and (next_state == (4,5))):
        return True

    else :
        return False

#check Grid Boundaries
def OutOfBoundaries(next_state, grid_size):
    if ((next_state[0] < 1) or (next_state[0] >= grid_size[0]) or (next_state[1] < 1) or (next_state[1] >= grid_size[1]) ):
        return True
    else:
        return False

#Moving Due to Special Gate
def MoveSpecialGate(next_state, gate_pos):
    if next_state in gate_pos:
        if next_state == gate_pos[0]:
            return gate_pos[1], True
        elif next_state == gate_pos[1]:
            return gate_pos[0], True

    return next_state, False

#Check finish game yet
def check_terminal_state(current_state, pass_requirement):
    #Always need to pass egg_beater first
    if (pass_requirement == False):
        return False

    terminal_states = [frying_pan_pos, oven_pos]
    if current_state in terminal_states:
        return True
    else:
        return False

# Implement a function to choose the next action based on the Q-table
def choose_action(current_state, prev_state, q_table, possible_actions, epsilon, isRandom, final_pos):
    #if epsilon high therefore learning, or when there are no learning history
    if (np.sum(q_table) == 0) or (np.random.uniform(0, 1) < epsilon):
        if (isRandom == True):
            #either take from random
            action = np.random.choice(possible_actions)
        else :
            #or use auclean to nearer
            action = move_towards_target(current_state, prev_state, final_pos, possible_actions)
    #if Epsilon low therefore not learning and get from q_table
    else:
        state_action = q_table[current_state[0], current_state[1]]
        action = possible_actions[np.argmax(state_action)]

    return action

def bfs(start, goal):
    queue = deque([start])
    visited = set([start])
    
    while queue:
        current = queue.popleft()
        
        if current == goal:
            return True  # goal found
        
        for next_state in get_valid_states(current):
            if next_state not in visited:
                visited.add(next_state)
                queue.append(next_state)
    
    return False  # goal not found

def get_valid_states(current_state):
    valid_states = []
    for action in actions:
        next_state = get_next_step(action, current_state)
        if next_state and not check_walls(current_state, next_state):
            valid_states.append(next_state)
    return valid_states

#Decide which beater to be taken
def move_towards_beater(current_state, egg_beater_pos):
    egg_beater_no = 0
    egg_beater_distance = (grid_size[0] * grid_size[1])

    for egg_beater_no_iter in range(len(egg_beater_pos)):
        distance = np.linalg.norm(np.array(current_state) - np.array(egg_beater_pos[egg_beater_no_iter]))
        if (distance < egg_beater_distance):
            egg_beater_no = egg_beater_no_iter
            egg_beater_distance = distance

    return egg_beater_no

#Push the action to closer to the oven/frying pan/beater -> final_pos = oven_pos for oven and frying_pan_pos for frying pan
def move_towards_target(current_state, prev_state, final_pos, possible_actions):
    best_action = None
    best_distance = float("inf")

    for action in possible_actions:
        if action == "up":
            next_state = (current_state[0]-1, current_state[1])
        elif action == "down":
            next_state = (current_state[0]+1, current_state[1])
        elif action == "left":
            next_state = (current_state[0], current_state[1]-1)
        elif action == "right":
            next_state = (current_state[0], current_state[1]+1)

        if ((check_walls(current_state, next_state) == False) and (bfs(next_state, final_pos))):
            distance = np.sqrt((next_state[0] - final_pos[0])**2 + (next_state[1] - final_pos[1])**2)
            if (prev_state == next_state):
                continue
            if distance < best_distance:
                best_distance = distance
                best_action = action

    return best_action

#get Next Step of Action :
def get_next_step(action, current_state):
    next_state = None

    if action == "up": # up
        next_state = (current_state[0]-1, current_state[1])
    elif action == "down": # down
        next_state = (current_state[0]+1, current_state[1])
    elif action == "left": # left
        next_state = (current_state[0], current_state[1]-1)
    elif action == "right": # right
        next_state = (current_state[0], current_state[1]+1)

    return next_state

#Move agent
def transition_function(current_state, prev_state, action, num_steps, fulfiled_prev, is_terminal):
    next_state = None
    reward = None
    #is_terminal = None
    #fulfiled_prev = True

    next_state = get_next_step(action, current_state)

    # Check if the next state is a wall
    if check_walls(current_state, next_state):
        #print("Hit the wall | " + str(num_steps) + " | prev_state = " + str(prev_state) + " current_state = " + str(current_state) + " next_state = " + str(next_state))
        if (prev_state == (0,0)):
            next_state = current_state #For the 1st time if the next state hit walls therefore you won't move instead of backtrack
        else :
            next_state = prev_state
    
    #Skip if out of boundaries
    if OutOfBoundaries(next_state, grid_size):
        #print("Out of Boundaries | " + str(num_steps) + " | prev_state = " + str(prev_state) + " current_state = " + str(current_state) + " next_state = " + str(next_state))
        if (prev_state == (0,0)):
            next_state = current_state #For the 1st time if the next state hit walls therefore you won't move instead of backtrack
        else :
            next_state = prev_state

    # Check if the next state is a special gate and update the state accordingly
    next_state, useSpecialGate = MoveSpecialGate(next_state, gate_pos)

    # Check if pick up egg_beater
    if (next_state in egg_beater_pos):
        #print("EGG BEATER FOUND | " + str(num_steps) + " | prev_state = " + str(prev_state) + " current_state = " + str(current_state) + " next_state = " + str(next_state))
        fulfiled_prev = True

    #check the game is final yet
    is_terminal = check_terminal_state(next_state, fulfiled_prev)

    # Get the reward for the current state, action and next state
    reward = get_reward(current_state, next_state, num_steps, prev_state, fulfiled_prev)

    return next_state, reward, is_terminal, fulfiled_prev, useSpecialGate

# Load the Q-table if it exists
try:
    q_table = np.load("q_table.npy")
except:
    q_table = np.zeros(grid_size)

for episode in range(num_episodes):
    savePathArray = [] #Save path to be printed in the end
    
    #define starting value
    current_state = starting_state
    num_of_steps = 0
    prev_state = (0,0)
    
    #Define starting check toggle
    is_terminal = False #isFoundStove
    fulfiled_prev = False #isFoundBeater

    #get egg beater that be targeted (nearest egg beater with euclian distance)
    egg_beater_target = egg_beater_pos[move_towards_beater(current_state, egg_beater_pos)]
    
    while not is_terminal:
        final_pos = egg_beater_target if (fulfiled_prev == False) else frying_pan_pos #oven_pos 

        # Choose an action based on the current state
        action = choose_action(current_state, prev_state, q_table, possible_actions, epsilon, isRandom, final_pos)

        # Perform the action and observe the next state and reward
        next_state, reward, is_terminal, fulfiled_prev, useSpecialGate = transition_function(current_state, prev_state, action, num_of_steps, fulfiled_prev, is_terminal)
        
        # Update the Q-value for the current state and action
        next_row, next_col = next_state
        q_table[current_state[0], current_state[1]] = (1 - alpha) * q_table[current_state[0], current_state[1]] + alpha * (reward + gamma * np.max(q_table[next_row, next_col]))

        savePathArray.append(current_state)
        
        prev_state = current_state
        current_state = next_state
        
        if (is_terminal):
            # Save the Q-table after training
            print("\nFinish in : " + str(num_of_steps) + " steps.\n" + str(savePathArray))
            np.save("q_table.npy", q_table)

        if (num_of_steps > 50):
            #print("BREAK EPISODE : " + str(episode) + " to prevent infinite loop | STEPS=" + str(num_of_steps) + " | prev_state = " + str(prev_state) + " current_state = " + str(current_state) + " next_state = " + str(next_state))
            break
        else :
            #print("EPISODE : " + str(episode) + " | STEPS=" + str(num_of_steps) + " | TARGET=" + str(final_pos) + " | prev_state = " + str(prev_state) + " current_state = " + str(current_state) + " next_state = " + str(next_state))
            pass
        
        num_of_steps += 1
        
        #If using special gate the steps was added 1 for declaring the usage of special gate
        if (useSpecialGate):
            num_of_steps += 1
            savePathArray.append('UseGate')

