import numpy as np
import heapq as hq
import cv2
import time

from typing import Dict, Tuple, List, Callable, Union
from numpy.typing import NDArray

#import matplotlib.pyplot as plt

import sys

R = 35 #Robot Wheel Radius in mm
r = 250 #Robot Radius in mm
L = 50 #Wheel Distance in mm

map_x = 5400
map_y = 3000
scale = 0.15



def get_delta_pose(current_x,current_y,theta_deg,u_l,u_r,dt,r,L):
    '''

    Args:
        u_l (float): left wheel RPM
        u_r (float): right wheel RPM
        theta (float): heading
        dt (float): time step
        r (float): wheel radius
        L (float): wheel base
    
    Returns:
        tuple: containing:
        - dx (float): change in x coord
        - dy (float): change in y coord
        - dtheta (float): change in heading
    
    '''
    #dx = (r/2)*(u_l+u_r)*np.cos(np.radians(theta*30))*dt
    #dy = (r/2)*(u_l+u_r)*np.sin(np.radians(theta*30))*dt
    #dtheta = np.degrees((r/L)*(u_r-u_l)*dt)
    #return dx,dy,dtheta

    # Convert RPM to rad/s
    ul_rad = (u_l * 2 * np.pi) / 60
    ur_rad = (u_r * 2 * np.pi) / 60

    # Compute linear and angular velocity
    v = r * (ul_rad + ur_rad) / 2 #mm/s18.326
    omega = r * (ur_rad - ul_rad) / L #rad/s0

    # Update pose using small step integration
    n_steps = max(1,int(dt / 0.01))
    x, y, theta = current_x, current_y, np.radians(theta_deg)

    trajectory = []

    for _ in range(n_steps):
        dx = v * np.cos(theta) * 0.01
        dy = v * np.sin(theta) * 0.01
        dtheta = omega * 0.01
        x += dx
        y += dy
        theta += dtheta
        trajectory.append((x,y,np.degrees(theta)%360))

    return trajectory



# Define function for visualizing environment

def visualize_environment(obstacles, clearances, start, goal, path, explored_nodes, trajectory_map):
    # Create a blank 5400x3000 white frame
    frame = np.ones((map_y, map_x, 3), dtype=np.uint8) * 255

    # Generate meshgrid of all (x, y) coordinates
    x_grid, y_grid = np.meshgrid(np.arange(map_x), np.arange(map_y))

    # Compute clearance area and display as gray
    for conditions in clearances.values():
        mask = np.ones_like(x_grid, dtype=bool)
        for cond in conditions:
            mask &= cond(x_grid, y_grid)
        frame[mask] = (150, 150, 150)

    # Compute obstacle area and display as black
    obstacle_mask = np.zeros_like(x_grid, dtype=bool)
    for conditions in obstacles.values():
        temp_mask = np.ones_like(x_grid, dtype=bool)
        for cond in conditions:
            temp_mask &= cond(x_grid, y_grid)
        obstacle_mask |= temp_mask
    frame[np.where(obstacle_mask)] = (0, 0, 0)

    # Flip to match coordinate system
    frame = cv2.flip(frame, 0)

    # Draw explored node trajectories
    for i, node in enumerate(explored_nodes):
        if node in trajectory_map:
            trajectory = trajectory_map[node]
            for j in range(len(trajectory) - 1):
                x1, y1, _ = trajectory[j]
                x2, y2, _ = trajectory[j + 1]

                y1_flipped = map_y - y1
                y2_flipped = map_y - y2

                cv2.line(frame, (int(x1), int(y1_flipped)), (int(x2), int(y2_flipped)),
                         (0, 200, 200), 1)
        else:
            print("not there")

        # Update display every 100 steps
        if i % 100 == 0:
            scale_frame = cv2.resize(frame, (int(map_x * scale), int(map_y * scale)), interpolation=cv2.INTER_LINEAR)
            cv2.imshow("A* Path Visualization", scale_frame)
            cv2.waitKey(1)

    # Draw path node trajectories
    if path is not None:
        for node in path:
            if node in trajectory_map:
                trajectory = trajectory_map[node]
                for j in range(len(trajectory) - 1):
                    x1, y1, _ = trajectory[j]
                    x2, y2, _ = trajectory[j + 1]

                    y1_flipped = map_y - y1
                    y2_flipped = map_y - y2

                    cv2.line(frame, (int(x1), int(y1_flipped)), (int(x2), int(y2_flipped)),
                             (255, 0, 0), 2)  # Blue path

    # Draw final start and goal points
    cv2.circle(frame, (int(start[0]), int(map_y - start[1])), 50, (0, 0, 255), -1)  # Red (start)
    cv2.circle(frame, (int(goal[0]), int(map_y - goal[1])), 50, (0, 255, 0), -1)  # Green (goal)

    scale_frame = cv2.resize(frame, (int(map_x * scale), int(map_y * scale)), interpolation=cv2.INTER_LINEAR)

    # Final visualization
    cv2.imshow("A* Path Visualization", scale_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



# Define function for determining whether a location is valid

def is_valid(x: float | int, y: float | int, clearances: Dict) -> bool:

    # If location is within obstacle constraints
    if any(all(constraint(x, y) for constraint in constraints) for constraints in clearances.values()):

        # Return invalid
        return False
    
    # If location is not within obstacle constraints
    else:
        
        # Return valid
        return True



# Define functions for gathering a pose

def get_start_pose(clearances: Dict) -> Tuple:

    # Loop until pose is valid
    while True:

        # Gather user input

        user_input = input(f"\nStart pose separated by commas in the format of: x, y, θ\n- x: 1 - {map_x}\n- y: 1 - {map_y}\n- θ: Intervals of 30\nEnter: ").strip()

        # default for quick testing
        if user_input == '':
            user_input = '100,100,0'


        if user_input is None:
            return ("Please enter a pose.")
        
        # Break user input into pose coordinates
        parts = user_input.split(",")
        
        # Ensure all three pose coordinates are present
        if len(parts) == 3:
            try:
                
                # Assign input coordinates
                x = float(parts[0].strip())
                y = float(parts[1].strip())
                theta = int(parts[2].strip())
                
                # If coordinates are within bounds
                if 1 <= x <= map_x and 1 <= y <= map_y and theta in [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360]:
                    
                    # Convert positional coordinates to 1 - n scale
                    x = x - 1
                    y = y - 1
                    
                    
                    # If location is not an obstacle, return pose
                    if is_valid(x + 1, y + 1, clearances):
                        return (x, y, (theta % 360))
                    
                    # Inform user of invalid location
                    else:
                        print("Sorry, this point is within the obstacle space. Try again.")
                
                # Inform user of invalid location
                else:
                    print("Invalid input. Please ensure both x and y are within the bounds of the space and theta is in [-60,-30,0,30,60].")
            
            # Inform user of invalid input format
            except ValueError:
                print("Invalid input. Please enter integers for x, y, and theta.")
        
        # Inform user of invalid input dimension
        else:
            print("Invalid input. Please enter exactly three integers separated by a comma.")

# Define functions for gathering a pose

def get_goal_pose(clearances: Dict) -> Tuple:

    # Loop until pose is valid
    while True:

        # Gather user input
        user_input = input(f"\nGoal pose separated by commas in the format of: x, y\n- x: 1 - {map_x}\n- y: 1 - {map_y}\nEnter: ").strip()

        # default for quick testing
        if user_input == '':
            user_input = '3000,500'

        if user_input is None:
            return ("Please enter a pose.")
        
        # Break user input into pose coordinates
        parts = user_input.split(",")
        
        # Ensure all two pose coordinates are present
        if len(parts) == 2:
            try:
                
                # Assign input coordinates
                x = float(parts[0].strip())
                y = float(parts[1].strip())
                # If coordinates are within bounds
                if 1 <= x <= map_x and 1 <= y <= map_y:
                    
                    # Convert positional coordinates to 1 - n scale
                    x = x - 1
                    y = y - 1
                    
                    # If location is not an obstacle, return pose
                    if is_valid(x + 1, y + 1, clearances):
                        return (x, y)
                    
                    # Inform user of invalid location
                    else:
                        print("Sorry, this point is within the obstacle space. Try again.")
                
                # Inform user of invalid location
                else:
                    print("Invalid input. Please ensure both x and y are within the bounds of the space.")
            
            # Inform user of invalid input format
            except ValueError:
                print("Invalid input. Please enter integers for x and y.")
        
        # Inform user of invalid input dimension
        else:
            print("Invalid input. Please enter exactly two integers separated by a comma.")


def get_wheel_rpms() -> Tuple:
    # Loop until values is valid
    while True:

        # Gather user input
        user_input = input("\nTwo positive wheel RPM values separated by commas in the format of: rpm1, rpm2\nEnter: ").strip()

        # default for quick testing
        if user_input == '':
            user_input = '5,10'

        if user_input is None:
            return ("Please enter a pair of wheel RPMs.")
        
        # Break user input into each rpm
        parts = user_input.split(",")
        
        # Ensure all values are present
        if len(parts) == 2:
            try:
                
                # Assign input rpms
                rpm1 = float(parts[0].strip())
                rpm2 = float(parts[1].strip())
                
                # If rpms are within bounds
                if rpm1 > 0 and rpm2 > 0:
                    return (rpm1,rpm2)
                                  
                # Inform user of invalid values
                else:
                    print("Invalid input. Please ensure both wheel RPM values are positive.")
            
            # Inform user of invalid input format
            except ValueError:
                print("Invalid input. Please enter numbers for both wheel RPM values.")
        
        # Inform user of invalid input dimension
        else:
            print("Invalid input. Please enter exactly two numbers separated by a comma.")

def get_clearance() -> int:
    # Loop until values is valid
    while True:

        # Gather user input
        user_input = input("\nClearance (in mm) of the robot [> 0]\nEnter: ").strip()

        # default for quick testing
        if user_input == '':
            user_input = '1'

        if user_input is None:
            return ("Please enter a clearance (as an integer).")

        try:
            
            # Assign input clearance
            clearance = int(user_input.strip())
            
            # If clearance is within bounds
            if clearance > 0:
                return clearance
                                
            # Inform user of invalid values
            else:
                print("Invalid input. Please ensure clearance is a positive integer.")
        
        # Inform user of invalid input format
        except ValueError:
            print("Invalid input. Please enter clearance as an integer.")
   

def a_star(start: Tuple[float, float, int], goal: Tuple[float, float], clearances: Dict, actions: List, map_size: Tuple[int, int] = (5400, 3000)) -> Union[List, None]:
    trajectory_map = {}
    # Mark start time
    start_time = time.time()
    debugging = 0           # show/hide outputs meant for debugging
    threshold = 250.0        # distance threshold to goal to consider success
    early_stop_on = False
    early_stop = 500000      # number of nodes to explore before quitting algorithm

    # Define function for computing heuristic
    def heuristic(node: Tuple[float, float, int], goal: Tuple[float, float]) -> float:
        return np.sqrt((node[0] - goal[0]) ** 2 + (node[1] - goal[1]) ** 2)

    # Define function for backtracking
    def backtrack(goal: Tuple[float, float, int], parent_map: Dict) -> List:
        path = []
        while goal in parent_map:
            path.append(goal)
            goal = parent_map[goal]
        path.reverse()
        return path

    # Define function for getting node neighbors
    def get_neighbors(node: Tuple[float, float, float], visited: np.ndarray, clearances: Dict, actions: List, map_size: Tuple[int, int] = (map_x, map_y)) -> List:
        
        # action specific parameters
        dt = 1.0                # time step (s)
        wheel_radius = R   
        wheel_base = L

        x, y, theta = node 
        theta_deg = theta % 360
        neighbors = []

        # for every action set generate new node
        #action_i = 0
        for ul, ur in actions:
            
            # get changes
            trajectory = get_delta_pose(u_l=ul,u_r=ur,theta_deg=theta_deg, dt=dt, r=wheel_radius, L=wheel_base,current_x=x,current_y=y)
            final_x, final_y, final_theta = trajectory[-1] 

            new_theta_30_index = int(round(final_theta / 30)) % 12
            int_x, int_y = int(round(final_x)), int(round(final_y))

            if 0 <= int_x < map_size[0] and 0 <= int_y < map_size[1]:
                if is_valid(final_x, final_y, clearances) and visited[int_y, int_x, new_theta_30_index] == 0:
                    visited[int_y, int_x, new_theta_30_index] = 1
                    neighbors.append((final_x, final_y, final_theta))
                    trajectory_map[(final_x, final_y, final_theta)] = trajectory

                    """else:
                        if debugging:
                            print(f'\nInvalid Node Created with action index {action_i}: ', new_x, new_y, new_theta, ' ints: ', int_x, int_y, int_theta)
                            print('Status -- visited: ', visited[int_y, int_x, int_theta] == 0)
                            print('Status -- is_valid: ', 0 <= int_x < map_size[0])
                    
            

            except Exception as e:
                print('Error: ', type(e), e)
                print('new_x: ', new_x, ' new_y: ', new_y, ' new_theta: ', new_theta_deg, '\ndx :', dx, " dy: ", dy, ' dtheta: ', dtheta, ' dtheta adjusted: ', dtheta, '\nx: ', x, ' y: ', y, ' theta: ', theta)
                #print('theta sum: ', theta_sum)
                #print('theta sum mod: ', new_theta_30)

                
                print('- - EXITING - -')
                sys.exit()
            """

            #action_i += 1

        return neighbors

    # Create configuration map for visited nodes
    visited = np.zeros((map_size[1], map_size[0], 12), dtype=np.uint8)

    # Initialize open list
    open_list = []
    hq.heappush(open_list, (0, start))

    # Initialize dictionary for storing parent information
    parent_map = {}

    # Initialize dictionary for storing cost information
    cost_map = {start: 0}

    # Initialize list for storing closed nodes and explored nodes
    closed_nodes = []
    explored_nodes = []

    # Loop until queue is empty
    try:
        while open_list:

            current_node_info = hq.heappop(open_list)
            current_node: Tuple[float, float, int] = current_node_info[1]

            # Add node to closed list
            closed_nodes.append(current_node)

            # Record explored node for visualization
            explored_nodes.append(current_node)
            #print(current_node)

            # Determine if solution is found
            if np.sqrt((current_node[0] - goal[0]) ** 2 + (current_node[1] - goal[1]) ** 2) <= threshold:

                # Mark end time
                end_time = time.time()

                print(f"Time to search: {end_time - start_time:.4f} seconds")

                # Backtrack to find path from goal
                return backtrack(current_node, parent_map), explored_nodes, trajectory_map
            
            # Loop through neighbors
            for neighbor in get_neighbors(current_node, visited, clearances, actions):

                # cost for action taken -- all actions valued at 1
                new_cost = cost_map[current_node] + 1  

                # checks if node cost can be reduce and updates it
                if neighbor not in cost_map or new_cost < cost_map[neighbor]:
                    cost_map[neighbor] = new_cost
                    total_cost = new_cost + heuristic(neighbor, goal)
                    hq.heappush(open_list, (total_cost, neighbor))
                    parent_map[neighbor] = current_node

            # logging
            if len(explored_nodes)%1000 == 0:
                print('Nodes Explored: ', len(explored_nodes))


            # early stop to give up -- for testing
            if early_stop_on:
                if len(explored_nodes) == early_stop:
                    break
    except KeyboardInterrupt:
        print('Force Quit')



    return None, explored_nodes, trajectory_map  # Return None if no path is found
        
    

# Define function for main execution

def main():

    base_clearance = 5
    
    # Define action set
    '''actions = [
        (1.25, 60),
        (1.25, 30),
        (1.25, 0),
        (1.25, -30),
        (1.25, -60)
    ]
    '''

    # Gather clearance
    print('======================================================')
    print('Project 3 Phase 2: Turtlebot A* Path Planner')
    user_clearance = get_clearance()

    clearance = base_clearance+user_clearance

    # Define obstacles
    obstacles = {

            "Obstacle 1": [
                lambda x, y: x >= 1000,
                lambda x, y: x <= 1100,
                lambda x, y: y >= 0,
                lambda x, y: y <= 2000
            ],

            "Obstacle 2": [
                lambda x, y: x >= 2100,
                lambda x, y: x <= 2200,
                lambda x, y: y >= 1000,
                lambda x, y: y <= 3000
            ],

            "Obstacle 3": [
                lambda x, y: x >= 3200,
                lambda x, y: x <= 3300,
                lambda x, y: y >= 0,
                lambda x, y: y <= 1000
            ],

            "Obstacle 4": [
                lambda x, y: x >= 3200,
                lambda x, y: x <= 3300,
                lambda x, y: y >= 2000,
                lambda x, y: y <= 3000
            ],

            "Obstacle 5": [
                lambda x, y: x >= 4300,
                lambda x, y: x <= 4400,
                lambda x, y: y >= 0,
                lambda x, y: y <= 2000
            ],

    }

    # Define clearances
    clearances = {

            "Clearance 1": [
                lambda x, y: x >= 1000-clearance,
                lambda x, y: x <= 1100+clearance,
                lambda x, y: y >= 0,
                lambda x, y: y <= 2000+clearance
            ],

            "Clearance 2": [
                lambda x, y: x >= 2100-clearance,
                lambda x, y: x <= 2200+clearance,
                lambda x, y: y >= 1000-clearance,
                lambda x, y: y <= 3000
            ],

            "Clearance 3": [
                lambda x, y: x >= 3200-clearance,
                lambda x, y: x <= 3300+clearance,
                lambda x, y: y >= 0,
                lambda x, y: y <= 1000+clearance
            ],

            "Clearance 4": [
                lambda x, y: x >= 3200-clearance,
                lambda x, y: x <= 3300+clearance,
                lambda x, y: y >= 2000-clearance,
                lambda x, y: y <= 3000
            ],

            "Clearance 5": [
                lambda x, y: x >= 4300-clearance,
                lambda x, y: x <= 4400+clearance,
                lambda x, y: y >= 0,
                lambda x, y: y <= 2000+clearance
            ],

            "Clearance 6": [
                lambda x, y: x >= 0,
                lambda x, y: x <= 10+clearance,
                lambda x, y: y >= 0,
                lambda x, y: y <= map_y
            ],

            "Clearance 7": [
                lambda x, y: x >= map_x-10-clearance,
                lambda x, y: x <= map_x,
                lambda x, y: y >= 0,
                lambda x, y: y <= map_y
            ],

            "Clearance 8": [
                lambda x, y: x >= 0,
                lambda x, y: x <= map_x,
                lambda x, y: y >= 0,
                lambda x, y: y <= 10+clearance
            ],

            "Clearance 9": [
                lambda x, y: x >= 0,
                lambda x, y: x <= map_x,
                lambda x, y: y >= map_y-10-clearance,
                lambda x, y: y <= map_y
            ],

    }

    # Gather start pose
    start = get_start_pose(clearances)

    # Gather goal pose
    goal = get_goal_pose(clearances)

    # Gather wheel RPMS
    rpms = get_wheel_rpms()

    actions = [
        (0,rpms[0]),
        (rpms[0],0),
        (rpms[0],rpms[0]),
        (0,rpms[1]),
        (rpms[1],0),
        (rpms[1],rpms[0]),
        (rpms[0],rpms[1]),
        (rpms[1],rpms[1])
    ]

    # Run search algorithm
    printstart = f"({start[0]+1}, {start[1]+1}, {start[2]*30})"
    printgoal = f"({goal[0]+1}, {goal[1]+1})"

    print('======================================================')
    print('Parameters:')
    print('- Clearance: ', user_clearance)
    print('- Start: ', printstart)
    print('- Goal: ', printgoal)
    print('- RPMS: ', rpms)
    print('======================================================')
    wait = input('Press Enter to Begin Algorithm')
    print('Running A* Algorithm with given parameters...')

    path, explored_nodes, trajectory_map = a_star(start, goal, clearances, actions)
    print('Finished')

    if path is None:
        print('NO PATH FOUND: -- Explored: ', len(explored_nodes), ' Nodes')

    # Visualize the environment
    visualize_environment(obstacles, clearances, start, goal, path, explored_nodes, trajectory_map)



def test():
    R = 35  # Updated wheel radius
    L = 160 # Updated wheelbase
    
    start = (499.0, 29.0, 30)
    print("Start:", start)
    
    # Test straight movement (5,5)
    traj = get_delta_pose(*start, 5, 5, 1.0, R, L)
    print("After moving (5,5):", traj[-1])  # Should move ~18.3mm forward
    
    # Test right turn (5,10)
    traj = get_delta_pose(*start, 5, 10, 1.0, R, L) 
    print("After moving (5,10):", traj[-1])  # Should curve right
    
    # Test pivot (0,5) 
    traj = get_delta_pose(*start, -5, 5, 1.0, R, L)
    print("After pivoting (0,5):", traj[-1])  # Should turn in place




    

# Execute script
if __name__ == "__main__":
    main()