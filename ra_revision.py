import numpy as np
import heapq as hq
import cv2

from typing import Dict, Tuple, List, Callable, Union
from numpy.typing import NDArray



# Define function for visualizing environment

import numpy as np
import cv2

def visualize_environment(obstacles, clearances, start, goal, path):
    """
    Displays the environment state efficiently with path arrows.

    Args:
        obstacles (Dict[str, List[Callable[[float, float], bool]]]): Algebraic functions bounding obstacles.
        clearances (Dict[str, List[Callable[[float, float], bool]]]): Algebraic functions bounding obstacle clearances.
        start (Tuple[int, int, int]): Start position (x, y, θ).
        goal (Tuple[int, int, int]): Goal position (x, y, θ).
        path (List[Tuple[int, int, int]]): Optimal path from A*.
        explored_nodes (List[Tuple[int, int, int]]): Explored nodes from A*.
    """

    # Create a blank 600x250 white frame
    frame = np.ones((250, 600, 3), dtype=np.uint8) * 255

    # Generate meshgrid of all (x, y) coordinates
    x_grid, y_grid = np.meshgrid(np.arange(600), np.arange(250))

    # Compute clearance areas in bulk
    for conditions in clearances.values():
        mask = np.ones_like(x_grid, dtype=bool)
        for cond in conditions:
            mask &= cond(x_grid, y_grid)  
        frame[mask] = (150, 150, 150)  # Gray for clearance

    # Compute obstacle areas efficiently
    obstacle_mask = np.zeros_like(x_grid, dtype=bool)
    for conditions in obstacles.values():
        temp_mask = np.ones_like(x_grid, dtype=bool)
        for cond in conditions:
            temp_mask &= cond(x_grid, y_grid)
        obstacle_mask |= temp_mask  
    frame[np.where(obstacle_mask)] = (0, 0, 0)  # Black for obstacles

    # Draw the optimal path using arrows
    for i in range(len(path) - 1):
        x1, y1, theta1 = path[i]
        x2, y2, theta2 = path[i + 1]
        
        # Compute arrow direction based on theta
        dx = int(5 * np.cos(np.radians(theta2 * 30)))  # Scale for better visibility
        dy = int(5 * np.sin(np.radians(theta2 * 30)))

        # Draw arrow from (x1, y1) to (x2, y2)
        cv2.arrowedLine(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 1, tipLength=0.75)

    # Draw start and goal points
    cv2.circle(frame, (int(start[0]), int(start[1])), 2, (0, 0, 255), -1)  # Blue (start)
    cv2.circle(frame, (int(goal[0]), int(goal[1])), 2, (0, 255, 0), -1)  # Yellow (goal)

    # Flip frame to match coordinate system
    frame = cv2.flip(frame, 0)

    #frame = cv2.resize(frame, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)

    # Display the environment with path arrows
    cv2.imshow("A* Path Visualization", frame)
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



# Define function for gathering a pose

def get_pose(location: str, clearances: Dict) -> Tuple:

    # Loop until pose is valid
    while True:

        # Gather user input
        user_input = input(f"{location} pose separated by commas in the format of: x, y, θ\n- x: 1 - 600\n- y: 1 - 250\n- θ: Intervals of 30, 60\nEnter: ").strip()

        # Return default start pose if input is empty
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
                if 1 <= x <= 600 and 1 <= y <=250 and theta in [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360]:
                    
                    # Convert positional coordinates to 1 - n scale
                    x = x - 1
                    y = y - 1
                    
                    # If location is not an obstacle, return pose
                    if is_valid(x + 1, y + 1, clearances):
                        return (x, y, theta)
                    
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



# Define function for performing A* algorithm

def a_star(start: Tuple[float, float, int], goal: Tuple[float, float, int], clearances: Dict, actions: List, map_size: Tuple[int, int]=(600, 250)): #-> Union[List, None]:



    # Define function for computing heuristic

    def heuristic(node: Tuple[float, float, int], goal: Tuple[float, float, int]) -> float:

        # Return Euclidean distance between node and goal in 3D CMap space
        return np.sqrt((node[0] - goal[0]) ** 2 + (node[1] - goal[1]) ** 2)



    # Define function for backtracking

    def backtrack(goal: Tuple[float, float, int], parent_map: Dict) -> List:

        # Initialize list for path
        path = []

        # Loop through parent map
        while goal in parent_map:

            # Add node to path
            path.append(goal)

            # Backtrack one node to parent
            goal = parent_map[goal]

        # Reverse path
        path.reverse()

        # Return path
        return path



    # Define function for getting node neighbors
    
    def get_neighbors(node: Tuple[float, int, int], visited: NDArray[np.uint8], clearances: Dict, actions: List, map_size: Tuple[int, int]=(600, 250)) -> List:

        # Unpack node information
        x, y, theta = node

        # Initialize list for storing neighbors
        neighbors = []

        # Loop through all possible actions
        for move, delta_angle in actions:

            # Calculate new orientation (0 - 11)
            new_theta = (theta + delta_angle // 30) % 12

            # Calculate new position
            new_x = x + move * np.cos(np.deg2rad(new_theta * 30))
            new_y = y + move * np.sin(np.deg2rad(new_theta * 30))

            # If neighbors are valid locations
            if is_valid(new_x, new_y, clearances):

                # Add location to neighbors
                neighbors.append((new_x, new_y, new_theta))

        # Return list of neighbors
        return neighbors



    # Create configuration map for visited nodes
    visited = np.zeros((map_size[1], map_size[0], 12), dtype=np.uint8)

    # Initialize open list
    open_list = []

    # Add start node to queue
    hq.heappush(open_list, (0, start))

    # Initialize dictionary for storing parent information
    parent_map = {}

    # Initialize dictionary for storing cost information
    cost_map = {start: 0}

    # Initialize list for storing closed nodes
    closed_nodes = []

    # Loop until queue is empty
    while open_list:

        # Get node of lowest total estimated cost
        current_node_info = hq.heappop(open_list)
        current_node: Tuple[float, float, int] = current_node_info[1]

        print(f"Closing node: {current_node}")

        # Add node to closed list
        closed_nodes.append(current_node)

        # Determine if solution is found
        if heuristic(current_node, goal) <= 1.0 and current_node[2] == goal[2]:
            
            print(f"Found solution at {current_node}")

            # Backtrack to find path from goal
            return backtrack(current_node, parent_map)
        
        # Loop through neighbors
        for neighbor in get_neighbors(current_node, visited, clearances, actions):

            # Add movement cost to node's C2C
            new_cost = cost_map[current_node] + 1

            # Add cost to cost map, or update if it is lower
            if neighbor not in cost_map or new_cost < cost_map[neighbor]:
                
                # Add / Update the cost
                cost_map[neighbor] = new_cost

                # Calculate the total estimated cost
                total_cost = new_cost + heuristic(neighbor, goal)

                # Push the neighbor to the open list
                hq.heappush(open_list, (total_cost, neighbor))

                # Update neighbor's parent with the current node
                parent_map[neighbor] = current_node

    # Return None if no path is found
    return None
        
    

# Define function for main execution

def main():


    # Define obstacles
    obstacles = {

            "Obstacle 1": [
                lambda x, y: x >= 26.25,
                lambda x, y: x <= 51.25,
                lambda x, y: y >= 50,
                lambda x, y: y <= 175
            ],

            "Obstacle 2": [
                lambda x, y: x >= 51.25,
                lambda x, y: x <= 81.25,
                lambda x, y: y >= 50,
                lambda x, y: y <= 75
            ],

            "Obstacle 3": [
                lambda x, y: x >= 51.25,
                lambda x, y: x <= 81.25,
                lambda x, y: y >= 100,
                lambda x, y: y <= 125
            ],

            "Obstacle 4": [
                lambda x, y: x >= 51.25,
                lambda x, y: x <= 81.25,
                lambda x, y: y >= 150,
                lambda x, y: y <= 175
            ],

            "Obstacle 5": [
                lambda x, y: x >= 96.25,
                lambda x, y: x <= 121.25,
                lambda x, y: y >= 50,
                lambda x, y: y <= 175
            ],

            "Obstacle 6": [
                lambda x, y: x >= 121.25,
                lambda x, y: x <= 136.25,
                lambda x, y: y >= -(3 + (1/3)) * x + (504 + (1/6)),
                lambda x, y: y <= -(3 + (1/3)) * x + (579 + (1/6))
            ],

            "Obstacle 7": [
                lambda x, y: x >= 136.25,
                lambda x, y: x <= 161.25,
                lambda x, y: y >= 50,
                lambda x, y: y <= 175,
            ],

            "Obstacle 8": [
                lambda x, y: x >= 176.25,
                lambda x, y: x <= 201.25,
                lambda x, y: y >= 50,
                lambda x, y: y <= 175
            ],

            "Obstacle 9": [
                lambda x, y: x >= 201.25,
                lambda x, y: (x - 201.25)**2 + (y - 150) ** 2 <= 625
            ],

            "Obstacle 10": [
                lambda x, y: x >= 241.25,
                lambda x, y: x <= 266.25,
                lambda x, y: y >= 50,
                lambda x, y: y <= 175
            ],

            "Obstacle 11": [
                lambda x, y: x >= 266.25,
                lambda x, y: x <= 297.5,
                lambda x, y: y >= 50,
                lambda x, y: y <= 175,
                lambda x, y: y <= -3.2 * x + 1027,
                lambda x, y: y >= -3.2 * x + 952
            ],

            "Obstacle 12": [
                lambda x, y: x >= 297.5,
                lambda x, y: x <= 328.75,
                lambda x, y: y >= 50,
                lambda x, y: y <= 175,
                lambda x, y: y <= 3.2 * x - 877,
                lambda x, y: y >= 3.2 * x - 952
            ],

            "Obstacle 13": [
                lambda x, y: x >= 328.75,
                lambda x, y: x <= 353.75,
                lambda x, y: y >= 50,
                lambda x, y: y <= 175,
            ],

            "Obstacle 14": [
                lambda x, y: (x - 406.25)**2 + (y - 87.5) ** 2 <= 1406.25,
            ],

            "Obstacle 15": [
                lambda x, y: (x - 476.5)**2 + (y - 87.5) ** 2 <= 11556.25,
                lambda x, y: (x - 406.25)**2 + (y - 87.5) ** 2 >= 1406.25,
                lambda x, y: (x - 476.5)**2 + (y - 87.5) ** 2 >= 6806.25,
                lambda x, y: y >= 87.5,
                lambda x, y: x <= 426.25,
            ],

            "Obstacle 16": [
                lambda x, y: (x - 496.25)**2 + (y - 87.5) ** 2 <= 1406.25,
            ],

            "Obstacle 17": [
                lambda x, y: (x - 566.25)**2 + (y - 87.5) ** 2 <= 11556.25,
                lambda x, y: (x - 496.25)**2 + (y - 87.5) ** 2 >= 1406.25,
                lambda x, y: (x - 566.25)**2 + (y - 87.5) ** 2 >= 6806.25,
                lambda x, y: y >= 87.5,
                lambda x, y: x <= 516.25,
            ],

            "Obstacle 18": [
                lambda x, y: x >= 548.75,
                lambda x, y: x <= 573.75,
                lambda x, y: y >= 50,
                lambda x, y: y <= 183
            ],

    }

    # Define clearances
    clearances = {

            "Clearance 1": [
                lambda x, y: x >= 21.25,
                lambda x, y: x <= 56.25,
                lambda x, y: y >= 45,
                lambda x, y: y <= 180
            ],

            "Clearance 2": [
                lambda x, y: x >= 56.25,
                lambda x, y: x <= 86.25,
                lambda x, y: y >= 45,
                lambda x, y: y <= 80
            ],

            "Clearance 3": [
                lambda x, y: x >= 56.25,
                lambda x, y: x <= 86.25,
                lambda x, y: y >= 95,
                lambda x, y: y <= 130
            ],

            "Clearance 4": [
                lambda x, y: x >= 56.25,
                lambda x, y: x <= 86.25,
                lambda x, y: y >= 145,
                lambda x, y: y <= 180
            ],

            "Clearance 5": [
                lambda x, y: x >= 91.25,
                lambda x, y: x <= 126.25,
                lambda x, y: y >= 45,
                lambda x, y: y <= 180,
                lambda x, y: y <= -89.12565661 * x + 11318.04697,
            ],

            "Clearance 6": [
                lambda x, y: x >= 124.9701533,
                lambda x, y: x <= 132.52984675,
                lambda x, y: y <= -(3 + (1/3)) * x + 596.5671771,
                lambda x, y: y >= -(3 + (1/3)) * x + 486.7661641
            ],

            "Clearance 7": [
                lambda x, y: x >= 131.25,
                lambda x, y: x <= 166.25,
                lambda x, y: y >= 45,
                lambda x, y: y <= 180,
                lambda x, y: y >= -89.12565315 * x + 11856.80915,
            ],

            "Clearance 8": [
                lambda x, y: x >= 171.25,
                lambda x, y: x <= 206.25,
                lambda x, y: y >= 45,
                lambda x, y: y <= 180
            ],

            "Clearance 9": [
                lambda x, y: x >= 201.25,
                lambda x, y: (x - 201.25)**2 + (y - 150) ** 2 <= 900
            ],

            "Clearance 10": [
                lambda x, y: x >= 236.25,
                lambda x, y: x <= 271.25,
                lambda x, y: y >= 45,
                lambda x, y: y <= 180,
                lambda x, y: y <= -85.165548 * x + 23168.391984
            ],

            "Clearance 11": [
                lambda x, y: x >= 269.92595457,
                lambda x, y: x <= 297.5,
                lambda x, y: y >= 45,
                lambda x, y: y <= 180,
                lambda x, y: y >= -3.2 * x + 935.2369454,
                lambda x, y: y <= -3.2 * x + 1043.763055
            ],

            "Clearance 12": [
                lambda x, y: x >= 297.5,
                lambda x, y: x <= 325.07404543,
                lambda x, y: y >= 45,
                lambda x, y: y <= 180,
                lambda x, y: y <= 3.2 * x - 860.2369454,
                lambda x, y: y >= 3.2 * x - 968.7630546
            ],

            "Clearance 13": [
                lambda x, y: x >= 323.75,
                lambda x, y: x <= 358.75,
                lambda x, y: y >= 45,
                lambda x, y: y <= 180,
                lambda x, y: y <= 85.165548 * x - 27505.10922
            ],

            "Clearance 14": [
                lambda x, y: (x - 406.25)**2 + (y - 87.5) ** 2 <= 1806.25,
            ],

            "Clearance 15": [
                lambda x, y: (x - 476.5)**2 + (y - 87.5) ** 2 <= 12656.25,
                lambda x, y: (x - 406.25)**2 + (y - 87.5) ** 2 >= 1806.25,
                lambda x, y: (x - 476.5)**2 + (y - 87.5) ** 2 >= 6006.25,
                lambda x, y: y >= 87.5,
                lambda x, y: x <= 431.25,
            ],

            "Clearance 16": [
                lambda x, y: (x - 496.25)**2 + (y - 87.5) ** 2 <= 1806.25,
            ],

            "Clearance 17": [
                lambda x, y: (x - 566.25)**2 + (y - 87.5) ** 2 <= 12556.25,
                lambda x, y: (x - 496.25)**2 + (y - 87.5) ** 2 >= 1806.25,
                lambda x, y: (x - 566.25)**2 + (y - 87.5) ** 2 >= 6006.25,
                lambda x, y: y >= 87.5,
                lambda x, y: x <= 521.25,
            ],

            "Clearance 18": [
                lambda x, y: x >= 543.75,
                lambda x, y: x <= 578.75,
                lambda x, y: y >= 45,
                lambda x, y: y <= 188
            ],

        }
    
    # Define action set
    actions = [
        (5, 60),
        (5, 30),
        (5, 0),
        (5, -30),
        (5, -60)
    ]

    # Gather start pose
    start = get_pose("Start", clearances)

    # Gather goal pose
    goal = get_pose("Goal", clearances)

    path = a_star(start, goal, clearances, actions)

    print(path)

    visualize_environment(obstacles, clearances, start, goal, path)


# Execute script

if __name__ == "__main__":
    main()