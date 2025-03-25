# Import NumPy for mathematical operations and typing for type hints
import numpy as np
from numpy.typing import NDArray
from typing import Union, Callable, Dict, List, Tuple

# Import heapq for A* algorithm
import heapq

# Import time for analyzing computation time
import time

# Import OpenCV for visualization
import cv2

# TODO - A* as function
# TODO - Define action set
# TODO - Result visualization
# TODO - Visited nodes function
# TODO - Is obstacle function



# Create class for node information

class Node():

    """
    Class for storing and processing node information.

    Attributes:
        location (tuple[Union[int, float], Union[int, float]]): Location of the node.
        angle (int): Angle of the node.
        arrow_rep_length (float): Visual arrow length.
        move_distance (int): Move action length.
    """

    # Initialize node attributes

    def __init__(self, location: Tuple[Union[int, float], Union[int, float]], angle: int) -> None:

        """
        Initializes the Node class.

        Args:
            location (tuple[Union[int, float], Union[int, float]]): Location of the node.
            angle (int): Angle of the node.
        """

        # Define pose attributes
        self.location = location
        self.angle = angle

        self.discrete_angle = int(angle / 30)

        # Visual arrow length
        self.arrow_rep_length = 0.2

        # Move action length
        self.move_distance = 0.5

        # Format angle
        self.format_angle()



    # Define method for returning pose information

    def get_pose(self) -> Tuple[Union[int, float], Union[int, float], int]:

        """
        Returns the pose information of the node.

        Returns:
            tuple[Union[int, float], Union[int, float], int]: Pose information of the node.
        """

        # Return pose information
        return round(float(self.location[0]), 2), round(float(self.location[1]), 2), int(self.angle)
    


    # Define method for returning location

    def get_location(self) -> Tuple[Union[int, float], Union[int, float]]:

        """
        Returns the location information of the node.

        Returns:
            tuple[Union[int, float], Union[int, float]]: Location information of the node.
        """

        # Return location information
        return self.location
    


    # Define method for returning angle

    def get_angle(self) -> Tuple[int, int]:

        """
        Returns the angle information of the node.

        Returns:
            int: Angle information of the node.
        """

        # Return angle information
        return self.angle, self.discrete_angle
    


    # Define method for returning arrow representation

    def get_arrow_rep(self) -> Tuple[Union[int, float], Union[int, float], Union[int, float], Union[int, float]]:

        """
        Returns the arrow representation of the node.

        Returns:
            tuple[Union[int, float], Union[int, float]]: Arrow representation of the node.
        """

        # Calculate arrow end point
        x_end = self.location[0] + self.arrow_rep_length * np.cos(np.radians(self.angle))
        y_end = self.location[1] + self.arrow_rep_length * np.sin(np.radians(self.angle))

        # Return arrow start and end points
        return self.location[0], self.location[1], x_end, y_end
    


    # Define method for performing action 1

    def action_1(self):

        # Change heading
        delta_angle = 60
        self.angle += delta_angle

        delta_discrete = 2
        self.discrete_angle += delta_discrete

        # Move
        self.move()

    

    # Define method for performing action 2

    def action_2(self):

        # Change heading
        delta_angle = 30
        self.angle += delta_angle

        delta_discrete = 1
        self.discrete_angle += delta_discrete



        # Move
        self.move()

    

    # Define method for performing action 3

    def action_3(self):

        # Move
        self.move()

    

    # Define method for performing action 4

    def action_4(self):

        # Change heading
        delta_angle = -30
        self.angle += delta_angle

        # Move
        self.move()



    # Define method for performing action 5

    def action_5(self):

        # Change heading
        delta_angle = -60
        self.angle += delta_angle

        # Move
        self.move()

    

    # Define method for moving

    def move(self):

        # Start location
        x_0 = self.location[0]
        y_0 = self.location[1]

        # End location
        x_d = self.move_distance * np.cos(np.radians(self.angle))
        y_d = self.move_distance * np.sin(np.radians(self.angle))

        # Update location
        self.location = (x_0 + x_d, y_0 + y_d)

        # Format angle
        self.format_angle()

    

    # Define method for formatting angle

    def format_angle(self) -> None:

        """
        Formats the angle to be between -180 and 180 degrees.
        """

        # Format discrete_angle to be on a scale from 0-11
        if self.discrete_angle > 11:
            self.discrete_angle = self.discrete_angle % 12
        elif self.discrete_angle < 0:
            self.discrete_angle = 12 + (self.discrete_angle % 12)


    
    # Define method to return node information

    def __str__(self) -> str:

        """
        Returns the string representation of the node.

        Returns:
            str: String representation of the node.
        """

        # Return node information
        return f"Position: ({self.location[0]}, {self.location[1]}) mm -- Angle: {self.angle} deg"



# Create class for representing 3D configuration map

class CMap():

    """
    Class for storing and processing configuration map information.

    Attributes:
        x_dim (int): X-dimension of the environment.
        y_dim (int): Y-dimension of the environment.
        z_dim (int): Z-dimension of the environment.
        c_map (NDArray[np.uint8]): Configuration map of the environment.
    """

    # Initialize CMap attributes

    def __init__(self) -> None:

        """
        Initializes the CMap class.

        Args:
            Node (Node): Node object to be checked.
        """

        # Assign environment dimensions
        self.x_dim = 1200 # 600 mm environment with 0.5 mm resolution
        self.y_dim = 500 # 250 mm environment with 0.5 mm resolution
        self.z_dim = 12 # 360 deg environment with 30 deg resolution

        # Create configuration map
        self.c_map = np.zeros((self.x_dim, self.y_dim, self.z_dim), dtype=np.uint8)

        # Define static obstacle instance
        self.obstacles: Dict[str, List[Callable[[Union[int, float], Union[int, float]], bool]]] = {

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

        # Define static clearance instance
        self.clearances: Dict[str, List[Callable[[Union[int, float], Union[int, float]], bool]]] = {

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

    

    def discretize(self, xlim=(0, 600), ylim=(0, 250)) -> NDArray[np.uint8]:

        """
        Converts algebraic inequality environment into a discretized environment.

        Args:
            xlim (tuple[int, int]): X-limits of environment. Defaults to (0, 600).
            ylim (tuple[int, int]): Y-limits of environment. Defaults to (0, 250).

        Returns:
            NDArray[np.uint8]: Binary grid of the discretized environment.
                               1 = Free space, 0 = Obstacle.
        """
        
        # Initialize grid as free space (1)
        grid = np.ones((ylim[1], xlim[1]), dtype=np.uint8)

        # Loop through x-values
        for x in range(xlim[0], xlim[1]):
            # Loop through y-values
            for y in range(ylim[0], ylim[1]):
                # Check if the (x, y) point is inside any obstacle
                if any(all(constraint(x, y) for constraint in constraints) for constraints in self.obstacles.values()):
                    grid[y, x] = 0  # Mark as obstacle

        # Scale the grid 2x to match
        scaled_grid = np.repeat(np.repeat(grid, 2, axis=0), 2, axis=1)

        # Stack scaled grid across a 3rd dimension 3 times
        full_3D_binary = np.stack([scaled_grid] * 3, axis=-1)

        # Store scaled binary map and 3D
        self.binary_map = scaled_grid
        self.binary_3D_map = full_3D_binary

        return scaled_grid, full_3D_binary



    # Define method for checking if a node already exists

    def check(self, Node: Node) -> int:

        """
        Checks if a node already exists in the configuration map.

        Args:
            Node (Node): Node object to be checked.

        Returns:
            int: 1 if node is valid, 0 if node is invalid.
        """

        # Get pose
        x, y, angle = Node.get_pose()

        # Convert pose to CMap frame
        x_idx, y_idx, angle_idx = self.to_cmap_frame(x, y, angle)

        # If node config does not exist
        if self.c_map[x_idx, y_idx, angle_idx] == 0:

            # Add to CMap
            self.c_map[x_idx, y_idx, angle_idx] = 1

            # Return 1 as valid
            return 1
        
        # If node does exist
        else:

            # Return 0 as invalid
            return 0
    


    # Define method for converting pose to CMap frame

    def to_cmap_frame(self, x: Union[int, float], y: Union[int, float], angle: int) -> Tuple[int, int, int]:

        """
        Converts pose to the CMap frame.

        Args:
            x (Union[int, float]): X-coordinate of the pose.
            y (Union[int, float]): Y-coordinate of the pose.
            angle (int): Angle of the pose.

        Returns:
            tuple[int, int, int]: _description_
        """

        # Convert x and y to CMap frame
        x_idx = int(np.round(x * 2))
        y_idx = int(np.round(y * 2))

        # Convert angle to CMap frame
        angle_idx = int(np.ceil(angle / 30.0 + 6) - 1)

        # Return CMap frame
        return x_idx, y_idx, angle_idx
    


    # Define method to visualize environment

    def visualize_environment(self) -> None:
    
        """
        Displays the state of the environment.

        Args:
            obstacles (Dict[str, List[Callable[[Union[int, float], Union[int, float]], bool]]]): Algebraic functions bounding obstacles.
            clearances (Dict[str, List[Callable[[Union[int, float], Union[int, float]], bool]]]): Algebraic functions bounding obstacle clearances.
        """
        
        # Create 600 x 250 blank white frame
        frame = np.ones((250, 600, 3), dtype=np.uint8) * 255

        # Loop through obstacle clearances
        for name, conditions in self.clearances.items():
            
            # Loop through x values
            for x in range(600):
                
                # Loop through y values
                for y in range(250):
                    
                    # If location fits all algebraic conditions
                    if all(cond(x, y) for cond in conditions):
                        
                        # Mark clearance location gray
                        frame[y, x] = (150, 150, 150)

        # Loop through obstacles
        for name, conditions in self.obstacles.items():
            
            # Loop through x values
            for x in range(600):
                
                # Loop through y values
                for y in range(250):
                    
                    # If location fits all algebraic conditions
                    if all(cond(x, y) for cond in conditions):
                        
                        # Mark obstacle location black
                        frame[y, x] = (0, 0, 0)

        # Flip frame to match environment coordinate system
        frame = cv2.flip(frame, 0)

        # Display environment
        cv2.imshow("Environment", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    

# Define function to check if a point is in an obstacle space

def is_valid(loc: Tuple, obstacle_arr: NDArray[np.uint8]) -> bool:
    
    """
    Determines whether a location is a valid point in free space.

    Args:
        loc (tuple): Location to be tested.
        obstacle_arr (NDArray[np.uint8]): Discretized, binary array of the environment state.

    Returns:
        bool: True if point is valid, False otherwise.
    """
    
    # If location is out of environment bounds
    if loc[0] < 0 or loc[0] > 599 or loc[1] < 0 or loc[1] > 249:
        return False
    
    # Return True for 1 (free space) False for 0 (obstacle space)
    return obstacle_arr[loc[0],loc[1]] == 1



# Define function to get user input for point

def get_point(loc: str,obstacle_arr: NDArray[np.uint8]) -> Tuple:
    
    """
    Prompts the user to enter a valid pose in the environment.

    Args:
        loc (str): Definition of state. "Start" or "Goal".
        obstacle_arr (NDArray[np.uint8]): Discretized, binary array of the environment state.

    Returns:
        tuple: User-defined pose in the format of (x, y, θ).
    """
    
    # Loop until inputs are valid
    while True:
        
        # Gather user input
        user_input = input(f"{loc} pose separated by commas in the format of: x, y, θ\n- x: 1 - 600\n- y: 1 - 250\n- θ: -60, -30, 0, 30, 60\nEnter: ").strip()
        
        # Return default start pose if input is empty
        if user_input == "" and loc == "start":
            return (6,6,0)
        
        # Return default goal pose of  if input is empty
        elif user_input == "" and loc == "goal":
            return (590,240,0)
        
        # Break user input into pose coordinates
        parts = user_input.split(",")
        
        # Ensure all three pose coordinates are present
        if len(parts) == 3:
            try:
                
                # Assign input coordinates
                x = int(parts[0].strip())
                y = int(parts[1].strip())
                theta = int(parts[2].strip())
                
                # If coordinates are within bounds
                if 1<=x<=600 and 1<=y<=250 and theta in [-60,-30,0,30,60]:
                    
                    # Convert positional coordinates to 1 - n scale
                    x = x-1
                    y = y-1
                    
                    # If location is not an obstacle, return pose
                    if is_valid((x,y), obstacle_arr):
                        return (x,y,theta)
                    
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

def a_star(start: Node, goal: Node, binary_map: NDArray[np.uint8]):

    # Define action set
    actions = [
        lambda node: node.action_1(),
        lambda node: node.action_2(),
        lambda node: node.action_3(),
        lambda node: node.action_4(),
        lambda node: node.action_5()
    ]

    # Define the heuristic
    def heuristic(node: Node, goal: Node) -> float:
        x1, y1, angle1 = node.get_pose()
        x2, y2, angle2 = goal.get_pose()
        return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (angle2 - angle1) ** 2)

    # Initialize open nodes
    open_list = []

    # Push start node into queue (C2C, Node, C2G)
    heapq.heappush(open_list, (0, start, heuristic(start, goal)))

    print(open_list)

    # Initialize parent dictionary to reconstruct the path
    parents = {start.get_pose(): None}

    # Initialize closed set to keep track of visited nodes
    closed_set = set()



# Define main execution

def main():

    # Create CMap instance
    cmap = CMap()

    # Discretize environment
    binary_map, full_3D_binary = cmap.discretize()

    # Get user input for start pose
    start_point = get_point("Start", binary_map)
    
    # Loop until the goal point is different from the start point
    while True:
        
        # Get user input for goal pose
        end_point = get_point("Goal", binary_map)
        
        # Break loop if poses are different
        if end_point != start_point:
            break
        
        # Inform user that poses must be different
        print("The goal point cannot be the same as the start point. Please try again.")

    # Convert start and end points to nodes
    start_node = Node((start_point[0], start_point[1]), start_point[2])
    end_node = Node((end_point[0], end_point[1]), end_point[2])

    a_star(start_node, end_node, full_3D_binary)

    # Visualize environment
    # cmap.visualize_environment()



# Execute the script

if __name__ == "__main__":
    main()