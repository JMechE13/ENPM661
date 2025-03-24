# Import NumPy for mathematical operations
import numpy as np

# Import type hinting
from numpy.typing import NDArray
from typing import Union, Callable, Dict, List, Tuple

# Import HeapQ for A* algorithm
import heapq

# Create configuration map class

class CMap():

    """
    Class for storing and processing configuration map information
    """

    # Initialize CMap attributes

    def __init__(self):
        
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



    # Define method for discretizing environment
    def discretize(self, x_lim: Tuple[int, int]=(0, 600), y_lim: Tuple[int, int]=(0, 250)) -> Tuple[NDArray[np.uint8], NDArray[np.uint8], NDArray[np.uint8]]: 

        """
        Converts algebraic inequality environment into discretized environment. Creates an additional 3D configuration space.
        """

        # Initialize grid as free space
        binary_map = np.ones((y_lim[1], x_lim[1]), dtype=np.uint8)

        # Loop through x values
        for x in range(x_lim[0], x_lim[1]):

            # Loop through y values
            for y in range(y_lim[0], y_lim[1]):

                # Check if location is inside clearances
                if any(all(constraint(x, y) for constraint in constraints) for constraints in self.clearances.values()):

                    # Mark as an obstacle
                    binary_map[y, x] = 0

        # Scale grid to match 0.5 mm resolution
        binary_map_scaled = np.repeat(np.repeat(binary_map, 2, axis=0), 2, axis=1)

        # Stack grid across third dimension to represent angle-space
        binary_map_3D = np.stack([binary_map_scaled] * 3, axis=-1)

        # Store non-scaled, scaled,  and 3D environments
        self.binary_map = binary_map
        self.binary_map_scaled = binary_map_scaled
        self.binary_map_3D = binary_map_3D

        # Return maps
        return binary_map, binary_map_scaled, binary_map_3D



class Node():
    """
    Class for storing and processing node information.
    """

    # Initialize Node attributes

    def __init__(self, location: Tuple[Union[int, float], Union[int, float]], angle: int) -> None:

        """
        Initializes the Node class.
        """

        # Define pose attributes
        self.location = location
        self.angle = angle

        # Discretize orientation for CMap
        self.discrete_angle = int(angle / 30)

    
    
    # Define method for formatting angle

    def format_angle(self, angle: int, discrete_angle: int):

        """
        Formats angle to keep within 0 - 359 degrees. Discretizes angle to 30 degree resolution.
        """

        # Normalize angle to be within 0 - 359 degrees
        self.angle = angle % 360

        # Normalize discrete angle to be within 0 - 11 steps
        self.discrete_angle = discrete_angle % 12

    

    # Define methods for action set

    def action_1(self):

        """
        Defines an action for a 60 degree CCW movement.
        """

        # Change heading
        self.angle += 60
        self.discrete_angle += 2
        self.format_angle(self.angle, self.discrete_angle)

        # Move
        self.move()

    def action_2(self):
        
        """
        Defines an action for a 30 degree CCW movement.
        """

        # Change heading
        self.angle += 30
        self.discrete_angle += 1
        self.format_angle(self.angle, self.discrete_angle)

        # Move
        self.move()

    def action_3(self):
        
        """
        Defines an action for a 0 degree movement.
        """

        # Move
        self.move()

    def action_4(self):
        
        """
        Defines an action for a 30 degree CW movement.
        """

        # Change heading
        self.angle -= 30
        self.discrete_angle -= 1
        self.format_angle(self.angle, self.discrete_angle)

        # Move
        self.move()

    def action_5(self):
        
        """
        Defines an action for a 60 degree CW movement.
        """

        # Change heading
        self.angle -= 60
        self.discrete_angle -= 2
        self.format_angle(self.angle, self.discrete_angle)

        # Move
        self.move()
    


    # Define method for returning node pose

    def get_discrete_pose(self) -> Tuple[Union[int, float], Union[int, float], int]:

        """
        Returns the pose information for the node
        """

        return round(float(self.location[0]), 2), round(float(self.location[1]), 2), int(self.discrete_angle)




# Define function for determining whether a point is valid

def is_valid(loc: Tuple[int, int], binary_map: NDArray[np.uint8]) -> bool:

    """
    Determines whether a location is a valid point in free space.
    """

    # If location is out of environment bounds
    if loc[0] < 0 or loc[0] > 599 or loc[1] < 0 or loc[1] > 249:
        return False
    
    # Return True for 1 (free space) False for 0 (obstacle space)
    return binary_map[loc[1], loc[0]] == 1



# Define function to get user input for a pose

def get_pose(loc: str, binary_map: NDArray[np.uint8]) -> Union[Tuple[int, int, int], None]:

    """
    Gathers user input for a pose.
    """

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
                if 1<=x<=600 and 1<=y<=250 and theta in [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360]:
                    
                    # Convert positional coordinates to 1 - n scale
                    x = x-1
                    y = y-1
                    
                    # If location is not an obstacle, return pose
                    if is_valid((x,y), binary_map):
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



# Define function for A* algorithm

def a_star(start: Node, goal: Node):
    
    """
    Performs A* path planning algorithm.
    """

    # Define action set
    actions = [
        start.action_1,
        start.action_2,
        start.action_3,
        start.action_4,
        start.action_5
    ]



    # Define function for computing C2G heuristic

    def heuristic(node: Node, goal: Node) -> float:

        """
        Computes the C2G heuristic for a node.
        """
        
        # Get pose information for current node
        node_pose = node.get_discrete_pose()

        # Get pose information for goal node
        goal_pose = goal.get_discrete_pose()

        # Return Euclidean distance between node and goal in 3D CMap space
        return np.sqrt((node_pose[0] - goal_pose[0]) ** 2 + (node_pose[1] - goal_pose[1]) ** 2 + (node_pose[2] - goal_pose[2]) ** 2)
    


    # Initialize open nodes and add start node
    open_nodes = []
    heapq.heappush(open_nodes, (0, start, heuristic(start, goal)))

    # Initialize parent dictionary
    parent_dict = {start.get_discrete_pose(): None}

    # Initialize closed nodes set
    closed_nodes = set()



# Define main execution

def main() -> None:

    # Create instance of configuration map
    cmap = CMap()

    # Discretize the environment
    binary_map, _, _ = cmap.discretize()

    # Get user input for start pose
    start_pose = get_pose("Start", binary_map)

    # Loop until the goal point is different from the start point
    while True:
        
        # Get user input for goal pose
        end_pose = get_pose("Goal", binary_map)
        
        # Break loop if poses are different
        if end_pose != start_pose:
            break
        
        # Inform user that poses must be different
        print("The goal point cannot be the same as the start point. Please try again.")

    # Convert start and end poses to nodes
    start_node = Node((start_pose[0], start_pose[1]), start_pose[2])
    end_node = Node((end_pose[0], end_pose[1]), end_pose[2])

    # Perform A* algorithm
    a_star(start_node, end_node)



# Execute script

if __name__ == "__main__":
    main()