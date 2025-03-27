import heapq
import numpy as np
from typing import Tuple, Union, Callable, Dict, List
from numpy.typing import NDArray
import cv2

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

    def visualize_environment(self, start, goal, path):
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
        start = start.get_discrete_pose()
        goal = goal.get_discrete_pose()

        # Create a blank 600x250 white frame
        frame = np.ones((250, 600, 3), dtype=np.uint8) * 255

        # Generate meshgrid of all (x, y) coordinates
        x_grid, y_grid = np.meshgrid(np.arange(600), np.arange(250))

        # Compute clearance areas in bulk
        for conditions in self.clearances.values():
            mask = np.ones_like(x_grid, dtype=bool)
            for cond in conditions:
                mask &= cond(x_grid, y_grid)  
            frame[mask] = (150, 150, 150)  # Gray for clearance

        # Compute obstacle areas efficiently
        obstacle_mask = np.zeros_like(x_grid, dtype=bool)
        for conditions in self.obstacles.values():
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

class Node():
    """
    Class for storing and processing node information.
    """

    def __init__(self, location: Tuple[Union[int, float], Union[int, float]], angle: int) -> None:
        """
        Initializes the Node class.
        """
        self.location = location
        self.angle = angle
        self.discrete_angle = int(angle / 30)
        self.move_distance = 0.5

    def format_angle(self, angle: int, discrete_angle: int):
        """
        Formats angle to keep within 0 - 359 degrees. Discretizes angle to 30-degree resolution.
        """
        self.angle = angle % 360
        self.discrete_angle = discrete_angle % 12

    def move(self):
        """
        Moves the node forward based on its current heading.
        """
        x0, y0 = self.location
        x_d = self.move_distance * np.cos(np.deg2rad(self.angle))
        y_d = self.move_distance * np.sin(np.deg2rad(self.angle))
        self.location = (x0 + x_d, y0 + y_d)

    def action_1(self):
        new_node = Node(self.location, self.angle + 60)
        new_node.format_angle(new_node.angle, new_node.discrete_angle)
        new_node.move()
        return new_node

    def action_2(self):
        new_node = Node(self.location, self.angle + 30)
        new_node.format_angle(new_node.angle, new_node.discrete_angle)
        new_node.move()
        return new_node

    def action_3(self):
        new_node = Node(self.location, self.angle)
        new_node.move()
        return new_node

    def action_4(self):
        new_node = Node(self.location, self.angle - 30)
        new_node.format_angle(new_node.angle, new_node.discrete_angle)
        new_node.move()
        return new_node

    def action_5(self):
        new_node = Node(self.location, self.angle - 60)
        new_node.format_angle(new_node.angle, new_node.discrete_angle)
        new_node.move()
        return new_node

    def get_discrete_pose(self) -> Tuple[int, int, int]:
        """
        Returns the discrete pose (rounded location and discrete angle).
        """
        return (int(round(self.location[0])), int(round(self.location[1])), int(self.discrete_angle))

    def __lt__(self, other):
        """Allows Node objects to be compared in the priority queue."""
        return (self.location, self.angle) < (other.location, other.angle)


def is_valid(loc: Tuple[int, int], binary_map: NDArray[np.uint8]) -> bool:
    """
    Determines whether a location is a valid point in free space.
    """
    x, y = loc  # Ensure we're only checking (x, y)
    if x < 0 or x > 599 or y < 0 or y > 249:
        return False
    return binary_map[int(round(y)), int(round(x))] == 1


def get_pose(loc: str, binary_map: NDArray[np.uint8]) -> Union[Tuple[int, int, int], None]:
    """
    Gathers user input for a pose.
    """
    while True:
        user_input = input(f"{loc} pose (x, y, θ): ").strip()
        if user_input == "" and loc == "Start":
            return (6, 6, 0)
        elif user_input == "" and loc == "Goal":
            return (590, 240, 0)

        parts = user_input.split(",")
        if len(parts) == 3:
            try:
                x, y, theta = map(int, map(str.strip, parts))
                if 1 <= x <= 600 and 1 <= y <= 250 and theta % 30 == 0:
                    x, y = x - 1, y - 1
                    if is_valid((x, y), binary_map):
                        return (x, y, theta)
                    else:
                        print("Point is in an obstacle. Try again.")
                else:
                    print("Invalid input. Ensure x ∈ [1,600], y ∈ [1,250], θ is multiple of 30.")
            except ValueError:
                print("Invalid format. Enter x, y, and θ as integers.")
        else:
            print("Invalid input. Enter exactly three values separated by a comma.")


def a_star(start: Node, goal: Node, cmap):
    """
    Performs A* path planning algorithm.
    """
    actions = [lambda node: node.action_1(), lambda node: node.action_2(), lambda node: node.action_3(),
               lambda node: node.action_4(), lambda node: node.action_5()]

    def heuristic(node: Node, goal: Node) -> float:
        """Heuristic considering both Euclidean distance and orientation change."""
        node_pose, goal_pose = node.get_discrete_pose(), goal.get_discrete_pose()
        distance = ((node_pose[0] - goal_pose[0]) ** 2 + (node_pose[1] - goal_pose[1]) ** 2) ** 0.5
        orientation_diff = min(abs(node_pose[2] - goal_pose[2]), 12 - abs(node_pose[2] - goal_pose[2]))
        return distance + 0.5 * orientation_diff  # Increased weight for orientation penalty

    if not is_valid(start.location, cmap.binary_map):
        print("Error: Start position is inside an obstacle!")
        return []
    if not is_valid(goal.location, cmap.binary_map):
        print("Error: Goal position is inside an obstacle!")
        return []

    open_nodes = []
    heapq.heappush(open_nodes, (heuristic(start, goal), start))

    parent_dict = {start.get_discrete_pose(): None}
    g_cost = {start.get_discrete_pose(): 0}
    closed_nodes = set()

    step = 0
    while open_nodes:

        if step%1000 == 0:
            print(f"step: {step} -- closed set: {len(closed_nodes)}")
        step += 1

        _, current_node = heapq.heappop(open_nodes)

        #print(f"Expanding Node: {current_node.get_discrete_pose()}")

        if current_node.get_discrete_pose() == goal.get_discrete_pose():
            path = []
            while current_node:
                path.append(current_node.get_discrete_pose())
                current_node = parent_dict[current_node.get_discrete_pose()]
            path.reverse()
            print("Path found!")
            return path

        closed_nodes.add(current_node.get_discrete_pose())

        for action in actions:
            new_node = action(current_node)
            new_pose = new_node.get_discrete_pose()

            #print(f"Generated New Node: {new_pose}")

            if new_pose in closed_nodes or not is_valid(new_node.location, cmap.binary_map):
                #print(f"Skipping Node {new_pose} (Invalid or in Closed Set)")
                continue

            tentative_g = g_cost[current_node.get_discrete_pose()] + 1

            if tentative_g < g_cost.get(new_pose, float('inf')):
                g_cost[new_pose] = tentative_g
                f_cost = tentative_g + heuristic(new_node, goal)
                heapq.heappush(open_nodes, (f_cost, new_node))
                parent_dict[new_pose] = current_node

    print("No path found!")
    return []


def main() -> None:
    cmap = CMap()
    binary_map, _, _ = cmap.discretize()

    start_pose = get_pose("Start", binary_map)
    start_node = Node(location=(start_pose[0], start_pose[1]), angle=start_pose[2])

    goal_pose = get_pose("Goal", binary_map)
    goal_node = Node(location=(goal_pose[0], goal_pose[1]), angle=goal_pose[2])

    path = a_star(start_node, goal_node, cmap)

    if path:
        print("Path found:")
        for node in path:
            ang = node[2]*30
            if ang>180:
                ang = ang - 360
            print(f"Location: {node[0]+1}, {node[1]+1}, Angle: {ang}")
        cmap.visualize_environment(start_node, goal_node, path)

    else:
        print("No path found.")

    


if __name__ == "__main__":
    main()
