# Import required packages

# Import NumPy for mathematical operations and typing for type hints
import numpy as np
from numpy.typing import NDArray
from typing import Union, Callable, Dict, List

# Import heapq for A* algorithm
import heapq

# Import time for analyzing computation time
import time

# Import OpenCV for visualization
import cv2

# TODO - User input for orientation
# TODO - A* as function
# TODO - Define action set
# TODO - Result visualization
# TODO - Visited nodes function
# TODO - Is obstacle function


# Define function for visualizing the environment

def visualize_environment(obstacles: Dict[str, List[Callable[[Union[int, float], Union[int, float]], bool]]],
                          clearances: Dict[str, List[Callable[[Union[int, float], Union[int, float]], bool]]]) -> None:
    
    """
    Displays the state of the environment.

    Args:
        obstacles (Dict[str, List[Callable[[Union[int, float], Union[int, float]], bool]]]): Algebraic functions bounding obstacles.
        clearances (Dict[str, List[Callable[[Union[int, float], Union[int, float]], bool]]]): Algebraic functions bounding obstacle clearances.
    """
    
    # Create 600 x 250 blank white frame
    frame = np.ones((250, 600, 3), dtype=np.uint8) * 255

    # Loop through obstacle clearances
    for name, conditions in clearances.items():
        
        # Loop through x values
        for x in range(600):
            
            # Loop through y values
            for y in range(250):
                
                # If location fits all algebraic conditions
                if all(cond(x, y) for cond in conditions):
                    
                    # Mark clearance location gray
                    frame[y, x] = (150, 150, 150)

    # Loop through obstacles
    for name, conditions in obstacles.items():
        
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



# Define function for discretizing environment

def discretize(clearances: dict) -> NDArray[np.uint8]:

    """
    Converts algebraic inequality environment into a discretized environment.

    Args:
        clearances (dict): Dictionary of clearance information. In the form of {"Clearance X": [inequality_1, ..., inequality_n]}
        xlim (tuple[int, int]): X-limits of environment. Defaults to (0, 600)
        xlim (tuple[int, int]): Y-limits of environment. Defaults to (0, 250)

    Returns:
        NDArray: Binary grid of the discretized environment. Clearances are marked with a 1.
    """

    # Create the grid
    xlim = (0, 600)
    ylim = (0, 250)

    # Initialize grid as empty space
    grid = np.ones((ylim[1], xlim[1]))

    # Loop through x-values
    for x in range(xlim[0], xlim[1]):

        # Loop through y-values
        for y in range(ylim[0], ylim[1]):

            # Initialze clearance state as false
            inside_clearance = False

            # Loop through obstacle dictionary
            for clearance, constraints in clearances.items():

                # If location meets all constraints
                if all(constraint(x, y) for constraint in constraints):

                    # Mark location as obstacle and break loop
                    inside_clearance = True
                    break

            # If location is an clearance
            if inside_clearance:
                
                # Label its location with a 0
                grid[y, x] = 0

    # Return binary discretized environment
    return grid.astype(np.uint8)

def get_point(loc,obstacle_arr):
    while True:
        user_input = input(f"Enter position and orientation for {loc} separated by commas, in format x,y,theta (x from 1 to 600, y from 1 to 250, theta as -60, -30, 0, 30, or 60): ").strip()
        if user_input == "" and loc == "start":
            return (6,6,0)
        elif user_input == "" and loc == "goal":
            return (590,240,0)
        parts = user_input.split(",")
        if len(parts) == 3:
            try:
                x = int(parts[0].strip())
                y = int(parts[1].strip())
                theta = int(parts[2].strip())
                if 1<=x<=600 and 1<=y<=250 and theta in [-60,-30,0,30,60]:
                    x = x-1
                    y = y-1
                    if obstacle_arr[x,y]:
                        return (x,y,theta)
                    else:
                        print("Sorry this point is within the obstacle space. Try again.")
                else:
                    print("Invalid input. Please ensure both x and y are within the bounds of the space and theta is in [-60,-30,0,30,60].")
            except ValueError:
                print("Invalid input. Please enter integers for x, y, and theta.")
        else:
            print("Invalid input. Please enter exactly three integers separated by a comma.")

# Define main execution

def main():

    # Define obstacles
    obstacles: Dict[str, List[Callable[[Union[int, float], Union[int, float]], bool]]] = {

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
    clearance_arr = discretize(clearances)
    start_point = get_point("start",clearance_arr)
    end_point = get_point("goal",clearance_arr)
    visualize_environment(obstacles, clearances)
    


# Execute the script

if __name__ == "__main__":
    main()