# ENPM661: Project 3 Phase 2 | Riley Albert, Adam Del Colliano, Joseph Shaheen

## Libaries and Dependencies Used
- `numpy`: For mathematical operations.
- `heapq`: For operations with A* algorithm queue.
- `cv2`: For visualizing the environment, search process, and path.
- `time`: For timing the search algorithm.
- `typing`: For type hinting.
- `numpy.typing`: For type hinting.
- `matplotlib.pyplot`: For verifying action set.

## Executing Script
When executing the script the user will be promped several times for the items below. To use default values simply click 'Enter' for each prompt:
- Clearance: thickness of obstacles in addition to their actual boundaries. Should be greater than 0 -- Default: 1 (mm)
- Start Pose: x and y coordinates followed by angular heading in 30 degree increments -- Default: 400,400,0 (mm,mm,deg)
- Goal: x and y coordinates of goal region center, which is a circle of radius 100 mm -- Default: 1500,500 (mm,mm)
- Wheel RPMS: two differnet values to represent the RPM of the robot wheels defining the action set -- Default 50,100 (RPMS)

After all prompts are completed a summary of parameters will be printed to the screen after which the user will need to press 'Enter' to begin the path planning algorithm. Every 1000 nodes searched a progress time stamp will be printed to relay that the algorithm is still running.

In both cases, where a path is found or not, a window will pop up displaying the actions taken by the algorithm to search for the path from start to goal. If a path is found, it will be highlighted in blue in the end, otherwise, only the actions will appear. In the terminal it will print whehter or not a path is found, followed by search time and animation viewing progress.


## Team Members
- Riley Albert:
    - Directory ID: ralbert8
    - UID: 120985195

- Adam Del Colliano:
    - Directory ID: adelcoll
    - UID: 115846982

- Joseph Shaheen:
    - Directory ID: jshaheen
    - UID: 116534321

## Repository Link
https://github.com/JMechE13/ENPM661
NOTE: This is for Phase 2, so make sure to look at that folder.