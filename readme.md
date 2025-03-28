# ENPM661: Project 3 Phase 1 | Riley Albert, Adam Del Colliano, Joseph Shaheen

## Libaries and Dependencies Used
- `numpy`: For mathematical operations.
- `heapq`: For operations with A* algorithm queue.
- `cv2`: For visualizing the environment, search process, and path.
- `time`: For timing the search algorithm.
- `typing`: For type hinting.
- `numpy.typing`: For type hinting.

## Executing Script
- Upon executing `a_star_albert_delcolliano_shaheen.py`, the user will be asked to input the start pose for the path. The input is bounded between 1-600 on the x-axis, 1-250 on the y-axis, and 0 - 330 in increments of 30 for the orientation in degrees.
- Once the start pose is successfully entered, the user will be prompted to enter the goal pose with the same conditions.
- After successfully entering both start and end poses, the A* search algorithm will begin finding the optimal path. Once located, the terminal will output the time to execute the search. Additionally, a window will open, animating the algorithm's search process. Once the animation is complete, the optimal path will be highlighted.

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