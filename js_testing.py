

# testing action set implementation
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


# class to represent each node's information
class Node():

    # initialze attributes
    def __init__(self, location, angle):

        self.location = location
        self.angle = angle

        # length of arrow for visualization 
        self.arrow_rep_length = 0.2

        # move distance of each action
        self.move_distance = 0.5

        self.format_angle()

    # return three values, position x y and orientation in degrees of node
    def get_pose(self):

        return self.location[0], self.location[1], self.angle

    # return node's lcoation
    def get_location(self):
        
        return self.location
    
    # return node's angle
    def get_angle(self):

        return self.angle
    
    # return 4 points representing start and end coords for an arrow visualization of the node
    def get_arrow_rep(self):

        x_end = self.arrow_rep_length*np.cos(np.deg2rad(self.angle))
        y_end = self.arrow_rep_length*np.sin(np.deg2rad(self.angle))
        

        return self.location[0], self.location[1], x_end, y_end
    
    # perform first action on node: move at +60 degree
    def action_1(self):
        
        delta_angle = 60
        self.angle += delta_angle

        self.move()

    # perform second action on node: move at +30 degree
    def action_2(self):

        delta_angle = 30
        self.angle += delta_angle

        self.move()

    # perform third action on node: move at 0 degree
    def action_3(self):

        self.move()

    # perform fourth action on node: move at -30 degree
    def action_4(self):

        delta_angle = -30
        self.angle += delta_angle

        self.move()

    # perform fifth action on node: move at -60 degree
    def action_5(self):

        delta_angle = -60
        self.angle += delta_angle

        self.move()
    
    # move distance in current direction
    def move(self):

        x0 = self.location[0]
        y0 = self.location[1]
        
        x_d = self.move_distance*np.cos(np.deg2rad(self.angle))
        y_d = self.move_distance*np.sin(np.deg2rad(self.angle))

        self.location = (x0 + x_d, y0 + y_d)

        self.format_angle()

    # defining rotatio: 0 along horizontal, CCW 0-> 180, CW 0-> -180 -- total domain (-180, 180)
    def format_angle(self):

        if self.angle > 180:
            self.angle =  -180 + (self.angle - 180)

        elif self.angle <= -180:
            self.angle = 180 - (self.angle + 180)

    # print node
    def __str__(self):

        return f'Position ({self.location[0]}, {self.location[1]}) mm -- Angle: {self.angle} deg'

# class to represent 3D configuration map
class CMap():

    def __init__(self):

        # threshold for x and y are 0.5 mms
        self.x_dim = 600*2              
        self.y_dim = 250*2

        # threshold for angles is 30 -> 360/30 = 12
        self.z_dim = 12

        self.cmap_dims = (self.x_dim, self.y_dim, self.z_dim)

        self.cmap = np.zeros(self.cmap_dims, dtype=np.uint8)


    # check the cmap for if node already exists
    def check(self, Node):

        x, y, angle = Node.get_pose()

        x_idx, y_idx, angle_idx = self.to_cmap_frame(x, y, angle)

        if self.cmap[x_idx, y_idx, angle_idx] == 0:

            # node config did not already exist, add to cmap -- return 1 as valid
            self.cmap[x_idx, y_idx, angle_idx] = 1
            return 1

        else:

            # node config already exists -- return 0 as invalid
            return 0

    # convert pose to cmap representation
    def to_cmap_frame(self, x, y, angle):

        x_idx  = int(np.round(x*2))
        y_idx = int(np.round(y*2))
        

        angle_idx = int(np.ceil(angle/30.0 + 6) - 1)

        #print('angle: ', angle, ' cmap idx: ', angle_idx)

        return x_idx, y_idx, angle_idx


def main():
    

    # initialize plot, cmap, and node testers
    plt.figure()
    
    cmap = CMap()

    # node definition, (x,y) and angle in degrees CCW + from horizontal
    vec = Node((1,1), 10)

    # testing actions, arrow visualizations and cmap checking
    print('\nInitial Pose Vector:')
    print(vec)
    print(vec.get_pose())
    arrow = vec.get_arrow_rep()
    plt.arrow(arrow[0], arrow[1], arrow[2], arrow[3], head_width = 0.1, head_length=0.1, fc='blue', ec='blue')

    vec.action_4()
    print('\nAfter action:')
    print(vec)
    print(vec.get_pose())
    arrow = vec.get_arrow_rep()
    plt.arrow(arrow[0], arrow[1], arrow[2], arrow[3], head_width = 0.1, head_length=0.1, fc='blue', ec='blue')


    # for i in range(-180, 180):
    #     vec = Node((1,1,), i)
    #     cmap.check(vec)

    print('\nCmap checks: (first accepted - 1, second failed - 0)')
    print(cmap.check(vec))   # returns 1 -- node did not exist, therefore check passes and is valid
    print(cmap.check(vec))   # return 0 -- node exists already, therefore fails check and is invalid





    plt.xlabel('x')
    plt.xlim(0,5)
    plt.ylabel('y')
    plt.ylim(0,5)
    plt.title('test')

    plt.show()

    return 0


if __name__ == '__main__':
    main()