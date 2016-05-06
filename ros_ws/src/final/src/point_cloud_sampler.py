import numpy as np
from featurize import window
from featurize import moment_arm

#set constants
max_gripper_width = .0825
min_gripping_width = .002 #.045 <- this is the actual value you want, but doesn't work for the test
x_diff = .0127 #points can be within 1/2 an inch in the x dir of each other
points_wanted = 7 #will only return one pair if <= 6


#takes in argument of a point cloud expressed in np array of shape (n x 3)
#returns points_wanted featurized grasps
def determine_grasp(point_cloud):
    possible_grasps = contact_pairs(point_cloud)
    print "possible_grasps: \n", possible_grasps
    featurized_grasps = featurize_pairs(possible_grasps, point_cloud)
    print "featurized_grasps shapes: \n", featurized_grasps.shape
    return featurized_grasps

#takes contact pairs and featurizes them in the window and moment arm format
def featurize_pairs(pairs_arr, pc):
    def featurize_pair(pair):
        p1, p2 = pair[:3], pair[3:]
        return np.hstack((window(p1,(p2-p1)/np.linalg.norm(p2-p1),pc), window(p2,(p1-p2)/np.linalg.norm(p2-p1),pc), moment_arm(p1,pc), moment_arm(p2,pc)))

    if len(pairs_arr.shape) == 1:
        return featurize_pair(pairs_arr)

    fp = []
    for x in range(pairs_arr.shape[0]):
        fp.append(featurize_pair(pairs_arr[x]))
    return np.asarray(fp)

def num_possible_connections(n):
    total = 0
    for i in range(n):
        total += i
    return n*(n-1)-total

#finds valid pairs of contact points and returns featurized versions
def contact_pairs(pc):
    tried_pairs = set()
    ret_points = None
    npc = num_possible_connections(pc.shape[0])
    while len(tried_pairs) < npc:
        #generate new point indices
        indices = np.random.choice(pc.shape[0], 2, replace=False)
        i, j = indices[0], indices[1]
        if (i,j) in tried_pairs or (j,i) in tried_pairs:
            continue
        tried_pairs.add((i,j))

        #filter points by distance
        p1, p2 = pc[i], pc[j]
        d = np.linalg.norm(p1-p2)
        if d > max_gripper_width or d < min_gripping_width:
            continue
        #filter by x value
        #TODO: remove this to generalize
        if abs(p1[0]-p2[0]) > x_diff:
            continue
        points = np.hstack((p1,p2))
        if ret_points is None:
            ret_points = points
        else:
            ret_points = np.row_stack((ret_points, points))
        if ret_points.shape[0] >= points_wanted:
            return ret_points
    print "tried to generate the asked possible pairs and failed"
    return ret_points

def testing():
    x = np.arange(120).reshape(40,3)/500.
    print determine_grasp(x)


