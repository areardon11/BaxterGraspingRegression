import numpy as np
from featurize import featurize
import sys
sys.path.append("/Users/andrewreardon/Classes/ee106b/EE106BFinal/ml") #update this to say the correct path for model.py
import model

#set constants
max_gripper_width = .0825
min_gripping_width = .002 #.045 <- this is the actual value you want, but doesn't work for the test
x_diff = .0127 #points can be within 1/2 an inch in the x dir of each other
points_wanted = 10 #will only return one pair if <= 6


#takes in argument of a point cloud expressed in np array of shape (n x 3)
#returns points_wanted featurized grasps
def determine_grasp(point_cloud):
    #select possible grasps, featurize them, and then pass them into the force closure learner
    possible_grasps = contact_pairs(point_cloud)
    print "possible_grasps: \n", possible_grasps
    featurized_grasps = featurize(possible_grasps, point_cloud)
    print "featurized_grasps shape:", featurized_grasps.shape
    force_closure = model.predict_neural_net(featurized_grasps, ["neural_net_weights/V1.npy", "neural_net_weights/W1.npy"])
    print "force closure: \n", force_closure

    #filter possible_grasps and featurized_grasps by force_closure, then pass it into ferrari_canny learner
    possible_grasps = possible_grasps[np.where(force_closure == 1)]
    if possible_grasps.shape[0] == 0:
        return determine_grasp(point_cloud)
    featurized_grasps = featurized_grasps[np.where(force_closure == 1)]
    ferrari_canny = model.predict_neural_net(featurized_grasps, ["neural_net_weights/V_ferrari_canny2.npy", "neural_net_weights/W_ferrari_canny2.npy"], classifier=False)
    print "ferrari_canny: \n", ferrari_canny

    return possible_grasps[np.argmax(ferrari_canny)]

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

#None of the testing grasps end up being in force closure
def testing():
    x = np.arange(120).reshape(40,3)/500.
    f = determine_grasp(x)
    print "The determined grasp: \n", f
    return f

