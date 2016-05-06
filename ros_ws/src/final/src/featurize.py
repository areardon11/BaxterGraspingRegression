import numpy as np
import numpy.linalg as la
from numpy.random import normal
from IPython import embed

#takes contact pairs and featurizes them in the window and moment arm format
def featurize(pairs_arr, pc, ma=False, w=.05, gs=15):
    """
    Returns a featurized array based on the pairs of points passed in using the below functions
    'pairs_arr' is (n x 6) np array where each row contains the two contact points
    'pc' is the point cloud as a np array (n x 3)
    'ma' is the argument for whether you want the full moment arms in the featurization or just the magnitude
    'w' and 'gs' are the window arguments
    """
    def featurize_pair(pair, full_ma):
        p1, p2 = pair[:3], pair[3:]
        if full_ma:
            return np.hstack((window(p1,(p2-p1)/la.norm(p2-p1),pc,w=w,gs=gs), window(p2,(p1-p2)/la.norm(p2-p1),pc,w=w,gs=gs), moment_arm(p1,pc), moment_arm(p2,pc)))
        return np.hstack((window(p1,(p2-p1)/la.norm(p2-p1),pc,w=w,gs=gs), window(p2,(p1-p2)/la.norm(p2-p1),pc,w=w,gs=gs), la.norm(moment_arm(p1,pc)), la.norm(moment_arm(p2,pc))))

    if len(pairs_arr.shape) == 1:
        return featurize_pair(pairs_arr, ma)

    fp = []
    for x in range(pairs_arr.shape[0]):
        fp.append(featurize_pair(pairs_arr[x], ma))
    return np.asarray(fp)

def window(c, n, pc, w=.05, gs=15):
    """
    Generates the window, around contact `c`,
    given point cloud `pc` and normal `n` (assumes normal is unit vector pointing toward other contact)
    `w` is in units of m
    `gs` is the grid size of the window
    """
    c = np.array(c)
    
    # Create plane
    n = n.tolist()
    U,D,V = la.svd([n])
    u, v = V[:, 1], V[:,2]
    
    n = np.array(n)
    
    # Create 3D to 2D mapping
    three_to_two = np.vstack((u,v))
    delta = w / float(gs)
    grid = np.zeros((15,15)) + .2
    # print("Delta is " + str(delta))
    for p in pc:
        
        # Project point onto plane
        p = np.array(p)
        a = p - c
        d = n.dot(a)
        if d < 0:
            sign = -1
        else:
            sign = 1
        p_hat = p - d*n
        
        # Reject points outside our window of size w
        p_2d = three_to_two.dot(p_hat - c)
        if abs(p_2d[0]) > w/2. or np.abs(p_2d[1]) > w/2.:
            continue
        
        # Create projection window using minimum depths
        coord = np.floor(p_2d / delta).astype(int)
        coord = (coord[0] + 7, -coord[1] + 7)
        residual = la.norm(p_hat - p)
        if residual < abs(grid[coord]):
            grid[coord] = sign * residual
                
    return grid.flatten()

def moment_arm(c, pc):
    o = np.average(pc, 0)
    return o - c

def test():
    c = normal(0.0, 0.001,size=3)
    n = normal(0.0, 0.001,size=3)
    n /= la.norm(n)
    pc = normal(0.0, 0.001, size=(100,3))
    print("contact: " + str(c))
    print("normal: "  + str(n))
    grid = window(c,n,pc)
    return grid
    

if __name__ == "__main__":
    test()
