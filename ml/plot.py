import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pickle

def preprocess(f, points):
    with open(f, 'rb') as f:
        data = pickle.load(f)
    data = np.vstack(data)

    nan = np.isnan(data)
    idx = np.where(nan != True)[0]
    idx = np.unique(idx)
    data = data[idx]
    idx = np.random.randint(0, data.shape[0], points)
    return data[idx]

def plot(data, c, ax):
    ax.scatter(data[:,0], data[:,1], data[:,2], c=c, marker='.')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('points', type=int)
    args = parser.parse_args()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    files = [('kinect2_pc2_read', 'b'), ('kinect1_pc2_read', 'r')]
    for f in files: 
        d = preprocess(f[0], args.points)
        plot(d,f[1], ax)
    plt.show()
