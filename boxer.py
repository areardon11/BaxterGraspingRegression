import numpy as np
from IPython import embed

def boxer(pc, bounds):
    potentials = []
    for i in range(3):
        print(i)
        pl = np.where(pc[:,i] >= bounds[0][0])[0]
        ph = np.where(pc[:,i] <= bounds[0][1])[0]
        p = np.intersect1d(pl, ph)
        potentials.append(p)
    p = np.intersect1d(potentials[0], potentials[1])
    p = np.intersect1d(p, potentials[2])
    return pc[p]

pc = np.random.normal(0.0, .1, (1000,3))
bounds = [[-.1,.1], [-.1,.1], [-.1,.1]]
fpc = boxer(pc, bounds)
