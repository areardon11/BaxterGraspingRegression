import numpy as np
import obj_file
import transformations
from numpy.linalg import norm
import pickle

SPRAY_BOTTLE_MESH_FILENAME = 'data/spray.obj'
new_points = [\
    ([-0.002968,  -0.014356, 0.082942],[-0.004784, 0.010934, 0.083253]),
    ([0.002201, 0.015478, 0.041976], [0.002619, -0.01669, 0.041517]),
    ([-0.004857, -0.015731, 0.016332], [-0.005288, 0.017409, 0.016805]),
    ([0.015356, 0.015923, 0.021635], [0.015765, -0.015494, 0.021186]),
    ([0.033505, -0.015957, -0.011488], [0.032806, 0.016866, -0.00687])    
]
def force_closure(contacts, normals, mu=.5):
    c1, c2 = contacts[:,0].flatten(), contacts[:,1].flatten()
    n1, n2 = -normals[:,0].flatten(), -normals[:,1].flatten()
    v = c2 - c1
    theta1 = abs(np.arccos(v.dot(n1) / (norm(n1) * norm(v))))
    v = c1 - c2
    theta2 = abs(np.arccos(v.dot(n2) / (norm(n2) * norm(v))))
    alpha = abs(np.arctan(mu))
    return int(theta1 <= alpha and theta2 <= alpha)

def find(vertices, v):
    i = 0
    for vertex in vertices:
        if vertex[0] == v[0] and vertex[1] == v[1] and vertex[2] == v[2]:
            return i
        i += 1
    return -1

def find_candidates(vertices, normals):
    candidates = []
    new_points = [\
        ([-0.002968,  -0.014356, 0.082942],[-0.004784, 0.010934, 0.083253]),
        ([0.002201, 0.015478, 0.041976], [0.002619, -0.01669, 0.041517]),
        ([-0.004857, -0.015731, 0.016332], [-0.005288, 0.017409, 0.016805]),
        ([0.015356, 0.015923, 0.021635], [0.015765, -0.015494, 0.021186]),
        ([0.033505, -0.015957, -0.011488], [0.032806, 0.016866, -0.00687])    
    ]
    return new_points

def contacts_to_baxter_hand_pose(contact1, contact2):
    c1 = np.array(contact1)
    c2 = np.array(contact2)

    # compute gripper center and axis
    center = 0.5 * (c1 + c2)
    y_axis = c2 - c1
    y_axis = y_axis / np.linalg.norm(y_axis)
    z_axis = np.array([y_axis[1], -y_axis[0], 0]) # the z axis will always be in the table plane for now
    z_axis = z_axis / np.linalg.norm(z_axis)
    x_axis = np.cross(y_axis, z_axis)

    # convert to hand pose
    R_obj_gripper = np.array([x_axis, y_axis, z_axis]).T
    t_obj_gripper = center
    T_obj_gripper = np.eye(4)
    T_obj_gripper[:3,:3] = R_obj_gripper
    T_obj_gripper[:3,3] = t_obj_gripper
    q_obj_gripper = transformations.quaternion_from_matrix(T_obj_gripper)

    return t_obj_gripper, q_obj_gripper 

if __name__ == '__main__':
    of = obj_file.ObjFile(SPRAY_BOTTLE_MESH_FILENAME)
    mesh = of.read()

    vertices = list(map(list, mesh.vertices))
    triangles = mesh.triangles
    normals = list(map(list,mesh.normals))

    print('Num vertices: ' + `len(vertices)`)
    print('Num triangles: '+ `len(triangles)`)
    print('Num normals: '+ `len(normals)`)

    # 1. Generate candidate pairs of contact points
    c = find_candidates(vertices, normals, potentials)
    # 2. Check for force closure
        # Pick a few points at random and see if Baxter likes those 
    # 3. Convert each grasp to a hand pose
    contact1 = vertices[0]
    contact2 = vertices[100]
    t_obj_gripper, q_obj_gripper = contacts_to_baxter_hand_pose(contact1, contact2)
    print('Translation: ' + `t_obj_gripper`)
    print('Rotation: '+ `q_obj_gripper`)

    # 4. Execute on the actual robot
